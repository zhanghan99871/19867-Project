import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


class Optimizer:
    def __init__(self, network, name):
        self.name = name 
        self.network = network
        self.model = gp.Model(name) if name is not None else gp.Model()

    def optimize(self, start, end, travel_reduce_ratio, time_limit=None, mip_gap=0.01):
        raise NotImplementedError("The optimize method must be implemented by the subclass.")

    def build_optimized_flow_matrices(self, result, start, end):
        raise NotImplementedError("The build_optimized_flow_matrices method must be implemented by the subclass.")

    def update_network_flow(self, optimized_flow):
        self.network.flow_matrix_dict = optimized_flow
        self.network.model.flow_matrix_dict = optimized_flow


class BinaryReduceEdgeOptimizer(Optimizer):
    def __init__(self, network, name="edge_reduce_binary"):
        super().__init__(network, name)

    def optimize(self, start, end, travel_reduce_ratio=0.2, time_limit=None, mip_gap=0.01):
        if not self.network.fited:
            raise ValueError("Network model must be fitted before optimization.")
        if not 0 <= travel_reduce_ratio <= 1:
            raise ValueError("travel_reduce_ratio must be between 0 and 1.")

        net, model = self.network, self.model
        start, end = int(start), int(end)
        n, T = int(net.num_nodes), end - start + 1
        N = np.asarray(net.N, dtype=float)
        beta, gamma, beta_travel = net.model.get()

        dates = pd.to_datetime(net.nodes[0].case_data["date"].iloc[start:end + 1])
        flow_keys = [pd.Timestamp(d).normalize() for d in dates.iloc[:-1]]
        F = [net.flow_matrix_dict[key] for key in flow_keys]

        initial_state = net.initial_state_from_data(start)
        initial_frac = initial_state / N[:, None]

        edges = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if sum(F[t][i, j] for t in range(T - 1)) > 0:
                    edges.append((i, j))

        print("number of travel edges:", len(edges))

        S = model.addVars(n, T, lb=0.0, ub=1.0, name="S")
        I = model.addVars(n, T, lb=0.0, ub=1.0, name="I")
        R = model.addVars(n, T, lb=0.0, ub=1.0, name="R")
        keep = model.addVars(edges, vtype=GRB.BINARY, name="keep")

        W = {}
        for i, j in edges:
            for t in range(T - 1):
                W[i, j, t] = model.addVar(lb=0.0, ub=1.0, name=f"W_{i}_{j}_{t}")

        model.update()

        for i in range(n):
            model.addConstr(S[i, 0] == initial_frac[i, 0])
            model.addConstr(I[i, 0] == initial_frac[i, 1])
            model.addConstr(R[i, 0] == initial_frac[i, 2])

        # Exact linearization for W = keep * I when keep is binary and 0 <= I <= 1.
        for i, j in edges:
            for t in range(T - 1):
                w, y = W[i, j, t], keep[i, j]
                model.addConstr(w <= I[j, t])
                model.addConstr(w <= y)
                model.addConstr(w >= I[j, t] - (1 - y))

        outgoing = {i: [] for i in range(n)}
        for i, j in edges:
            outgoing[i].append(j)

        for t in range(T - 1):
            Ft = F[t]
            for i in range(n):
                local_infection = beta[i] * S[i, t] * I[i, t]
                travel_infection = gp.QuadExpr()
                for j in outgoing[i]:
                    flow = Ft[i, j]
                    if flow == 0:
                        continue
                    coeff = beta_travel[i] * flow / N[i]
                    travel_infection += coeff * S[i, t] * W[i, j, t]

                infection = local_infection + travel_infection
                recovery = gamma[i] * I[i, t]

                model.addConstr(S[i, t + 1] == S[i, t] - infection)
                model.addConstr(I[i, t + 1] == I[i, t] + infection - recovery)
                model.addConstr(R[i, t + 1] == R[i, t] + recovery)
                model.addConstr(S[i, t + 1] + I[i, t + 1] + R[i, t + 1] == 1.0)

        edge_flow = {(i, j): sum(F[t][i, j] for t in range(T - 1)) for i, j in edges}
        total_travel = sum(edge_flow.values())
        M = travel_reduce_ratio * total_travel

        model.addConstr(
            gp.quicksum(edge_flow[i, j] * (1 - keep[i, j]) for i, j in edges) <= M,
            name="travel_budget",
        )

        model.setObjective(
            gp.quicksum(N[i] * (1 - S[i, T - 1]) for i in range(n)),
            GRB.MINIMIZE,
        )

        model.Params.NonConvex = 2
        model.Params.MIPGap = mip_gap
        if time_limit is not None:
            model.Params.TimeLimit = time_limit

        model.optimize()

        if model.SolCount == 0:
            print("No feasible solution found.")
            return None

        disabled_edges = []
        for i, j in edges:
            if keep[i, j].X < 0.5:
                disabled_edges.append((i, j, edge_flow[i, j]))

        total_removed_flow = sum(flow for _, _, flow in disabled_edges)

        print("objective:", model.ObjVal)
        print("total travel:", total_travel)
        print("allowed reduction:", M)
        print("removed travel:", total_removed_flow)
        print("disabled edges:")
        for i, j, flow in disabled_edges:
            print(net.node_ids[i], "->", net.node_ids[j], "flow:", flow)

        return {
            "objective": model.ObjVal,
            "disabled_edges": disabled_edges,
            "removed_flow": total_removed_flow,
            "total_travel": total_travel,
            "S": np.array([[S[i, t].X * N[i] for t in range(T)] for i in range(n)]),
            "I": np.array([[I[i, t].X * N[i] for t in range(T)] for i in range(n)]),
            "R": np.array([[R[i, t].X * N[i] for t in range(T)] for i in range(n)]),
        }

    def build_optimized_flow_matrices(self, result, start, end):
        net = self.network
        optimized_flow = {date: F.copy() for date, F in net.flow_matrix_dict.items()}
        dates = pd.to_datetime(net.nodes[0].case_data["date"].iloc[int(start):int(end)])
        flow_keys = [pd.Timestamp(d).normalize() for d in dates]

        for i, j, _ in result["disabled_edges"]:
            for date in flow_keys:
                optimized_flow[date][i, j] = 0.0

        return optimized_flow


class ContinuousReduceEdgeOptimizer(Optimizer):
    def __init__(self, network, name="edge_reduce_continuous"):
        super().__init__(network, name)

    def optimize(self, start, end, travel_reduce_ratio=0.2, time_limit=None, mip_gap=0.01):
        if not self.network.fited:
            raise ValueError("Network model must be fitted before optimization.")
        if not 0 <= travel_reduce_ratio <= 1:
            raise ValueError("travel_reduce_ratio must be between 0 and 1.")

        net, model = self.network, self.model
        start, end = int(start), int(end)
        n, T = int(net.num_nodes), end - start + 1
        N = np.asarray(net.N, dtype=float)
        beta, gamma, beta_travel = net.model.get()

        dates = pd.to_datetime(net.nodes[0].case_data["date"].iloc[start:end + 1])
        flow_keys = [pd.Timestamp(d).normalize() for d in dates.iloc[:-1]]
        F = [net.flow_matrix_dict[key] for key in flow_keys]

        initial_state = net.initial_state_from_data(start)
        initial_frac = initial_state / N[:, None]

        edges = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if sum(F[t][i, j] for t in range(T - 1)) > 0:
                    edges.append((i, j))

        print("number of travel edges:", len(edges))

        S = model.addVars(n, T, lb=0.0, ub=1.0, name="S")
        I = model.addVars(n, T, lb=0.0, ub=1.0, name="I")
        R = model.addVars(n, T, lb=0.0, ub=1.0, name="R")
        keep = model.addVars(edges, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="keep")

        W = {}
        for i, j in edges:
            for t in range(T - 1):
                W[i, j, t] = model.addVar(lb=0.0, ub=1.0, name=f"W_{i}_{j}_{t}")

        model.update()

        for i in range(n):
            model.addConstr(S[i, 0] == initial_frac[i, 0])
            model.addConstr(I[i, 0] == initial_frac[i, 1])
            model.addConstr(R[i, 0] == initial_frac[i, 2])

        # Bilinear equality W = keep * I.
        for i, j in edges:
            for t in range(T - 1):
                model.addQConstr(W[i, j, t] == keep[i, j] * I[j, t])

        outgoing = {i: [] for i in range(n)}
        for i, j in edges:
            outgoing[i].append(j)

        for t in range(T - 1):
            Ft = F[t]
            for i in range(n):
                local_infection = beta[i] * S[i, t] * I[i, t]
                travel_infection = gp.QuadExpr()
                for j in outgoing[i]:
                    flow = Ft[i, j]
                    if flow == 0:
                        continue
                    coeff = beta_travel[i] * flow / N[i]
                    travel_infection += coeff * S[i, t] * W[i, j, t]

                infection = local_infection + travel_infection
                recovery = gamma[i] * I[i, t]

                model.addConstr(S[i, t + 1] == S[i, t] - infection)
                model.addConstr(I[i, t + 1] == I[i, t] + infection - recovery)
                model.addConstr(R[i, t + 1] == R[i, t] + recovery)
                model.addConstr(S[i, t + 1] + I[i, t + 1] + R[i, t + 1] == 1.0)

        edge_flow = {(i, j): sum(F[t][i, j] for t in range(T - 1)) for i, j in edges}
        total_travel = sum(edge_flow.values())
        M = travel_reduce_ratio * total_travel

        model.addConstr(
            gp.quicksum(edge_flow[i, j] * (1 - keep[i, j]) for i, j in edges) <= M,
            name="travel_budget",
        )

        model.setObjective(
            gp.quicksum(N[i] * (1 - S[i, T - 1]) for i in range(n)),
            GRB.MINIMIZE,
        )

        model.Params.NonConvex = 2
        model.Params.MIPGap = mip_gap
        if time_limit is not None:
            model.Params.TimeLimit = time_limit

        model.optimize()

        if model.SolCount == 0:
            print("No feasible solution found.")
            return None

        reduced_edges = []
        for i, j in edges:
            keep_ratio = keep[i, j].X
            reduce_ratio = 1.0 - keep_ratio
            reduced_flow = edge_flow[i, j] * reduce_ratio
            if reduce_ratio > 1e-6:
                reduced_edges.append((i, j, keep_ratio, reduce_ratio, reduced_flow))

        total_removed_flow = sum(x[4] for x in reduced_edges)

        print("objective:", model.ObjVal)
        print("total travel:", total_travel)
        print("allowed reduction:", M)
        print("actual reduced travel:", total_removed_flow)
        print("reduced edges:")
        for i, j, keep_ratio, reduce_ratio, reduced_flow in reduced_edges:
            print(net.node_ids[i], "->", net.node_ids[j],
                  f"keep={keep_ratio:.4f}",
                  f"reduce={reduce_ratio:.4f}",
                  f"reduced_flow={reduced_flow:.2f}")

        return {
            "objective": model.ObjVal,
            "reduced_edges": reduced_edges,
            "removed_flow": total_removed_flow,
            "total_travel": total_travel,
            "S": np.array([[S[i, t].X * N[i] for t in range(T)] for i in range(n)]),
            "I": np.array([[I[i, t].X * N[i] for t in range(T)] for i in range(n)]),
            "R": np.array([[R[i, t].X * N[i] for t in range(T)] for i in range(n)]),
        }

    def build_optimized_flow_matrices(self, result, start, end):
        net = self.network
        optimized_flow = {date: F.copy() for date, F in net.flow_matrix_dict.items()}
        dates = pd.to_datetime(net.nodes[0].case_data["date"].iloc[int(start):int(end)])
        flow_keys = [pd.Timestamp(d).normalize() for d in dates]

        for i, j, keep_ratio, _, _ in result["reduced_edges"]:
            for date in flow_keys:
                optimized_flow[date][i, j] *= keep_ratio

        return optimized_flow

class BinaryReduceNodeOptimizer(Optimizer):
    def __init__(self, network, name="node_reduce_binary"):
        super().__init__(network, name)

    def optimize(self, start, end, travel_reduce_ratio=0.2, time_limit=None, mip_gap=0.01):
        if not self.network.fited:
            raise ValueError("Network model must be fitted before optimization.")
        if not 0 <= travel_reduce_ratio <= 1:
            raise ValueError("travel_reduce_ratio must be between 0 and 1.")

        net, model = self.network, self.model
        start, end = int(start), int(end)
        n, T = int(net.num_nodes), end - start + 1
        N = np.asarray(net.N, dtype=float)
        beta, gamma, beta_travel = net.model.get()

        dates = pd.to_datetime(net.nodes[0].case_data["date"].iloc[start:end + 1])
        flow_keys = [pd.Timestamp(d).normalize() for d in dates.iloc[:-1]]
        F = [net.flow_matrix_dict[key] for key in flow_keys]

        initial_state = net.initial_state_from_data(start)
        initial_frac = initial_state / N[:, None]

        edges = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if sum(F[t][i, j] for t in range(T - 1)) > 0:
                    edges.append((i, j))

        print("number of nodes:", n)
        print("number of travel edges:", len(edges))

        S = model.addVars(n, T, lb=0.0, ub=1.0, name="S")
        I = model.addVars(n, T, lb=0.0, ub=1.0, name="I")
        R = model.addVars(n, T, lb=0.0, ub=1.0, name="R")

        # active[i] = 1: node keeps travel
        # active[i] = 0: disable all travel related to node i
        active = model.addVars(n, vtype=GRB.BINARY, name="active")

        # keep[i,j] = active[i] AND active[j]
        keep = model.addVars(edges, vtype=GRB.BINARY, name="keep")

        # W[i,j,t] = keep[i,j] * I[j,t]
        W = {}
        for i, j in edges:
            for t in range(T - 1):
                W[i, j, t] = model.addVar(lb=0.0, ub=1.0, name=f"W_{i}_{j}_{t}")

        model.update()

        for i in range(n):
            model.addConstr(S[i, 0] == initial_frac[i, 0])
            model.addConstr(I[i, 0] == initial_frac[i, 1])
            model.addConstr(R[i, 0] == initial_frac[i, 2])

        # keep[i,j] = active[i] AND active[j]
        for i, j in edges:
            model.addConstr(keep[i, j] <= active[i])
            model.addConstr(keep[i, j] <= active[j])
            model.addConstr(keep[i, j] >= active[i] + active[j] - 1)

        # W = keep * I
        for i, j in edges:
            for t in range(T - 1):
                w, y = W[i, j, t], keep[i, j]
                model.addConstr(w <= I[j, t])
                model.addConstr(w <= y)
                model.addConstr(w >= I[j, t] - (1 - y))

        outgoing = {i: [] for i in range(n)}
        for i, j in edges:
            outgoing[i].append(j)

        # Normalized SIR dynamics
        for t in range(T - 1):
            Ft = F[t]

            for i in range(n):
                local_infection = beta[i] * S[i, t] * I[i, t]
                travel_infection = gp.QuadExpr()

                for j in outgoing[i]:
                    flow = Ft[i, j]
                    if flow == 0:
                        continue

                    coeff = beta_travel[i] * flow / N[i]
                    travel_infection += coeff * S[i, t] * W[i, j, t]

                infection = local_infection + travel_infection
                recovery = gamma[i] * I[i, t]

                model.addConstr(S[i, t + 1] == S[i, t] - infection)
                model.addConstr(I[i, t + 1] == I[i, t] + infection - recovery)
                model.addConstr(R[i, t + 1] == R[i, t] + recovery)
                model.addConstr(S[i, t + 1] + I[i, t + 1] + R[i, t + 1] == 1.0)

        # Travel budget
        edge_flow = {
            (i, j): sum(F[t][i, j] for t in range(T - 1))
            for i, j in edges
        }

        total_travel = sum(edge_flow.values())
        M = travel_reduce_ratio * total_travel

        model.addConstr(
            gp.quicksum(
                edge_flow[i, j] * (1 - keep[i, j])
                for i, j in edges
            ) <= M,
            name="travel_budget",
        )

        # Minimize final cumulative infected population
        model.setObjective(
            gp.quicksum(
                N[i] * (1 - S[i, T - 1])
                for i in range(n)
            ),
            GRB.MINIMIZE,
        )

        model.Params.NonConvex = 2
        model.Params.MIPGap = mip_gap

        if time_limit is not None:
            model.Params.TimeLimit = time_limit

        model.optimize()

        if model.SolCount == 0:
            print("No feasible solution found.")
            return None

        disabled_nodes = [
            i for i in range(n)
            if active[i].X < 0.5
        ]

        disabled_edges = [
            (i, j, edge_flow[i, j])
            for i, j in edges
            if keep[i, j].X < 0.5
        ]

        total_removed_flow = sum(
            flow for _, _, flow in disabled_edges
        )

        print("objective:", model.ObjVal)
        print("total travel:", total_travel)
        print("allowed reduction:", M)
        print("removed travel:", total_removed_flow)

        print("disabled nodes:")
        for i in disabled_nodes:
            print(net.node_ids[i], net.nodes[i].name)

        print("disabled edges:")
        for i, j, flow in disabled_edges:
            print(
                net.node_ids[i],
                "->",
                net.node_ids[j],
                "flow:",
                flow,
            )

        return {
            "objective": model.ObjVal,
            "disabled_nodes": disabled_nodes,
            "disabled_edges": disabled_edges,
            "removed_flow": total_removed_flow,
            "total_travel": total_travel,
            "S": np.array([
                [S[i, t].X * N[i] for t in range(T)]
                for i in range(n)
            ]),
            "I": np.array([
                [I[i, t].X * N[i] for t in range(T)]
                for i in range(n)
            ]),
            "R": np.array([
                [R[i, t].X * N[i] for t in range(T)]
                for i in range(n)
            ]),
        }

    def build_optimized_flow_matrices(self, result, start, end):
        net = self.network

        optimized_flow = {
            date: F.copy()
            for date, F in net.flow_matrix_dict.items()
        }

        dates = pd.to_datetime(
            net.nodes[0].case_data["date"].iloc[int(start):int(end)]
        )

        flow_keys = [
            pd.Timestamp(d).normalize()
            for d in dates
        ]

        for i in result["disabled_nodes"]:
            for date in flow_keys:
                optimized_flow[date][i, :] = 0.0
                optimized_flow[date][:, i] = 0.0

        return optimized_flow
    
class BinaryClusterOptimizer(Optimizer):
    def __init__(self, network, name="cluster_binary"):
        super().__init__(network, name)

    def optimize(self, start, end, K=3, travel_reduce_ratio=0.2, time_limit=None, mip_gap=0.01):
        if not self.network.fited:
            raise ValueError("Network model must be fitted before optimization.")
        if not 0 <= travel_reduce_ratio <= 1:
            raise ValueError("travel_reduce_ratio must be between 0 and 1.")
        if K < 1:
            raise ValueError("K must be at least 1.")

        net, model = self.network, self.model
        start, end = int(start), int(end)
        n, T = int(net.num_nodes), end - start + 1
        K = min(int(K), n)

        N = np.asarray(net.N, dtype=float)
        beta, gamma, beta_travel = net.model.get()

        dates = pd.to_datetime(net.nodes[0].case_data["date"].iloc[start:end + 1])
        flow_keys = [pd.Timestamp(d).normalize() for d in dates.iloc[:-1]]
        F = [net.flow_matrix_dict[key] for key in flow_keys]

        initial_state = net.initial_state_from_data(start)
        initial_frac = initial_state / N[:, None]

        edges = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if sum(F[t][i, j] for t in range(T - 1)) > 0:
                    edges.append((i, j))

        print("number of nodes:", n)
        print("maximum clusters:", K)
        print("number of travel edges:", len(edges))

        # Normalized SIR state
        S = model.addVars(n, T, lb=0.0, ub=1.0, name="S")
        I = model.addVars(n, T, lb=0.0, ub=1.0, name="I")
        R = model.addVars(n, T, lb=0.0, ub=1.0, name="R")

        # assign[i,k] = 1 iff node i belongs to cluster k
        assign = model.addVars(n, K, vtype=GRB.BINARY, name="assign")

        # used[k] = 1 iff cluster k is used
        used = model.addVars(K, vtype=GRB.BINARY, name="used")

        # same[i,j,k] = assign[i,k] AND assign[j,k]
        same = model.addVars(
            [(i, j, k) for i, j in edges for k in range(K)],
            vtype=GRB.BINARY,
            name="same",
        )

        # keep[i,j] = 1 iff i,j are in same cluster
        keep = model.addVars(edges, vtype=GRB.BINARY, name="keep")

        # W[i,j,t] = keep[i,j] * I[j,t]
        W = {}
        for i, j in edges:
            for t in range(T - 1):
                W[i, j, t] = model.addVar(
                    lb=0.0, ub=1.0, name=f"W_{i}_{j}_{t}"
                )

        model.update()

        # Initial state
        for i in range(n):
            model.addConstr(S[i, 0] == initial_frac[i, 0])
            model.addConstr(I[i, 0] == initial_frac[i, 1])
            model.addConstr(R[i, 0] == initial_frac[i, 2])

        # Every node belongs to exactly one cluster
        for i in range(n):
            model.addConstr(
                gp.quicksum(assign[i, k] for k in range(K)) == 1
            )

        # Cluster usage
        for k in range(K):
            for i in range(n):
                model.addConstr(assign[i, k] <= used[k])

            model.addConstr(
                used[k] <= gp.quicksum(assign[i, k] for i in range(n))
            )

        # Symmetry breaking: used clusters are 0,1,2,... consecutively
        for k in range(K - 1):
            model.addConstr(used[k] >= used[k + 1])

        # Optional symmetry breaking
        model.addConstr(assign[0, 0] == 1)

        # same[i,j,k] = assign[i,k] AND assign[j,k]
        for i, j in edges:
            for k in range(K):
                y = same[i, j, k]

                model.addConstr(y <= assign[i, k])
                model.addConstr(y <= assign[j, k])
                model.addConstr(
                    y >= assign[i, k] + assign[j, k] - 1
                )

            # Because each node belongs to exactly one cluster,
            # this sum is either 0 or 1
            model.addConstr(
                keep[i, j]
                == gp.quicksum(same[i, j, k] for k in range(K))
            )

        # W = keep * I
        for i, j in edges:
            for t in range(T - 1):
                w, y = W[i, j, t], keep[i, j]

                model.addConstr(w <= I[j, t])
                model.addConstr(w <= y)
                model.addConstr(w >= I[j, t] - (1 - y))

        outgoing = {i: [] for i in range(n)}
        for i, j in edges:
            outgoing[i].append(j)

        # Normalized SIR dynamics
        for t in range(T - 1):
            Ft = F[t]

            for i in range(n):
                local_infection = beta[i] * S[i, t] * I[i, t]
                travel_infection = gp.QuadExpr()

                for j in outgoing[i]:
                    flow = Ft[i, j]

                    if flow == 0:
                        continue

                    coeff = beta_travel[i] * flow / N[i]

                    travel_infection += (
                        coeff
                        * S[i, t]
                        * W[i, j, t]
                    )

                infection = local_infection + travel_infection
                recovery = gamma[i] * I[i, t]

                model.addConstr(
                    S[i, t + 1]
                    == S[i, t] - infection
                )

                model.addConstr(
                    I[i, t + 1]
                    == I[i, t] + infection - recovery
                )

                model.addConstr(
                    R[i, t + 1]
                    == R[i, t] + recovery
                )

                model.addConstr(
                    S[i, t + 1]
                    + I[i, t + 1]
                    + R[i, t + 1]
                    == 1.0
                )

        # Travel budget
        edge_flow = {
            (i, j): sum(
                F[t][i, j]
                for t in range(T - 1)
            )
            for i, j in edges
        }

        total_travel = sum(edge_flow.values())
        M = travel_reduce_ratio * total_travel

        model.addConstr(
            gp.quicksum(
                edge_flow[i, j]
                * (1 - keep[i, j])
                for i, j in edges
            ) <= M,
            name="travel_budget",
        )

        # Minimize new infections during this interval.
        # Since the initial S is fixed, this is equivalent
        # to minimizing final cumulative infections.
        model.setObjective(
            gp.quicksum(
                N[i]
                * (
                    initial_frac[i, 0]
                    - S[i, T - 1]
                )
                for i in range(n)
            ),
            GRB.MINIMIZE,
        )

        model.Params.NonConvex = 2
        model.Params.MIPGap = mip_gap

        if time_limit is not None:
            model.Params.TimeLimit = time_limit

        model.optimize()

        if model.SolCount == 0:
            print("No feasible solution found.")
            return None

        # Cluster assignment
        node_cluster = []

        for i in range(n):
            cluster_id = next(
                k for k in range(K)
                if assign[i, k].X > 0.5
            )

            node_cluster.append(cluster_id)

        clusters = {}

        for i, cluster_id in enumerate(node_cluster):
            clusters.setdefault(cluster_id, []).append(i)

        disabled_edges = [
            (
                i,
                j,
                edge_flow[i, j],
            )
            for i, j in edges
            if keep[i, j].X < 0.5
        ]

        total_removed_flow = sum(
            flow
            for _, _, flow in disabled_edges
        )

        print("objective:", model.ObjVal)
        print("number of clusters:", len(clusters))
        print("total travel:", total_travel)
        print("allowed reduction:", M)
        print("removed travel:", total_removed_flow)

        print("clusters:")

        for cluster_id, members in clusters.items():
            names = [
                net.nodes[i].name
                for i in members
            ]

            print(
                f"cluster {cluster_id}:",
                names,
            )

        print("disabled edges:")

        for i, j, flow in disabled_edges:
            print(
                net.node_ids[i],
                "->",
                net.node_ids[j],
                "flow:",
                flow,
            )

        return {
            "objective": model.ObjVal,
            "node_cluster": node_cluster,
            "clusters": clusters,
            "disabled_edges": disabled_edges,
            "removed_flow": total_removed_flow,
            "total_travel": total_travel,
            "S": np.array([
                [
                    S[i, t].X * N[i]
                    for t in range(T)
                ]
                for i in range(n)
            ]),
            "I": np.array([
                [
                    I[i, t].X * N[i]
                    for t in range(T)
                ]
                for i in range(n)
            ]),
            "R": np.array([
                [
                    R[i, t].X * N[i]
                    for t in range(T)
                ]
                for i in range(n)
            ]),
        }

    def build_optimized_flow_matrices(self, result, start, end):
        net = self.network

        optimized_flow = {
            date: F.copy()
            for date, F in net.flow_matrix_dict.items()
        }

        dates = pd.to_datetime(
            net.nodes[0].case_data["date"]
            .iloc[int(start):int(end)]
        )

        flow_keys = [
            pd.Timestamp(d).normalize()
            for d in dates
        ]

        node_cluster = result["node_cluster"]

        for date in flow_keys:
            F = optimized_flow[date]

            for i in range(net.num_nodes):
                for j in range(net.num_nodes):
                    if node_cluster[i] != node_cluster[j]:
                        F[i, j] = 0.0

        return optimized_flow