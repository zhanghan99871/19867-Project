import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import os 
import time 


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
    
    def reset_network_flow(self, flow):
        self.network.flow_matrix_dict = flow
        self.network.model.flow_matrix_dict = flow 


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

    def optimize(self, start, end, K=3, travel_reduce_ratio=None, time_limit=None, mip_gap=0.01):
        if not self.network.fited:
            raise ValueError("Network model must be fitted before optimization.")
        if travel_reduce_ratio is not None and not 0 <= travel_reduce_ratio <= 1:
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
        
        min_cluster_population = np.sum(N) / (K + 1)
 
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
            # If node i is assigned to k, cluster k must be used
            for i in range(n):
                model.addConstr(assign[i, k] <= used[k])

            # If cluster k is used, it must contain at least one node
            model.addConstr(
                used[k] <= gp.quicksum(
                    assign[i, k]
                    for i in range(n)
                )
            )

            # Minimum population for every used cluster
            cluster_population = gp.quicksum(
                N[i] * assign[i, k]
                for i in range(n)
            )

            model.addConstr(
                cluster_population
                >= min_cluster_population * used[k]
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
        if travel_reduce_ratio is not None:
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
        if travel_reduce_ratio is not None:
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

    def visualize_clusters(self, result, start, end, save_path=None, show_removed=False):
        net = self.network
        start, end = int(start), int(end)
        node_cluster = result["node_cluster"]

        dates = pd.to_datetime(
            net.nodes[0].case_data["date"].iloc[start:end]
        )
        flow_keys = [pd.Timestamp(d).normalize() for d in dates]

        # Aggregate flow over optimization horizon
        total_flow = np.zeros((net.num_nodes, net.num_nodes))
        for date in flow_keys:
            total_flow += net.flow_matrix_dict[date]

        G = nx.DiGraph()

        # Add nodes
        for i, node in enumerate(net.nodes):
            G.add_node(
                i,
                name=node.name,
                cluster=node_cluster[i],
                population=node.total_population,
            )

        # Add edges
        for i in range(net.num_nodes):
            for j in range(net.num_nodes):
                if i == j or total_flow[i, j] <= 0:
                    continue

                same_cluster = (
                    node_cluster[i] == node_cluster[j]
                )

                if same_cluster or show_removed:
                    G.add_edge(
                        i,
                        j,
                        flow=total_flow[i, j],
                        kept=same_cluster,
                    )

        # Layout
        pos = nx.spring_layout(
            G,
            seed=42,
            weight="flow",
            k=1.5,
        )

        plt.figure(figsize=(14, 10))

        # Node size proportional to population
        populations = np.array([
            net.nodes[i].total_population
            for i in range(net.num_nodes)
        ], dtype=float)

        node_sizes = (
            800
            + 3000
            * populations
            / populations.max()
        )

        node_colors = [
            node_cluster[i]
            for i in range(net.num_nodes)
        ]

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.tab10,
            alpha=0.9,
        )

        labels = {
            i: net.nodes[i].name
            for i in range(net.num_nodes)
        }

        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            font_size=9,
        )

        kept_edges = [
            (u, v)
            for u, v, d in G.edges(data=True)
            if d["kept"]
        ]

        removed_edges = [
            (u, v)
            for u, v, d in G.edges(data=True)
            if not d["kept"]
        ]

        # Scale edge width by log(flow)
        def edge_width(edges):
            if not edges:
                return []

            flows = np.array([
                G[u][v]["flow"]
                for u, v in edges
            ])

            log_flow = np.log1p(flows)

            return (
                0.5
                + 3.0
                * log_flow
                / log_flow.max()
            )

        # Internal edges
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=kept_edges,
            width=edge_width(kept_edges),
            alpha=0.6,
            arrows=True,
            arrowsize=12,
        )

        # Removed cross-cluster edges
        if show_removed:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=removed_edges,
                width=edge_width(removed_edges),
                alpha=0.2,
                style="dashed",
                arrows=True,
                arrowsize=10,
            )

        plt.title(
            f"Optimized Travel Bubbles "
            f"({len(result['clusters'])} clusters)"
        )

        plt.axis("off")
        plt.tight_layout()

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            plt.savefig(
                path,
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.show()

        plt.close()

    def visualize_cluster_population(self, result, save_path=None):
        net = self.network

        cluster_ids = sorted(result["clusters"].keys())

        populations = [
            sum(
                net.nodes[i].total_population
                for i in result["clusters"][k]
            )
            for k in cluster_ids
        ]

        labels = [
            f"Cluster {k}"
            for k in cluster_ids
        ]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, populations)

        plt.ylabel("Population")
        plt.title("Population by Travel Bubble")
        plt.tight_layout()

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300)
        else:
            plt.show()

        plt.close()


def compare_optimizers(
    network,
    start,
    end,
    travel_reduce_ratio=0.2,
    K=3,
    time_limit=600,
    mip_gap=0.05,
    output_dir="../results/network",
):
    os.makedirs(output_dir, exist_ok=True)

    original_flow = {
        date: F.copy()
        for date, F in network.flow_matrix_dict.items()
    }

    def set_flow(flow):
        copied = {
            date: F.copy()
            for date, F in flow.items()
        }
        network.flow_matrix_dict = copied
        network.model.flow_matrix_dict = copied

    def infected_curve(states):
        # sum_node I(t)
        return np.sum(states[:, :, 1], axis=1)

    records = []
    curves = {}

    dates = pd.to_datetime(
        network.nodes[0].case_data["date"].iloc[start:end + 1]
    )

    # =========================================================
    # 1. Baseline
    # =========================================================
    set_flow(original_flow)

    baseline_pred = network.predict(
        start=start,
        end=end,
    )

    baseline_new_infected = network.sum_infected(
        baseline_pred
    )

    curves["Baseline"] = infected_curve(
        baseline_pred
    )

    records.append({
        "name": "Baseline",
        "new_infected": baseline_new_infected,
        "removed_flow": 0.0,
        "removed_ratio": 0.0,
        "runtime": 0.0,
    })

    # =========================================================
    # 2. Uniform reduction
    # =========================================================
    reduced_flow = {
        date: F * (1 - travel_reduce_ratio)
        for date, F in original_flow.items()
    }

    set_flow(reduced_flow)

    even_pred = network.predict(
        start=start,
        end=end,
    )

    even_new_infected = network.sum_infected(
        even_pred
    )

    curves["Uniform Reduction"] = infected_curve(
        even_pred
    )

    records.append({
        "name": "Uniform Reduction",
        "new_infected": even_new_infected,
        "removed_flow": None,
        "removed_ratio": travel_reduce_ratio,
        "runtime": 0.0,
    })

    # Restore before optimization
    set_flow(original_flow)

    # =========================================================
    # 3. Optimizers
    # =========================================================
    optimizer_specs = [
        (
            "Binary Edge",
            BinaryReduceEdgeOptimizer,
        ),
        (
            "Continuous Edge",
            ContinuousReduceEdgeOptimizer,
        ),
        (
            "Binary Node",
            BinaryReduceNodeOptimizer,
        ),
        (
            "Cluster",
            BinaryClusterOptimizer,
        ),
    ]

    for display_name, optimizer_class in optimizer_specs:

        print("\n" + "=" * 70)
        print("Running:", display_name)
        print("=" * 70)

        # VERY IMPORTANT:
        # each optimizer starts from exactly the same original flow
        set_flow(original_flow)

        optimizer = optimizer_class(network)

        t0 = time.time()

        if optimizer_class is BinaryClusterOptimizer:
            result = optimizer.optimize(
                start=start,
                end=end,
                K=K,
                travel_reduce_ratio=None,
                time_limit=time_limit,
                mip_gap=mip_gap,
            )
        else:
            result = optimizer.optimize(
                start=start,
                end=end,
                travel_reduce_ratio=travel_reduce_ratio,
                time_limit=time_limit,
                mip_gap=mip_gap,
            )

        runtime = time.time() - t0

        if result is None:
            print(display_name, "failed.")

            records.append({
                "name": display_name,
                "new_infected": None,
                "removed_flow": None,
                "removed_ratio": None,
                "runtime": runtime,
            })

            continue

        optimized_flow = (
            optimizer.build_optimized_flow_matrices(
                result,
                start=start,
                end=end,
            )
        )

        set_flow(optimized_flow)

        optimized_pred = network.predict(
            start=start,
            end=end,
        )

        new_infected = network.sum_infected(
            optimized_pred
        )

        curves[display_name] = infected_curve(
            optimized_pred
        )

        removed_flow = result.get(
            "removed_flow",
            0.0,
        )

        total_travel = result.get(
            "total_travel",
            0.0,
        )

        removed_ratio = (
            removed_flow / total_travel
            if total_travel > 0
            else 0.0
        )

        records.append({
            "name": display_name,
            "new_infected": new_infected,
            "removed_flow": removed_flow,
            "removed_ratio": removed_ratio,
            "runtime": runtime,
        })

    # Restore network after comparison
    set_flow(original_flow)

    # =========================================================
    # 4. Save comparison text
    # =========================================================
    summary_path = os.path.join(
        output_dir,
        "optimizer_compare.txt",
    )

    with open(summary_path, "w") as f:
        f.write("Optimizer Comparison\n")
        f.write("=" * 80 + "\n")

        f.write(
            f"Start: {dates.iloc[0]}\n"
        )
        f.write(
            f"End: {dates.iloc[-1]}\n"
        )
        f.write(
            f"Travel reduction budget: "
            f"{travel_reduce_ratio:.4f}\n"
        )
        f.write(
            f"Cluster K: {K}\n"
        )
        f.write("\n")

        f.write(
            f"{'Method':<22}"
            f"{'New Infected':>18}"
            f"{'Reduction':>15}"
            f"{'Travel Reduce':>18}"
            f"{'Runtime(s)':>15}\n"
        )

        f.write("-" * 88 + "\n")

        for record in records:
            name = record["name"]

            if record["new_infected"] is None:
                f.write(
                    f"{name:<22}"
                    f"{'FAILED':>18}"
                    f"{'-':>15}"
                    f"{'-':>18}"
                    f"{record['runtime']:>15.2f}\n"
                )
                continue

            reduction = (
                baseline_new_infected
                - record["new_infected"]
            )

            f.write(
                f"{name:<22}"
                f"{record['new_infected']:>18.2f}"
                f"{reduction:>15.2f}"
                f"{record['removed_ratio']:>17.2%}"
                f"{record['runtime']:>15.2f}\n"
            )

        f.write("\n")

        f.write(
            f"Baseline new infected: "
            f"{baseline_new_infected:.2f}\n"
        )

        f.write("\nRelative infection reduction:\n")

        for record in records:
            if (
                record["new_infected"] is None
                or record["name"] == "Baseline"
            ):
                continue

            reduction_ratio = (
                baseline_new_infected
                - record["new_infected"]
            ) / baseline_new_infected

            f.write(
                f"{record['name']}: "
                f"{reduction_ratio:.2%}\n"
            )

    # =========================================================
    # 5. Plot sum-node infected curves
    # =========================================================
    plt.figure(figsize=(12, 7))

    for name, curve in curves.items():
        plt.plot(
            dates,
            curve,
            label=name,
            linewidth=2,
        )

    plt.xlabel("Date")
    plt.ylabel("Total Currently Infected")
    plt.title("Sum-node Infected Population by Intervention")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.xticks(rotation=30)
    plt.tight_layout()

    figure_path = os.path.join(
        output_dir,
        "optimizer_compare_infected.png",
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nComparison saved to:")
    print(summary_path)
    print(figure_path)

    return records, curves