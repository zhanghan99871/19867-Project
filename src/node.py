import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
import os 
from model import SIR, SIRD, NetworkSIRSimple
import pandas as pd

class Node:
    def __init__(self, id, name=None, mode = "fit", model_type="SIRD", case_data=None, total_population=0, period=14):
        # mode can be "fit" or "simulate"
        self.id = id
        self.name = name 
        self.mode = mode
        self.model_type = model_type
        self.period = period
        self.fited = False
        self.model = None
        if model_type == "SIRD":
            self.params = {
                "beta": 0.25,
                "gamma": 0.1,
                "mu": 0.005
            }
            self.model = SIRD((self.params["beta"], self.params["gamma"], self.params["mu"]), total_population)
        elif model_type == "SIR":
            self.params = {
                "beta": 0.25,
                "gamma": 0.1
            }
            self.model = SIR((self.params["beta"], self.params["gamma"]), total_population)
        else:
            raise NotImplementedError(
                f"Model type {model_type} is not implemented."
            )
        self.case_data = case_data
        self.total_population = total_population
    
    def get(self):
        return self.params.copy()

    def step(self, state, theta, t):
        N = self.total_population
        self.model.reset(theta)
        return self.model.step(state, t)

    def run(self, initial_state, T, theta=None):
        if self.model_type == "SIR":
            if theta is None:
                theta = np.array([
                    self.params["beta"],
                    self.params["gamma"]
                ])
            states = np.zeros((T, 3))
            states[0] = initial_state
        
        elif self.model_type == "SIRD":
            if theta is None:
                theta = np.array([
                    self.params["beta"],
                    self.params["gamma"],
                    self.params["mu"]
                ])

            states = np.zeros((T, 4))
            states[0] = initial_state

        for t in range(T - 1):
            states[t + 1] = self.step(
                states[t],
                theta,
                t
            )

        return states

    def objective(self, theta, initial_state, T, C_obs, D_obs):

        states = self.run(
            initial_state=initial_state,
            T=T,
            theta=theta
        )
        if not np.all(np.isfinite(states)):
            return 1e20

        if np.any(states < 0):
            return 1e20
        
        if self.model_type == "SIR":
            S_pred = states[:, 0]
            R_pred = states[:, 2]

            # cumulative confirmed cases
            C_pred = self.total_population - S_pred

            C_obs = C_obs.to_numpy()

            C_scale = max(C_obs.max(), 1)

            case_loss = np.mean(
                ((C_pred - C_obs) / C_scale) ** 2
            )

            return case_loss

        elif self.model_type == "SIRD":
            S_pred = states[:, 0]
            R_pred = states[:, 2]
            D_pred = states[:, 3]

            # cumulative confirmed cases
            C_pred = self.total_population - S_pred

            C_obs = C_obs.to_numpy()
            D_obs = D_obs.to_numpy()

            C_scale = max(C_obs.max(), 1)
            D_scale = max(D_obs.max(), 1)

            case_loss = np.mean(
                ((C_pred - C_obs) / C_scale) ** 2
            )

            death_loss = np.mean(
                ((D_pred - D_obs) / D_scale) ** 2
            )

            return case_loss + death_loss

    def analyze(self, result):
        if self.model_type == "SIR":
            beta_hat, gamma_hat = result.x
            self.params["beta"] = beta_hat
            self.params["gamma"] = gamma_hat
            self.fited = result.success

            print("success:", result.success)
            print("loss:", result.fun)
            print("beta:", beta_hat)
            print("gamma:", gamma_hat)
        elif self.model_type == "SIRD":
            beta_hat, gamma_hat, mu_hat = result.x
            self.params["beta"] = beta_hat
            self.params["gamma"] = gamma_hat
            self.params["mu"] = mu_hat
            self.fited = result.success
            
            print("success:", result.success)
            print("loss:", result.fun)
            print("beta:", beta_hat)
            print("gamma:", gamma_hat)
            print("mu:", mu_hat)
        else:
            raise NotImplementedError(
                f"Model type {self.model_type} is not implemented."
            )

    def fit_model(self, start, end, bounds=[
                (1e-6, 1.0),   # beta
                (1e-6, 1.0),   # gamma
                (1e-8, 0.2)    # mu
            ]):
        self.case_data["new_cases"] = (
            self.case_data["confirmed"]
            .diff()
            .clip(lower=0)
        )

        self.case_data["I_est"] = (
            self.case_data["new_cases"]
            .rolling(self.period, min_periods=1)
            .sum()
        )
        if self.model_type == "SIR":
            initial_state = np.array([
                self.total_population - self.case_data["confirmed"].iloc[start],
                self.case_data["I_est"].iloc[start],
                self.case_data["confirmed"].iloc[start] - self.case_data["I_est"].iloc[start]
            ])
            
            x0 = np.array([
                self.params["beta"],
                self.params["gamma"]
            ])
            bounds = [
                (1e-6, 1.0),   # beta
                (1e-6, 1.0)    # gamma
            ]
        elif self.model_type == "SIRD":
            initial_state = np.array([
                self.total_population - self.case_data["confirmed"].iloc[start],
                self.case_data["I_est"].iloc[start],
                self.case_data["confirmed"].iloc[start] - self.case_data["I_est"].iloc[start] - self.case_data["deaths"].iloc[start],
                self.case_data["deaths"].iloc[start]
            ])
            
            x0 = np.array([
                self.params["beta"],
                self.params["gamma"],
                self.params["mu"]
            ])
            
        else:
            raise NotImplementedError(
                f"Model type {self.model_type} is not implemented."
            )

        result = minimize(
            self.objective,
            x0=x0,
            args=(initial_state, end - start + 1, self.case_data["confirmed"].iloc[start:end+1], self.case_data["deaths"].iloc[start:end+1]),
            method="L-BFGS-B",
            bounds=bounds
        )

        self.analyze(result)

        return result
    
    def predict(self, start, end, initial_state=None, theta=None):
        if not self.fited and self.mode == "fit":
            print("Model has not been fitted yet. Please call fit_model() first.")
            return
        if end is None:
            end = len(self.case_data) - 1
        if self.mode == "simulate":
            if initial_state is None or theta is None:
                raise ValueError("For simulation mode, initial_state and theta must be provided.")
        else:
            if self.model_type == "SIR":
                initial_state = np.array([
                    self.total_population - self.case_data["confirmed"].iloc[start],
                    self.case_data["I_est"].iloc[start],
                    self.case_data["confirmed"].iloc[start] - self.case_data["I_est"].iloc[start]
                ])
                theta = (self.params["beta"], self.params["gamma"])
            elif self.model_type == "SIRD":
                initial_state = np.array([
                    self.total_population - self.case_data["confirmed"].iloc[start],
                    self.case_data["I_est"].iloc[start],
                    self.case_data["confirmed"].iloc[start] - self.case_data["I_est"].iloc[start] - self.case_data["deaths"].iloc[start],
                    self.case_data["deaths"].iloc[start]
                ])
                theta = (self.params["beta"], self.params["gamma"], self.params["mu"])
        states = self.run(
            initial_state=initial_state,
            T=end - start + 1,
            theta=theta
        )
        return states
    
    def plot_results(self, states, start, end=None, save_path=None):
        if end is None:
            end = len(self.case_data) - 1

        C_pred = self.total_population - states[:, 0]

        plt.figure(figsize=(12, 6))
        if self.mode == "fit":
            plt.plot(
                self.case_data["date"].iloc[start:end+1],
                self.case_data["confirmed"].iloc[start:end+1],
                label="JHU confirmed"
            )

            plt.plot(
                self.case_data["date"].iloc[start:end+1],
                C_pred,
                "--",
                label="SIRD predicted"
            )
        elif self.mode == "simulate":
            plt.plot(
                states[:, 0],
                label="Susceptible"
            )
            plt.plot(
                states[:, 1],
                label="Infected"
            )
            plt.plot(
                states[:, 2],
                label="Recovered"
            )
        if self.model_type == "SIRD":
            D_pred = states[:, 3]
            if self.mode == "fit":
                plt.plot(
                    self.case_data["date"].iloc[start:end+1],
                    self.case_data["deaths"].iloc[start:end+1],
                    label="JHU deaths"
                )

                plt.plot(
                    self.case_data["date"].iloc[start:end+1],
                    D_pred,
                    "--",
                    label="SIRD predicted"
                )
            elif self.mode == "simulate":
                plt.plot(
                    states[:, 3],
                    label="Deceased"
                )

        plt.xlabel("Date")
        plt.ylabel("Cumulative cases")
        plt.legend()
        plt.xticks(rotation=30)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.title(f"{self.mode.capitalize()} {self.model_type} Model for {self.id}")
        if save_path:
            path = Path(save_path + f"/{self.mode.capitalize()}/{self.model_type}")
            os.makedirs(path, exist_ok=True)
            plt.savefig(path / f"{self.id}.png")
        else:
            plt.show()
        plt.close()
            
class Clusters:
    def __init__(self, nodes):
        self.nodes = nodes
        self.sum_node = Node(id="sum_node", model_type=self.nodes[0].model_type, total_population=sum(node.total_population for node in nodes), mode="simulate")
    
    def get_all(self):
        result = {}
        for node in self.nodes:
            node_params = node.get()
            if node.model_type == "SIRD":
                result[node.id] = tuple(node_params[i] for i in ["beta", "gamma", "mu"])
            else:
                result[node.id] = tuple(node_params[i] for i in ["beta", "gamma"])
        return result
        
    def random_initializer(self, beta_range=(0.1, 0.5), gamma_range=(0.05, 0.2), mu_range=(0.001, 0.01)):
        for node in self.nodes:
            print(f"Randomly initializing model for {node.id}...")
            if node.model_type == "SIR":
                node.params["beta"] = np.random.uniform(*beta_range)
                node.params["gamma"] = np.random.uniform(*gamma_range)
            elif node.model_type == "SIRD":
                node.params["beta"] = np.random.uniform(*beta_range)
                node.params["gamma"] = np.random.uniform(*gamma_range)
                node.params["mu"] = np.random.uniform(*mu_range)
            print(f"Finished initializing model for {node.id}.")
    
    def fit_all(self, start, end):
        for node in self.nodes:
            print(f"Fitting model for {node.id}...")
            node.fit_model(start, end)
            print(f"Finished fitting model for {node.id}.")
            
    def predict_all(self, start, end, initial_states=None, thetas=None):
        predictions = {}

        for node in self.nodes:
            print(f"Predicting for {node.id}...")

            states = node.predict(start, end, 
                initial_state=(
                    initial_states[node.id]
                    if initial_states is not None
                    else None
                ),
                theta=(
                    thetas[node.id]
                    if thetas is not None
                    else None
                ),
            )

            predictions[node.id] = states
            print(f"Finished predicting for {node.id}.")

        return predictions

    def plot_all(self, predictions, start, end=None, save_path=None, plot_sum=True):
        for node in self.nodes:
            print(f"Plotting results for node {node.id}...")
            node.plot_results(predictions[node.id], start, end, save_path)
            print(f"Finished plotting results for node {node.id}.")
        if plot_sum:
            # Aggregate predictions for the sum_node
            predictions[self.sum_node.id] = sum(
                predictions[node.id] for node in self.nodes
            )
            print(f"Plotting results for node {self.sum_node.id}...")
            self.sum_node.plot_results(predictions[self.sum_node.id], start, end, save_path)
            print(f"Finished plotting results for node {self.sum_node.id}.")
        
            
class Network:
    def __init__(self, nodes, flow_matrix_dict, mode="fit", method="euler"):
        self.nodes = nodes
        self.node_ids = [node.id for node in nodes]
        self.num_nodes = len(nodes)
        self.fited = False
        self.mode = mode

        self.N = np.array([node.total_population for node in nodes], dtype=float)
        beta = np.array([node.params["beta"] for node in nodes], dtype=float)
        gamma = np.array([node.params["gamma"] for node in nodes], dtype=float)
        beta_travel = np.full(self.num_nodes, 1e-3, dtype=float)

        self.flow_matrix_dict = self._prepare_flow_matrices(flow_matrix_dict)
        self.model = NetworkSIRSimple(
            theta=(beta, gamma, beta_travel),
            N=self.N,
            flow_matrix_dict=self.flow_matrix_dict,
            method=method,
        )
        self.sum_node = Node(id="sum_node", model_type=self.nodes[0].model_type, total_population=sum(node.total_population for node in nodes), mode="simulate")
        
    def set_beta_travel(self, beta_travel):
        if np.isscalar(beta_travel):
            x0 = np.full(self.num_nodes, beta_travel, dtype=float)
        else:
            x0 = np.asarray(beta_travel, dtype=float)

        if x0.shape != (self.num_nodes,):
            raise ValueError(
                f"beta_travel must have shape ({self.num_nodes},), got {x0.shape}"
            )

        self.model.beta_travel = x0

    def _prepare_flow_matrices(self, flow_matrix_dict):
        prepared = {}
        node_ids = [str(x) for x in self.node_ids]

        for date, F in sorted(flow_matrix_dict.items()):
            if not hasattr(F, "reindex"):
                raise TypeError(f"Flow matrix for {date} must be a pandas DataFrame.")

            F = F.copy()
            F.index = F.index.astype(str)
            F.columns = F.columns.astype(str)
            F = F.reindex(index=node_ids, columns=node_ids, fill_value=0.0)
            F = F.to_numpy(dtype=float)

            if F.shape != (self.num_nodes, self.num_nodes):
                raise ValueError(
                    f"Flow matrix on {date} has shape {F.shape}, "
                    f"expected ({self.num_nodes}, {self.num_nodes})."
                )

            prepared[pd.Timestamp(date).normalize()] = F

        return prepared

    def _refresh_local_params(self):
        beta = np.array([node.params["beta"] for node in self.nodes], dtype=float)
        gamma = np.array([node.params["gamma"] for node in self.nodes], dtype=float)
        self.model.reset((beta, gamma, self.model.beta_travel))

    def _get_fit_data(self, start, end):
        ref_dates = pd.to_datetime(
            self.nodes[0].case_data["date"].iloc[start:end + 1]
        ).reset_index(drop=True)

        confirmed = []
        for node in self.nodes:
            node_dates = pd.to_datetime(
                node.case_data["date"].iloc[start:end + 1]
            ).reset_index(drop=True)

            if not np.array_equal(ref_dates.to_numpy(), node_dates.to_numpy()):
                raise ValueError(f"Date range for node {node.id} does not match.")

            confirmed.append(
                node.case_data["confirmed"].iloc[start:end + 1].to_numpy(dtype=float)
            )

        C_obs = np.column_stack(confirmed)
        flow_keys = [pd.Timestamp(date).normalize() for date in ref_dates.iloc[:-1]]

        missing = [date for date in flow_keys if date not in self.flow_matrix_dict]
        if missing:
            raise ValueError(f"Missing flow matrices: {missing[:10]}")

        return C_obs, flow_keys

    def initial_state_from_data(self, start):
        states = []

        for node in self.nodes:
            if "I_est" not in node.case_data.columns:
                node.case_data["new_cases"] = node.case_data["confirmed"].diff().clip(lower=0)
                node.case_data["I_est"] = (
                    node.case_data["new_cases"]
                    .rolling(node.period, min_periods=1)
                    .sum()
                )

            confirmed = node.case_data["confirmed"].iloc[start]
            I = node.case_data["I_est"].iloc[start]
            S = node.total_population - confirmed
            R = confirmed - I
            states.append([S, I, R])

        return np.asarray(states, dtype=float)

    def run(self, initial_state, theta, T, flow_keys=None):
        initial_state = np.asarray(initial_state, dtype=float)

        if initial_state.shape != (self.num_nodes, 3):
            raise ValueError(
                f"Expected initial_state shape ({self.num_nodes}, 3), "
                f"got {initial_state.shape}"
            )

        if flow_keys is None:
            flow_keys = list(self.flow_matrix_dict.keys())

        if len(flow_keys) < T - 1:
            raise ValueError(f"Need {T - 1} flow matrices, got {len(flow_keys)}")

        states = np.zeros((T, self.num_nodes, 3), dtype=float)
        states[0] = initial_state
        self.model.reset(theta)

        for k in range(T - 1):
            S = states[k, :, 0]
            I = states[k, :, 1]
            R = states[k, :, 2]
            S_next, I_next, R_next = self.model.step((S, I, R), flow_keys[k])
            states[k + 1, :, 0] = S_next
            states[k + 1, :, 1] = I_next
            states[k + 1, :, 2] = R_next

        return states

    def objective(self, theta_opt, initial_state, flow_keys, gamma, C_obs):
        theta_opt = np.asarray(theta_opt, dtype=float)

        # theta_opt =
        # [beta_1, ..., beta_n,
        #  beta_travel_1, ..., beta_travel_n]
        if theta_opt.shape != (2 * self.num_nodes,):
            return 1e20

        beta = theta_opt[:self.num_nodes]
        beta_travel = theta_opt[self.num_nodes:]

        states = self.run(
            initial_state,
            (beta, gamma, beta_travel),
            T=C_obs.shape[0],
            flow_keys=flow_keys,
        )

        if not np.all(np.isfinite(states)):
            return 1e20

        if np.any(states < 0):
            return 1e20

        C_pred = self.N[None, :] - states[:, :, 0]

        scales = np.maximum(C_obs.max(axis=0), 1.0)

        error = (C_pred - C_obs) / scales

        return np.mean(error ** 2)

    def analyze(self, result, show=False):
        theta_opt = np.asarray(result.x, dtype=float)

        n = self.num_nodes

        if theta_opt.shape != (2 * n,):
            raise ValueError(
                f"Expected result.x shape ({2 * n},), "
                f"got {theta_opt.shape}"
            )

        beta = theta_opt[:n]
        beta_travel = theta_opt[n:2 * n]

        # gamma is fixed during network fitting
        _, gamma, _ = self.model.get()

        self.model.reset((beta, gamma, beta_travel))

        # update beta stored in individual nodes too
        for i, node in enumerate(self.nodes):
            node.params["beta"] = beta[i]

        self.fited = (
            result.success
            and np.isfinite(result.fun)
            and result.fun < 1e19
        )

        print("Network fitting")
        print("success:", result.success)
        print("valid fit:", self.fited)
        print("loss:", result.fun)

        if show:
            for i, node in enumerate(self.nodes):
                print(
                    f"{node.id}: "
                    f"beta={beta[i]:.6f}, "
                    f"gamma={gamma[i]:.6f}, "
                    f"beta_travel={beta_travel[i]:.6f}"
                )

    def fit_model(
        self,
        start,
        end,
        beta0=None,
        beta_travel0=None,
        beta_bounds=(1e-6, 1.0),
        beta_travel_bounds=(1e-8, 1.0),
    ):
        self._refresh_local_params()

        initial_state = self.initial_state_from_data(start)

        C_obs, flow_keys = self._get_fit_data(
            start,
            end,
        )

        # Current model parameters
        beta_current, gamma, beta_travel_current = self.model.get()

        # Initial beta
        if beta0 is None:
            beta_init = beta_current.copy()

        elif np.isscalar(beta0):
            beta_init = np.full(self.num_nodes, beta0, dtype=float)

        else:
            beta_init = np.asarray(beta0, dtype=float)

        # Initial beta_travel
        if beta_travel0 is None:
            beta_travel_init = beta_travel_current.copy()

        elif np.isscalar(beta_travel0):
            beta_travel_init = np.full(self.num_nodes, beta_travel0, dtype=float)

        else:
            beta_travel_init = np.asarray(beta_travel0, dtype=float)

        if beta_init.shape[0] != self.num_nodes:
            raise ValueError(
                f"beta0 must have shape "
                f"({self.num_nodes},), "
                f"got {beta_init.shape}"
            )

        if beta_travel_init.shape[0] != self.num_nodes:
            raise ValueError(
                f"beta_travel0 must have shape "
                f"({self.num_nodes},), "
                f"got {beta_travel_init.shape}"
            )

        # [beta, beta_travel]
        x0 = np.concatenate([
            beta_init,
            beta_travel_init,
        ])

        bounds = (
            [beta_bounds] * self.num_nodes
            +
            [beta_travel_bounds] * self.num_nodes
        )

        initial_loss = self.objective(
            x0,
            initial_state,
            flow_keys,
            gamma,
            C_obs,
        )
        result = minimize(
            self.objective,
            x0=x0,
            args=(
                initial_state,
                flow_keys,
                gamma, 
                C_obs,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 1000,
                "maxfun": 100000,
            },
        )

        self.analyze(result)

        return result

    def predict(self, start, end, initial_state=None, theta=None, flow_keys=None):
        if not self.fited and self.mode == "fit":
            print("Network model has not been fitted yet.")
            return

        if self.mode == "simulate":
            if initial_state is None or theta is None:
                raise ValueError("Simulation mode requires initial_state and theta.")

        if initial_state is None:
            initial_state = self.initial_state_from_data(start)

        if theta is None:
            theta = self.model.get()

        if flow_keys is None and self.mode == "fit":
            _, flow_keys = self._get_fit_data(start, end)

        return self.run(
            initial_state,
            theta,
            T=end - start + 1,
            flow_keys=flow_keys,
        )

    def split_predictions(self, states):
        return {
            node_id: states[:, i, :]
            for i, node_id in enumerate(self.node_ids)
        }

    def plot_all(self, states, start, end=None, save_path=None, plot_sum=True):
        predictions = self.split_predictions(states)

        for node in self.nodes:
            print(f"Plotting results for node {node.id}...")
            node.plot_results(
                predictions[node.id],
                start=start,
                end=end,
                save_path=save_path,
            )
            print(f"Finished plotting results for node {node.id}.")
        
        # Plot the aggregated results for the sum_node
        if plot_sum:
            print(f"Plotting results for node {self.sum_node.id}...")
            sum_states = np.sum(states, axis=1)
            self.sum_node.plot_results(
                sum_states,
                start=start,
                end=end,
                save_path=save_path,
            )
            print(f"Finished plotting results for node {self.sum_node.id}.")