import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
import os 
from model import SIR, SIRD, NetworkSIRSimple
import pandas as pd

class Node:
    def __init__(self, id, mode = "fit", model_type="SIRD", case_data=None, total_population=0, period=14):
        # mode can be "fit" or "simulate"
        self.id = id
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
                (1e-6, 5.0),   # beta
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
                (1e-6, 5.0),   # beta
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
            
class Clusters:
    def __init__(self, nodes):
        self.nodes = nodes
        
    def random_initializer(self):
        for node in self.nodes:
            print(f"Randomly initializing model for {node.id}...")
            if node.model_type == "SIR":
                node.params["beta"] = np.random.uniform(0.1, 0.5)
                node.params["gamma"] = np.random.uniform(0.05, 0.2)
            elif node.model_type == "SIRD":
                node.params["beta"] = np.random.uniform(0.1, 0.5)
                node.params["gamma"] = np.random.uniform(0.05, 0.2)
                node.params["mu"] = np.random.uniform(0.001, 0.01)
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
            states = node.predict(start, end, initial_states=initial_states[node.id] if initial_states is not None else None, thetas=thetas[node.id] if thetas is not None else None)
            predictions[node.id] = states
            print(f"Finished predicting for {node.id}.")
        return predictions

    def plot_all(self, predictions, start, end=None, save_path=None):
        for node in self.nodes:
            print(f"Plotting results for {node.id}...")
            node.plot_results(predictions[node.id], start, end, save_path)
            print(f"Finished plotting results for {node.id}.")
            
class Network:
    def __init__(
        self,
        nodes,
        flow_matrix_dict, 
        mode="fit", 
        method="euler"
    ):
        self.nodes = nodes
        self.node_ids = [node.id for node in nodes]
        self.num_nodes = len(nodes)

        self.fited = False
        self.mode = mode

        self.N = np.array(
            [node.total_population for node in nodes],
            dtype=float,
        )

        beta = np.array(
            [node.params["beta"] for node in nodes],
            dtype=float,
        )

        gamma = np.array(
            [node.params["gamma"] for node in nodes],
            dtype=float,
        )
        
        beta_travel = np.array(
            [0.1 for _ in nodes],
            dtype=float
        )

        self.flow_matrix_dict = (
            self._prepare_flow_matrices(
                flow_matrix_dict
            )
        )

        self.model = NetworkSIRSimple(
                theta=(beta, gamma, beta_travel),
                N=self.N,
                flow_matrix_dict=self.flow_matrix_dict,
                method=method
            )

    def _prepare_flow_matrices(self, flow_matrix_dict):
        """
        Prepare daily flow matrices for NetworkSIRSimple.

        Parameters
        ----------
        flow_matrix_dict : dict
            {
                pd.Timestamp: pd.DataFrame
            }

            Each DataFrame has:
                index   = origin node IDs
                columns = destination node IDs

        Returns
        -------
        dict
            {
                pd.Timestamp: np.ndarray
            }

            Every matrix has shape
                (num_nodes, num_nodes)

            and follows exactly the ordering in self.node_ids.
        """

        prepared = {}

        # Make node IDs consistent with DataFrame labels
        node_ids = [str(node_id) for node_id in self.node_ids]

        for date in sorted(flow_matrix_dict.keys()):

            F = flow_matrix_dict[date]

            if not hasattr(F, "reindex"):
                raise TypeError(
                    f"Flow matrix for {date} must be a pandas DataFrame."
                )

            # Make index/column types consistent
            F = F.copy()
            F.index = F.index.astype(str)
            F.columns = F.columns.astype(str)

            # Reorder matrix to exactly match Network node ordering.
            #
            # Missing nodes automatically get flow = 0.
            F = F.reindex(
                index=node_ids,
                columns=node_ids,
                fill_value=0.0,
            )

            # Convert to numpy for fast simulation
            F = F.to_numpy(dtype=float)

            if F.shape != (self.num_nodes, self.num_nodes):
                raise ValueError(
                    f"Flow matrix on {date} has shape {F.shape}, "
                    f"expected ({self.num_nodes}, {self.num_nodes})."
                )

            prepared[pd.Timestamp(date)] = F

        return prepared

    def _refresh_local_params(self):
        """
        Reload beta/gamma from the Node objects.

        Useful if Network was created before the nodes were fitted.
        """
        beta = np.array(
            [node.params["beta"] for node in self.nodes],
            dtype=float,
        )

        gamma = np.array(
            [node.params["gamma"] for node in self.nodes],
            dtype=float,
        )

        self.model.beta = beta
        self.model.gamma = gamma

    def _get_fit_data(self, start, end):
        """
        Collect observed confirmed cases and corresponding flow dates.

        Returns
        -------
        C_obs : ndarray, shape (T, num_nodes)
            Observed cumulative confirmed cases.

        flow_keys : list[pd.Timestamp]
            Dates corresponding to each transition:
                t -> t+1

            Therefore len(flow_keys) = T - 1
        """
        if end <= start:
            raise ValueError("end must be greater than start.")

        # Use first node as the reference date sequence
        ref_dates = pd.to_datetime(
            self.nodes[0].case_data["date"].iloc[start:end + 1]
        ).reset_index(drop=True)

        confirmed = []

        for node in self.nodes:
            node_dates = pd.to_datetime(
                node.case_data["date"].iloc[start:end + 1]
            ).reset_index(drop=True)

            # Make sure every node represents the same dates
            if not np.array_equal(
                ref_dates.to_numpy(),
                node_dates.to_numpy(),
            ):
                raise ValueError(
                    f"Date range for node {node.id} "
                    "does not match the other nodes."
                )

            confirmed.append(
                node.case_data["confirmed"]
                .iloc[start:end + 1]
                .to_numpy(dtype=float)
            )

        # shape:
        #     (T, num_nodes)
        C_obs = np.column_stack(confirmed)

        # A simulation of T states needs T-1 transitions
        flow_keys = [
            pd.Timestamp(date).normalize()
            for date in ref_dates.iloc[:-1]
        ]

        missing = [
            date
            for date in flow_keys
            if date not in self.flow_matrix_dict
        ]

        if missing:
            raise ValueError(
                "Missing flow matrices for dates: "
                + ", ".join(str(d.date()) for d in missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )

        return C_obs, flow_keys

    def initial_state_from_data(self, start):
        states = []

        for node in self.nodes:

            if "I_est" not in node.case_data.columns:

                node.case_data["new_cases"] = (
                    node.case_data["confirmed"]
                    .diff()
                    .clip(lower=0)
                )

                node.case_data["I_est"] = (
                    node.case_data["new_cases"]
                    .rolling(
                        node.period,
                        min_periods=1,
                    )
                    .sum()
                )

            confirmed = (
                node.case_data["confirmed"]
                .iloc[start]
            )

            I = (
                node.case_data["I_est"]
                .iloc[start]
            )

            S = (
                node.total_population
                - confirmed
            )

            R = confirmed - I

            states.append([S, I, R])

        return np.asarray(
            states,
            dtype=float,
        )

    def step(self, states, theta, t):
        self.model.reset(theta)
        return self.model.step(states, t)
    
    def run(
        self,
        initial_state,
        theta, 
        T,
        flow_keys=None,
    ):
        initial_state = np.asarray(
            initial_state,
            dtype=float,
        )

        if initial_state.shape != (
            self.num_nodes,
            3,
        ):
            raise ValueError(
                f"Expected initial_state shape "
                f"({self.num_nodes}, 3), "
                f"got {initial_state.shape}"
            )

        states = np.zeros(
            (
                T,
                self.num_nodes,
                3,
            ),
            dtype=float,
        )

        states[0] = initial_state

        if flow_keys is None:
            flow_keys = list(
                self.flow_matrix_dict.keys()
            )

        if len(flow_keys) < T - 1:
            raise ValueError(
                f"Need {T - 1} flow matrices, "
                f"got {len(flow_keys)}"
            )

        for k in range(T - 1):

            S = states[k, :, 0]
            I = states[k, :, 1]
            R = states[k, :, 2]

            (
                S_next,
                I_next,
                R_next,
            ) = self.model.step(
                (S, I, R),
                theta, 
                flow_keys[k],
            )

            states[k + 1, :, 0] = S_next
            states[k + 1, :, 1] = I_next
            states[k + 1, :, 2] = R_next

        return states

    def objective(
        self,
        theta,
        initial_state,
        flow_keys,
        C_obs,
    ):
        """
        Objective for fitting network transmission strength.

        Parameters
        ----------
        theta : array-like
            theta[0] = beta_travel

        initial_state : ndarray
            shape (num_nodes, 3)

        flow_keys : list
            Flow matrix date for each simulation transition.

        C_obs : ndarray
            shape (T, num_nodes)
            Observed cumulative confirmed cases.
        """

        beta_travel = float(theta[0])

        # Update network transmission parameter
        self.model.beta_travel = beta_travel

        T = C_obs.shape[0]

        states = self.run(
            initial_state=initial_state,
            T=T,
            flow_keys=flow_keys,
        )

        # Invalid numerical simulation
        if not np.all(np.isfinite(states)):
            return 1e20

        if np.any(states < 0):
            return 1e20

        # states shape:
        #   (T, num_nodes, 3)
        #
        # cumulative infected:
        #
        #   C = N - S
        #
        # because there is no migration.
        S_pred = states[:, :, 0]

        C_pred = (
            self.N[None, :]
            - S_pred
        )

        # Normalize each state separately.
        #
        # Otherwise large-population states like CA/TX
        # dominate small states.
        scales = np.maximum(
            C_obs.max(axis=0),
            1.0,
        )

        normalized_error = (
            C_pred - C_obs
        ) / scales[None, :]

        # Equal weight for each state
        loss_per_node = np.mean(
            normalized_error ** 2,
            axis=0,
        )

        loss = np.mean(loss_per_node)

        return loss
    
    def analyze(self, result):
        self.model.reset(result.x)

        self.fited = result.success

        print("Network fitting")
        print("----------------")
        print("success:", result.success)
        print("loss:", result.fun)

    def fit_model(
        self,
        start,
        end,
        beta_travel0=None,
        bounds=(1e-8, 5.0),
    ):
        """
        Fit beta_travel while keeping local beta_i and gamma_i fixed.

        Parameters
        ----------
        start : int
            Start index in node case_data.

        end : int
            End index in node case_data, inclusive.

        beta_travel0 : float, optional
            Initial value for beta_travel.

        bounds : tuple
            Bounds for beta_travel.
        """

        # Make sure local parameters reflect the
        # most recently fitted Node values
        self._refresh_local_params()

        # Construct initial S/I/R for every state
        initial_state = (
            self.initial_state_from_data(start)
        )

        # Observations and matching mobility dates
        C_obs, flow_keys = self._get_fit_data(
            start,
            end,
        )

        if beta_travel0 is None:
            beta_travel0 = self.model.beta_travel

        x0 = np.array(
            [beta_travel0],
            dtype=float,
        )

        result = minimize(
            self.objective,
            x0=x0,
            args=(
                initial_state,
                flow_keys,
                C_obs,
            ),
            method="L-BFGS-B",
            bounds=[bounds],
        )

        self.analyze(result)

        return result

    def predict(
        self,
        start,
        end,
        initial_state=None,
        theta=None, 
        flow_keys=None,
    ):
        if not self.fited and self.mode == "fit":
            print("Network model has not been fitted yet. Please call fit_model() first.")
            return
        if self.mode == "simulate":
            if initial_state is None or theta is None:
                raise ValueError("For simulation mode, initial_state and theta must be provided.")
        
        if initial_state is None:
            initial_state = (
                self.initial_state_from_data(start)
            )
        if theta is None: 
            theta = self.model.get()

        T = end - start + 1

        return self.run(
            initial_state=initial_state,
            theta=theta,
            T=T,
            flow_keys=flow_keys,
        )

    def split_predictions(self, states):
        return {
            node_id: states[:, i, :]
            for i, node_id
            in enumerate(self.node_ids)
        }

    def plot_all(
        self,
        states,
        start,
        end=None,
        save_path=None,
    ):
        predictions = (
            self.split_predictions(states)
        )

        for node in self.nodes:
            node.plot_results(
                predictions[node.id],
                start=start,
                end=end,
                save_path=save_path,
            )