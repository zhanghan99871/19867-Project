import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Node:
    def __init__(self, id, model_type="SIRD", case_data=None, total_population=0, period=14):
        self.id = id
        self.model_type = model_type
        self.period = period
        if model_type == "SIRD":
            self.params = {
                "beta": 0.25,
                "gamma": 0.1,
                "mu": 0.005
            }
        elif model_type == "SIR":
            self.params = {
                "beta": 0.25,
                "gamma": 0.1
            }
        else:
            raise NotImplementedError(
                f"Model type {model_type} is not implemented."
            )

        self.case_data = case_data
        self.total_population = total_population

        self.population_flow_data = {}

    def sird_step(self, state, theta):
        """
        One-day Euler update.
        """
        N = self.total_population

        if self.model_type == "SIR":
            S, I, R = state
            beta, gamma = theta

            new_infected = beta * S * I / N
            new_recovered = gamma * I

            return np.array([
                S - new_infected,
                I + new_infected - new_recovered,
                R + new_recovered
            ])
        elif self.model_type == "SIRD":
            S, I, R, D = state
            beta, gamma, mu = theta

            new_infected = beta * S * I / N
            new_recovered = gamma * I
            new_deaths = mu * I

            return np.array([
                S - new_infected,
                I + new_infected - new_recovered - new_deaths,
                R + new_recovered,
                D + new_deaths
            ])
        else:
            raise NotImplementedError(
                f"Model type {self.model_type} is not implemented."
            )

    def sird_simulate(self, initial_state, T, theta=None):
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
            states[t + 1] = self.sird_step(
                states[t],
                theta
            )

        return states

    def objective(self, theta, initial_state, T, C_obs, D_obs):

        states = self.sird_simulate(
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

    def analyzer(self, result):
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

    def fit_model(self, start, end):
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
            
            bounds = [
                (1e-6, 5.0),   # beta
                (1e-6, 1.0),   # gamma
                (1e-8, 0.2)    # mu
            ]

        result = minimize(
            self.objective,
            x0=x0,
            args=(initial_state, end - start + 1, self.case_data["confirmed"].iloc[start:end+1], self.case_data["deaths"].iloc[start:end+1]),
            method="L-BFGS-B",
            bounds=bounds
        )

        self.analyzer(result)

        return result
    
    def plot_results(self, start, end=None, save_path=None):
        if end is None:
            end = len(self.case_data) - 1
        if not self.fited:
            print("Model has not been fitted yet. Please call fit_model() first.")
            return
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
        states = self.sird_simulate(
            initial_state=initial_state,
            T=end - start + 1,
            theta=theta
        )

        C_pred = self.total_population - states[:, 0]

        plt.figure(figsize=(12, 6))

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
        if self.model_type == "SIRD":
            D_pred = states[:, 3]
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

        plt.xlabel("Date")
        plt.ylabel("Cumulative cases")
        plt.legend()
        plt.xticks(rotation=30)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.title(f"Fitted SIR Model for {self.id}")
        if save_path:
            plt.savefig(save_path + f"/{self.id}_{self.model_type}.png")
        else:
            plt.show()