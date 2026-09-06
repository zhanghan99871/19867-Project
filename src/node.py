import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Node:
    def __init__(self, id, model_type="SIRD", case_data=None, total_population=0):
        self.id = id
        if model_type == "SIRD":
            self.params = {
                "beta": 0.25,
                "gamma": 0.1,
                "mu": 0.005
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

        S, I, R, D = state
        beta, gamma, mu = theta

        N = self.total_population

        new_infected = beta * S * I / N
        new_recovered = gamma * I
        new_deaths = mu * I

        return np.array([
            S - new_infected,
            I + new_infected - new_recovered - new_deaths,
            R + new_recovered,
            D + new_deaths
        ])

    def sird_simulate(self, initial_state, T, theta=None):

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

    def objective(self, theta, initial_state):

        T = len(self.case_data)

        states = self.sird_simulate(
            initial_state=initial_state,
            T=T,
            theta=theta
        )

        S_pred = states[:, 0]
        D_pred = states[:, 3]

        # cumulative confirmed cases
        C_pred = self.total_population - S_pred

        C_obs = self.case_data["confirmed"].to_numpy()
        D_obs = self.case_data["deaths"].to_numpy()

        C_scale = max(C_obs.max(), 1)
        D_scale = max(D_obs.max(), 1)

        case_loss = np.mean(
            ((C_pred - C_obs) / C_scale) ** 2
        )

        death_loss = np.mean(
            ((D_pred - D_obs) / D_scale) ** 2
        )

        return case_loss + death_loss

    def fit_model(self):
        initial_state = np.array([
            self.total_population - self.case_data["confirmed"].iloc[0],
            self.case_data["confirmed"].iloc[0],
            0,
            0
        ])
        x0 = np.array([
            self.params["beta"],
            self.params["gamma"],
            self.params["mu"]
        ])

        result = minimize(
            self.objective,
            x0=x0,
            args=(initial_state,),
            method="L-BFGS-B",
            bounds=[
                (1e-6, 2.0),   # beta
                (1e-6, 1.0),   # gamma
                (1e-8, 0.2)    # mu
            ]
        )

        beta_hat, gamma_hat, mu_hat = result.x

        # save fitted parameters
        self.params["beta"] = beta_hat
        self.params["gamma"] = gamma_hat
        self.params["mu"] = mu_hat

        print("success:", result.success)
        print("loss:", result.fun)
        print("beta:", beta_hat)
        print("gamma:", gamma_hat)
        print("mu:", mu_hat)

        return result
    
    def plot_results(self):
        initial_state = np.array([
            self.total_population - self.case_data["confirmed"].iloc[0],
            self.case_data["confirmed"].iloc[0],
            0,
            0
        ])
        print("Initial state:", initial_state)
        states = self.sird_simulate(
            initial_state=initial_state,
            T=len(self.case_data),
            theta=(self.params["beta"], self.params["gamma"], self.params["mu"])
        )

        C_pred = self.total_population - states[:, 0]
        D_pred = states[:, 3]

        plt.figure(figsize=(10, 5))

        plt.plot(
            self.case_data["date"],
            self.case_data["confirmed"],
            label="JHU confirmed"
        )

        plt.plot(
            self.case_data["date"],
            C_pred,
            "--",
            label="SIRD predicted"
        )

        plt.xlabel("Date")
        plt.ylabel("Cumulative cases")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))

        plt.plot(
            self.case_data["date"],
            self.case_data["deaths"],
            label="JHU deaths"
        )

        plt.plot(
            self.case_data["date"],
            D_pred,
            "--",
            label="SIRD predicted"
        )

        plt.xlabel("Date")
        plt.ylabel("Cumulative deaths")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()