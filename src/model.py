import numpy as np
class SIR:
    def __init__(self, theta, N, method="euler"):
        self.beta, self.gamma = theta[0], theta[1]
        self.N = N
        self.method = method
        
    def reset(self, theta):
        self.beta, self.gamma = theta[0], theta[1]
    
    def Euler_step(self, states, t):
        S, I, R = states
        new_infected = (self.beta * S * I) / self.N
        new_recovered = self.gamma * I

        S_next = S - new_infected
        I_next = I + new_infected - new_recovered
        R_next = R + new_recovered

        return S_next, I_next, R_next

    def RK4_step(self, states, t):
        S, I, R = states

        def dSdt(S, I):
            return -self.beta * S * I / self.N

        def dIdt(S, I):
            return self.beta * S * I / self.N - self.gamma * I

        def dRdt(I):
            return self.gamma * I

        k1_S = dSdt(S, I)
        k1_I = dIdt(S, I)
        k1_R = dRdt(I)

        k2_S = dSdt(S + 0.5 * k1_S, I + 0.5 * k1_I)
        k2_I = dIdt(S + 0.5 * k1_S, I + 0.5 * k1_I)
        k2_R = dRdt(I + 0.5 * k1_I)

        k3_S = dSdt(S + 0.5 * k2_S, I + 0.5 * k2_I)
        k3_I = dIdt(S + 0.5 * k2_S, I + 0.5 * k2_I)
        k3_R = dRdt(I + 0.5 * k2_I)

        k4_S = dSdt(S + k3_S, I + k3_I)
        k4_I = dIdt(S + k3_S, I + k3_I)
        k4_R = dRdt(I + k3_I)

        S_next = S + (k1_S + 2*k2_S + 2*k3_S + k4_S) / 6
        I_next = I + (k1_I + 2*k2_I + 2*k3_I + k4_I) / 6
        R_next = R + (k1_R + 2*k2_R + 2*k3_R + k4_R) / 6

        return S_next, I_next, R_next

    def step(self, states, t):
        if self.method == "euler":
            return self.Euler_step(states, t)
        elif self.method == "rk4":
            return self.RK4_step(states, t)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
class SIRD(SIR):
    def __init__(self, theta, N, method="euler"):
        super().__init__(theta[:2], N, method)
        self.mu = theta[2]
    
    def reset(self, theta):
        super().reset(theta[:2])
        self.mu = theta[2]

    def Euler_step(self, states, t):
        S, I, R, D = states
        new_infected = (self.beta * S * I) / self.N
        new_recovered = self.gamma * I
        new_deaths = self.mu * I

        S_next = S - new_infected
        I_next = I + new_infected - new_recovered - new_deaths
        R_next = R + new_recovered
        D_next = D + new_deaths

        return S_next, I_next, R_next, D_next

    def RK4_step(self, states, t):
        S, I, R, D = states

        def dSdt(S, I):
            return -self.beta * S * I / self.N

        def dIdt(S, I):
            return self.beta * S * I / self.N - self.gamma * I - self.mu * I

        def dRdt(I):
            return self.gamma * I

        def dDdt(I):
            return self.mu * I

        k1_S = dSdt(S, I)
        k1_I = dIdt(S, I)
        k1_R = dRdt(I)
        k1_D = dDdt(I)

        k2_S = dSdt(S + 0.5 * k1_S, I + 0.5 * k1_I)
        k2_I = dIdt(S + 0.5 * k1_S, I + 0.5 * k1_I)
        k2_R = dRdt(I + 0.5 * k1_I)
        k2_D = dDdt(I + 0.5 * k1_I)

        k3_S = dSdt(S + 0.5 * k2_S, I + 0.5 * k2_I)
        k3_I = dIdt(S + 0.5 * k2_S, I + 0.5 * k2_I)
        k3_R = dRdt(I + 0.5 * k2_I)
        k3_D = dDdt(I + 0.5 * k2_I)

        k4_S = dSdt(S + k3_S, I + k3_I)
        k4_I = dIdt(S + k3_S, I + k3_I)
        k4_R = dRdt(I + k3_I)
        k4_D = dDdt(I + k3_I)
        
        S_next = S + (k1_S + 2*k2_S + 2*k3_S + k4_S) / 6
        I_next = I + (k1_I + 2*k2_I + 2*k3_I + k4_I) / 6
        R_next = R + (k1_R + 2*k2_R + 2*k3_R + k4_R) / 6
        D_next = D + (k1_D + 2*k2_D + 2*k3_D + k4_D) / 6
        
        return S_next, I_next, R_next, D_next
    
class SIVR(SIR):
    def __init__(self, theta, N, method="euler"):
        super().__init__(theta[:2], N, method)
        self.sigma = theta[2]
    
    def reset(self, theta):
        super().reset(theta[:2])
        self.sigma = theta[2]

    def Euler_step(self, states, t):
        S, I, R, V = states
        new_infected = (self.beta * S * I) / self.N
        new_recovered = self.gamma * I
        new_vaccinated = self.sigma * S

        S_next = S - new_infected - new_vaccinated
        I_next = I + new_infected - new_recovered
        R_next = R + new_recovered
        V_next = V + new_vaccinated

        return S_next, I_next, R_next, V_next

    def RK4_step(self, states, t):
        S, I, R, V = states

        def dSdt(S, I):
            return -self.beta * S * I / self.N - self.sigma * S

        def dIdt(S, I):
            return self.beta * S * I / self.N - self.gamma * I

        def dRdt(I):
            return self.gamma * I

        def dVdt(S):
            return self.sigma * S

        k1_S = dSdt(S, I)
        k1_I = dIdt(S, I)
        k1_R = dRdt(I)
        k1_V = dVdt(S)

        k2_S = dSdt(S + 0.5 * k1_S, I + 0.5 * k1_I)
        k2_I = dIdt(S + 0.5 * k1_S, I + 0.5 * k1_I)
        k2_R = dRdt(I + 0.5 * k1_I)
        k2_V = dVdt(S + 0.5 * k1_S)

        k3_S = dSdt(S + 0.5 * k2_S, I + 0.5 * k2_I)
        k3_I = dIdt(S + 0.5 * k2_S, I + 0.5 * k2_I)
        k3_R = dRdt(I + 0.5 * k2_I)
        k3_V = dVdt(S + 0.5 * k2_S)
        
        k4_S = dSdt(S + k3_S, I + k3_I)
        k4_I = dIdt(S + k3_S, I + k3_I)
        k4_R = dRdt(I + k3_I)   
        k4_V = dVdt(S + k3_S)
        
        S_next = S + (k1_S + 2*k2_S + 2*k3_S + k4_S) / 6
        I_next = I + (k1_I + 2*k2_I + 2*k3_I + k4_I) / 6
        R_next = R + (k1_R + 2*k2_R + 2*k3_R + k4_R) / 6
        V_next = V + (k1_V + 2*k2_V + 2*k3_V + k4_V) / 6
        
        return S_next, I_next, R_next, V_next
    
import numpy as np


class NetworkSIRSimple:
    """
    Network SIR without migration.

    lambda_i =
        beta_i * I_i / N_i
        +
        beta_travel_i *
        sum_j (F_ij / N_i) * (I_j / N_j)

    states = (S, I, R)
    """

    def __init__(
        self,
        theta,
        N,
        flow_matrix_dict,
        method="euler",
    ):
        self.N = np.asarray(N, dtype=float)
        self.flow_matrix_dict = flow_matrix_dict
        self.method = method
        self.reset(theta)

    def reset(self, theta):
        beta, gamma, beta_travel = theta

        self.beta = np.asarray(beta, dtype=float)
        self.gamma = np.asarray(gamma, dtype=float)
        self.beta_travel = np.asarray(
            beta_travel,
            dtype=float,
        )

        n = len(self.N)

        for name, x in [
            ("beta", self.beta),
            ("gamma", self.gamma),
            ("beta_travel", self.beta_travel),
        ]:
            if x.shape != (n,):
                raise ValueError(
                    f"{name} must have shape ({n},), "
                    f"got {x.shape}"
                )

    def get(self):
        return (
            self.beta.copy(),
            self.gamma.copy(),
            self.beta_travel.copy(),
        )

    def _flow_matrix(self, t):
        F = np.asarray(
            self.flow_matrix_dict[t],
            dtype=float,
        ).copy()

        if F.shape != (len(self.N), len(self.N)):
            raise ValueError(
                f"Invalid flow matrix shape: {F.shape}"
            )

        np.fill_diagonal(F, 0)

        return F

    def _rhs(self, states, t):
        S, I, R = np.asarray(states, dtype=float)

        # Infection prevalence I_i / N_i
        prevalence = I / self.N

        # F_ij / N_i
        F = self._flow_matrix(t)
        mobility = F / self.N[:, None]

        # Force of infection
        lambda_local = self.beta * prevalence

        lambda_travel = self.beta_travel * (mobility @ prevalence)

        infection = S * (lambda_local + lambda_travel)

        recovery = self.gamma * I

        return np.array([
            -infection,
            infection - recovery,
            recovery,
        ])

    def Euler_step(self, states, t):
        y = np.asarray(states, dtype=float)

        return tuple(
            y + self._rhs(y, t)
        )

    def RK4_step(self, states, t):
        y = np.asarray(states, dtype=float)

        k1 = self._rhs(y, t)
        k2 = self._rhs(y + k1 / 2, t)
        k3 = self._rhs(y + k2 / 2, t)
        k4 = self._rhs(y + k3, t)

        return tuple(
            y
            + (
                k1
                + 2 * k2
                + 2 * k3
                + k4
            ) / 6
        )

    def step(self, states, t):
        if self.method == "euler":
            return self.Euler_step(states, t)

        if self.method == "rk4":
            return self.RK4_step(states, t)

        raise ValueError(
            f"Unknown method: {self.method}"
        )