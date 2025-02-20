import numpy as np
from scipy.integrate import odeint

def dz_dt(state, t, theta):
    """
    Lotka-Voltera equations. Real positive parameters 'alpha', 'beta', 'gamma',
    'delta' describes the interaction of two species.
    """
    u = state[0] # prey population
    v = state[1] # predator population
    alpha, beta, gamma, delta = (
        theta[..., 0], # prey growth rate
        theta[..., 1], # predation rate
        theta[..., 2], # predator growth rate
        theta[..., 3], # predator death rate
    )
    du_dt = (alpha - beta * v) * u
    dv_dt = (-gamma + delta * u) * v
    return np.stack([du_dt, dv_dt])

class LotkaVolteraModel:
    """
    Defines Lotka-Volterra predator-prey model with 4 parameters
    """
    def __init__(self, state0=None, time_grid=None):
        if state0 is None:
            state0 = [10.0, 5.0] # Initital prey and predator populations
        if time_grid is None:
            time_grid = np.arange(0.0, 100.0, 0.1)

        self._state0 = state0
        self._t = time_grid

    def evaluate(self, theta):
        """
        Simulate Lotka-Voltera system for given parameters.
        params: array of parameter sets [alpha, beta, gamma, delta]
        Returns: population trajectories for each parameter set
        """
        results = []
        for t in theta:
            alpha, beta, gamma, delta = theta
            output = odeint(LotkaVolteraModel,
                        self._state0,
                        self._t,
                        args=(alpha, beta, gamma, delta))
            results.append(output[:, 0])
        
        return np.array(results)