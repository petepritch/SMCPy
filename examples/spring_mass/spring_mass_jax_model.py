import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

# ------------------------------------------------------
# Helper function to use scipy integrator in model class


def mass_spring(state, t, K, g):
    """
    Return velocity/acceleration given velocity/position and values for
    stiffness and mass
    """
    # unpack the state vector
    x = state[0]
    xd = state[1]

    # compute acceleration xdd
    xdd = -K * x + g

    # return the two state derivatives
    return [xd, xdd]


# ------------------------------------------------------
class SpringMassModel:
    """
    Defines Spring Mass model with 2 free params (spring stiffness, k & mass, m)
    """

    def __init__(self, state0=None, time_grid=None):
        if state0 is None:
            state0 = jnp.array([0.0, 0.0])
        else:
            state0 = jnp.array(state0)
        if time_grid is None:
            time_grid = jnp.arange(0.0, 10.0, 0.1)
        else:
            time_grid = jnp.array(time_grid)

        self._state0 = state0
        self._t = time_grid

    def evaluate(self, params):
        """
        Simulate spring mass system for given spring constant. Returns state
        (position, velocity) at all points in time grid
        """

        def _simulate_single(param):
            K, g = param

            def _ode_func(state, t):
                return mass_spring(state, t, K, g)

            solution = odeint(_ode_func, self._state0, self._t)

            return solution[:, 0]

        vmap_simulate = jax.vmap(_simulate_single)
        params_array = jnp.array(params)

        if params_array.ndim == 1:
            return _simulate_single(params_array)
        else:
            return vmap_simulate(params_array)
