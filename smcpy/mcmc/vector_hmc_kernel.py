import numpy as np
import jax
import jax.numpy as jnp
import blackjax

from .kernel_base import KernelBase


class HMCKernel(KernelBase):
    """
    A HMC kernel implementation using BlackJax's HMC.
    Documentation: https://blackjax-devs.github.io/blackjax/


    Parameters
    ----------
    mcmc_object : VectorMCMC object
        The VectorMCMC object that contains the model evaluations
    param_order : list or tuple
        The order of parameters
    path : GeometricPath, optional
        Path object for transitioning between distributions
    rng : numpy.random.Generator, optional
        Random number generator
    target_accept : float, optional
        Target acceptance ratio
    """

    def __init__(self, jax_mcmc, param_order, path=None, rng=None, target_accept=0.75):
        super().__init__(None, param_order, path, rng)
        self._jax_mcmc = jax_mcmc
        self.target_accept = target_accept
        seed = int(self.rng.integers(0, 2**32 - 1))
        self._jax_rng_key = jax.random.PRNGKey(seed)

    def mutate_particles(self, param_dict, num_samples, cov, phi=None):
        if phi is None:
            phi = self.path.phi

        param_array = self._conv_param_dict_to_array(param_dict)
        jax_params = jnp.array(param_array)

        inv_mass_matrix = jnp.linalg.inv(jnp.array(cov) + 1e-6 * jnp.eye(cov.shape[0]))
        step_size = self._tune_step_size(jax_params, inv_mass_matrix, phi)
        num_integration_steps = self._tune_num_steps(
            jax_params, inv_mass_matrix, step_size, phi
        )

        (
            self._jax_rng_key,
            positions,
            log_likes,
        ) = self._jax_mcmc.run_hmc(
            self._jax_rng_key,
            jax_params,
            num_samples,
            step_size,
            inv_mass_matrix,
            num_integration_steps,
            phi=phi,
        )

        new_params_np = np.array(positions)
        log_likes_np = np.array(log_likes).reshape(-1, 1)
        new_param_dict = self._conv_param_array_to_dict(new_params_np)
        return new_param_dict, log_likes_np

    def _tune_step_size(self, init_params, inv_mass_matrix, phi, n_iter=50):
        """
        Dual averaging adaptation for step size
        """
        import math

        log_prob_fn = lambda p: self._jax_mcmc.evaluate_log_posterior(p, phi)
        hmc = lambda eps: blackjax.hmc(
            log_prob_fn, eps, inv_mass_matrix, num_integration_steps=2
        )

        eps = 1.0
        mu = math.log(eps)
        epsbar, Hbar = 1.0, 0.0
        gamma, t0, kappa = 0.05, 10.0, 0.75

        init_param = init_params[0] if init_params.ndim > 1 else init_params
        state = hmc(eps).init(init_param)

        rng_key = self._jax_rng_key

        for t in range(1, n_iter + 1):
            rng_key, sub_key = jax.random.split(rng_key)
            kernel = hmc(eps)
            new_state, info = kernel.step(sub_key, state)

            acc = float(info.acceptance_rate)

            if t < 10:
                Hbar = (1 - 1 / (t + t0)) * Hbar + (1 / (t + t0)) * (
                    self.target_accept - acc
                )
                logeps = mu - (jnp.sqrt(t) / gamma) * Hbar
                eps = jnp.exp(jnp.clip(logeps, -10, 3))

                w = t**-kappa
                epsbar = jnp.exp(w * logeps + (1 - w) * jnp.log(epsbar))

            state = new_state

        return float(jnp.clip(epsbar, 1e-5, 5.0))

    def _tune_num_steps(
        self, init_params, inv_mass_matrix, step_size, phi, taus=(1.0, 2.0, 3.0)
    ):
        """
        Pick num_integration_steps = tau/eps with best ESJD/grad
        """
        log_prob_fn = lambda p: self._jax_mcmc.evaluate_log_posterior(p, phi)
        best_tau, best_score = taus[0], -jnp.inf

        for tau in taus:
            L = int(jnp.ceil(tau / step_size))
            hmc = blackjax.hmc(
                log_prob_fn, step_size, inv_mass_matrix, num_integration_steps=L
            )
            state = hmc.init(init_params[0])
            _, subkey = jax.random.split(self._jax_rng_key)
            new_state, _ = hmc.step(subkey, state)
            esjd = jnp.sum((new_state.position - state.position) ** 2)
            score = esjd / L
            if score > best_score:
                best_tau, best_score = tau, score

        return int(jnp.ceil(best_tau / step_size))

    def sample_from_prior(self, num_samples):
        self._jax_rng_key, sub_key = jax.random.split(self._jax_rng_key)
        samples = self._jax_mcmc.sample_from_priors(sub_key, num_samples)
        param_array = np.array(samples)
        return self._conv_param_array_to_dict(param_array)

    def sample_from_proposal(self, num_samples):
        if not self.has_proposal():
            raise ValueError("No proposal distribution available")

        param_array = self.path.proposal.rvs(num_samples, random_state=self.rng)
        return self._conv_param_array_to_dict(param_array)

    def get_log_likelihoods(self, param_dict):
        param_array = self._conv_param_dict_to_array(param_dict)
        jax_params = jnp.array(param_array)
        log_likes = self._jax_mcmc.evaluate_log_likelihood(jax_params)
        return np.array(log_likes).reshape(-1, 1)

    def get_log_priors(self, param_dict):
        param_array = self._conv_param_dict_to_array(param_dict)
        jax_params = jnp.array(param_array)
        log_priors = self._jax_mcmc.evaluate_log_priors(jax_params)
        return np.sum(np.array(log_priors), axis=1).reshape(-1, 1)

    def set_mcmc_rng(self, rng):
        seed = int(rng.integers(0, 2**32 - 1))
        self._jax_rng_key = jax.random.PRNGKey(seed)

    def _conv_param_array_to_dict(self, param_array):
        return dict(zip(self._param_order, param_array.T))

    def _conv_param_dict_to_array(self, param_dict):
        dim0 = 1
        if not isinstance(param_dict[self._param_order[0]], (int, float)):
            dim0 = len(param_dict[self._param_order[0]])
        param_array = np.zeros((dim0, len(self._param_order)))

        for i, k in enumerate(self._param_order):
            if param_dict[k].dtype == "object":
                param_array = param_array.astype("object")
            param_array[:, i] = param_dict[k]
        return param_array
