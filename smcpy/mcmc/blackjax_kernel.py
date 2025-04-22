import numpy as np
import jax
import jax.numpy as jnp

from .kernel_base import MCMCKernel


class BlackJaxKernel(MCMCKernel):
    """
    A MCMC kernel implementation using BlackJax's RMH.
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
    """

    def __init__(
        self,
        jax_mcmc,
        param_order,
        path=None,
        rng=None,
        algorithm="rmh",
        sampler_kwargs=None,
    ):
        super().__init__(None, param_order, path, rng)
        self._jax_mcmc = jax_mcmc

        self.algorithm = algorithm
        self.sampler_kwargs = sampler_kwargs or {}

        if algorithm == "rmh" and "sigma" not in self.sampler_kwargs:
            self.sampler_kwargs["sigma"] = jnp.eye(len(param_order)) * 0.1

        seed = int(self.rng.integers(0, 2**32 - 1))
        self._jax_rng_key = jax.random.PRNGKey(seed)

    def mutate_particles(self, param_dict, num_samples, cov, phi=None):
        if phi is None:
            phi = self.path.phi

        param_array = self._conv_param_dict_to_array(param_dict)
        jax_params = jnp.array(param_array)

        if self.algorithm == "rmh":
            self.sampler_kwargs["sigma"] = jnp.array(cov)

        (
            self._jax_rng_key,
            positions,
            log_likes,
        ) = self._jax_mcmc.run_mcmc(
            self._jax_rng_key,
            self.algorithm,
            jax_params,
            num_samples,
            sampler_kwargs=self.sampler_kwargs,
            phi=phi,
        )

        new_params_np = np.array(positions)
        log_likes_np = np.array(log_likes).reshape(-1, 1)
        new_param_dict = self._conv_param_array_to_dict(new_params_np)
        return new_param_dict, log_likes_np

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
