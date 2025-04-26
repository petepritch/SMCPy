import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import jax
import jax.numpy as jnp
import blackjax
from functools import partial


class JAXMCMC:
    """
    :param model: maps inputs to outputs
    :type model: callable
    :param data: data corresponding to model outputs
    :type data: 1D array
    :param priors: list of JAX distributions
    :type priors: list of objects
    :log_like_func: JAX function that takes inputs, model, data, and
        hyperparameters and returns log likelihoods (default is normal)
    :type log_like_func: callable
    :log_like_args: any fixed parameters that define the likelihood
        function (e.g., standard deviation for a Gaussian likelihood).
    :type log_like_args: 1D array or None
    """

    def __init__(
        self,
        model,
        data,
        priors,
        log_like_args=None,
        log_like_func=None,
    ):
        self.model = model
        self.data = jnp.array(data)
        self.priors = priors

        if log_like_func is None:
            self.log_like_func = self._normal_log_like
        else:
            self.log_like_func = log_like_func

        self.log_like_args = log_like_args

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_model(self, params):
        return self.model(params)

    @partial(jax.jit, static_argnums=(0,))
    def _normal_log_like(self, predicted, data, sigma):
        # Original implementation is written differently, precision reasons?
        residuals = data - predicted
        log_like = -0.5 * jnp.sum(residuals**2) / (sigma**2)
        log_like -= len(data) * jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
        return log_like

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_log_likelihood(self, params):
        eval_single = lambda p: self.log_like_func(
            self.evaluate_model(p), self.data, self.log_like_args
        )
        return jax.vmap(eval_single)(params) if params.ndim > 1 else eval_single(params)

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_log_priors(self, params):
        def eval_priors_single(p):
            return jnp.array(
                [prior.log_prob(p[i]) for i, prior in enumerate(self.priors)]
            )

        return (
            jax.vmap(eval_priors_single)(params)
            if params.ndim > 1
            else eval_priors_single(params)
        )

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_log_posterior(self, params, phi=1.0):
        if params.ndim == 1:
            log_like = self.evaluate_log_likelihood(params)
            log_priors = self.evaluate_log_priors(params)
            return jnp.sum(log_priors) + phi * log_like
        else:
            return jax.vmap(lambda p: self.evaluate_log_posterior(p, phi))(params)

    def sample_from_priors(self, rng_key, num_samples):
        samples = []

        for prior in self.priors:
            rng_key, subkey = jax.random.split(rng_key)
            samples.append(prior.sample(seed=subkey, sample_shape=num_samples))

        return jnp.column_stack(samples)

    def run_mcmc(
        self,
        rng_key,
        algorithm,
        initial_params,
        num_samples,
        sampler_kwargs=None,
        phi=1.0,
    ):
        """Run MCMC using BlackJax with vectorized particle processing"""
        if sampler_kwargs is None:
            sampler_kwargs = {}

        def inference_loop(rng_key, kernel, initial_state, num_samples):

            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)

            return states

        def inference_loop_multiple_chains(
            rng_key, kernel, initial_state, num_samples, num_chains
        ):

            @jax.jit
            def one_step(states, rng_key):
                keys = jax.random.split(rng_key, num_chains)
                states, _ = jax.vmap(kernel)(keys, states)
                return states, states

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)

            return states

        log_prob_fn = jax.jit(lambda p: self.evaluate_log_posterior(p, phi))

        if initial_params.ndim == 1:
            initial_params = initial_params.reshape(1, -1)

        batch_size, num_params = initial_params.shape

        if algorithm == "rmh":

            sigma = sampler_kwargs.pop("sigma", jnp.eye(num_params) * 0.1)
            rw = blackjax.additive_step_random_walk(
                log_prob_fn, blackjax.mcmc.random_walk.normal(sigma)
            )
            initial_states = jax.vmap(rw.init, in_axes=(0))(initial_params)
            rng_key, sample_key = jax.random.split(rng_key)
            new_states = inference_loop_multiple_chains(
                sample_key, rw.step, initial_states, num_samples, batch_size
            )

        elif algorithm == "nuts":

            inv_mass_matrix = sampler_kwargs.pop("inv_mass_matrix", jnp.eye(num_params))
            step_size = sampler_kwargs.pop("step_size", 1e-3)
            nuts = blackjax.nuts(log_prob_fn, step_size, inv_mass_matrix)
            initial_states = jax.vmap(nuts.init, in_axes=(0))(initial_params)
            rng_key, sample_key = jax.random.split(rng_key)
            new_states = inference_loop_multiple_chains(
                sample_key, nuts.step, initial_states, num_samples, batch_size
            )

        elif algorithm == "mala":

            step_size = sampler_kwargs.pop("step_size", 1e-3)
            mala = blackjax.mala(log_prob_fn, step_size)
            initial_states = jax.vmap(mala.init, in_axes=(0))(initial_params)
            rng_key, sample_key = jax.random.split(rng_key)
            new_states = inference_loop_multiple_chains(
                sample_key, mala.step, initial_states, num_samples, batch_size
            )

        else:
            raise NotImplementedError(
                f"Vectorized implementation for {algorithm} not implemented."
            )

        try:
            positions = new_states.position
        except (AttributeError, TypeError):
            positions = new_states

        if hasattr(positions, "ndim") and positions.ndim == 3:
            final_positions = positions[-1]
        else:
            final_positions = positions

        log_likes = self.evaluate_log_likelihood(final_positions)
        return jax.random.fold_in(rng_key, 0), final_positions, log_likes
