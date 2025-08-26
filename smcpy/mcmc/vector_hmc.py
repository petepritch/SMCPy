import os, multiprocessing

os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={multiprocessing.cpu_count()}"
)

import jax
import jax.numpy as jnp
import blackjax
from functools import partial


class VectorHMC:
    def __init__(self, model, data, priors, log_like_args=None, log_like_func=None):
        self.model = model
        self.data = jnp.array(data)
        self.priors = priors
        self.log_like_func = log_like_func or self._normal_log_like
        self.log_like_args = log_like_args

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_model(self, params):
        return self.model(params)

    @partial(jax.jit, static_argnums=(0,))
    def _normal_log_like(self, predicted, data, sigma):
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

    def run_hmc(
        self,
        rng_key,
        initial_params,
        num_samples,
        step_size,
        inv_mass_matrix,
        num_integration_steps,
        phi=1.0,
    ):
        """Run HMC for multiple chains (particles)"""
        log_prob_fn = jax.jit(lambda p: self.evaluate_log_posterior(p, phi))

        num_chains = multiprocessing.cpu_count()

        def inference_loop(rng_key, kernel, initial_state, num_samples):

            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)

            return states

        inference_loop_multiple_chains = jax.pmap(
            inference_loop,
            in_axes=(0, None, 0, None),
            static_broadcasted_argnums=(1, 3),
        )

        hmc = blackjax.hmc(
            log_prob_fn, step_size, inv_mass_matrix, num_integration_steps
        )

        if initial_params.ndim == 1:
            initial_params = jnp.tile(initial_params, (num_chains, 1))
        elif initial_params.shape[0] != num_chains:
            if initial_params.shape[0] == 1:
                initial_params = jnp.tile(initial_params, (num_chains, 1))
            else:
                initial_params = initial_params[:num_chains]

        initial_states = jax.vmap(hmc.init, in_axes=(0,))(initial_params)

        ### PMAP STARTS HERE ###

        rng_key, sample_key = jax.random.split(rng_key)
        sample_keys = jax.random.split(sample_key, num_chains)

        pmap_states = inference_loop_multiple_chains(
            sample_keys, hmc.step, initial_states, num_samples
        )

        final_positions = pmap_states.position[:, -1, :].block_until_ready()

        print(final_positions.shape)

        log_likes = self.evaluate_log_likelihood(final_positions)
        return jax.random.fold_in(rng_key, 0), final_positions, log_likes

        ### VMAP starts here ###
        # @jax.jit
        # def one_step(states, rng_key):
        #     keys = jax.random.split(rng_key, batch_size)
        #     states, _ = jax.vmap(hmc.step)(keys, states)
        #     return states, states

        # keys = jax.random.split(rng_key, num_samples)
        # _, states = jax.lax.scan(one_step, initial_states, keys)

        # print(states.position[-1])
        # final_positions = (
        #     states.position[-1]
        #     if hasattr(states.position, "ndim") and states.position.ndim == 3
        #     else states.position
        # )
        # log_likes = self.evaluate_log_likelihood(final_positions)
        # return jax.random.fold_in(rng_key, 0), final_positions, log_likes
