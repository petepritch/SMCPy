import jax
import jax.numpy as jnp
import blackjax
from functools import partial
import multiprocessing


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

        # curried log posterior with temperature
        def log_prob_fn(p):
            return self.evaluate_log_posterior(p, phi)

        if initial_params.ndim == 1:
            initial_params = initial_params.reshape(1, -1)

        batch_size, num_params = initial_params.shape

        if algorithm == "rmh":

            sigma = sampler_kwargs.pop("sigma", jnp.eye(num_params) * 0.1)
            rw = blackjax.additive_step_random_walk(
                log_prob_fn, blackjax.mcmc.random_walk.normal(sigma)
            )

            def inference_loop(rng_key, kernel, initial_state, num_samples, num_chains):

                @jax.jit
                def one_step(states, rng_key):
                    keys = jax.random.split(rng_key, num_chains)
                    states, _ = jax.vmap(kernel)(keys, states)
                    return states, states

                keys = jax.random.split(rng_key, num_samples)
                _, states = jax.lax.scan(one_step, initial_state, keys)

                return states

            initial_states = jax.vmap(rw.init, in_axes=(0))(initial_params)

            rng_key, sample_key = jax.random.split(rng_key)
            new_states = inference_loop(
                sample_key, rw.step, initial_states, num_samples, batch_size
            )

        else:
            raise NotImplementedError(
                f"Vectorized implementation for {algorithm} not implemented."
            )

        positions = new_states.position[-1]
        log_likes = self.evaluate_log_likelihood(positions)
        return jax.random.fold_in(rng_key, 0), positions, log_likes

        # if algorithm == "rmh":
        #     # Change this to "cov" for consistency?
        #     sigma = sampler_kwargs.pop("sigma", jnp.eye(num_params) * 0.1)

        #     states = initial_params
        #     log_probs = jax.vmap(log_prob_fn)(states)

        #     for step in range(num_samples):
        #         rng_key, step_key = jax.random.split(rng_key)
        #         particle_keys = jax.random.split(step_key, batch_size)

        #         proposal_noise = jax.random.normal(
        #             step_key, shape=(batch_size, num_params)
        #         ) * jnp.sqrt(jnp.diag(sigma))
        #         proposals = states + proposal_noise

        #         proposal_log_probs = jax.vmap(log_prob_fn)(proposals)

        #         log_accept_ratios = proposal_log_probs - log_probs

        #         accept_keys = jax.random.split(rng_key, batch_size)
        #         u = jax.random.uniform(rng_key, shape=(batch_size,))

        #         accept = log_accept_ratios > jnp.log(u)
        #         accept = jnp.expand_dims(accept, axis=1)

        #         states = jnp.where(accept, proposals, states)
        #         log_probs = jnp.where(accept[:, 0], proposal_log_probs, log_probs)

        #     final_positions = states
        # else:
        #     raise NotImplementedError(
        #         f"Vectorized implementation for {algorithm} not available yet. "
        #         "Use 'rmh' or implement a custom version."
        #     )
        # print(f"Final positions shape: {final_positions.shape}")
        # log_likes = self.evaluate_log_likelihood(final_positions)
        # print(f"Log likes shape: {log_likes.shape}")
        # return jax.random.fold_in(rng_key, 0), final_positions, log_likes
