import numpy as np
import jax.numpy as jnp
import blackjax
import jax

from abc import ABC, abstractmethod
from tqdm import tqdm

from .mcmc_base import MCMCBase

class BlackJaxMCMC(MCMCBase):
    """
    BlackJax MCMC interface
    
    BlackJax is a library of samplers for JAX that works on CPU as well as GPU.
    Documentation: https://blackjax-devs.github.io/blackjax/
    """
    
    def __init__(
        self, 
        model, 
        data, 
        priors, 
        log_like_args, 
        log_like_func, 
        sampler_type='nuts', 
        sampler_kwargs=None):
        """
        Initialize BlackJax MCMC sampler.

        Parameters
        ----------

        Returns
        -------
        """
        super().__init__(model, data, priors, log_like_args, log_like_func)

        self.sampler_type = sampler_type
        self.sampler_kwargs = sampler_kwargs or {}
        self._setup_blackjax_sampler()

    def evaluate_model(self, inputs):
        """Implement required by ABC"""
        return self._eval_model(inputs)
    
    def _setup_blackjax_sampler(self):
        """Configure sampler"""
        samplers = {
            'nuts': blackjax.nuts,
            'mala': blackjax.mala,
            'hmc': blackjax.hmc,
            'rmhmc': blackjax.rmhmc,
            'ghmc': blackjax.ghmc 
        }
    
        if self.sampler_type not in samplers:
            raise ValueError(f"Unsupported sampler: {self.sampler_type}. "
                            f"Choose from: {list(samplers.keys())}")
        
        self.blackjax_sampler = samplers[self.sampler_type]
    
    def _log_posterior_fn(self, params):
        """JAX-compatible log posterior function for BlackJax"""
        np_params = np.array(params).reshape(1, -1)

        log_priors = self.evaluate_log_priors(np_params)
        log_like = self.evaluate_log_likelihood(np_params)

        log_post = self.evaluate_log_posterior(log_like, log_priors)
        return float(log_post[0])

    def blackjax_sampling(
        self,
        initial_params, 
        num_samples,
        num_warmup=1000,
        step_size=0.1,
        target_acceptance=0.8,
        progress_bar=True
    ):
        """
        Run BlackJax sampler to generate posterior samples

        Parameters
        ----------

        Returns
        -------
        """
        if len(initial_params.shape) > 1:
            if initial_params.shape[0] > 1:
                print("Warning: Multiple initial states provided. Using only first.")
            initial_params = initial_params[0]

        log_post_fn = jax.jit(self._log_posterior_fn)

        if self.sampler_type == "nuts":
            # Window adaptation for NUTS
            adaptation = blackjax.window_adaptation(
                self.blackjax_sampler,
                log_post_fn,
                num_warmup,
                target_acceptance_rate=target_acceptance,
                **self.sampler_kwargs
            )

            initial_state, kernel, _ = adaptation.run(
                jax.random.PRNGKey(0),
                initial_params,
                step_size
            )
        
        else:
            # All other samplers
            kernel = self.blackjax_sampler(log_post_fn, step_size, **self.sampler_kwargs)
            initial_state = kernel.init(initial_params)

        samples = np.zeros((1, len(initial_params),  num_samples + 1))
        samples[0, :, 0] = initial_params

        state = initial_state
        for i in tqdm(range(1, num_samples + 1), disable=not progress_bar):
            state, _ = kernel.step(jax.random.PRNGKey(i), state)
            samples[0, :, i] = np.array(state.position)

        return samples 

    def metropolis(self, inputs, num_samples, cov, adapt_interval=None,
                adapt_delay=0, progress_bar=False, **kwargs):
        """
        Override metropolis method to use BlackJax sampler by default

        Added kwargs
        ------------
        use_native: use native MCMCBase metropolis implementation instead of 
            BlackJax
        num_warmup: number of warmpup/adaptation steps for BlackJax
        """
        if kwargs.pop('use_native', False):
            return super().metropolis(inputs, num_samples, cov, adapt_interval,
                                    adapt_delay, progress_bar, **kwargs)
        
        num_warmup = kwargs.pop('num_warmup', 1000)
        step_size = kwargs.pop('step_size', 0.1)
        target_acceptance = kwargs.pop('target_acceptance', 0.8)

        return self.blackjax_sampling(
            inputs,
            num_samples,
            num_warmup=num_warmup,
            step_size=step_size,
            target_acceptance=target_acceptance,
            progress_bar=progress_bar
        )