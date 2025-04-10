import numpy as np
import jax
import jax.numpy as jnp
import blackjax

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
    def __init__(self, mcmc_object, param_order, path=None, rng=None):
        super().__init__(mcmc_object, param_order, path, rng)
        self._mcmc.evaluate_log_posterior = self.path.logpdf
        
        # For BlackJax
        seed = int(self.rng.integers(0, 2**32 - 1))
        self._jax_rng_key = jax.random.PRNGKey(seed)
    
    def mutate_particles(self, param_dict, num_samples, cov):
        param_array = self._conv_param_dict_to_array(param_dict)
        num_particles = param_array.shape[0]
        
        new_params = np.zeros_like(param_array)
        log_likes = np.zeros(num_particles)
        
        phi = self.path.phi
        
        # IMPORTANT: THIS DOES NOT USE JIT
        def log_prob_fn(params):
            params_np = np.array(params).reshape(1, -1)
            log_prior = np.sum(self._mcmc.evaluate_log_priors(params_np))
            log_like = self._mcmc.evaluate_log_likelihood(params_np)[0]
            
            return float(log_prior + phi * log_like)
        
        # IMPORTANT: THIS DOES USE JIT
        def log_prob_fn_jax(params):

            def log_prob_fn_np(params_flat):
                params_np = np.array(params_flat).reshape(1, -1)
                log_prior = np.sum(self._mcmc.evaluate_log_priors(params_np))
                log_like = self._mcmc.evaluate_log_likelihood(params_np)[0]
                
                return float(log_prior + phi * log_like)
        
            result_shape_dtype = jax.ShapeDtypeStruct((), jnp.float32)
            
            
            return jax.pure_callback(log_prob_fn_np, result_shape_dtype, params)
        
        # JIT compile the log probability function
        # Not being used as it won't work with NumPy conversion
        jitted_log_prob = jax.jit(log_prob_fn_jax)
        
        # Extract standard deviations for the random walk
        std_devs = np.sqrt(np.diag(cov))
        
        # Proposal generator function 
        def proposal_generator(rng_key, position):
            """Generate a new position by adding Gaussian noise"""
            return position + std_devs * jax.random.normal(rng_key, shape=position.shape)
        
        # Replace log_prob_fn with jitted_log_prob if workaround exists
        # Then maybe can vectorize with vmap()?
        rmh = blackjax.rmh(jitted_log_prob, proposal_generator)
        
        for i in range(num_particles):
            current_params = param_array[i]
            
            position = jnp.array(current_params)
            current_key = self._jax_rng_key
            
            state = rmh.init(position)
            
            for _ in range(num_samples):
                current_key, step_key = jax.random.split(current_key)
                state, info = rmh.step(step_key, state)
                
            new_params[i] = np.array(state.position)
            log_likes[i] = self._mcmc.evaluate_log_likelihood(new_params[i].reshape(1, -1))[0]
            
            self._jax_rng_key = current_key
        
        new_param_dict = self._conv_param_array_to_dict(new_params)
        
        return new_param_dict, log_likes
    
    def sample_from_prior(self, num_samples):
        param_array = self._mcmc.sample_from_priors(num_samples)
        return self._conv_param_array_to_dict(param_array)
    
    def sample_from_proposal(self, num_samples):
        if not self.has_proposal():
            raise ValueError("No proposal distribution available")
        
        param_array = self.path.proposal.rvs(num_samples, random_state=self.rng)
        return self._conv_param_array_to_dict(param_array)
    
    def get_log_likelihoods(self, param_dict):
        param_array = self._conv_param_dict_to_array(param_dict)
        return self._mcmc.evaluate_log_likelihood(param_array)
    
    def get_log_priors(self, param_dict):
        param_array = self._conv_param_dict_to_array(param_dict)
        log_priors = self._mcmc.evaluate_log_priors(param_array)
        return np.sum(log_priors, axis=1).reshape(-1, 1)
    
    def set_mcmc_rng(self, rng):
        self._mcmc.rng = rng
        
        # Also update JAX RNG key
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