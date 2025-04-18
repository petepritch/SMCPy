import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import distrax
from spring_mass_jax_model import SpringMassModel
from smcpy import AdaptiveSampler, VectorMCMC

# Import the simple BlackJax kernel
from smcpy.mcmc.blackjax_kernel import BlackJaxKernel
from smcpy.mcmc.blackjax_mcmc import JAXMCMC

# Initialize model
state0 = jnp.array([0.0, 0.0])  # initial conditions
measure_t_grid = jnp.arange(0.0, 5.0, 0.2)  # time
model = SpringMassModel(state0, measure_t_grid)

# Load data
std_dev = 0.5
displacement_data = jnp.array(np.genfromtxt(Path(__file__).parent / "noisy_data.txt"))

# Define prior distributions & MCMC kernel
key = jax.random.PRNGKey(0)
priors = [
    distrax.Uniform(low=0.0, high=10.0),
    distrax.Uniform(low=0.0, high=10.0),
]
# Create the VectorMCMC object
vector_mcmc = JAXMCMC(model.evaluate, displacement_data, priors, std_dev)

# Create the simple BlackJax kernel
mcmc_kernel = BlackJaxKernel(vector_mcmc, ["K", "g"])

# SMC sampling
smc = AdaptiveSampler(mcmc_kernel)
step_list, mll_list = smc.sample(num_particles=500, num_mcmc_samples=5, target_ess=0.8)

# Display results
print(f"marginal log likelihood = {mll_list[-1]}")
print(f"parameter means = {step_list[-1].compute_mean()}")
print("true parameters = {'K': 1.67, 'g': 4.62}")
