import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

num_chains = multiprocessing.cpu_count()

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import distrax
from spring_mass_jax_model import SpringMassModel
from smcpy import AdaptiveSampler, VectorMCMC

from smcpy.mcmc.vector_hmc import VectorHMC
from smcpy.mcmc.vector_hmc_kernel import HMCKernel

state0 = jnp.array([0.0, 0.0])  # initial conditions
measure_t_grid = jnp.arange(0.0, 5.0, 0.2)  # time
model = SpringMassModel(state0, measure_t_grid)

sigma = 0.5
from datetime import date

displacement_data = jnp.array(np.genfromtxt(Path(__file__).parent / "noisy_data.txt"))

key = jax.random.key(int(date.today().strftime("%Y%m%d")))
priors = [
    distrax.Uniform(low=0.0, high=10.0),
    distrax.Uniform(low=0.0, high=10.0),
]

vector_hmc = VectorHMC(model.evaluate, displacement_data, priors, sigma)

mcmc_kernel = HMCKernel(vector_hmc, ["K", "g"])

smc = AdaptiveSampler(mcmc_kernel)
step_list, mll_list = smc.sample(
    num_particles=num_chains, num_mcmc_samples=1, target_ess=0.8
)

print(f"marginal log likelihood = {mll_list[-1]}")
print(f"parameter means = {step_list[-1].compute_mean()}")
print("true parameters = {'K': 1.67, 'g': 4.62}")
