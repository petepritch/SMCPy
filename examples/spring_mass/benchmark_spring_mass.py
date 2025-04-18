import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from smcpy import AdaptiveSampler
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy.mcmc.blackjax_kernel import BlackJaxKernel

from spring_mass_jax_model import SpringMassModel as SpringMassJAXModel
from spring_mass_model import SpringMassModel as SpringMassModel

from smcpy import VectorMCMC
from smcpy.mcmc.blackjax_mcmc import JAXMCMC

import distrax
from scipy.stats import uniform


def benchmark_numpy():
    print("Running benchmark with Numpy kernel...")

    state0 = [0.0, 0.0]
    measure_t_grid = np.arange(0.0, 5.0, 0.2)
    model = SpringMassModel(state0, measure_t_grid)

    std_dev = 0.5
    displacement_data = np.genfromtxt(Path(__file__).parent / "noisy_data.txt")

    priors = [uniform(0, 10), uniform(0, 10)]
    vector_mcmc = VectorMCMC(model.evaluate, displacement_data, priors, std_dev)
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=("K", "g"))

    smc = AdaptiveSampler(mcmc_kernel)
    start_time = time.time()

    step_list, mll_list = smc.sample(
        num_particles=500, num_mcmc_samples=5, target_ess=0.8
    )
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"NumPy elapsed time: {elapsed:.2f} seconds")
    print(f"NumPy marginal log likelihood = {mll_list[-1]}")
    print(f"NumPy parameter means = {step_list[-1].compute_mean()}")

    return elapsed, mll_list[-1], step_list[-1].compute_mean()


def benchmark_jax():

    print("Running benchmark with JAX kernel...")

    state0 = jnp.array([0.0, 0.0])
    measure_t_grid = jnp.arange(0.0, 5.0, 0.2)
    model = SpringMassJAXModel(state0, measure_t_grid)

    std_dev = 0.5
    displacement_data = jnp.array(
        np.genfromtxt(Path(__file__).parent / "noisy_data.txt")
    )

    priors = [
        distrax.Uniform(low=0.0, high=10.0),
        distrax.Uniform(low=0.0, high=10.0),
    ]

    vector_mcmc = JAXMCMC(model.evaluate, displacement_data, priors, std_dev)
    mcmc_kernel = BlackJaxKernel(vector_mcmc, ["K", "g"])

    start_time = time.time()

    smc = AdaptiveSampler(mcmc_kernel)
    step_list, mll_list = smc.sample(
        num_particles=500, num_mcmc_samples=5, target_ess=0.8
    )

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"JAX elapsed time: {elapsed:.2f} seconds")
    print(f"JAX marginal log likelihood = {mll_list[-1]}")
    print(f"JAX parameter means = {step_list[-1].compute_mean()}")
    return elapsed, mll_list[-1], step_list[-1].compute_mean()


def compare_results(numpy_results, jax_results):
    numpy_time, numpy_mll, numpy_params = numpy_results
    jax_time, jax_mll, jax_params = jax_results

    speedup = numpy_time / jax_time

    print("\n=== BENCHMARK RESULTS ===")
    print(f"Numpy time: {numpy_time:.2f} seconds")
    print(f"JAX time: {jax_time:.2f} seconds")
    print(f"Speedup: {speedup:.2f}x")

    print("\n=== ACCURACY COMPARISON ===")
    print(f"Numpy MLL: {numpy_mll:.4f}")
    print(f"JAX MLL: {jax_mll:.4f}")
    print(f"Difference in MLL: {abs(numpy_mll - jax_mll):.4f}")

    print("\nParameter Estimates")
    for param in numpy_params:
        print(
            f"{param}: NumPy = {numpy_params[param]:.4f}, JAX = {jax_params[param]:.4f}, "
            f"Diff = {abs(numpy_params[param] - jax_params[param]):.4f}"
        )


if __name__ == "__main__":
    numpy_results = benchmark_numpy()
    jax_results = benchmark_jax()
    compare_results(numpy_results, jax_results)
    print("\nTrue parameters = {'K': 1.67, 'g': 4.62}")
