import jax
import jax.numpy as jnp
import pytest

from smcpy.mcmc.blackjax_mcmc import *


@pytest.fixture
def stub_model():
    x = jnp.array([1.0, 2.0, 3.0])

    def evlauation(q):
        if q.ndim == 1:
            params = q[None, :]
        else:
            params = q
        assert params.ndim == 2
        assert params.shape[1] == 3

        a, b, c = params[:, 0], params[:, 1], params[:, 2]
        output = (a[:, None] * 2 + b[:, None] * 3.25 - c[:, None] ** 2) * x

        return output[0] if 1 in params.shape else output

    return evlauation


@pytest.fixture
def data():
    return jnp.array([5.0, 4.0, 9.0])


@pytest.fixture
def priors():
    import distrax

    return [
        distrax.Normal(loc=0.0, scale=2.0),
        distrax.Normal(loc=0.0, scale=2.0),
        distrax.Normal(loc=0.0, scale=2.0),
    ]


@pytest.fixture
def mcmc_instance(stub_model, data, priors):
    return JAXMCMC(
        model=stub_model,
        data=data,
        priors=priors,
        log_like_func=None,
        log_like_args=0.5,
    )


def test_model_evaluation(stub_model):
    params = jnp.array([1.0, 1.0, 1.0])
    results = stub_model(params)
    expected = jnp.array([1.0 * 2 + 1.0 * 3.25 - 1.0**2]) * jnp.array([1.0, 2.0, 3.0])
    assert jnp.allclose(results, expected)

    batch_params = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    batch_result = stub_model(batch_params)
    assert batch_result.shape == (2, 3)


def test_mcmc_instance_creation(mcmc_instance):
    assert mcmc_instance is not None
    assert hasattr(mcmc_instance, "evaluate_log_posterior")


def test_normal_log_like(mcmc_instance):
    predicted = jnp.array([5.0, 4.0, 9.0])
    data = jnp.array([5.0, 4.0, 9.0])
    sigma = 1.0

    log_like = mcmc_instance._normal_log_like(predicted, data, sigma)

    n = len(data)
    expected_log_like = -n * jnp.log(sigma * jnp.sqrt(2 * jnp.pi))

    assert jnp.allclose(log_like, expected_log_like)

    predicted_offset = jnp.array([6.0, 5.0, 10.0])
    log_like_offset = mcmc_instance._normal_log_like(predicted_offset, data, sigma)

    assert log_like_offset < log_like

    residuals = data - predicted_offset
    expected_log_like_offset = -0.5 * jnp.sum(residuals**2) / (sigma**2)
    expected_log_like_offset -= n * jnp.log(sigma * jnp.sqrt(2 * jnp.pi))

    assert jnp.allclose(log_like_offset, expected_log_like_offset)


def test_evaluate_log_likelihood(mcmc_instance):
    params = jnp.array([1.0, 1.0, 1.0])

    original_evaluate_model = mcmc_instance.evaluate_model

    def mock_evlaute_model(params):
        return jnp.array([5.0, 4.0, 9.0])

    mcmc_instance.evaluate_model = mock_evlaute_model

    try:
        log_like = mcmc_instance.evaluate_log_likelihood(params)

        n = len(mcmc_instance.data)
        expected_log_like = -n * jnp.log(0.5 * jnp.sqrt(2 * jnp.pi))

        assert jnp.allclose(log_like, expected_log_like)

        batch_params = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        batch_log_like = mcmc_instance.evaluate_log_likelihood(batch_params)

        assert batch_log_like.shape == (2,)
        assert jnp.allclose(
            batch_log_like, jnp.array([expected_log_like, expected_log_like])
        )

    finally:
        mcmc_instance.evaluate_model = original_evaluate_model
