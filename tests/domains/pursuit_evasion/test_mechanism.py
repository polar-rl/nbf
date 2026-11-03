import jax
import jax.numpy as jnp
import numpy as np
import pytest
from embedex.distributions.builder import create_model

from nbf.domains.pursuit_evasion.eval.base import make_env_kn, override_sigma
from nbf.domains.pursuit_evasion.policies import sample_train_component_policy
from nbf.eval.base import rollout_episodes, split_T
from nbf.eval.samples import nbf_crossover_stream_batch, pf_crossover_stream_batch
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.particle import ParticleFilter

N, T, NP, NREF = 2, 4, 16, 8


def _setup(domain, k):
    cfg = domain.config([f'data.num_evaders={k}'])
    policy = sample_train_component_policy(jax.random.PRNGKey(3), 'random_walk')
    env = make_env_kn(k, T, (policy,) * k, sigma_obs_min=0.01, sigma_obs_max=0.05)
    cfg.model.input_dim = 2 * k
    cfg.model.likelihood_dim = env.likelihood_feature_dim
    model = create_model(cfg.model, dequantize=False, key=jax.random.PRNGKey(0))
    state0_b, obs0_b, states_next_b, _, _ = rollout_episodes(env, T, N, jax.random.PRNGKey(1))
    episodes = override_sigma(env, state0_b, obs0_b, states_next_b, 0.02, jax.random.PRNGKey(2))
    return env, model, episodes


def _keys(seed):
    rk, sk, samp_k = jax.random.split(jax.random.PRNGKey(seed), 3)
    return jax.random.split(rk, N), split_T(sk, N, T), split_T(samp_k, N, T)


@pytest.mark.parametrize('k', [1, 2])
def test_crossover_streams(k, domains):
    env, model, episodes = _setup(domains['pursuit_evasion'], k)
    nbf = NeuralBayesianFilter(model, num_particles=NP)
    pf = ParticleFilter(num_particles=NP)

    pf_samples, regen, lp_regen, ess_pf = pf_crossover_stream_batch(
        env, pf, nbf, model, *episodes, *_keys(4), NREF
    )
    nbf_samples, resampled, lp_nbf, ess_nbf = nbf_crossover_stream_batch(
        env, nbf, model, *episodes, *_keys(5), NREF
    )
    for arr in (pf_samples, regen, nbf_samples, resampled):
        assert arr.shape == (N, T, NREF, k, 2)
        assert bool(jnp.all(jnp.isfinite(arr)))
    for lp in (lp_regen, lp_nbf):
        assert lp.shape == (N, T)
        assert bool(jnp.all(jnp.isfinite(lp)))
    for ess in (ess_pf, ess_nbf):
        assert ess.shape == (N, T)
        assert bool(jnp.all((ess > 0.0) & (ess <= 1.0 + 1e-6)))
    assert bool(jnp.all(pf_samples >= 0.0))
    assert not np.allclose(np.asarray(pf_samples), np.asarray(regen))
