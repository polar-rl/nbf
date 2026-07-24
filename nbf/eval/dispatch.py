import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from nbf.eval.base import batch_size, parse_method_spec, split_T, time_len
from nbf.eval.metrics import DENSITY_METRIC_NAMES, density_metrics_from_logps
from nbf.eval.samples import (
    gold_smc_reference_sample_stream_batch,
    nbf_reference_sample_stream_batch,
    nbf_samples_density_stream_batch,
    pf_reference_sample_stream_batch,
    pf_samples_density_stream_batch,
)
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.particle import ParticleFilter, resolve_pf_bandwidths

ORACLE_PF_COUNT = int(os.environ.get('NBF_ORACLE_PF_COUNT', 100_000))
GOLD_SMC_COUNT = int(os.environ.get('NBF_GOLD_SMC_COUNT', 100_000))
GOLD_SMC_MCMC_STEPS = int(os.environ.get('NBF_GOLD_SMC_MCMC_STEPS', 4))
GOLD_SMC_MCMC_STEP_SIZE = float(os.environ.get('NBF_GOLD_SMC_MCMC_STEP_SIZE', 0.3))
NUM_REF_SAMPLES = int(os.environ.get('NBF_NUM_REF_SAMPLES', 1024))
NUM_NEGATIVES = int(os.environ.get('NBF_NUM_NEGATIVES', 256))


def compute_pf_reference_samples(
    env,
    pf_count: int,
    num_ref_samples: int,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    key: Array,
):
    pf = ParticleFilter(num_particles=pf_count)
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    rk, sk, samp_k = jax.random.split(key, 3)
    reset_keys = jax.random.split(rk, N)
    step_keys_b = jax.random.split(sk, N * T).reshape(N, T, -1)
    sample_keys_b = jax.random.split(samp_k, N * T).reshape(N, T, -1)
    return pf_reference_sample_stream_batch(
        env,
        pf,
        state0_b,
        obs0_b,
        states_next_b,
        obs_prev_b,
        obs_next_b,
        reset_keys,
        step_keys_b,
        sample_keys_b,
        num_ref_samples,
    )


def compute_gold_smc_reference_samples(
    env,
    pf_count: int,
    num_ref_samples: int,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    key: Array,
    *,
    mcmc_steps: int = 4,
    mcmc_step_size: float = 0.3,
):
    pf = ParticleFilter(num_particles=pf_count)
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    rk, sk, rej_k, samp_k = jax.random.split(key, 4)
    reset_keys = jax.random.split(rk, N)
    step_keys_b = jax.random.split(sk, N * T).reshape(N, T, -1)
    rejuvenate_keys_b = jax.random.split(rej_k, N * T).reshape(N, T, -1)
    sample_keys_b = jax.random.split(samp_k, N * T).reshape(N, T, -1)
    return gold_smc_reference_sample_stream_batch(
        env,
        pf,
        state0_b,
        obs0_b,
        states_next_b,
        obs_prev_b,
        obs_next_b,
        reset_keys,
        step_keys_b,
        rejuvenate_keys_b,
        sample_keys_b,
        num_ref_samples,
        mcmc_steps,
        mcmc_step_size,
    )


def compute_method_samples(
    spec: str,
    *,
    env,
    embedding_model,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    num_ref_samples: int,
    key: Array,
    oracle_samples: Array | None = None,
    extra_samplers=None,
) -> Array:
    kind, n = parse_method_spec(spec)
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    if kind == 'pf_oracle':
        return compute_gold_smc_reference_samples(
            env,
            GOLD_SMC_COUNT,
            num_ref_samples,
            state0_b,
            obs0_b,
            states_next_b,
            obs_prev_b,
            obs_next_b,
            key,
            mcmc_steps=GOLD_SMC_MCMC_STEPS,
            mcmc_step_size=GOLD_SMC_MCMC_STEP_SIZE,
        )
    if kind == 'pf':
        return compute_pf_reference_samples(
            env,
            n,
            num_ref_samples,
            state0_b,
            obs0_b,
            states_next_b,
            obs_prev_b,
            obs_next_b,
            key,
        )
    if kind == 'nbf':
        nbf = NeuralBayesianFilter(embedding_model, num_particles=n)
        rk, sk, samp_k = jax.random.split(key, 3)
        return nbf_reference_sample_stream_batch(
            env,
            nbf,
            state0_b,
            obs0_b,
            states_next_b,
            obs_prev_b,
            obs_next_b,
            jax.random.split(rk, N),
            split_T(sk, N, T),
            split_T(samp_k, N, T),
            num_ref_samples,
        )
    if extra_samplers and kind in extra_samplers:
        return extra_samplers[kind](
            n,
            env=env,
            embedding_model=embedding_model,
            state0_b=state0_b,
            obs0_b=obs0_b,
            states_next_b=states_next_b,
            obs_prev_b=obs_prev_b,
            obs_next_b=obs_next_b,
            num_ref_samples=num_ref_samples,
            key=key,
        )
    raise ValueError(f'Unknown method spec {spec!r}')


def cache_path_for(
    output_dir: str,
    k: int,
    scenario: str,
    num_episodes: int,
    episode_length: int,
    num_ref_samples: int,
    rollout_seed: int,
) -> str:
    fname = f'oracle_smc_k{k}_{scenario}_N{num_episodes}_T{episode_length}_S{num_ref_samples}_seed{rollout_seed}.npz'
    return os.path.join(output_dir, '_oracle_cache', fname)


def get_or_compute_oracle(
    *,
    cache_path: str,
    env,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    num_ref_samples: int,
    key: Array,
) -> Array:
    if os.path.exists(cache_path):
        with np.load(cache_path) as d:
            return jnp.asarray(d['samples'])
    samples = compute_gold_smc_reference_samples(
        env,
        GOLD_SMC_COUNT,
        num_ref_samples,
        state0_b,
        obs0_b,
        states_next_b,
        obs_prev_b,
        obs_next_b,
        key,
        mcmc_steps=GOLD_SMC_MCMC_STEPS,
        mcmc_step_size=GOLD_SMC_MCMC_STEP_SIZE,
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, samples=np.asarray(samples))
    return samples


def compute_method_samples_and_density(
    spec: str,
    *,
    env,
    embedding_model,
    k: int,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    neg_pts_b: Array,
    num_ref_samples: int,
    key: Array,
    oracle_samples: Array | None = None,
    pf_bandwidth=None,
    extra_density_samplers=None,
) -> tuple[Array, dict[str, Array] | None]:
    kind, n = parse_method_spec(spec)
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)

    if kind == 'nbf':
        nbf = NeuralBayesianFilter(embedding_model, num_particles=n)
        rk, sk, samp_k = jax.random.split(key, 3)
        samples, lp_pos, lp_neg = nbf_samples_density_stream_batch(
            env,
            nbf,
            embedding_model,
            state0_b,
            obs0_b,
            states_next_b,
            obs_prev_b,
            obs_next_b,
            jax.random.split(rk, N),
            split_T(sk, N, T),
            split_T(samp_k, N, T),
            neg_pts_b,
            num_ref_samples,
        )
    elif kind == 'pf':
        bws = resolve_pf_bandwidths(pf_bandwidth, k, [n])
        pf = ParticleFilter(num_particles=n)
        rk, sk, samp_k = jax.random.split(key, 3)
        samples, lp_pos, lp_neg = pf_samples_density_stream_batch(
            env,
            pf,
            state0_b,
            obs0_b,
            states_next_b,
            obs_prev_b,
            obs_next_b,
            jax.random.split(rk, N),
            split_T(sk, N, T),
            split_T(samp_k, N, T),
            neg_pts_b,
            num_ref_samples,
            bws[int(n)],
        )
    elif extra_density_samplers and kind in extra_density_samplers:
        samples, lp_pos, lp_neg = extra_density_samplers[kind](
            n,
            env=env,
            embedding_model=embedding_model,
            k=k,
            state0_b=state0_b,
            obs0_b=obs0_b,
            states_next_b=states_next_b,
            obs_prev_b=obs_prev_b,
            obs_next_b=obs_next_b,
            neg_pts_b=neg_pts_b,
            num_ref_samples=num_ref_samples,
            key=key,
        )
    else:
        raise ValueError(f'compute_method_samples_and_density: unsupported {spec!r}')

    m = density_metrics_from_logps(lp_pos, lp_neg)
    return samples, {name: m[name] for name in DENSITY_METRIC_NAMES}
