import os

import jax
from jax import Array

from nbf.eval.base import (
    batch_size,
    chunked_stream,
    parse_method_spec,
    split_T,
    time_len,
)
from nbf.eval.metrics import DENSITY_METRIC_NAMES, density_metrics_from_logp
from nbf.eval.samples import (
    nbf_reference_sample_stream_batch,
    nbf_samples_density_stream_batch,
    pf_reference_sample_stream_batch,
)
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.particle import ParticleFilter

NUM_REF_SAMPLES = int(os.environ.get('NBF_NUM_REF_SAMPLES', 1024))


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
    *,
    chunk_size: int = 0,
    desc: str | None = None,
):
    pf = ParticleFilter(num_particles=pf_count)
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    rk, sk, samp_k = jax.random.split(key, 3)
    reset_keys = jax.random.split(rk, N)
    step_keys_b = jax.random.split(sk, N * T).reshape(N, T, -1)
    sample_keys_b = jax.random.split(samp_k, N * T).reshape(N, T, -1)
    return chunked_stream(
        lambda *a: pf_reference_sample_stream_batch(env, pf, *a),
        (state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b),
        (reset_keys, step_keys_b, sample_keys_b),
        (num_ref_samples,),
        N,
        chunk_size,
        desc=desc,
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
    extra_samplers=None,
    chunk_size: int = 0,
    desc: str | None = None,
) -> Array:
    """(N, T, num_ref_samples, *state) draws from the method's belief at every step."""
    kind, n = parse_method_spec(spec)
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
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
            chunk_size=chunk_size,
            desc=desc,
        )
    if kind == 'nbf':
        nbf = NeuralBayesianFilter(embedding_model, num_particles=n)
        rk, sk, samp_k = jax.random.split(key, 3)
        return chunked_stream(
            lambda *a: nbf_reference_sample_stream_batch(env, nbf, *a),
            (state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b),
            (jax.random.split(rk, N), split_T(sk, N, T), split_T(samp_k, N, T)),
            (num_ref_samples,),
            N,
            chunk_size,
            desc=desc,
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
            chunk_size=chunk_size,
            desc=desc,
        )
    raise ValueError(f'Unknown method spec {spec!r}')


def compute_method_samples_and_density(
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
    extra_density_samplers=None,
    chunk_size: int = 0,
    desc: str | None = None,
    density_model=None,
) -> tuple[Array, dict[str, Array]]:
    """Samples as in `compute_method_samples` plus the per-step NLL of the true state.

    Only methods with a native density (NBF, and the extra density samplers a domain
    registers, e.g. EnKF or GMF) are supported; particle filters have no density.
    """
    kind, n = parse_method_spec(spec)
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    episode_trees = (state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b)

    if kind == 'nbf':
        dmodel = density_model if density_model is not None else embedding_model
        nbf = NeuralBayesianFilter(embedding_model, num_particles=n)
        rk, sk, samp_k = jax.random.split(key, 3)
        samples, lp_true = chunked_stream(
            lambda *a: nbf_samples_density_stream_batch(env, nbf, dmodel, *a),
            episode_trees,
            (jax.random.split(rk, N), split_T(sk, N, T), split_T(samp_k, N, T)),
            (num_ref_samples,),
            N,
            chunk_size,
            desc=desc,
        )
    elif extra_density_samplers and kind in extra_density_samplers:
        samples, lp_true = extra_density_samplers[kind](
            n,
            env=env,
            embedding_model=embedding_model,
            density_model=density_model,
            state0_b=state0_b,
            obs0_b=obs0_b,
            states_next_b=states_next_b,
            obs_prev_b=obs_prev_b,
            obs_next_b=obs_next_b,
            num_ref_samples=num_ref_samples,
            key=key,
            chunk_size=chunk_size,
            desc=desc,
        )
    else:
        raise ValueError(f'compute_method_samples_and_density: unsupported {spec!r}')

    m = density_metrics_from_logp(lp_true)
    return samples, {name: m[name] for name in DENSITY_METRIC_NAMES}
