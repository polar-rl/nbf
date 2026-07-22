import time

import equinox as eqx
import jax
import numpy as np
from jax import Array

from nbf.domains.pursuit_evasion.eval.base import parse_method_spec, split_T
from nbf.domains.pursuit_evasion.eval.dispatch import compute_method_samples
from nbf.filters.nbf import NeuralBayesianFilter


@eqx.filter_jit
def _nbf_filter_collect_embeddings_stream_batch(
    env,
    nbf: NeuralBayesianFilter,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    reset_keys: Array,
    step_keys_b: Array,
):

    def _one(state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys):
        embedding = nbf.reset(env, state0, obs0, rkey)

        def _step(embedding, xs):
            sk, state_t, prev_obs_t, next_obs_t = xs
            embedding, _ = nbf.update(env, embedding, state_t, prev_obs_t, 0, next_obs_t, sk)
            return embedding, embedding

        _, embs = jax.lax.scan(_step, embedding, (step_keys, states_next, obs_prev, obs_next))
        return embs

    return jax.vmap(_one)(
        state0_b,
        obs0_b,
        states_next_b,
        obs_prev_b,
        obs_next_b,
        reset_keys,
        step_keys_b,
    )


@eqx.filter_jit
def _nbf_sample_only_stream_batch(
    nbf: NeuralBayesianFilter,
    embeddings_b,
    states_next_b,
    sample_keys_b: Array,
    num_ref_samples: int,
):
    """Draw samples from precomputed embeddings; no filter."""

    def _one(embeddings, states, sample_keys):
        def _step(_, x):
            emb, st, sk = x
            samples = nbf.sample(emb, st, num_ref_samples, sk)
            return None, samples.positions

        _, samps = jax.lax.scan(_step, None, (embeddings, states, sample_keys))
        return samps

    return jax.vmap(_one)(embeddings_b, states_next_b, sample_keys_b)


def time_method_walltime(
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
    num_warmup: int = 1,
    num_repeats: int = 3,
    key: Array,
) -> dict[str, float]:
    keys = jax.random.split(key, num_warmup + num_repeats)
    for i in range(num_warmup):
        out = compute_method_samples(
            spec,
            env=env,
            embedding_model=embedding_model,
            state0_b=state0_b,
            obs0_b=obs0_b,
            states_next_b=states_next_b,
            obs_prev_b=obs_prev_b,
            obs_next_b=obs_next_b,
            num_ref_samples=num_ref_samples,
            key=keys[i],
        )
        jax.block_until_ready(out)

    times = []
    for j in range(num_repeats):
        t0 = time.perf_counter()
        out = compute_method_samples(
            spec,
            env=env,
            embedding_model=embedding_model,
            state0_b=state0_b,
            obs0_b=obs0_b,
            states_next_b=states_next_b,
            obs_prev_b=obs_prev_b,
            obs_next_b=obs_next_b,
            num_ref_samples=num_ref_samples,
            key=keys[num_warmup + j],
        )
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times)
    result = {
        'mean_s': float(arr.mean()),
        'std_s': float(arr.std()),
        'median_s': float(np.median(arr)),
        'all_s': arr.tolist(),
    }

    kind, _ = parse_method_spec(spec)
    if kind == 'nbf':
        breakdown = _time_nbf_breakdown(
            spec,
            env=env,
            embedding_model=embedding_model,
            state0_b=state0_b,
            obs0_b=obs0_b,
            states_next_b=states_next_b,
            obs_prev_b=obs_prev_b,
            obs_next_b=obs_next_b,
            num_ref_samples=num_ref_samples,
            num_warmup=num_warmup,
            num_repeats=num_repeats,
            key=jax.random.fold_in(key, 17),
        )
        result.update(breakdown)
    return result


def _time_nbf_breakdown(
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
    num_warmup: int,
    num_repeats: int,
    key: Array,
) -> dict[str, float]:
    n = parse_method_spec(spec)[1]
    nbf = NeuralBayesianFilter(embedding_model, num_particles=n)
    N = state0_b.positions.shape[0]
    T = obs_prev_b.ranges.shape[1]

    def _run_filter(k):
        rk, sk = jax.random.split(k)
        return _nbf_filter_collect_embeddings_stream_batch(
            env,
            nbf,
            state0_b,
            obs0_b,
            states_next_b,
            obs_prev_b,
            obs_next_b,
            jax.random.split(rk, N),
            split_T(sk, N, T),
        )

    keys_outer = jax.random.split(key, num_warmup + num_repeats)
    embeddings_b = None
    for i in range(num_warmup):
        embeddings_b = _run_filter(keys_outer[i])
        jax.block_until_ready(embeddings_b)
    filter_times = []
    for j in range(num_repeats):
        t0 = time.perf_counter()
        embeddings_b = _run_filter(keys_outer[num_warmup + j])
        jax.block_until_ready(embeddings_b)
        filter_times.append(time.perf_counter() - t0)

    def _run_sample(k):
        return _nbf_sample_only_stream_batch(
            nbf, embeddings_b, states_next_b, split_T(k, N, T), num_ref_samples
        )

    keys_sample = jax.random.split(jax.random.fold_in(key, 31), num_warmup + num_repeats)
    for i in range(num_warmup):
        out = _run_sample(keys_sample[i])
        jax.block_until_ready(out)
    sample_times = []
    for j in range(num_repeats):
        t0 = time.perf_counter()
        out = _run_sample(keys_sample[num_warmup + j])
        jax.block_until_ready(out)
        sample_times.append(time.perf_counter() - t0)

    farr = np.asarray(filter_times)
    sarr = np.asarray(sample_times)
    return {
        'filter_mean_s': float(farr.mean()),
        'filter_std_s': float(farr.std()),
        'filter_all_s': farr.tolist(),
        'sample_mean_s': float(sarr.mean()),
        'sample_std_s': float(sarr.std()),
        'sample_all_s': sarr.tolist(),
    }
