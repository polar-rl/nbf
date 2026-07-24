import jax
from jax import Array

from nbf.domains.pursuit_evasion.eval.base import make_pe_gmf, sample_background_batch
from nbf.domains.pursuit_evasion.eval.samples import (
    gmf_reference_sample_stream_batch,
    gmf_samples_density_stream_batch,
)
from nbf.eval.base import batch_size, split_T, time_len
from nbf.eval.dispatch import (
    cache_path_for,
    get_or_compute_oracle,
)
from nbf.eval.dispatch import (
    compute_method_samples as _compute_method_samples,
)
from nbf.eval.dispatch import (
    compute_method_samples_and_density as _compute_method_samples_and_density,
)


def _gmf_sampler(
    n,
    *,
    env,
    state0_b,
    obs_prev_b,
    states_next_b,
    obs_next_b,
    num_ref_samples,
    key,
    **_,
):
    gmf = make_pe_gmf(num_evaders=env.num_evaders, M=n)
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    ik, samp_k = jax.random.split(key)
    return gmf_reference_sample_stream_batch(
        gmf,
        env.obstacles,
        env.policy,
        states_next_b,
        obs_next_b,
        jax.random.split(ik, N),
        split_T(samp_k, N, T),
        num_ref_samples,
    )


def _gmf_density_sampler(
    n,
    *,
    env,
    k,
    state0_b,
    obs_prev_b,
    states_next_b,
    obs_next_b,
    neg_pts_b,
    num_ref_samples,
    key,
    **_,
):
    gmf = make_pe_gmf(num_evaders=k, M=n)
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    ik, samp_k = jax.random.split(key)
    return gmf_samples_density_stream_batch(
        env,
        gmf,
        states_next_b,
        obs_next_b,
        jax.random.split(ik, N),
        split_T(samp_k, N, T),
        neg_pts_b,
        num_ref_samples,
    )


PE_EXTRA_SAMPLERS = {'gmf': _gmf_sampler}
PE_EXTRA_DENSITY_SAMPLERS = {'gmf': _gmf_density_sampler}


def compute_method_samples(spec: str, **kwargs) -> Array:
    return _compute_method_samples(spec, extra_samplers=PE_EXTRA_SAMPLERS, **kwargs)


def compute_method_samples_and_density(spec: str, **kwargs):
    return _compute_method_samples_and_density(
        spec, extra_density_samplers=PE_EXTRA_DENSITY_SAMPLERS, **kwargs
    )


def prepare_oracle_and_negatives(
    *,
    out_dir: str,
    env,
    k: int,
    scenario: str,
    num_episodes: int,
    episode_length: int,
    seed: int,
    num_ref_samples: int,
    num_negatives: int,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    oracle_key: Array,
    neg_key: Array,
) -> tuple[Array, Array]:
    cache = cache_path_for(
        out_dir, k, scenario, num_episodes, episode_length, num_ref_samples, seed
    )
    oracle = get_or_compute_oracle(
        cache_path=cache,
        env=env,
        state0_b=state0_b,
        obs0_b=obs0_b,
        states_next_b=states_next_b,
        obs_prev_b=obs_prev_b,
        obs_next_b=obs_next_b,
        num_ref_samples=num_ref_samples,
        key=oracle_key,
    )
    neg_pts_b = sample_background_batch(
        jax.random.split(neg_key, num_episodes * episode_length).reshape(
            num_episodes, episode_length, -1
        ),
        num_negatives,
        k,
    )
    return oracle, neg_pts_b
