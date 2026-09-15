import jax
from jax import Array

from nbf.domains.pursuit_evasion.eval.base import make_pe_gmf
from nbf.domains.pursuit_evasion.eval.samples import (
    gmf_reference_sample_stream_batch,
    gmf_samples_density_stream_batch,
)
from nbf.eval.base import batch_size, chunked_stream, split_T, time_len
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
    chunk_size=0,
    desc=None,
    **_,
):
    gmf = make_pe_gmf(num_evaders=env.num_evaders, M=n)
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    ik, samp_k = jax.random.split(key)
    return chunked_stream(
        lambda *a: gmf_reference_sample_stream_batch(gmf, env.obstacles, env.policy, *a),
        (states_next_b, obs_next_b),
        (jax.random.split(ik, N), split_T(samp_k, N, T)),
        (num_ref_samples,),
        N,
        chunk_size,
        desc=desc,
    )


def _gmf_density_sampler(
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
    return gmf_samples_density_stream_batch(
        env,
        gmf,
        states_next_b,
        obs_next_b,
        jax.random.split(ik, N),
        split_T(samp_k, N, T),
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
