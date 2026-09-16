import functools
import os

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from nbf.domains.lorenz96.env import (
    Lorenz96Env,
    conditional_gaussian_redraw,
    rk4_window,
)
from nbf.eval.base import batch_size, chunked_stream, split_T, time_len
from nbf.eval.dispatch import (
    compute_method_samples as _compute_method_samples,
)
from nbf.eval.dispatch import filter_sampler
from nbf.eval.dispatch import (
    compute_method_samples_and_density as _compute_method_samples_and_density,
)
from nbf.eval.samples import (
    nbf_reference_sample_stream_batch,
    nbf_samples_density_stream_batch,
    pf_reference_sample_stream_batch,
)
from nbf.filters.enkf import EnsembleKalmanFilter, LocalEnsembleTransformKalmanFilter, ring_taper
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.proposals import OptimalProposal
from nbf.filters.particle import RegularizedParticleFilter, systematic_indices
from nbf.filters.resample_move import ResampleMoveParticleFilter

PF_REG_BANDWIDTH_SCALE = float(os.environ.get('NBF_PF_REG_BANDWIDTH_SCALE', 1.0))
PF_RM_MOVES = int(os.environ.get('NBF_PF_RM_MOVES', 0))
PF_RM_STEP_SIZE = float(os.environ.get('NBF_PF_RM_STEP_SIZE', 0.3))
ENKF_INFLATION = float(os.environ.get('NBF_ENKF_INFLATION', 1.05))
ENKF_LOC_RADIUS = float(os.environ.get('NBF_ENKF_LOC_RADIUS', 4.0))


def make_pf_reg(n: int) -> RegularizedParticleFilter:
    return RegularizedParticleFilter(num_particles=n, bandwidth_scale=PF_REG_BANDWIDTH_SCALE)


def make_pf_rm(n: int) -> ResampleMoveParticleFilter:
    return ResampleMoveParticleFilter(
        num_particles=n, num_moves=PF_RM_MOVES, step_size=PF_RM_STEP_SIZE
    )


def make_enkf(m: int) -> EnsembleKalmanFilter:
    return EnsembleKalmanFilter(num_members=m, inflation=ENKF_INFLATION, loc_radius=ENKF_LOC_RADIUS)


def make_letkf(m: int) -> LocalEnsembleTransformKalmanFilter:
    return LocalEnsembleTransformKalmanFilter(
        num_members=m, inflation=ENKF_INFLATION, loc_radius=ENKF_LOC_RADIUS
    )


@eqx.filter_jit
def pf_op_sample_stream_batch(
    env: Lorenz96Env,
    state0_b,
    obs_values_b: Array,
    prior_keys: Array,
    step_keys_b: Array,
    num_particles: int,
    num_ref_samples: int,
):
    """Optimal-proposal particle filter (marginal weights at the propagated means,
    conjugate redraw of the observed coordinates, systematic resampling every step)."""
    obs_idx = env.obs_idx

    def _one(state0, obs_values, prior_key, step_keys):
        forcing, sigma_obs = state0.forcing, state0.sigma_obs
        parts = state0.center + env.sigma_init * jax.random.normal(
            prior_key, (num_particles, env.K)
        )

        def _step(parts, xs):
            k, y = xs
            _, res_key, rej_key, samp_key = jax.random.split(k, 4)
            means = rk4_window(parts, forcing, env.dt, env.substeps, env.linear)
            marg_var = env.sigma_dyn**2 + sigma_obs**2
            log_w = -0.5 * jnp.sum((means[:, obs_idx] - y) ** 2, axis=-1) / marg_var
            w = jax.nn.softmax(log_w)
            idx = systematic_indices(w, res_key)
            parts = conditional_gaussian_redraw(
                means[idx], y, obs_idx, env.sigma_dyn, sigma_obs, rej_key
            )
            samp_idx = jax.random.randint(samp_key, (num_ref_samples,), 0, num_particles)
            return parts, parts[samp_idx]

        _, samps = jax.lax.scan(_step, parts, (step_keys, obs_values))
        return samps

    return jax.vmap(_one)(state0_b, obs_values_b, prior_keys, step_keys_b)


@eqx.filter_jit
def enkf_samples_density_stream_batch(
    env: Lorenz96Env,
    filt: EnsembleKalmanFilter,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    reset_keys: Array,
    step_keys_b: Array,
    sample_keys_b: Array,
    num_ref_samples: int,
):
    taper = ring_taper(env.K, jnp.arange(env.K), filt.loc_radius)

    def _gauss_logp(pset, x_true):
        X = env.state_array(pset.particles)
        mu = jnp.mean(X, axis=0)
        A = X - mu
        cov = taper * (A.T @ A / (X.shape[0] - 1)) + 1e-4 * jnp.eye(env.K)
        chol = jnp.linalg.cholesky(cov)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

        def logpdf(x):
            sol = jax.scipy.linalg.solve_triangular(chol, x - mu, lower=True)
            return -0.5 * (jnp.sum(sol * sol) + log_det + env.K * jnp.log(2.0 * jnp.pi))

        return logpdf(x_true)

    def _one(state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys, sample_keys):
        pset = filt.reset(env, state0, obs0, rkey)

        def _step(pset, xs):
            sk, samp_k, state_t, prev_obs_t, next_obs_t = xs
            pset, _, _, _ = filt.update(env, pset, state_t, prev_obs_t, 0, next_obs_t, sk)
            samps = env.state_array(filt.sample(pset, state_t, num_ref_samples, samp_k))
            lp_true = _gauss_logp(pset, env.state_array(state_t))
            return pset, (samps, lp_true)

        _, (samps, lp_true) = jax.lax.scan(
            _step, pset, (step_keys, sample_keys, states_next, obs_prev, obs_next)
        )
        return samps, lp_true

    return jax.vmap(_one)(
        state0_b,
        obs0_b,
        states_next_b,
        obs_prev_b,
        obs_next_b,
        reset_keys,
        step_keys_b,
        sample_keys_b,
    )


def _enkf_density_sampler(
    n,
    *,
    env,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    num_ref_samples,
    key,
    chunk_size=0,
    desc=None,
    make=make_enkf,
    **_,
):
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    filt = make(n)
    rk, sk, samp_k = jax.random.split(key, 3)
    return chunked_stream(
        lambda *a: enkf_samples_density_stream_batch(env, filt, *a),
        (state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b),
        (jax.random.split(rk, N), split_T(sk, N, T), split_T(samp_k, N, T)),
        (num_ref_samples,),
        N,
        chunk_size,
        desc=desc,
    )


def _nbf_op_sampler(
    n,
    *,
    env,
    embedding_model,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    num_ref_samples,
    key,
    chunk_size=0,
    desc=None,
    **_,
):
    filt = NeuralBayesianFilter(embedding_model, n, proposal=OptimalProposal())
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    rk, sk, samp_k = jax.random.split(key, 3)
    return chunked_stream(
        lambda *a: nbf_reference_sample_stream_batch(env, filt, *a),
        (state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b),
        (jax.random.split(rk, N), split_T(sk, N, T), split_T(samp_k, N, T)),
        (num_ref_samples,),
        N,
        chunk_size,
        desc=desc,
    )


def _nbf_op_density_sampler(
    n,
    *,
    env,
    embedding_model,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    num_ref_samples,
    key,
    density_model=None,
    chunk_size=0,
    desc=None,
    **_,
):
    filt = NeuralBayesianFilter(embedding_model, n, proposal=OptimalProposal())
    dmodel = density_model if density_model is not None else embedding_model
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    rk, sk, samp_k = jax.random.split(key, 3)
    return chunked_stream(
        lambda *a: nbf_samples_density_stream_batch(env, filt, dmodel, *a),
        (state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b),
        (jax.random.split(rk, N), split_T(sk, N, T), split_T(samp_k, N, T)),
        (num_ref_samples,),
        N,
        chunk_size,
        desc=desc,
    )


def _pf_op_sampler(
    n,
    *,
    env,
    state0_b,
    obs_next_b,
    num_ref_samples,
    key,
    chunk_size=0,
    desc=None,
    **_,
):
    # The particle filter under the same locally optimal proposal NBF-OP uses.
    N = batch_size(state0_b)
    T = time_len(obs_next_b)
    prior_key, step_key = jax.random.split(key)
    return chunked_stream(
        lambda *a: pf_op_sample_stream_batch(env, *a),
        (state0_b, obs_next_b.values),
        (jax.random.split(prior_key, N), split_T(step_key, N, T)),
        (n, num_ref_samples),
        N,
        chunk_size,
        desc=desc,
    )


L96_EXTRA_SAMPLERS = {
    'pf_reg': filter_sampler(make_pf_reg),
    'pf_rm': filter_sampler(make_pf_rm),
    'enkf': filter_sampler(make_enkf),
    'letkf': filter_sampler(make_letkf),
    'nbf_op': _nbf_op_sampler,
    'pf_op': _pf_op_sampler,
}

L96_EXTRA_DENSITY_SAMPLERS = {
    'enkf': _enkf_density_sampler,
    'letkf': functools.partial(_enkf_density_sampler, make=make_letkf),
    'nbf_op': _nbf_op_density_sampler,
}


def compute_method_samples(spec: str, **kwargs) -> Array:
    return _compute_method_samples(spec, extra_samplers=L96_EXTRA_SAMPLERS, **kwargs)


def compute_method_samples_and_density(spec: str, **kwargs):
    return _compute_method_samples_and_density(
        spec, extra_density_samplers=L96_EXTRA_DENSITY_SAMPLERS, **kwargs
    )
