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
from nbf.eval.dispatch import (
    compute_method_samples_and_density as _compute_method_samples_and_density,
)
from nbf.eval.samples import (
    nbf_reference_sample_stream_batch,
    nbf_samples_density_stream_batch,
    pf_reference_sample_stream_batch,
    pf_samples_density_stream_batch,
)
from nbf.filters.enkf import EnsembleKalmanFilter, ring_taper
from nbf.filters.nbf_op import OptimalProposalNeuralBayesianFilter
from nbf.filters.particle import (
    RegularizedParticleFilter,
    resolve_pf_bandwidths,
    systematic_indices,
)
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


@eqx.filter_jit
def gold_smc_sample_stream_batch(
    env: Lorenz96Env,
    state0_b,
    obs_values_b: Array,
    prior_keys: Array,
    step_keys_b: Array,
    num_particles: int,
    num_ref_samples: int,
):
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
    neg_pts_b: Array,
    num_ref_samples: int,
):
    taper = ring_taper(env.K, jnp.arange(env.K), filt.loc_radius)

    def _gauss_logp(pset, x_true, x_neg):
        X = env.state_array(pset.particles)
        mu = jnp.mean(X, axis=0)
        A = X - mu
        cov = taper * (A.T @ A / (X.shape[0] - 1)) + 1e-4 * jnp.eye(env.K)
        chol = jnp.linalg.cholesky(cov)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

        def logpdf(x):
            sol = jax.scipy.linalg.solve_triangular(chol, x - mu, lower=True)
            return -0.5 * (jnp.sum(sol * sol) + log_det + env.K * jnp.log(2.0 * jnp.pi))

        logps = jax.vmap(logpdf)(jnp.concatenate([x_true[None], x_neg], axis=0))
        return logps[0], logps[1:]

    def _one(state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys, sample_keys, neg_pts):
        pset = filt.reset(env, state0, obs0, rkey)

        def _step(pset, xs):
            sk, samp_k, state_t, prev_obs_t, next_obs_t, neg = xs
            pset, _, _, _ = filt.update(env, pset, state_t, prev_obs_t, 0, next_obs_t, sk)
            samps = env.state_array(filt.sample(pset, state_t, num_ref_samples, samp_k))
            lp_pos, lp_neg = _gauss_logp(pset, env.state_array(state_t), neg)
            return pset, (samps, lp_pos, lp_neg)

        _, (samps, lp_pos, lp_neg) = jax.lax.scan(
            _step,
            pset,
            (step_keys, sample_keys, states_next, obs_prev, obs_next, neg_pts),
        )
        return samps, lp_pos, lp_neg

    return jax.vmap(_one)(
        state0_b,
        obs0_b,
        states_next_b,
        obs_prev_b,
        obs_next_b,
        reset_keys,
        step_keys_b,
        sample_keys_b,
        neg_pts_b,
    )


@eqx.filter_jit
def sample_l96_negatives_batch(env: Lorenz96Env, keys_b: Array, num_negatives: int) -> Array:
    lo = env.norm_mean - 4.0 * env.norm_std
    hi = env.norm_mean + 4.0 * env.norm_std
    return jax.vmap(
        jax.vmap(lambda k: jax.random.uniform(k, (num_negatives, env.K), minval=lo, maxval=hi))
    )(keys_b)


def _filter_sampler(make_filter):
    def _sampler(
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
        **_,
    ):
        N = batch_size(state0_b)
        T = time_len(obs_prev_b)
        filt = make_filter(n)
        rk, sk, samp_k = jax.random.split(key, 3)
        return chunked_stream(
            lambda *a: pf_reference_sample_stream_batch(env, filt, *a),
            (state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b),
            (jax.random.split(rk, N), split_T(sk, N, T), split_T(samp_k, N, T)),
            (num_ref_samples,),
            N,
            chunk_size,
            desc=desc,
        )

    return _sampler


def _kde_density_sampler(make_filter):
    def _sampler(
        n,
        *,
        env,
        k,
        state0_b,
        obs0_b,
        states_next_b,
        obs_prev_b,
        obs_next_b,
        neg_pts_b,
        num_ref_samples,
        key,
        chunk_size=0,
        desc=None,
        **_,
    ):
        bws = resolve_pf_bandwidths(None, k, [n])
        N = batch_size(state0_b)
        T = time_len(obs_prev_b)
        filt = make_filter(n)
        rk, sk, samp_k = jax.random.split(key, 3)
        return chunked_stream(
            lambda *a: pf_samples_density_stream_batch(env, filt, *a),
            (state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b),
            (jax.random.split(rk, N), split_T(sk, N, T), split_T(samp_k, N, T), neg_pts_b),
            (num_ref_samples, bws[int(n)]),
            N,
            chunk_size,
            desc=desc,
        )

    return _sampler


def _enkf_density_sampler(
    n,
    *,
    env,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    neg_pts_b,
    num_ref_samples,
    key,
    chunk_size=0,
    desc=None,
    **_,
):
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    filt = make_enkf(n)
    rk, sk, samp_k = jax.random.split(key, 3)
    return chunked_stream(
        lambda *a: enkf_samples_density_stream_batch(env, filt, *a),
        (state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b),
        (jax.random.split(rk, N), split_T(sk, N, T), split_T(samp_k, N, T), neg_pts_b),
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
    filt = OptimalProposalNeuralBayesianFilter(embedding_model, num_particles=n)
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
    neg_pts_b,
    num_ref_samples,
    key,
    density_model=None,
    chunk_size=0,
    desc=None,
    **_,
):
    filt = OptimalProposalNeuralBayesianFilter(embedding_model, num_particles=n)
    dmodel = density_model if density_model is not None else embedding_model
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    rk, sk, samp_k = jax.random.split(key, 3)
    return chunked_stream(
        lambda *a: nbf_samples_density_stream_batch(env, filt, dmodel, *a),
        (state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b),
        (jax.random.split(rk, N), split_T(sk, N, T), split_T(samp_k, N, T), neg_pts_b),
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
    # The gold-SMC construction at a small budget: the strongest sample-based
    # baseline under the same locally optimal proposal NBF-OP uses.
    N = batch_size(state0_b)
    T = time_len(obs_next_b)
    prior_key, step_key = jax.random.split(key)
    return chunked_stream(
        lambda *a: gold_smc_sample_stream_batch(env, *a),
        (state0_b, obs_next_b.values),
        (jax.random.split(prior_key, N), split_T(step_key, N, T)),
        (n, num_ref_samples),
        N,
        chunk_size,
        desc=desc,
    )


L96_EXTRA_SAMPLERS = {
    'pf_reg': _filter_sampler(make_pf_reg),
    'pf_rm': _filter_sampler(make_pf_rm),
    'enkf': _filter_sampler(make_enkf),
    'nbf_op': _nbf_op_sampler,
    'pf_op': _pf_op_sampler,
}

L96_EXTRA_DENSITY_SAMPLERS = {
    'pf_reg': _kde_density_sampler(make_pf_reg),
    'pf_rm': _kde_density_sampler(make_pf_rm),
    'enkf': _enkf_density_sampler,
    'nbf_op': _nbf_op_density_sampler,
}


def compute_method_samples(spec: str, **kwargs) -> Array:
    return _compute_method_samples(spec, extra_samplers=L96_EXTRA_SAMPLERS, **kwargs)


def compute_method_samples_and_density(spec: str, **kwargs):
    return _compute_method_samples_and_density(
        spec, extra_density_samplers=L96_EXTRA_DENSITY_SAMPLERS, **kwargs
    )


def compute_l96_gold_smc_samples(
    env: Lorenz96Env,
    pf_count: int,
    num_ref_samples: int,
    state0_b,
    obs_next_b,
    key: Array,
    *,
    chunk_size: int = 0,
    desc: str | None = None,
):
    N = batch_size(state0_b)
    T = time_len(obs_next_b)
    prior_key, step_key = jax.random.split(key)
    return chunked_stream(
        lambda *a: gold_smc_sample_stream_batch(env, *a),
        (state0_b, obs_next_b.values),
        (jax.random.split(prior_key, N), split_T(step_key, N, T)),
        (pf_count, num_ref_samples),
        N,
        chunk_size,
        desc=desc,
    )
