import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from nbf.filters.gmf import GMF


@eqx.filter_jit
def _gmf_logp_step(weights, mus, sigmas, x_true: Array) -> Array:
    """Log-density of the Gaussian mixture at the true state."""
    x_flat = x_true.reshape(-1)  # (2k,)
    n = x_flat.shape[-1]
    safe_w = weights / (jnp.sum(weights) + 1e-30)
    log_w = jnp.log(safe_w + 1e-30)

    def comp_logpdf(mu, sig, x):
        sig_sym = 0.5 * (sig + sig.T) + 1e-6 * jnp.eye(n)
        L = jnp.linalg.cholesky(sig_sym)
        diff = x - mu
        sol = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return -0.5 * (jnp.sum(sol * sol) + log_det + n * jnp.log(2.0 * jnp.pi))

    def logpdf(x):
        per_comp = jax.vmap(lambda mu, sig: comp_logpdf(mu, sig, x))(mus, sigmas)
        return jax.scipy.special.logsumexp(log_w + per_comp)

    return logpdf(x_flat)


@eqx.filter_jit
def _gmf_sample_mixture(weights, mus, sigmas, S: int, k: int, key: Array) -> Array:
    pick_key, noise_key = jax.random.split(key)
    safe_w = weights / (jnp.sum(weights) + 1e-30)
    idx = jax.random.choice(pick_key, mus.shape[0], (S,), p=safe_w)
    mu_sel = mus[idx]
    sig_sel = sigmas[idx]
    n = mu_sel.shape[-1]
    # Cholesky with jitter for numerical stability under near-degenerate modes
    jitter = 1e-6 * jnp.eye(n)
    L_chol = jnp.linalg.cholesky(sig_sel + jitter[None])
    eps = jax.random.normal(noise_key, (S, n))
    draws = mu_sel + jnp.einsum('sij,sj->si', L_chol, eps)
    return draws.reshape(S, k, 2)


@eqx.filter_jit
def gmf_reference_sample_stream_batch(
    gmf: GMF,
    obstacles,
    policy_tuple,
    states_next_b,
    obs_next_b,
    init_keys: Array,
    sample_keys_b: Array,
    num_ref_samples: int,
):
    k = gmf.ukf.k

    def _one(states_next, obs_next, ikey, sample_keys):
        w0, mu0, sg0 = gmf.init(key=ikey)

        def _step(carry, xs):
            w, mu, sg = carry
            samp_k, _state_t, next_obs_t = xs
            w, mu, sg = gmf.step(w, mu, sg, policy_tuple, obstacles, next_obs_t)
            samps = _gmf_sample_mixture(w, mu, sg, num_ref_samples, k, samp_k)
            return (w, mu, sg), samps

        _, samps = jax.lax.scan(_step, (w0, mu0, sg0), (sample_keys, states_next, obs_next))
        return samps

    return jax.vmap(_one)(
        states_next_b,
        obs_next_b,
        init_keys,
        sample_keys_b,
    )


@eqx.filter_jit
def gmf_samples_density_stream_batch(
    env,
    gmf: GMF,
    states_next_b,
    obs_next_b,
    init_keys: Array,
    sample_keys_b: Array,
    num_ref_samples: int,
):
    k = gmf.ukf.k

    def _one(states_next, obs_next, ikey, sample_keys):
        w0, mu0, sg0 = gmf.init(key=ikey)

        def _step(carry, xs):
            w, mu, sg = carry
            samp_k, state_t, next_obs_t = xs
            w, mu, sg = gmf.step(w, mu, sg, env.policy, env.obstacles, next_obs_t)
            samps = _gmf_sample_mixture(w, mu, sg, num_ref_samples, k, samp_k)
            lp_true = _gmf_logp_step(w, mu, sg, state_t.positions)
            return (w, mu, sg), (samps, lp_true)

        _, (samps, lp_true) = jax.lax.scan(
            _step, (w0, mu0, sg0), (sample_keys, states_next, obs_next)
        )
        return samps, lp_true

    return jax.vmap(_one)(states_next_b, obs_next_b, init_keys, sample_keys_b)
