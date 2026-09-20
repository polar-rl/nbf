"""Gaussian Mixture Filter.

A bank of M UKF-style Gaussian components with weights. Step: each component
runs UKF predict + likelihood-weighted update, then symmetric-KL merging
within ``kl_merge`` and weight pruning below ``weight_prune`` collapse the
mixture back toward ``M`` components.

Domain-agnostic: pass an initial ``(init_mu, init_sigma)``, plus the same
process/observation/dynamics inputs that ``UKF`` takes.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from nbf.filters.kalman import UKF, _sigma_points


def _ukf_update_with_loglik(ukf: UKF, mu_pred, sigma_pred, observation):
    sigmas = _sigma_points(mu_pred, sigma_pred, ukf.lam)
    pred_obs, y, R = ukf._predicted_and_observed(sigmas, mu_pred, observation)
    z_pred = jnp.sum(ukf.wm[:, None] * pred_obs, axis=0)
    dz = pred_obs - z_pred
    S = jnp.einsum('i,ij,ik->jk', ukf.wc, dz, dz) + R
    dx = sigmas - mu_pred
    Pxz = jnp.einsum('i,ij,ik->jk', ukf.wc, dx, dz)

    S_inv = jnp.linalg.inv(S)
    K = Pxz @ S_inv

    innovation = y - z_pred
    mu = mu_pred + K @ innovation
    sigma = sigma_pred - K @ S @ K.T
    sigma = 0.5 * (sigma + sigma.T)

    S_sym = 0.5 * (S + S.T)
    _, logdet = jnp.linalg.slogdet(S_sym)
    quad = innovation @ jnp.linalg.solve(S_sym, innovation)
    d = y.shape[0]
    log_lik = -0.5 * (d * jnp.log(2.0 * jnp.pi) + logdet + quad)
    return mu, sigma, log_lik


def _observation_present(observation) -> jax.Array:
    """The no-observation sentinel is a negative first range."""
    return jnp.reshape(observation.ranges, (-1,))[0] >= 0.0


def _moment_match(w_i, mu_i, sig_i, w_j, mu_j, sig_j):
    w_new = w_i + w_j
    safe = jnp.where(w_new > 0.0, w_new, 1.0)
    a_i = w_i / safe
    a_j = w_j / safe
    mu_new = a_i * mu_i + a_j * mu_j
    di = mu_i - mu_new
    dj = mu_j - mu_new
    sig_new = a_i * (sig_i + jnp.outer(di, di)) + a_j * (sig_j + jnp.outer(dj, dj))
    sig_new = 0.5 * (sig_new + sig_new.T)
    return w_new, mu_new, sig_new


def _kl_gaussian(mu_p, sig_p, mu_q, sig_q):
    n = mu_p.shape[0]
    sig_q_sym = 0.5 * (sig_q + sig_q.T) + 1e-8 * jnp.eye(n)
    sig_p_sym = 0.5 * (sig_p + sig_p.T) + 1e-8 * jnp.eye(n)
    sig_q_inv = jnp.linalg.inv(sig_q_sym)
    diff = mu_q - mu_p
    tr = jnp.trace(sig_q_inv @ sig_p_sym)
    quad = diff @ sig_q_inv @ diff
    _, ldet_q = jnp.linalg.slogdet(sig_q_sym)
    _, ldet_p = jnp.linalg.slogdet(sig_p_sym)
    return 0.5 * (tr + quad - n + ldet_q - ldet_p)


def _sym_kl(mu_i, sig_i, mu_j, sig_j):
    return 0.5 * (_kl_gaussian(mu_i, sig_i, mu_j, sig_j) + _kl_gaussian(mu_j, sig_j, mu_i, sig_i))


class GMF(eqx.Module):
    M: int = eqx.field(static=True)
    ukf: UKF
    init_mu: jax.Array
    init_sigma: jax.Array
    jitter_scale: float = eqx.field(static=True)
    kl_merge: float = eqx.field(static=True)
    weight_prune: float = eqx.field(static=True)

    def __init__(
        self,
        num_evaders: int,
        sigma_dyn: float,
        sigma_obs: float,
        sensor_pos: jax.Array,
        propagate_fn: Callable,
        init_mu: jax.Array,
        init_sigma: jax.Array,
        M: int = 1,
        jitter_scale: float = 0.0,
        kl_merge: float = 0.5,
        weight_prune: float = 0.01,
        observation_sigma_scale: float = 1.0,
        obs_model=None,
        heading_draws: int = 0,
    ):
        self.M = M
        self.ukf = UKF(
            num_evaders=num_evaders,
            sigma_dyn=sigma_dyn,
            sigma_obs=sigma_obs,
            sensor_pos=sensor_pos,
            propagate_fn=propagate_fn,
            observation_sigma_scale=observation_sigma_scale,
            obs_model=obs_model,
            heading_draws=heading_draws,
        )
        self.init_mu = jnp.asarray(init_mu)
        self.init_sigma = jnp.asarray(init_sigma)
        self.jitter_scale = jitter_scale
        self.kl_merge = kl_merge
        self.weight_prune = weight_prune

    def init(self, key=None):
        mus = jnp.tile(self.init_mu[None], (self.M, 1))
        sigmas = jnp.tile(self.init_sigma[None], (self.M, 1, 1))
        if key is not None and self.M > 1 and self.jitter_scale > 0.0:
            jitter = self.jitter_scale * jax.random.normal(key, mus.shape)
            mus = mus + jitter
        weights = jnp.full((self.M,), 1.0 / self.M)
        return weights, mus, sigmas

    def predict(self, weights, mus, sigmas, policy, obstacles, key=None):
        if key is None:
            mus_p, sigmas_p = jax.vmap(lambda mu, sg: self.ukf.predict(mu, sg, policy, obstacles))(
                mus, sigmas
            )
        else:
            keys = jax.random.split(key, self.M)
            mus_p, sigmas_p = jax.vmap(
                lambda mu, sg, kk: self.ukf.predict(mu, sg, policy, obstacles, kk)
            )(mus, sigmas, keys)
        return weights, mus_p, sigmas_p

    def update(self, weights, mus, sigmas, observation):
        def one(mu, sigma):
            return _ukf_update_with_loglik(self.ukf, mu, sigma, observation)

        mus_u, sigmas_u, log_liks = jax.vmap(one)(mus, sigmas)

        log_w = jnp.log(jnp.clip(weights, min=1e-30)) + log_liks
        log_w = log_w - jnp.max(log_w)
        new_w = jnp.exp(log_w) * (weights > 0.0)
        new_w = new_w / (jnp.sum(new_w) + 1e-30)
        return new_w, mus_u, sigmas_u

    def merge(self, weights, mus, sigmas):
        def body(carry, i):
            w, mu, sg = carry
            kls = jax.vmap(lambda j: _sym_kl(mu[i], sg[i], mu[j], sg[j]))(jnp.arange(self.M))
            mask = (jnp.arange(self.M) > i) & (w > 0.0) & (w[i] > 0.0) & (kls < self.kl_merge)
            kls_masked = jnp.where(mask, kls, jnp.inf)
            j = jnp.argmin(kls_masked)
            do_merge = jnp.any(mask)

            w_new, mu_new, sg_new = _moment_match(w[i], mu[i], sg[i], w[j], mu[j], sg[j])
            w_i = jnp.where(do_merge, w_new, w[i])
            mu_i = jnp.where(do_merge, mu_new, mu[i])
            sg_i = jnp.where(do_merge, sg_new, sg[i])
            w_j = jnp.where(do_merge, 0.0, w[j])

            w = w.at[i].set(w_i).at[j].set(w_j)
            mu = mu.at[i].set(mu_i)
            sg = sg.at[i].set(sg_i)
            return (w, mu, sg), None

        (w, mu, sg), _ = jax.lax.scan(body, (weights, mus, sigmas), jnp.arange(self.M))
        total = jnp.sum(w)
        w = w / jnp.where(total > 0.0, total, 1.0)
        return w, mu, sg

    def prune(self, weights, mus, sigmas):
        keep = weights >= self.weight_prune
        w = jnp.where(keep, weights, 0.0)
        total = jnp.sum(w)
        w = w / jnp.where(total > 0.0, total, 1.0)
        return w, mus, sigmas

    def step(self, weights, mus, sigmas, policy, obstacles, observation, key=None):
        w, mu, sg = self.predict(weights, mus, sigmas, policy, obstacles, key)

        def with_obs(args):
            w, mu, sg = self.update(*args, observation)
            w, mu, sg = self.merge(w, mu, sg)
            return self.prune(w, mu, sg)

        return jax.lax.cond(_observation_present(observation), with_obs, lambda a: a, (w, mu, sg))
