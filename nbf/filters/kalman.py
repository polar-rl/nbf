"""Unscented Kalman Filter.

Domain-agnostic: process noise, observation noise, sensor position, and the
deterministic dynamics propagator are all constructor inputs. Callers (e.g. PE)
supply their own; see ``nbf.domains.pursuit_evasion.eval.base.make_pe_gmf`` for
how the pursuit-evasion benchmark assembles its values.

The observation model is range-only: ``z = ||pos - sensor_pos||`` per evader,
with isotropic Gaussian noise. State is the flat ``2k``-vector of evader xy
positions (k = ``num_evaders``).

Precision follows the JAX session: by default float32, or float64 if the caller
enables ``jax_enable_x64`` in their entrypoint. The small ``_ALPHA`` here leads
to catastrophic cancellation in the sigma-point weights at float32, so enabling
x64 is recommended for production use.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

_ALPHA = 1e-3
_BETA = 2.0
_KAPPA = 0.0


def _ukf_weights(n: int):
    lam = _ALPHA**2 * (n + _KAPPA) - n
    wm = jnp.full(2 * n + 1, 0.5 / (n + lam))
    wm = wm.at[0].set(lam / (n + lam))
    wc = jnp.full(2 * n + 1, 0.5 / (n + lam))
    wc = wc.at[0].set(lam / (n + lam) + (1 - _ALPHA**2 + _BETA))
    return wm, wc, jnp.asarray(lam)


def _sigma_points(mu, sigma, lam):
    n = mu.shape[0]
    sigma_sym = 0.5 * (sigma + sigma.T)
    factor = (n + lam) * sigma_sym
    eigvals, eigvecs = jnp.linalg.eigh(factor)
    eigvals = jnp.clip(eigvals, min=1e-8)
    sqrt_m = eigvecs * jnp.sqrt(eigvals)[None, :]
    return jnp.concatenate([mu[None], mu + sqrt_m.T, mu - sqrt_m.T], axis=0)


class UKF(eqx.Module):
    n: int = eqx.field(static=True)
    k: int = eqx.field(static=True)
    wm: jax.Array
    wc: jax.Array
    lam: jax.Array
    Q: jax.Array
    R: jax.Array
    sensor_pos: jax.Array
    propagate_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        num_evaders: int,
        sigma_dyn: float,
        sigma_obs: float,
        sensor_pos: jax.Array,
        propagate_fn: Callable,
        observation_sigma_scale: float = 1.0,
    ):
        self.n = 2 * num_evaders
        self.k = num_evaders
        self.wm, self.wc, self.lam = _ukf_weights(self.n)
        self.Q = sigma_dyn**2 * jnp.eye(self.n)
        scaled = sigma_obs * observation_sigma_scale
        self.R = scaled**2 * jnp.eye(self.k)
        self.sensor_pos = jnp.asarray(sensor_pos)
        self.propagate_fn = propagate_fn

    def predict(self, mu, sigma, policy, obstacles):
        sigmas = _sigma_points(mu, sigma, self.lam)
        k = self.k

        # `policy` may be a length-k tuple (one per evader) or a single policy
        # broadcast across all evaders.
        policy_tuple = policy if isinstance(policy, tuple) else (policy,) * k
        propagate = self.propagate_fn

        def prop(pt):
            pos = pt.reshape(k, 2)
            nxt_list = [propagate(pos[e], policy_tuple[e], obstacles) for e in range(k)]
            return jnp.stack(nxt_list, axis=0).reshape(-1)

        propagated = jax.vmap(prop)(sigmas)
        mu_pred = jnp.sum(self.wm[:, None] * propagated, axis=0)
        diff = propagated - mu_pred
        sigma_pred = jnp.einsum('i,ij,ik->jk', self.wc, diff, diff) + self.Q
        return mu_pred, sigma_pred

    def update(self, mu_pred, sigma_pred, observation):
        sigmas = _sigma_points(mu_pred, sigma_pred, self.lam)
        k = self.k
        sensor_pos = self.sensor_pos

        def obs_fn(pt):
            pos = pt.reshape(k, 2)
            return jnp.linalg.norm(pos - sensor_pos, axis=-1)

        pred_obs = jax.vmap(obs_fn)(sigmas)
        z_pred = jnp.sum(self.wm[:, None] * pred_obs, axis=0)

        dz = pred_obs - z_pred
        S = jnp.einsum('i,ij,ik->jk', self.wc, dz, dz) + self.R

        dx = sigmas - mu_pred
        Pxz = jnp.einsum('i,ij,ik->jk', self.wc, dx, dz)

        K = Pxz @ jnp.linalg.inv(S)
        innovation = observation.ranges - z_pred
        mu = mu_pred + K @ innovation
        sigma = sigma_pred - K @ S @ K.T
        sigma = 0.5 * (sigma + sigma.T)
        return mu, sigma

    def step(self, mu, sigma, policy, obstacles, observation):
        mu_pred, sigma_pred = self.predict(mu, sigma, policy, obstacles)
        return self.update(mu_pred, sigma_pred, observation)
