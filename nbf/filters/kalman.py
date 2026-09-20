from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

_ALPHA = 1.0
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
    obs_model: object  # None, or a pytree with predicted / vector / noise_std
    observation_sigma_scale: float = eqx.field(static=True)
    heading_draws: int = eqx.field(static=True)

    def __init__(
        self,
        num_evaders: int,
        sigma_dyn: float,
        sigma_obs: float,
        sensor_pos: jax.Array,
        propagate_fn: Callable,
        observation_sigma_scale: float = 1.0,
        obs_model=None,
        heading_draws: int = 0,
    ):
        self.n = 2 * num_evaders
        self.k = num_evaders
        self.wm, self.wc, self.lam = _ukf_weights(self.n)
        self.Q = sigma_dyn**2 * jnp.eye(self.n)
        scaled = sigma_obs * observation_sigma_scale
        self.R = scaled**2 * jnp.eye(self.k)
        self.sensor_pos = jnp.asarray(sensor_pos)
        self.propagate_fn = propagate_fn
        self.obs_model = obs_model
        self.observation_sigma_scale = float(observation_sigma_scale)
        self.heading_draws = int(heading_draws)
        if self.heading_draws > 0 and not callable(propagate_fn):
            raise ValueError(
                'heading_draws > 0 needs a keyed propagate_fn(pos, policy, obstacles, key)'
            )

    def predict(self, mu, sigma, policy, obstacles, key=None):
        sigmas = _sigma_points(mu, sigma, self.lam)
        k = self.k

        # `policy` may be a length-k tuple (one per evader) or a single policy
        # broadcast across all evaders.
        policy_tuple = policy if isinstance(policy, tuple) else (policy,) * k
        propagate = self.propagate_fn

        if self.heading_draws > 0 and key is not None:
            H = self.heading_draws

            def prop_draw(pt, draw_key):
                pos = pt.reshape(k, 2)
                keys = jax.random.split(draw_key, k)
                nxt = [propagate(pos[e], policy_tuple[e], obstacles, keys[e]) for e in range(k)]
                return jnp.stack(nxt, axis=0).reshape(-1)

            draw_keys = jax.random.split(key, H)
            # (P, H, n): every sigma point under H heading draws.
            draws = jax.vmap(lambda pt: jax.vmap(lambda kk: prop_draw(pt, kk))(draw_keys))(sigmas)
            propagated = jnp.mean(draws, axis=1)
            dev = draws - propagated[:, None, :]
            heading_cov = jnp.mean(jnp.einsum('phi,phj->pij', dev, dev), axis=0) / max(H - 1, 1)
            extra = heading_cov
        else:

            def prop(pt):
                pos = pt.reshape(k, 2)
                nxt_list = [propagate(pos[e], policy_tuple[e], obstacles) for e in range(k)]
                return jnp.stack(nxt_list, axis=0).reshape(-1)

            propagated = jax.vmap(prop)(sigmas)
            extra = 0.0

        mu_pred = jnp.sum(self.wm[:, None] * propagated, axis=0)
        diff = propagated - mu_pred
        sigma_pred = jnp.einsum('i,ij,ik->jk', self.wc, diff, diff) + self.Q + extra
        return mu_pred, sigma_pred

    def _predicted_and_observed(self, sigmas, mu_pred, observation):
        if self.obs_model is None:
            k = self.k
            sensor_pos = self.sensor_pos

            def obs_fn(pt):
                pos = pt.reshape(k, 2)
                return jnp.linalg.norm(pos - sensor_pos, axis=-1)

            return jax.vmap(obs_fn)(sigmas), observation.ranges, self.R
        pred = self.obs_model.predicted(sigmas, mu_pred)
        y = self.obs_model.vector(observation)
        sig = self.obs_model.noise_std(observation) * self.observation_sigma_scale
        return pred, y, sig**2 * jnp.eye(y.shape[0])

    def update(self, mu_pred, sigma_pred, observation):
        sigmas = _sigma_points(mu_pred, sigma_pred, self.lam)
        pred_obs, y, R = self._predicted_and_observed(sigmas, mu_pred, observation)
        z_pred = jnp.sum(self.wm[:, None] * pred_obs, axis=0)

        dz = pred_obs - z_pred
        S = jnp.einsum('i,ij,ik->jk', self.wc, dz, dz) + R

        dx = sigmas - mu_pred
        Pxz = jnp.einsum('i,ij,ik->jk', self.wc, dx, dz)

        K = Pxz @ jnp.linalg.inv(S)
        innovation = y - z_pred
        mu = mu_pred + K @ innovation
        sigma = sigma_pred - K @ S @ K.T
        sigma = 0.5 * (sigma + sigma.T)
        return mu, sigma

    def step(self, mu, sigma, policy, obstacles, observation, key=None):
        mu_pred, sigma_pred = self.predict(mu, sigma, policy, obstacles, key)
        return self.update(mu_pred, sigma_pred, observation)
