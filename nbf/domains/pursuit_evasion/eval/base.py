import jax.numpy as jnp

from nbf.domains.pursuit_evasion.env import (
    SENSOR_POS,
    SIGMA_DYN,
    SIGMA_OBS_MAX,
    L,
    PursuitEvasionEnv,
    propagate_deterministic,
)
from nbf.filters.gmf import GMF


def make_pe_gmf(
    num_evaders: int,
    M: int = 1,
    observation_sigma_scale: float = 1.0,
    kl_merge: float = 0.5,
    weight_prune: float = 0.01,
) -> GMF:
    n = 2 * num_evaders
    return GMF(
        num_evaders=num_evaders,
        sigma_dyn=SIGMA_DYN,
        sigma_obs=SIGMA_OBS_MAX,
        sensor_pos=SENSOR_POS,
        propagate_fn=propagate_deterministic,
        init_mu=jnp.full(n, L / 2),
        init_sigma=((L / 2) ** 2 * jnp.eye(n)),
        M=M,
        jitter_scale=0.1 * L,
        kl_merge=kl_merge,
        weight_prune=weight_prune,
        observation_sigma_scale=observation_sigma_scale,
    )


def make_env_kn(k: int, episode_length: int, policy_tuple) -> PursuitEvasionEnv:
    return PursuitEvasionEnv(
        num_evaders=k,
        time_limit=episode_length,
        policy=policy_tuple,
        shuffle_observations=True,
    )
