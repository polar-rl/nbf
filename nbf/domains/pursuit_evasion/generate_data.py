import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from tqdm import tqdm

from nbf.data import load_arrays
from nbf.domains.pursuit_evasion.config import (
    Config,
    DataConfig,
    load_config,
    resolve_runtime_paths,
)
from nbf.domains.pursuit_evasion.env import PursuitEvasionEnv
from nbf.domains.pursuit_evasion.policies import sample_training_policies
from nbf.filters.particle import ParticleSet


def data_dirname(num_evaders: int) -> str:
    """Directory holding streaming .npy files."""
    return f'pe_belief_data_k{num_evaders}'


def _init_pf(env: PursuitEvasionEnv, num_pf_particles: int, key: Array) -> ParticleSet:
    init_keys = jax.random.split(key, num_pf_particles)
    particle_states = jax.vmap(env.generator)(init_keys)
    weights = jnp.full(num_pf_particles, 1.0 / num_pf_particles)
    return ParticleSet(particle_states, weights)


@eqx.filter_jit
def _episode_batch(
    env: PursuitEvasionEnv,
    num_steps: int,
    num_pf_particles: int,
    num_samples: int,
    resample_threshold: float,
    batch_keys: Array,
) -> tuple[Array, Array]:
    def one_episode(key):
        reset_key, init_key, row0_key, scan_key = jax.random.split(key, 4)
        state = env.generator(reset_key)
        pset = _init_pf(env, num_pf_particles, init_key)
        step_keys = jax.random.split(scan_key, num_steps)

        def _subsample(p, key):
            idx = jax.random.choice(key, num_pf_particles, (num_samples,), replace=False)
            positions = p.particles.positions[idx]
            w_raw = p.weights[idx]
            return positions, w_raw / (jnp.sum(w_raw) + 1e-30)

        p0, w0 = _subsample(pset, row0_key)

        def step(carry, k):
            cur_state, cur_pset = carry
            step_k, sim_k, post_k, resample_k = jax.random.split(k, 4)

            cur_state, timestep = env.step(cur_state, 0, step_k)
            obs = timestep.observation

            cur_pset, _ = cur_pset.simulate(env, 0, obs, sim_k)
            cur_pset = jax.lax.cond(
                cur_pset.neff() < num_pf_particles * resample_threshold,
                lambda: cur_pset.systematic_resample(resample_k),
                lambda: cur_pset,
            )

            particles_tp1, weights_tp1 = _subsample(cur_pset, post_k)

            return (cur_state, cur_pset), (particles_tp1, weights_tp1)

        _, (ptp1, wtp1) = jax.lax.scan(step, (state, pset), step_keys)
        ptp1 = jnp.concatenate([p0[None], ptp1], axis=0)
        wtp1 = jnp.concatenate([w0[None], wtp1], axis=0)
        return ptp1, wtp1

    return jax.vmap(one_episode)(batch_keys)


def _generate_pf(
    data_cfg: DataConfig,
    num_samples: int,
    *,
    key: Array,
    out_dir: str,
) -> None:
    k = data_cfg.num_evaders
    num_policies = data_cfg.num_policies
    eps_per_scenario = data_cfg.num_episodes_per_policy

    key, policy_key = jax.random.split(key)
    library = sample_training_policies(policy_key, num_policies)

    rows_per_episode = data_cfg.episode_length + 1
    total = num_policies * eps_per_scenario * rows_per_episode
    pt_shape = (total, num_samples, 2 * k)
    w_shape = (total, num_samples)
    block = eps_per_scenario * rows_per_episode

    os.makedirs(out_dir, exist_ok=True)
    ptp1_out = np.lib.format.open_memmap(
        os.path.join(out_dir, 'particles_tp1.npy'), mode='w+', dtype=np.float16, shape=pt_shape
    )
    wtp1_out = np.lib.format.open_memmap(
        os.path.join(out_dir, 'weights_tp1.npy'), mode='w+', dtype=np.float32, shape=w_shape
    )

    for s in tqdm(range(num_policies), desc='Scenarios (PF)'):
        key, sel_key, batch_key = jax.random.split(key, 3)
        sel_idx = jax.random.randint(sel_key, (k,), 0, len(library))
        episode_policies = tuple(library[int(i)][1] for i in sel_idx)

        env = PursuitEvasionEnv(
            num_evaders=k,
            time_limit=data_cfg.episode_length,
            policy=episode_policies,
            shuffle_observations=data_cfg.shuffle_observations,
            interaction_radius=data_cfg.interaction_radius,
            interaction_strength=data_cfg.interaction_strength,
        )

        ep_keys = jax.random.split(batch_key, eps_per_scenario)
        ptp1, wtp1 = _episode_batch(
            env,
            data_cfg.episode_length,
            data_cfg.num_pf_particles,
            num_samples,
            0.5,
            ep_keys,
        )
        start = s * block
        stop = start + block
        ptp1_out[start:stop] = np.asarray(ptp1).reshape(block, num_samples, 2 * k)
        wtp1_out[start:stop] = np.asarray(wtp1).reshape(block, num_samples)

    ptp1_out.flush()
    wtp1_out.flush()


def generate_data(
    data_cfg: DataConfig,
    num_samples: int,
    *,
    key: Array,
    out_dir: str,
) -> None:
    _generate_pf(data_cfg, num_samples, key=key, out_dir=out_dir)


def load_data(data_path: str, num_evaders: int = 1) -> dict[str, np.ndarray]:
    """Load a previously generated belief-data directory of ``.npy`` files."""
    dpath = os.path.join(data_path, data_dirname(num_evaders))
    return load_arrays(dpath, ('particles_tp1', 'weights_tp1'))


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate pursuit-evasion belief data.')
    parser.add_argument('--config', type=str, default=None, help='YAML config path')
    parser.add_argument(
        '--set', nargs='*', default=[], help='Override config values: key.path=value'
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config is not None else Config()

    if args.set:
        from nbf.config import apply_overrides

        cfg = apply_overrides(cfg, args.set)
        cfg = resolve_runtime_paths(cfg)

    data_path = cfg.data_path
    os.makedirs(data_path, exist_ok=True)

    num_evaders = cfg.data.num_evaders
    out_path = os.path.join(data_path, data_dirname(num_evaders))

    if os.path.exists(out_path):
        print(f'Data already exists at {out_path} — skipping generation.')
        return

    print(
        f'Generating belief data via particle filter '
        f'(k={num_evaders}, N={cfg.data.num_pf_particles}) ...'
    )
    key = jax.random.PRNGKey(cfg.seed)
    key, data_key = jax.random.split(key)
    generate_data(
        cfg.data,
        num_samples=cfg.training.num_samples,
        key=data_key,
        out_dir=out_path,
    )
    print(f'Saved triples to {out_path}/')


if __name__ == '__main__':
    main()
