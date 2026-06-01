import equinox as eqx
import jax
import jax.numpy as jnp

from nbf.domains.pursuit_evasion.env import SENSOR_POS, L

KIND_GOAL = 0
KIND_RW = 1
KIND_EVASIVE = 2
KIND_PATROL = 3
KIND_WALL = 4
KIND_ORBIT = 5

MAX_WAYPOINTS = 4


class Policy(eqx.Module):
    kind: jax.Array
    point: jax.Array
    waypoints: jax.Array
    n_waypoints: jax.Array
    concentration: jax.Array

    def __call__(self, position, key):
        return jax.lax.switch(
            self.kind,
            (
                lambda: _goal_call(self, position, key),
                lambda: _rw_call(self, position, key),
                lambda: _evasive_call(self, position, key),
                lambda: _patrol_call(self, position, key),
                lambda: _wall_call(self, position, key),
                lambda: _orbit_call(self, position, key),
            ),
        )


def _goal_call(p: Policy, position, key):
    diff = p.point - position
    mean_angle = jnp.arctan2(diff[1], diff[0])
    return mean_angle + jax.random.normal(key) / p.concentration


def _rw_call(p: Policy, position, key):
    del p, position
    return jax.random.uniform(key, minval=0.0, maxval=2 * jnp.pi)


def _evasive_call(p: Policy, position, key):
    diff = position - p.point
    mean_angle = jnp.arctan2(diff[1], diff[0])
    return mean_angle + jax.random.normal(key) * 0.3


def _patrol_call(p: Policy, position, key):
    raw_dists = jnp.linalg.norm(p.waypoints - position, axis=-1)
    valid = jnp.arange(MAX_WAYPOINTS) < p.n_waypoints
    dists = jnp.where(valid, raw_dists, jnp.inf)
    sorted_idx = jnp.argsort(dists)
    target_idx = jax.lax.cond(
        dists[sorted_idx[0]] < 0.5,
        lambda: sorted_idx[1],
        lambda: sorted_idx[0],
    )
    diff = p.waypoints[target_idx] - position
    mean_angle = jnp.arctan2(diff[1], diff[0])
    return mean_angle + jax.random.normal(key) / p.concentration


def _wall_call(p: Policy, position, key):
    del p
    dists = jnp.array(
        [
            position[0],
            L - position[0],
            position[1],
            L - position[1],
        ]
    )
    nearest = jnp.argmin(dists)
    angles = jnp.array([jnp.pi / 2, 0.0, -jnp.pi / 2, jnp.pi])
    return angles[nearest] + jax.random.normal(key) * 0.2


def _orbit_call(p: Policy, position, key):
    diff = position - p.point
    angle = jnp.arctan2(diff[1], diff[0])
    tangent = angle + jnp.pi / 2.0
    return tangent + jax.random.normal(key) / p.concentration


def _zero_point() -> jax.Array:
    return jnp.zeros((2,), dtype=jnp.float32)


def _zero_waypoints() -> jax.Array:
    return jnp.zeros((MAX_WAYPOINTS, 2), dtype=jnp.float32)


def make_goal_policy(goal, concentration: float = 3.0) -> Policy:
    return Policy(
        kind=jnp.asarray(KIND_GOAL, dtype=jnp.int32),
        point=jnp.asarray(goal, dtype=jnp.float32),
        waypoints=_zero_waypoints(),
        n_waypoints=jnp.asarray(0, dtype=jnp.int32),
        concentration=jnp.asarray(concentration, dtype=jnp.float32),
    )


def make_random_walk_policy() -> Policy:
    return Policy(
        kind=jnp.asarray(KIND_RW, dtype=jnp.int32),
        point=_zero_point(),
        waypoints=_zero_waypoints(),
        n_waypoints=jnp.asarray(0, dtype=jnp.int32),
        concentration=jnp.asarray(1.0, dtype=jnp.float32),
    )


def make_evasive_policy(sensor_pos=SENSOR_POS) -> Policy:
    return Policy(
        kind=jnp.asarray(KIND_EVASIVE, dtype=jnp.int32),
        point=jnp.asarray(sensor_pos, dtype=jnp.float32),
        waypoints=_zero_waypoints(),
        n_waypoints=jnp.asarray(0, dtype=jnp.int32),
        concentration=jnp.asarray(1.0, dtype=jnp.float32),
    )


def make_patrol_policy(waypoints, concentration: float = 3.0) -> Policy:
    wp = jnp.asarray(waypoints, dtype=jnp.float32)
    n = wp.shape[0]
    if n > MAX_WAYPOINTS:
        raise ValueError(f'patrol policy supports at most {MAX_WAYPOINTS} waypoints, got {n}')
    padded = jnp.zeros((MAX_WAYPOINTS, 2), dtype=jnp.float32).at[:n].set(wp)
    return Policy(
        kind=jnp.asarray(KIND_PATROL, dtype=jnp.int32),
        point=_zero_point(),
        waypoints=padded,
        n_waypoints=jnp.asarray(n, dtype=jnp.int32),
        concentration=jnp.asarray(concentration, dtype=jnp.float32),
    )


def make_wall_hugger_policy() -> Policy:
    return Policy(
        kind=jnp.asarray(KIND_WALL, dtype=jnp.int32),
        point=_zero_point(),
        waypoints=_zero_waypoints(),
        n_waypoints=jnp.asarray(0, dtype=jnp.int32),
        concentration=jnp.asarray(1.0, dtype=jnp.float32),
    )


def make_orbit_policy(center, concentration: float = 5.0) -> Policy:
    return Policy(
        kind=jnp.asarray(KIND_ORBIT, dtype=jnp.int32),
        point=jnp.asarray(center, dtype=jnp.float32),
        waypoints=_zero_waypoints(),
        n_waypoints=jnp.asarray(0, dtype=jnp.int32),
        concentration=jnp.asarray(concentration, dtype=jnp.float32),
    )


def sample_training_policies(key, num_policies: int = 100):
    policies: list[tuple[str, Policy]] = []
    keys = jax.random.split(key, num_policies)

    n_goal = round(0.4 * num_policies)
    n_rw = round(0.2 * num_policies)
    n_patrol = round(0.2 * num_policies)
    n_evasive = num_policies - n_goal - n_rw - n_patrol

    idx = 0
    for i in range(n_goal):
        k1, k2 = jax.random.split(keys[idx])
        goal = jax.random.uniform(k1, (2,), minval=0.0, maxval=L)
        conc = float(jax.random.uniform(k2, (), minval=1.0, maxval=5.0))
        policies.append((f'goal_{i}', make_goal_policy(goal, conc)))
        idx += 1

    for i in range(n_rw):
        policies.append((f'rw_{i}', make_random_walk_policy()))
        idx += 1

    for i in range(n_patrol):
        k1, k2 = jax.random.split(keys[idx])
        n_wp = int(jax.random.randint(k1, (), 2, MAX_WAYPOINTS + 1))
        wp = jax.random.uniform(k2, (n_wp, 2), minval=0.5, maxval=L - 0.5)
        policies.append((f'patrol_{i}', make_patrol_policy(wp)))
        idx += 1

    for i in range(n_evasive):
        policies.append((f'evasive_{i}', make_evasive_policy()))
        idx += 1

    return policies


OOD_POLICIES: dict[str, Policy] = {
    'opposite_corner_A': make_goal_policy(jnp.array([1.0, 1.0]), concentration=5.0),
    'opposite_corner_B': make_goal_policy(jnp.array([9.0, 9.0]), concentration=5.0),
    'evasive': make_evasive_policy(),
    'patrol_diamond': make_patrol_policy(
        jnp.array([[2.0, 5.0], [5.0, 8.0], [8.0, 5.0], [5.0, 2.0]])
    ),
    'wall_hugger': make_wall_hugger_policy(),
}


OOD_MARGINAL_COMPONENTS = ('wall', 'orbit', 'sharp_goal', 'diffuse_goal')
OOD_MARGINAL_WEIGHTS = (0.35, 0.25, 0.25, 0.15)


def sample_ood_component_policy(key, component: str) -> Policy:
    if component == 'wall':
        return make_wall_hugger_policy()
    if component == 'orbit':
        center = jax.random.uniform(key, (2,), minval=1.0, maxval=L - 1.0)
        return make_orbit_policy(center, concentration=5.0)
    if component == 'sharp_goal':
        k1, k2 = jax.random.split(key)
        goal = jax.random.uniform(k1, (2,), minval=0.0, maxval=L)
        conc = float(jax.random.uniform(k2, (), minval=5.0, maxval=10.0))
        return make_goal_policy(goal, conc)
    if component == 'diffuse_goal':
        k1, k2 = jax.random.split(key)
        goal = jax.random.uniform(k1, (2,), minval=0.0, maxval=L)
        conc = float(jax.random.uniform(k2, (), minval=0.3, maxval=1.0))
        return make_goal_policy(goal, conc)
    raise ValueError(f'unknown OOD component {component!r}')


def sample_ood_marginal_policy(key) -> Policy:
    k_choice, k_param = jax.random.split(key)
    weights = jnp.asarray(OOD_MARGINAL_WEIGHTS, dtype=jnp.float32)
    idx = int(jax.random.categorical(k_choice, jnp.log(weights)))
    return sample_ood_component_policy(k_param, OOD_MARGINAL_COMPONENTS[idx])


TRAIN_MIXTURE_COMPONENTS = ('goal', 'random_walk', 'patrol', 'evasive')
TRAIN_MIXTURE_WEIGHTS = (0.4, 0.2, 0.2, 0.2)


def sample_train_component_policy(key, component: str) -> Policy:
    if component == 'goal':
        k1, k2 = jax.random.split(key)
        goal = jax.random.uniform(k1, (2,), minval=0.0, maxval=L)
        conc = float(jax.random.uniform(k2, (), minval=1.0, maxval=5.0))
        return make_goal_policy(goal, conc)
    if component == 'random_walk':
        return make_random_walk_policy()
    if component == 'patrol':
        k1, k2 = jax.random.split(key)
        n_wp = int(jax.random.randint(k1, (), 2, MAX_WAYPOINTS + 1))
        wp = jax.random.uniform(k2, (n_wp, 2), minval=0.5, maxval=L - 0.5)
        return make_patrol_policy(wp)
    if component == 'evasive':
        return make_evasive_policy()
    raise ValueError(f'unknown training component {component!r}')


def sample_train_mixture_policy(key) -> Policy:
    """Single per-evader draw from the training mixture (40/20/20/20)."""
    k_choice, k_param = jax.random.split(key)
    weights = jnp.asarray(TRAIN_MIXTURE_WEIGHTS, dtype=jnp.float32)
    idx = int(jax.random.categorical(k_choice, jnp.log(weights)))
    return sample_train_component_policy(k_param, TRAIN_MIXTURE_COMPONENTS[idx])


def sample_in_dist_tuple(key, num_evaders: int) -> tuple[Policy, ...]:
    """k iid policies from the training mixture (matches NBF training joint)."""
    keys = jax.random.split(key, num_evaders)
    return tuple(sample_train_mixture_policy(ki) for ki in keys)


def sample_one_evader_ood_tuple(key, num_evaders: int) -> tuple[Policy, ...]:
    if num_evaders < 1:
        raise ValueError(f'num_evaders must be >= 1, got {num_evaders}')
    ood_key, train_key = jax.random.split(key)
    ood_pol = sample_ood_marginal_policy(ood_key)
    if num_evaders == 1:
        return (ood_pol,)
    train_keys = jax.random.split(train_key, num_evaders - 1)
    train_pols = tuple(sample_train_mixture_policy(ki) for ki in train_keys)
    return (ood_pol, *train_pols)


def ood_marginal_scenarios(key, num_evaders: int) -> dict[str, tuple[Policy, ...]]:
    out: dict[str, tuple[Policy, ...]] = {}
    comp_keys = jax.random.split(key, len(OOD_MARGINAL_COMPONENTS) + 1)
    for ck, comp in zip(comp_keys[:-1], OOD_MARGINAL_COMPONENTS, strict=False):
        evader_keys = jax.random.split(ck, num_evaders)
        out[f'ood_{comp}'] = tuple(sample_ood_component_policy(ek, comp) for ek in evader_keys)
    mix_keys = jax.random.split(comp_keys[-1], num_evaders)
    out['ood_mixture'] = tuple(sample_ood_marginal_policy(ek) for ek in mix_keys)
    return out


def joint_correlated_scenarios(num_evaders: int) -> dict[str, tuple[Policy, ...]]:
    if num_evaders < 2:
        return {}
    diamond = make_patrol_policy(jnp.array([[2.0, 5.0], [5.0, 8.0], [8.0, 5.0], [5.0, 2.0]]))
    goal_a = make_goal_policy(jnp.array([1.0, 1.0]), concentration=5.0)
    goal_b = make_goal_policy(jnp.array([9.0, 9.0]), concentration=5.0)
    converge_goal = make_goal_policy(jnp.array([5.0, 5.0]), concentration=5.0)
    evasive = make_evasive_policy()
    wall = make_wall_hugger_policy()

    def alternate(a, b):
        return tuple(a if i % 2 == 0 else b for i in range(num_evaders))

    return {
        'all_evasive': (evasive,) * num_evaders,
        'all_wall_hugger': (wall,) * num_evaders,
        'goal_split': alternate(goal_a, goal_b),
        'mixed_evasive_patrol': alternate(evasive, diamond),
        'converge': (converge_goal,) * num_evaders,
    }
