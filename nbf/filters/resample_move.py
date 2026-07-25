import equinox as eqx
import jax
import jax.numpy as jnp
from jumanji.env import Environment

from nbf.base_types import Observation, State
from nbf.filters.particle import ParticleFilter, ParticleSet, systematic_indices


def rw_mh_move(env, x, means, observation, num_steps, step_size, key):
    def log_target(x):
        return env.transition_log_prob(means, x) + env.observation_log_prob(x, observation)

    def body(carry, key_iter):
        x, cur_log = carry
        prop_key, acc_key = jax.random.split(key_iter)
        prop = env.constrain(x + step_size * jax.random.normal(prop_key, x.shape))
        prop_log = log_target(prop)
        u = jax.random.uniform(acc_key, prop_log.shape)
        accept = jnp.log(u + 1e-30) < prop_log - cur_log
        accept_b = accept.reshape(accept.shape + (1,) * (x.ndim - 1))
        new_x = jnp.where(accept_b, prop, x)
        new_log = jnp.where(accept, prop_log, cur_log)
        return (new_x, new_log), None

    (x, _), _ = jax.lax.scan(body, (x, log_target(x)), jax.random.split(key, num_steps))
    return x


class ResampleMoveParticleFilter(ParticleFilter):
    """SIR with a post-resample move against p(x_t | x_{t-1}, y_t).

    ``num_moves == 0`` uses the env's exact conjugate move (``env.conjugate_move``);
    ``num_moves > 0`` runs that many RW-MH steps against the same target via the
    ``transition_log_prob`` / ``observation_log_prob`` hooks.
    """

    num_moves: int = 0
    step_size: float = 0.3

    def __repr__(self) -> str:
        return f'PF-RM ({self.num_particles})'

    @eqx.filter_jit
    def update(
        self,
        env: Environment,
        particle_set: ParticleSet,
        state: State,
        observation: Observation,
        action: int,
        next_observation: Observation,
        key: jax.Array,
    ) -> tuple[ParticleSet, jax.Array, jax.Array, jax.Array]:
        if self.num_moves == 0 and not hasattr(env, 'conjugate_move'):
            raise ValueError('num_moves == 0 requires an env with a conjugate_move hook')
        simulate_key, resample_key, move_key = jax.random.split(key, 3)
        means = jax.vmap(env.propagate_mean_array)(particle_set.particles)
        particle_set, simulation_fallback = particle_set.simulate(
            env, action, next_observation, simulate_key
        )

        def _resample_and_move():
            idx = systematic_indices(particle_set.weights, resample_key)
            particles = jax.tree.map(lambda x: x[idx], particle_set.particles)

            def _move():
                if self.num_moves == 0:
                    return env.conjugate_move(means[idx], next_observation, move_key)
                return rw_mh_move(
                    env,
                    env.state_array(particles),
                    means[idx],
                    next_observation,
                    self.num_moves,
                    self.step_size,
                    move_key,
                )

            moved = jax.lax.cond(
                env.observation_present(next_observation),
                _move,
                lambda: env.state_array(particles),
            )
            particles = env.replace_state_array(particles, moved)
            n = particle_set.weights.shape[0]
            return ParticleSet(particles, jnp.full(n, 1.0 / n))

        particle_set = jax.lax.cond(
            particle_set.neff() < self.num_particles * self.resample_factor,
            _resample_and_move,
            lambda: particle_set,
        )
        return particle_set, particle_set.neff(), jnp.array(False), simulation_fallback
