"""Print a fingerprint of one NBF reset+update per filter class on fixed keys.

Run at two commits and diff the output: identical lines mean the filter's random
stream and arithmetic are unchanged across a refactor (the guard bets relies on).
    uv run python scripts/check_filter_bitwise.py
"""

import hashlib
import os

import jax
import numpy as np
from embedex.distributions.builder import create_model

from nbf.domains.lorenz96.env import build_lorenz96_env
from nbf.domains.lorenz96.train_gold import load_config, simulate_episodes
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.nbf_op import OptimalProposalNeuralBayesianFilter

SMOKE = os.path.join(
    os.path.dirname(__file__), '..', 'nbf', 'domains', 'lorenz96', 'configs', 'smoke_l96_gold.yaml'
)


def fingerprint(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes())
    return h.hexdigest()[:16]


def main() -> None:
    cfg = load_config(SMOKE, [])
    cfg.data.norm_mean, cfg.data.norm_std = 2.5, 3.9
    env = build_lorenz96_env(cfg.data)
    cfg.model.input_dim = cfg.model.output_dim = cfg.data.K
    model = create_model(cfg.model, dequantize=False, key=jax.random.PRNGKey(0))
    state0, obs0, states_next, obs_prev, obs_next, _ = simulate_episodes(env, 1, jax.random.PRNGKey(1))
    one = lambda t: jax.tree.map(lambda x: x[0], t)  # noqa: E731
    s0, o0 = one(state0), one(obs0)
    st, op, on = (jax.tree.map(lambda x: x[0, 0], t) for t in (states_next, obs_prev, obs_next))
    for cls in (NeuralBayesianFilter, OptimalProposalNeuralBayesianFilter):
        filt = cls(model, 32)
        emb = filt.reset(env, s0, o0, jax.random.PRNGKey(2))
        emb1, neff, f1, f2 = filt.update(env, emb, st, op, 0, on, jax.random.PRNGKey(3))
        pset, *_ = filt.step(env, emb, st, op, 0, on, jax.random.PRNGKey(4))
        print(
            f'{cls.__name__}: reset={fingerprint(emb)} update={fingerprint(emb1, neff)} '
            f'step={fingerprint(pset.particles.x, pset.weights)}'
        )


if __name__ == '__main__':
    main()
