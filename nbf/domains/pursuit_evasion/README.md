# Pursuit-Evasion Filtering Experiments

Range-only target tracking with a stationary sensor in a 10×10 grid with obstacles. Supports `k = 1, 3, 5` evaders. Compares Neural Bayesian Filtering (NBF) against particle filters and a Gaussian mixture filter.

## Files

| File | Purpose |
|---|---|
| `env.py` | `PursuitEvasionEnv` — dynamics, range-only observations, obstacles, soft inter-evader repulsion, optional shuffled-observation likelihood |
| `policies.py` | Evader policy library (goal, random walk, patrol, evasive, wall-hugger, orbit), the training mixture, and the named OOD scenarios |
| `train.py` | On-policy NBF training: episodes are simulated on the fly under policies drawn from the training mixture, the filter is rolled out with the current parameters, and the generator objective is evaluated at the true joint state (loop in `nbf/train.py`) |
| `run_grid.py` | Evaluation grid: `id` (training-mixture components), `ood` (held-out policies), `sigma` (pinned observation noise past the training range); one shard per (cell, subcell, seed), `--merge` builds the tables |
| `eval/` | Evaluation helpers: GMF construction, sample and density streams, method dispatch, sigma override |
| `visualize.py` | Plots: metric-vs-time, dimension scaling, OOD bar charts, filter heatmaps |
| `config.py` | `DataConfig` (evader count, episode length, interaction, shuffling) |

There is no data generation step and no reference posterior: the trainer needs only the simulator, and evaluation scores every method on the true state (NLL for native densities, RMSE, CRPS, coverage, spread).

## Multi-evader (k > 1)

For `k > 1` two mechanisms make the joint posterior genuinely non-factorisable, so a tracker that maintains independent per-evader marginals cannot succeed:

- **Pairwise soft repulsion** in `env.transition_joint`: each evader is pushed away from any neighbour closer than `interaction_radius` with magnitude proportional to `interaction_strength`. Per-evader policies are still sampled independently from the training mixture each episode — all coupling comes from this interaction term, not from shared intent.
- **Shuffled range observations** (`shuffle_observations: true`): the sensor returns the `k` ranges in a random order at every step. The likelihood in `observation_weights` is marginalised exactly over the `k!` permutations (`k ≤ 5` keeps this cheap).

The NBF set-transformer embedding is permutation-invariant over particles, which matches both the joint-state symmetry and the shuffled-observation likelihood.

## Configs

`configs/nbf_k{1,3,5}.yaml` — one config per evader count: model (embedding + flow-matching generator), training and rollout hyperparameters, and `save_path` (relative to the repo root). `configs/smoke_k1.yaml` is the CPU smoke config used by `scripts/smoke_k1.sh`.

## Running

```bash
# Train the NBF embedding on-policy
python -m nbf.domains.pursuit_evasion.train \
    --config nbf/domains/pursuit_evasion/configs/nbf_k1.yaml

# Evaluate one cell locally (id / ood / sigma), or one shard per array task on a cluster
python -m nbf.domains.pursuit_evasion.run_grid id \
    --model_dir results/pursuit_evasion/k1/nbf --output_dir results/pursuit_evasion/k1/grid
python -m nbf.domains.pursuit_evasion.run_grid --list_tasks --output_dir /tmp/x   # array size
python -m nbf.domains.pursuit_evasion.run_grid --merge --t_min 2 \
    --model_dir results/pursuit_evasion/k1/nbf --output_dir results/pursuit_evasion/k1/grid
```
