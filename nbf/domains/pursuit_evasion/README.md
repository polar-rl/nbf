# Pursuit-Evasion Filtering Experiments

Range-only target tracking with a stationary sensor in a 10×10 grid with obstacles. Supports `k = 1, 2, 3, 5` evaders. Compares Neural Bayesian Filtering (NBF) against standard particle filters and UKF baselines.

## Files

| File | Purpose |
|---|---|
| `env.py` | `PursuitEvasionEnv` — dynamics, range-only observations, obstacles, soft inter-evader repulsion, optional shuffled-observation likelihood |
| `policies.py` | Evader policy library (goal, random walk, patrol, evasive, wall-hugger) and named OOD k-tuples (`ood_scenarios`) |
| `kalman.py` | Unscented Kalman Filter baseline (deliberately weak; Gaussian misfit on multimodal posteriors) |
| `gmf.py` | Gaussian mixture filter baseline |
| `pf_kde_bandwidths.py` | KDE bandwidth utilities for particle-filter density evaluation |
| `generate_data.py` | Belief data generation: joint particle filter at large particle count, treated as the reference posterior |
| `train.py` | NBF embedding model training (delegates the training loop to `embedex.distributions.training`) |
| `evaluate.py` | Evaluation dispatcher — runs one or more of `experiments/experiment_{1..8}` |
| `visualize.py` | Plots: JS-vs-time, dimension scaling, OOD bar charts, filter heatmaps |
| `config.py` | PE-specific config dataclasses; reads `configs/nbf_k*.yaml` |
| `eval_utils.py`, `eval_utils_kn.py` | Shared evaluation helpers (rollouts, embedding loading, k>1 specifics) |
| `experiments/experiment_{1..8}.py` | Individual evaluation scenarios dispatched by `evaluate.py` |
| `experiments/experiment_diag.py` | Early-step diagnostic for k=1 |

## Multi-evader (k > 1)

For `k > 1` two mechanisms make the joint posterior genuinely non-factorisable, so a tracker that maintains independent per-evader marginals cannot succeed:

- **Pairwise soft repulsion** in `env.transition_joint`: each evader is pushed away from any neighbour closer than `interaction_radius` with magnitude proportional to `interaction_strength`. Per-evader policies are still sampled independently from the policy library each episode — all coupling comes from this interaction term, not from shared intent.
- **Shuffled range observations** (`shuffle_observations: true`): the sensor returns the `k` ranges in a random order at every step. The likelihood in `observation_weights` is marginalised exactly over the `k!` permutations (`k ≤ 5` keeps this cheap).

The reference posterior at every `k` is a large joint particle filter (`num_pf_particles`, default 100000). The NBF set-transformer embedding is permutation-invariant over particles, which matches both the joint-state symmetry and the shuffled-observation likelihood.

UKF is included as a baseline at all `k`. It is *not* expected to do well: the range-only likelihood is already multimodal at `k=1` (annulus around the sensor), and at `k>1` the posterior is additionally a mixture over the `k!` permutation-consistent labellings under shuffled observations. A single Gaussian cannot represent either, which is exactly the gap NBF is meant to close.

## Configs

`configs/nbf_k{1,2,3,5}.yaml` — one config per evader count. Each defines the model (embedding + flow matching generator), training hyperparameters, data generation parameters, and the `data_path` / `save_path` roots (relative to the working directory).

## Running

All commands assume the current working directory is the repo root.

```bash
# 1. Generate reference belief data
python -m nbf.domains.pursuit_evasion.generate_data \
    --config nbf/domains/pursuit_evasion/configs/nbf_k1.yaml

# 2. Train the NBF embedding
python -m nbf.domains.pursuit_evasion.train \
    --config nbf/domains/pursuit_evasion/configs/nbf_k1.yaml

# 3. Evaluate (dispatches experiments E1..E8; see --help for filtering)
python -m nbf.domains.pursuit_evasion.evaluate \
    --config nbf/domains/pursuit_evasion/configs/nbf_k1.yaml \
    --experiments 1,2 \
    --model_dir results/pursuit_evasion/k1/nbf/<run_name>
```

`evaluate.py` takes per-experiment flags (`--e1_methods`, `--e2_ks`, `--e5_sigma_methods`, …); see `python -m nbf.domains.pursuit_evasion.evaluate --help` for the full list. Experiments 7 and 8 are PF-only sweeps and do not need an embedding model.
