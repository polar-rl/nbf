# nbf

`nbf` implements Neural Bayesian Filtering (NBF), a Bayesian filter that tracks beliefs under partial observability by representing them as embeddings. At each step, it samples particles from a learned conditional generator, propagates them through the dynamics and embeds the propagated set, weighted by the likelihood of the new observation.

This repository accompanies the paper [Neural Bayesian Filtering](https://arxiv.org/abs/2510.03614), which proposes the filter. The commit tagged `2510.03614v2` marks the code used for its experiments; later commits may differ. [Belief Embedding Tree Search](https://arxiv.org/abs/2610.15046) builds on `nbf`.

## Installation

`nbf` requires Python 3.13 and is installed with [uv](https://docs.astral.sh/uv/), which also fetches Python if needed. Clone the repository and let `uv` set up the environment:

```bash
git clone https://github.com/polar-rl/nbf.git
cd nbf
uv sync
```

This also installs [`embedex`](https://github.com/polar-rl/embedex) from GitHub, at the commit tagged `2510.03614v2`. NBF uses `embedex`'s distribution embeddings to represent beliefs.

Two extras add optional dependencies, alone or together (`uv sync --extra gpu --extra dev`):

- `gpu`: JAX with CUDA 12 support (Linux only).
- `dev`: the development tools (pytest, ruff and pre-commit).

### Contributing

Install with `--extra dev` and enable the pre-commit hooks by running `uv run pre-commit install`. Run the tests with `uv run pytest`, or `uv run pytest -m "not slow"` to skip the slowest ones.

To develop `embedex` and `nbf` together, clone `embedex` next to `nbf` and install your clone in place of the pinned one: in `pyproject.toml`, comment out the GitHub source of `embedex` under `[tool.uv.sources]`, uncomment the local source below it and run `uv sync`. Edits to the clone then take effect without reinstalling. The `no-local-sources` pre-commit hook refuses to commit `pyproject.toml` or `uv.lock` while they point at a local clone, so restore them with `git checkout -- pyproject.toml uv.lock` and run `uv sync` before committing.

To move `nbf` to another version of `embedex`, change the `tag` of its source in `pyproject.toml` and run `uv sync`.

## Contents

- `nbf/filters/`: NBF and the baseline filters it is compared against (particle filters, ensemble Kalman filters and a Gaussian mixture filter).
- `nbf/train.py`: the on-policy training loop shared by every domain.
- `nbf/eval/`: the evaluation grid, its metrics and its plots.
- `nbf/domains/`: the domains used in the experiments, Lorenz-96 and pursuit-evasion, each with its environment, training entry point and evaluation grid.

## Running the experiments

The configs of the paper's experiments live in the `configs/` directories of the two domains. Train a model with a config, then score it and the baselines with the domain's evaluation grid. Models and tables are written under `results/`:

```bash
uv run python -m nbf.domains.lorenz96.train --config nbf/domains/lorenz96/configs/nbf_l96_hard_k20.yaml
uv run python -m nbf.domains.lorenz96.run_grid l2 --model_dir results/k20/nbf_hard --output_dir results/k20/grid

uv run python -m nbf.domains.pursuit_evasion.train --config nbf/domains/pursuit_evasion/configs/nbf_k1_sharp.yaml
uv run python -m nbf.domains.pursuit_evasion.run_grid id --model_dir results/pursuit_evasion/k1_sharp/nbf --output_dir results/pursuit_evasion/k1_sharp/grid
```

`--set key.path=value` overrides any training config field, for example `--set seed=1`. The evaluation grids take a cell: `l1`, `l2` or `sigma` for Lorenz-96, and `id`, `ood` or `sigma` for pursuit-evasion.

## Citation

```bibtex
@misc{solinas2026nbf,
    title={Neural Bayesian Filtering},
    author={Christopher Solinas and Radovan Haluska and David Sychrovsky and Finbarr Timbers and Nolan Bard and Michael Buro and Martin Schmid and Nathan R. Sturtevant and Michael Bowling},
    year={2026},
    eprint={2510.03614},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
}
```
