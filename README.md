# nbf

Neural Bayesian Filtering: amortized Bayesian filters built on learned distribution embeddings, together with the pursuit-evasion domain they were developed against.

## Installation

`nbf` currently requires Python 3.13 or 3.14. Clone the repository and let `uv` set up the environment:

```bash
git clone git@github.com:polar-rl/nbf.git
cd nbf
uv sync  # optionally add `--extra gpu` for GPU support
```

If you plan to contribute, install with `--extra dev` and enable the pre-commit hooks by running `uv run pre-commit install`.
