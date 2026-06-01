import os
import sys
import types
from dataclasses import dataclass, fields

import yaml

__all__ = [
    'TrainingConfig',
    'apply_overrides',
    'build_dataclass',
    'load_config',
    'parse_args',
]


@dataclass
class TrainingConfig:
    num_iterations: int = 30000
    learning_rate: float = 1e-3
    warmup_steps: int = 0
    end_learning_rate: float = 1e-5
    batch_size: int = 32
    checkpoint_interval: int = 1000
    log_interval: int = 100
    num_samples: int = 128
    seed: int = 0
    clip_grad_norm: float | None = None


def _type_allows_none(anno) -> bool:
    """Check if a type annotation includes None (e.g. ``str | None``)."""
    return isinstance(anno, types.UnionType) and type(None) in anno.__args__


def build_dataclass(cls, data: dict):
    valid = {f.name for f in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in valid:
            continue
        f = next(f for f in fields(cls) if f.name == key)
        anno = f.type
        if (value is None or value == 'None') and _type_allows_none(anno):
            kwargs[key] = None
        elif (
            isinstance(value, dict)
            and isinstance(anno, type)
            and hasattr(anno, '__dataclass_fields__')
        ):
            kwargs[key] = build_dataclass(anno, value)
        elif anno in (int, float, str, bool) and not isinstance(value, anno):
            kwargs[key] = anno(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(config_cls, path: str):
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    config = build_dataclass(config_cls, data)
    if hasattr(config, 'save_path') and not os.path.isabs(config.save_path):
        config.save_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(path)), config.save_path)
        )
    return config


def apply_overrides(config, overrides: list[str]):
    import ast
    import copy

    config = copy.deepcopy(config)
    for override in overrides:
        if '=' not in override:
            print(f'Warning: ignoring malformed override: {override}', file=sys.stderr)
            continue
        key_path, value_str = override.split('=', 1)
        parts = key_path.split('.')

        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)

        field_name = parts[-1]
        current = getattr(obj, field_name)

        if isinstance(current, bool):
            parsed = value_str.lower() in ('true', '1', 'yes')
        elif isinstance(current, int):
            parsed = int(value_str)
        elif isinstance(current, float):
            parsed = float(value_str)
        elif isinstance(current, list):
            try:
                parsed = ast.literal_eval(value_str)
                if not isinstance(parsed, list):
                    parsed = [parsed]
            except (ValueError, SyntaxError):
                parsed = [value_str]
        elif current is None:
            parsed = value_str if value_str.lower() != 'null' else None
        else:
            parsed = value_str

        setattr(obj, field_name, parsed)

    return config


def parse_args(config_cls, description: str = 'Train and evaluate model.'):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument(
        '--set', nargs='*', default=[], help='Override config values: key.path=value'
    )
    args = parser.parse_args()
    config = load_config(config_cls, args.config) if args.config is not None else config_cls()

    if args.set:
        config = apply_overrides(config, args.set)

    return config
