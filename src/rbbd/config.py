"""Load and merge YAML configs and compute the canonical config hash.

A tier config (e.g. `configs/smoke.yaml`) is deep-merged over `configs/base.yaml`
from the same directory: mappings merge key by key, every other value (lists
included) is replaced by the tier value. The config hash is `make_key` over the
merged mapping minus presentation-only sections (`paths`, `logging`), so moving
the artifact root or changing log verbosity never invalidates caches, while any
experimental setting does (ARCHITECTURE §4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rbbd.utils.cache import make_key

BASE_NAME = "base.yaml"
PRESENTATION_KEYS = frozenset({"paths", "logging"})
REQUIRED_KEYS = ("run_name", "seed")


class ConfigError(ValueError):
    """Raised for unreadable or incomplete configs."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict: `override` merged into `base` (mappings recurse, others replace)."""
    merged: dict[str, Any] = {k: v for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_hash(data: Mapping[str, Any]) -> str:
    """16-hex hash of the experimental content of a merged config."""
    return make_key({k: v for k, v in data.items() if k not in PRESENTATION_KEYS})


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"config {path} must be a mapping at top level")
    return loaded


@dataclass(frozen=True)
class Config:
    """A merged, hashed configuration. `data` must be treated as read-only."""

    data: dict[str, Any]
    path: Path
    hash: str
    overrides: tuple[str, ...] = ()

    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up `a.b.c` in the nested mapping, returning `default` if any level is missing."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


def apply_override(data: dict[str, Any], assignment: str) -> None:
    """Apply one `a.b.c=value` override in place; `value` is parsed as YAML.

    Intermediate mappings are created when missing; a non-mapping on the path is an
    error. Used for decisions taken at run time (e.g. epochs chosen by a probe rule,
    D-076), so the effective config, and therefore its hash, records them.
    """
    key, sep, raw = assignment.partition("=")
    if not sep or not key.strip():
        raise ConfigError(f"override must look like a.b.c=value, got {assignment!r}")
    parts = key.strip().split(".")
    node: Any = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {assignment!r}: {part!r} is not a mapping")
    node[parts[-1]] = yaml.safe_load(raw)


def load(path: str | Path, overrides: Sequence[str] = ()) -> Config:
    """Load `path`, merging it over `base.yaml` from the same directory when it is not the base,
    then apply `overrides` (`a.b.c=value`) before hashing."""
    path = Path(path)
    tier = _read_yaml(path)
    if path.name == BASE_NAME:
        data = tier
    else:
        base_path = path.parent / BASE_NAME
        if not base_path.exists():
            raise ConfigError(f"{path} needs {base_path} to merge over")
        data = deep_merge(_read_yaml(base_path), tier)
    for assignment in overrides:
        apply_override(data, assignment)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ConfigError(f"config {path} is missing required keys: {missing}")
    return Config(data=data, path=path, hash=config_hash(data), overrides=tuple(overrides))
