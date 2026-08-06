from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

LATENT_ARCHITECTURES = frozenset({"latent_v2", "gated_coarse_v3", "anchored_gated_v4"})


def uses_latent_tokens(config: dict[str, Any]) -> bool:
    """Whether this configuration requires cached SkyCLIP region tokens."""
    return config.get("model", {}).get("architecture") in LATENT_ARCHITECTURES


def uses_text_gated_coarse(config: dict[str, Any]) -> bool:
    return config.get("model", {}).get("architecture") in {"gated_coarse_v3", "anchored_gated_v4"}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def ensure_parent(path: str | Path) -> Path:
    value = Path(path)
    value.parent.mkdir(parents=True, exist_ok=True)
    return value


def ensure_dir(path: str | Path) -> Path:
    value = Path(path)
    value.mkdir(parents=True, exist_ok=True)
    return value
