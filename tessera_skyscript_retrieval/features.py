from __future__ import annotations

from pathlib import Path

import numpy as np


def suffix_for_limit(limit: int | None) -> str:
    return f"_{limit}" if limit is not None else ""


def feature_paths(config: dict, limit: int | None = None) -> dict[str, Path]:
    suffix = suffix_for_limit(limit)
    tessera_dir = Path(config["tessera"]["cache_dir"])
    skyclip_dir = Path(config["skyclip"]["cache_dir"])
    return {
        "descriptors": tessera_dir / f"descriptors{suffix}.npy",
        "descriptor_rows": tessera_dir / f"row_ids{suffix}.npy",
        "highres": skyclip_dir / f"highres_features{suffix}.npy",
        "highres_tokens": skyclip_dir / f"highres_tokens{suffix}.npy",
        "text": skyclip_dir / f"text_features{suffix}.npy",
        "titles": skyclip_dir / f"titles{suffix}.parquet",
    }


def load_feature_arrays(config: dict, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = feature_paths(config, limit)
    required = (paths["descriptors"], paths["descriptor_rows"], paths["highres"], paths["text"], paths["titles"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing feature cache(s): " + ", ".join(missing))
    descriptors = np.load(paths["descriptors"], mmap_mode="r")
    highres = np.load(paths["highres"], mmap_mode="r")
    text = np.load(paths["text"], mmap_mode="r")
    if len(descriptors) != len(highres):
        raise ValueError(f"cache row mismatch: descriptors={len(descriptors)}, highres={len(highres)}")
    return descriptors, highres, text


def load_latent_feature_arrays(
    config: dict,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = feature_paths(config, limit)
    descriptors, highres, text = load_feature_arrays(config, limit)
    if not paths["highres_tokens"].is_file():
        raise FileNotFoundError(f"missing high-resolution token cache: {paths['highres_tokens']}")
    highres_tokens = np.load(paths["highres_tokens"], mmap_mode="r")
    if len(highres_tokens) != len(highres):
        raise ValueError(
            f"cache row mismatch: highres_tokens={len(highres_tokens)}, highres={len(highres)}"
        )
    expected = (
        int(config["model"]["latent_count"]),
        int(config["model"]["latent_dim"]),
    )
    if highres_tokens.shape[1:] != expected:
        raise ValueError(f"expected highres token shape Nx{expected}, got {highres_tokens.shape}")
    return descriptors, highres, highres_tokens, text
