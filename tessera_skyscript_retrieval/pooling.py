from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

from .config import ensure_dir
from .data import load_prepared


def adaptive_average_pool(chip: np.ndarray, level: int) -> np.ndarray:
    height, width, channels = chip.shape
    pooled = np.empty((level, level, channels), dtype=np.float32)
    values = chip.astype(np.float32, copy=False)
    for row in range(level):
        row_start = math.floor(row * height / level)
        row_end = math.ceil((row + 1) * height / level)
        for col in range(level):
            col_start = math.floor(col * width / level)
            col_end = math.ceil((col + 1) * width / level)
            pooled[row, col] = values[row_start:row_end, col_start:col_end].mean(axis=(0, 1))
    return pooled


def tessera_descriptor(chip: np.ndarray, levels: tuple[int, ...], include_std: bool) -> np.ndarray:
    if chip.ndim != 3 or chip.shape[-1] != 128:
        raise ValueError(f"expected HxWx128 chip, got {chip.shape}")
    values = [adaptive_average_pool(chip, level).reshape(-1) for level in levels]
    if include_std:
        values.append(chip.astype(np.float32, copy=False).std(axis=(0, 1)))
    descriptor = np.concatenate(values).astype(np.float32)
    if not np.isfinite(descriptor).all():
        raise ValueError("descriptor contains non-finite values")
    return descriptor


def _pool_path(args: tuple[str, tuple[int, ...], bool]) -> np.ndarray:
    path, levels, include_std = args
    chip = np.load(path, allow_pickle=False)
    return tessera_descriptor(chip, levels, include_std)


def cache_tessera(config: dict[str, Any], limit: int | None = None) -> Path:
    frame = load_prepared(config)
    if limit is not None:
        frame = frame.iloc[:limit]
    cfg = config["tessera"]
    levels = tuple(map(int, cfg["pyramid_levels"]))
    include_std = bool(cfg["include_std"])
    expected_dim = (sum(level * level for level in levels) + int(include_std)) * int(cfg["embedding_dim"])
    if expected_dim != int(cfg["descriptor_dim"]):
        raise ValueError(f"descriptor_dim={cfg['descriptor_dim']} but pooling produces {expected_dim}")
    output_dir = ensure_dir(cfg["cache_dir"])
    suffix = f"_{limit}" if limit is not None else ""
    output = output_dir / f"descriptors{suffix}.npy"
    features = open_memmap(output, mode="w+", dtype=np.float16, shape=(len(frame), expected_dim))
    workers = max(1, int(cfg.get("workers", 8)))
    tasks = ((path, levels, include_std) for path in frame["chip_path"])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, descriptor in enumerate(tqdm(executor.map(_pool_path, tasks, chunksize=32), total=len(frame), desc="cache-tessera")):
            features[index] = descriptor.astype(np.float16)
            if index and index % 10000 == 0:
                features.flush()
    features.flush()
    np.save(output_dir / f"row_ids{suffix}.npy", frame["row_id"].to_numpy(np.int64))
    print(f"cached {len(frame)} TESSERA descriptors at {output} with shape {features.shape}")
    return output
