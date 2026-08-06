from __future__ import annotations

import hashlib
import math
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import ensure_parent


def stable_bucket(value: str, seed: int, modulo: int = 100) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def normalize_title(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _parse_sample(row: dict[str, Any], metadata_root: str) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    filepath = str(row["filepath"])
    image_folder = filepath.split("/", 1)[0]
    if not image_folder.startswith("images"):
        raise ValueError(f"unexpected filepath: {filepath}")
    part = int(image_folder.removeprefix("images"))
    pieces = sample_id.rsplit("_", 2)
    if len(pieces) != 3:
        raise ValueError(f"unexpected sample id: {sample_id}")
    source, year_suffix = pieces[-2], pieces[-1]
    year = int(year_suffix) + (2000 if int(year_suffix) < 100 else 0)
    meta_path = Path(metadata_root) / f"meta{part}" / f"{sample_id}.pickle"
    with meta_path.open("rb") as stream:
        metadata = pickle.load(stream)
    bbox = metadata.get("bbox", metadata.get("box"))
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"missing bbox: {meta_path}")
    west, south, east, north = map(float, bbox)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(f"invalid bbox: {meta_path}: {bbox}")
    center_lon = (west + east) / 2.0
    center_lat = (south + north) / 2.0
    split_group = f"{math.floor(center_lon * 10)}_{math.floor(center_lat * 10)}"
    return {
        "sample_id": sample_id,
        "part": part,
        "source": source,
        "year": year,
        "metadata_path": str(meta_path),
        "bbox_west": west,
        "bbox_south": south,
        "bbox_east": east,
        "bbox_north": north,
        "center_lon": center_lon,
        "center_lat": center_lat,
        "split_group": split_group,
    }


def _inspect_record(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    row, paths = args
    parsed = _parse_sample(row, paths["metadata_root"])
    image_path = Path(paths["image_root"]) / str(row["filepath"])
    chip_path = Path(paths["chip_root"]) / f"{row['sample_id']}.npy"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not chip_path.is_file():
        raise FileNotFoundError(chip_path)
    chip = np.load(chip_path, mmap_mode="r", allow_pickle=False)
    if chip.ndim != 3 or chip.shape[-1] != 128:
        raise ValueError(f"unexpected TESSERA chip shape {chip.shape}: {chip_path}")
    parsed.update({
        "image_path": str(image_path),
        "chip_path": str(chip_path),
        "title": normalize_title(row["title"]),
        "chip_h": int(chip.shape[0]),
        "chip_w": int(chip.shape[1]),
    })
    return parsed


def _inspect_record_safe(args: tuple[dict[str, Any], dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return _inspect_record(args), None
    except (OSError, ValueError, TypeError, KeyError, IndexError, EOFError, pickle.UnpicklingError) as error:
        return None, str(error)


def prepare_manifest(config: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    cfg = config["data"]
    source = pd.read_csv(cfg["source_manifest"], usecols=["sample_id", "filepath", "title"])
    if source["sample_id"].duplicated().any():
        raise ValueError("source manifest contains duplicate sample_id values")
    allowed = set(cfg["allowed_sources"])
    source_code = source["sample_id"].str.rsplit("_", n=2).str[-2]
    source = source.loc[source_code.isin(allowed)].reset_index(drop=True)
    if limit is not None:
        source = source.iloc[:limit].copy()
    paths = {
        "image_root": str(cfg["image_root"]),
        "chip_root": str(cfg["chip_root"]),
        "metadata_root": str(cfg["metadata_root"]),
    }
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    workers = min(32, max(1, int(config["tessera"].get("workers", 8))))
    rows = source.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = executor.map(_inspect_record_safe, ((row, paths) for row in rows), chunksize=64)
        for index, (record, error) in enumerate(tqdm(futures, total=len(rows), desc="prepare")):
            if error is not None:
                failures.append(f"row={index}: {error}")
            else:
                records.append(record)
    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(f"failed to prepare {len(failures)} rows:\n{preview}")
    frame = pd.DataFrame.from_records(records)
    frame["title_id"], title_values = pd.factorize(frame["title"], sort=True)
    frame["title_id"] = frame["title_id"].astype(np.int32)
    seed = int(cfg["split_seed"])
    train_cut = int(cfg["train_percent"])
    val_cut = train_cut + int(cfg["val_percent"])
    group_buckets = frame["split_group"].map(lambda value: stable_bucket(value, seed))
    base_split = np.where(group_buckets < train_cut, "train", np.where(group_buckets < val_cut, "val", "test"))
    frame["split"] = base_split

    counts = frame["title"].value_counts()
    eligible = counts[counts >= int(cfg["oov_min_samples"])].index
    oov_titles = {
        title for title in eligible
        if stable_bucket(f"oov:{title}", seed) < int(cfg["oov_percent"])
    }
    oov_mask = frame["title"].isin(oov_titles)
    frame.loc[oov_mask & frame["split"].eq("test"), "split"] = "oov_test"
    frame.loc[oov_mask & ~frame["split"].eq("oov_test"), "split"] = "excluded_oov"
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))

    output = ensure_parent(cfg["prepared_manifest"])
    frame.to_parquet(output, index=False)
    audit = {
        "rows": len(frame),
        "titles": len(title_values),
        "sources": frame["source"].value_counts().sort_index().to_dict(),
        "splits": frame["split"].value_counts().sort_index().to_dict(),
        "oov_titles": len(oov_titles),
    }
    pd.Series(audit).to_json(output.with_suffix(".summary.json"), indent=2, force_ascii=False)
    print(audit)
    return frame


def load_prepared(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["data"]["prepared_manifest"])
    if not path.is_file():
        raise FileNotFoundError(f"prepared manifest not found; run prepare first: {path}")
    frame = pd.read_parquet(path)
    expected = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["row_id"].to_numpy(), expected):
        raise ValueError("prepared manifest row_id must be contiguous and ordered")
    return frame
