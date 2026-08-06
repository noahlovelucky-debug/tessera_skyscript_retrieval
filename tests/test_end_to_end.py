from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tessera_skyscript_retrieval.evaluation import evaluate
from tessera_skyscript_retrieval.indexing import (
    build_index,
    retrieve_by_modality,
    search,
)
from tessera_skyscript_retrieval.training import train_adapter


def test_train_evaluate_and_build_index(tmp_path: Path):
    rng = np.random.default_rng(7)
    rows = 48
    title_ids = np.arange(rows) % 4
    splits = np.array(["train"] * 32 + ["val"] * 8 + ["test"] * 8)
    frame = pd.DataFrame({
        "row_id": np.arange(rows), "sample_id": [f"sample_{i}" for i in range(rows)],
        "title": [f"title {value}" for value in title_ids], "title_id": title_ids,
        "split": splits, "source": "CH", "year": 2021,
        "image_path": [f"image_{i}.jpg" for i in range(rows)],
        "chip_path": [f"chip_{i}.npy" for i in range(rows)],
        "bbox_west": 0.0, "bbox_south": 0.0, "bbox_east": 0.1, "bbox_north": 0.1,
        "center_lon": 0.05, "center_lat": 0.05,
    })
    manifest = tmp_path / "prepared.parquet"; frame.to_parquet(manifest, index=False)
    tessera_dir = tmp_path / "tessera"; sky_dir = tmp_path / "sky"; run_dir = tmp_path / "run"
    tessera_dir.mkdir(); sky_dir.mkdir()
    descriptors = rng.normal(size=(rows, 32)).astype(np.float16)
    text = rng.normal(size=(4, 16)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    highres = (text[title_ids] + 0.02 * rng.normal(size=(rows, 16))).astype(np.float32)
    highres /= np.linalg.norm(highres, axis=1, keepdims=True)
    np.save(tessera_dir / "descriptors.npy", descriptors)
    np.save(tessera_dir / "row_ids.npy", np.arange(rows))
    np.save(sky_dir / "highres_features.npy", highres.astype(np.float16))
    np.save(sky_dir / "text_features.npy", text.astype(np.float16))
    pd.DataFrame({"title_id": np.arange(4), "title": [f"title {i}" for i in range(4)]}).to_parquet(sky_dir / "titles.parquet")
    config = {
        "data": {"prepared_manifest": str(manifest)},
        "tessera": {"descriptor_dim": 32, "cache_dir": str(tessera_dir)},
        "skyclip": {"cache_dir": str(sky_dir)},
        "model": {"hidden_dim": 32, "common_dim": 16, "dropout": 0.0, "temperature": 0.07},
        "training": {
            "output_dir": str(run_dir), "epochs": 1, "titles_per_batch": 2,
            "samples_per_title": 2, "learning_rate": 1e-3, "temperature_learning_rate": 1e-3,
            "weight_decay": 0.0, "warmup_ratio": 0.0, "max_grad_norm": 1.0,
            "mixed_precision": False, "early_stopping_patience": 2,
            "max_validation_candidates": 8, "seed": 42, "workers": 0,
            "semantic_weight": 1.0, "pair_distill_weight": 0.5, "relation_distill_weight": 0.25,
        },
        "evaluation": {
            "checkpoint": str(run_dir / "best.pt"), "k_values": [1, 5],
            "max_exact_pairs": 8, "report": str(run_dir / "evaluation.json"),
            "candidate_chunk_size": 32,
        },
        "index": {"output_dir": str(tmp_path / "index")},
    }
    checkpoint = train_adapter(config)
    assert checkpoint.is_file()
    report = evaluate(config)
    assert report["splits"]["test"]["text_to_highres"]["Hit@1"] == 1.0
    index_dir = build_index(config)
    assert (index_dir / "metadata.parquet").is_file()
    assert np.load(index_dir / "tessera_features.npy").shape == (rows, 16)


def test_retrieve_by_modality_and_combined_search(tmp_path: Path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "title": ["alpha", "beta", "gamma"],
            "source": ["CH", "US", "FI"],
            "year": [2020, 2021, 2022],
            "image_path": ["a.jpg", "b.jpg", "c.jpg"],
            "chip_path": ["a.npy", "b.npy", "c.npy"],
            "bbox_west": [0.0, 1.0, 2.0],
            "bbox_south": [0.0, 1.0, 2.0],
            "bbox_east": [0.1, 1.1, 2.1],
            "bbox_north": [0.1, 1.1, 2.1],
            "center_lon": [0.05, 1.05, 2.05],
            "center_lat": [0.05, 1.05, 2.05],
        }
    )
    frame.to_parquet(index_dir / "metadata.parquet", index=False)
    highres = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float16)
    tessera = np.array([[0.9, 0.1], [1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    np.save(index_dir / "highres_features.npy", highres)
    np.save(index_dir / "tessera_features.npy", tessera)
    (index_dir / "index.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "tessera_skyscript_retrieval.indexing.encode_query",
        lambda *_args, **_kwargs: np.array([1.0, 0.0], dtype=np.float32),
    )
    config = {
        "model": {},
        "evaluation": {"candidate_chunk_size": 2},
        "index": {"output_dir": str(index_dir), "prompt_templates": ["{query}"]},
    }

    grouped = retrieve_by_modality(config, "alpha", ("highres", "tessera"), 2)
    assert [row["sample_id"] for row in grouped["highres"]] == ["a", "b"]
    assert [row["sample_id"] for row in grouped["tessera"]] == ["b", "a"]
    combined = search(config, "alpha", "both", 2)
    assert len(combined["results"]) == 2
    assert combined["results"][0]["sample_id"] in {"a", "b"}
    assert combined["results"][0]["rank"] == 1
