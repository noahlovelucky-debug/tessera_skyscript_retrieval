from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import ensure_parent
from .data import load_prepared
from .features import load_feature_arrays
from .metrics import paired_recall, semantic_retrieval_metrics
from .model import build_adapter


@torch.inference_mode()
def encode_tessera(model, descriptors: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    output = np.empty((len(descriptors), model.projection[-1].out_features), dtype=np.float32)
    model.eval()
    for start in tqdm(range(0, len(descriptors), batch_size), desc="encode-tessera"):
        batch = torch.from_numpy(np.asarray(descriptors[start:start + batch_size], dtype=np.float32)).to(device)
        output[start:start + len(batch)] = model(batch).cpu().numpy()
    return output


def _semantic_block(text, highres, tessera, labels, k_values):
    unique = np.unique(labels)
    queries = np.asarray(text[unique], dtype=np.float32)
    high_values = semantic_retrieval_metrics(queries, unique, highres, labels, k_values)
    tessera_values = semantic_retrieval_metrics(queries, unique, tessera, labels, k_values)
    high_centroids = np.stack([highres[labels == label].mean(axis=0) for label in unique])
    cross_values = semantic_retrieval_metrics(high_centroids, unique, tessera, labels, k_values)
    return {"text_to_highres": high_values, "text_to_tessera": tessera_values, "highres_to_tessera_semantic": cross_values}


def evaluate(config: dict, limit: int | None = None, checkpoint_path: str | None = None) -> dict:
    if config.get("model", {}).get("architecture") in {"latent_v2", "gated_coarse_v3", "anchored_gated_v4"}:
        from .latent_evaluation import evaluate_latent

        return evaluate_latent(config, limit, checkpoint_path)
    frame = load_prepared(config)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    descriptors, highres, text = load_feature_arrays(config, limit)
    checkpoint_value = Path(checkpoint_path or config["evaluation"]["checkpoint"])
    if limit is not None and checkpoint_path is None:
        checkpoint_value = Path(config["training"]["output_dir"]) / f"best_{limit}.pt"
    checkpoint = torch.load(checkpoint_value, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_adapter(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    tessera = encode_tessera(model, descriptors, device)
    highres = F.normalize(torch.from_numpy(np.asarray(highres, dtype=np.float32)), dim=-1).numpy()
    k_values = tuple(map(int, config["evaluation"]["k_values"]))
    report = {"checkpoint": str(checkpoint_value), "splits": {}}
    for split in ("test", "oov_test"):
        positions = frame.index[frame["split"].eq(split)].to_numpy(np.int64)
        if not len(positions):
            continue
        labels = frame["title_id"].iloc[positions].to_numpy(np.int64)
        report["splits"][split] = _semantic_block(
            text, highres[positions], tessera[positions], labels, k_values
        )
    exact_positions = frame.index[frame["split"].isin(["test", "oov_test"])].to_numpy(np.int64)
    maximum = int(config["evaluation"]["max_exact_pairs"])
    exact_positions = exact_positions[:maximum]
    if len(exact_positions):
        report["paired_highres_to_tessera"] = paired_recall(
            highres[exact_positions], tessera[exact_positions], k_values
        )
    report_path = ensure_parent(config["evaluation"]["report"])
    if limit is not None:
        report_path = report_path.with_name(f"{report_path.stem}_{limit}{report_path.suffix}")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
