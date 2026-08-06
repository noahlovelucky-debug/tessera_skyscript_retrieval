from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import ensure_parent, uses_text_gated_coarse
from .data import load_prepared
from .features import load_latent_feature_arrays
from .latent import encode_latent_features, encode_text_gates, encode_text_latents
from .metrics import (
    gated_coarse_topk,
    paired_recall,
    semantic_retrieval_metrics,
    semantic_topk_metrics,
)
from .model import build_latent_model


def _latent_semantic_block(
    model,
    text,
    teacher_highres,
    highres_global,
    highres_latents,
    tessera,
    labels,
    k_values,
    config,
    device,
):
    unique = np.unique(labels)
    text_global = np.asarray(text[unique], dtype=np.float32)
    text_latents = encode_text_latents(model, text_global, device)
    text_gates = encode_text_gates(model, text_global, device)
    if uses_text_gated_coarse(config):
        ranked_indices = gated_coarse_topk(
            text_global,
            text_latents,
            text_gates,
            highres_global,
            highres_latents,
            max(k_values),
            device,
            query_batch_size=int(config["evaluation"].get("gated_query_batch_size", 16)),
            candidate_chunk_size=int(config["evaluation"].get("gated_candidate_chunk_size", 8192)),
        )
    else:
        from .metrics import late_interaction_prefilter_topk

        ranked_indices = late_interaction_prefilter_topk(
            text_latents,
            highres_latents,
            text_global,
            highres_global,
            max(k_values),
            int(config["index"].get("fine_prefilter", 1000)),
            float(config["evaluation"]["fine_weight"]),
            device,
            query_batch_size=int(
                config["evaluation"].get("fine_prefilter_query_batch_size", 8)
            ),
        )
    highres_centroids = np.stack(
        [highres_global[labels == label].mean(axis=0) for label in unique]
    )
    return {
        "text_to_highres_teacher": semantic_retrieval_metrics(
            text_global, unique, teacher_highres, labels, k_values
        ),
        "text_to_highres": semantic_retrieval_metrics(
            text_global, unique, highres_global, labels, k_values
        ),
        "text_to_highres_gated" if uses_text_gated_coarse(config) else "text_to_highres_fine": semantic_topk_metrics(
            ranked_indices, unique, labels, k_values
        ),
        "text_to_tessera": semantic_retrieval_metrics(
            text_global, unique, tessera, labels, k_values
        ),
        "highres_to_tessera_semantic": semantic_retrieval_metrics(
            highres_centroids, unique, tessera, labels, k_values
        ),
    }


def evaluate_latent(
    config: dict,
    limit: int | None = None,
    checkpoint_path: str | None = None,
) -> dict:
    frame = load_prepared(config)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    descriptors, teacher_highres, region_tokens, text = load_latent_feature_arrays(
        config, limit
    )
    checkpoint_value = Path(checkpoint_path or config["evaluation"]["checkpoint"])
    if limit is not None and checkpoint_path is None:
        checkpoint_value = Path(config["training"]["output_dir"]) / f"best_{limit}.pt"
    checkpoint = torch.load(checkpoint_value, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_latent_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    k_values = tuple(map(int, config["evaluation"]["k_values"]))
    report = {
        "checkpoint": str(checkpoint_value),
        "architecture": config["model"].get("architecture", "latent_v2"),
        "trainable_parameters": int(
            checkpoint.get(
                "trainable_parameters",
                sum(parameter.numel() for parameter in model.parameters()),
            )
        ),
        "representations": {
            "highres_latents": [
                int(config["model"]["latent_count"]),
                int(config["model"]["latent_dim"]),
            ],
            "global": int(config["model"]["common_dim"]),
        },
        "retrieval_pipeline": (
            {
                "full_token_scan": True,
                "global_weight": "text_mlp_gate",
                "local_weight": "1 - text_mlp_gate",
                "local_score": "max similarity over 8 tokens per candidate",
            }
            if uses_text_gated_coarse(config)
            else {
                "highres_global_prefilter": int(config["index"].get("fine_prefilter", 1000)),
                "fine_weight": float(config["evaluation"]["fine_weight"]),
                "global_weight": 1.0 - float(config["evaluation"]["fine_weight"]),
            }
        ),
        "splits": {},
    }
    encoded_by_split = {}
    for split in ("test", "oov_test"):
        positions = frame.index[frame["split"].eq(split)].to_numpy(np.int64)
        if not len(positions):
            continue
        teacher = F.normalize(
            torch.from_numpy(
                np.asarray(teacher_highres[positions], dtype=np.float32)
            ),
            dim=-1,
        ).numpy()
        tessera, highres_global, highres_latents = encode_latent_features(
            model,
            np.asarray(descriptors[positions]),
            teacher,
            np.asarray(region_tokens[positions]),
            device,
        )
        labels = frame["title_id"].iloc[positions].to_numpy(np.int64)
        report["splits"][split] = _latent_semantic_block(
            model,
            text,
            teacher,
            highres_global,
            highres_latents,
            tessera,
            labels,
            k_values,
            config,
            device,
        )
        encoded_by_split[split] = (highres_global, tessera)
    exact_highres = []
    exact_tessera = []
    maximum = int(config["evaluation"]["max_exact_pairs"])
    for split in ("test", "oov_test"):
        if split not in encoded_by_split:
            continue
        highres_values, tessera_values = encoded_by_split[split]
        remaining = maximum - sum(len(values) for values in exact_highres)
        if remaining <= 0:
            break
        exact_highres.append(highres_values[:remaining])
        exact_tessera.append(tessera_values[:remaining])
    if exact_highres:
        report["paired_highres_to_tessera"] = paired_recall(
            np.concatenate(exact_highres), np.concatenate(exact_tessera), k_values
        )
    report_path = ensure_parent(config["evaluation"]["report"])
    if limit is not None:
        report_path = report_path.with_name(
            f"{report_path.stem}_{limit}{report_path.suffix}"
        )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
