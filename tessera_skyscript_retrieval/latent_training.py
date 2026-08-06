from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import ensure_dir, uses_text_gated_coarse
from .data import load_prepared
from .features import load_latent_feature_arrays
from .latent import encode_latent_features, encode_text_gates, encode_text_latents
from .losses import latent_alignment_losses
from .metrics import (
    gated_coarse_topk,
    semantic_retrieval_metrics,
    semantic_topk_metrics,
)
from .model import build_latent_model
from .training import TitleBalancedBatchSampler, _cosine_schedule


class LatentAlignmentDataset(Dataset):
    def __init__(
        self,
        frame,
        descriptors,
        highres,
        highres_tokens,
        text,
        split: str,
    ) -> None:
        self.positions = frame.index[frame["split"].eq(split)].to_numpy(np.int64)
        self.frame = frame.iloc[self.positions].reset_index(drop=True)
        self.descriptors = descriptors
        self.highres = highres
        self.highres_tokens = highres_tokens
        self.text = text
        self.by_title: dict[int, list[int]] = defaultdict(list)
        for local_index, title_id in enumerate(self.frame["title_id"]):
            self.by_title[int(title_id)].append(local_index)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        position = int(self.positions[index])
        title_id = int(self.frame.iloc[index]["title_id"])
        return {
            "descriptor": np.asarray(self.descriptors[position], dtype=np.float32),
            "highres": np.asarray(self.highres[position], dtype=np.float32),
            "highres_tokens": np.asarray(
                self.highres_tokens[position], dtype=np.float32
            ),
            "text": np.asarray(self.text[title_id], dtype=np.float32),
            "title_id": title_id,
        }


def _collate_latent(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "descriptors": torch.from_numpy(
            np.stack([row["descriptor"] for row in batch])
        ),
        "highres": torch.from_numpy(np.stack([row["highres"] for row in batch])),
        "highres_tokens": torch.from_numpy(
            np.stack([row["highres_tokens"] for row in batch])
        ),
        "text": torch.from_numpy(np.stack([row["text"] for row in batch])),
        "title_ids": torch.tensor(
            [row["title_id"] for row in batch], dtype=torch.long
        ),
    }


@torch.inference_mode()
def _validation_metrics(
    model,
    dataset: LatentAlignmentDataset,
    device: torch.device,
    maximum: int,
    config: dict,
) -> dict[str, float]:
    model.eval()
    count = min(len(dataset), maximum)
    positions = dataset.positions[:count]
    descriptors = np.asarray(dataset.descriptors[positions], dtype=np.float32)
    teacher = np.asarray(dataset.highres[positions], dtype=np.float32)
    tokens = np.asarray(dataset.highres_tokens[positions], dtype=np.float32)
    tessera, highres_global, highres_latents = encode_latent_features(
        model, descriptors, teacher, tokens, device
    )
    labels = dataset.frame["title_id"].iloc[:count].to_numpy(np.int64)
    unique = np.unique(labels)
    text_global = np.asarray(dataset.text[unique], dtype=np.float32)
    text_latents = encode_text_latents(model, text_global, device)
    k_values = tuple(map(int, config["training"].get("monitor_k_values", (10, 100))))
    tessera_metrics = semantic_retrieval_metrics(
        text_global, unique, tessera, labels, k_values
    )
    highres_global_metrics = semantic_retrieval_metrics(
        text_global, unique, highres_global, labels, k_values
    )
    text_gates = encode_text_gates(model, text_global, device)
    if uses_text_gated_coarse(config):
        highres_indices = gated_coarse_topk(
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

        highres_indices = late_interaction_prefilter_topk(
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
    highres_rank_metrics = semantic_topk_metrics(
        highres_indices, unique, labels, k_values
    )
    primary_keys = [
        f"{metric}@{k}" for k in k_values for metric in ("Precision", "nDCG")
    ]
    primary_score = float(
        np.mean(
            [tessera_metrics[key] for key in primary_keys]
            + [highres_rank_metrics[key] for key in primary_keys]
        )
    )
    result = {"primary_score": primary_score}
    for prefix, values in (
        ("tessera", tessera_metrics),
        ("highres_global", highres_global_metrics),
        ("highres_gated" if uses_text_gated_coarse(config) else "highres_fine", highres_rank_metrics),
    ):
        for key in primary_keys:
            result[f"{prefix}_{key}"] = float(values[key])
    return result


def train_latent_model(config: dict, limit: int | None = None) -> Path:
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    frame = load_prepared(config)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    descriptors, highres, highres_tokens, text = load_latent_feature_arrays(
        config, limit
    )
    train_set = LatentAlignmentDataset(
        frame, descriptors, highres, highres_tokens, text, "train"
    )
    val_set = LatentAlignmentDataset(
        frame, descriptors, highres, highres_tokens, text, "val"
    )
    if not len(train_set) or not len(val_set):
        raise ValueError(
            f"training and validation splits must be non-empty: "
            f"{len(train_set)}, {len(val_set)}"
        )
    cfg = config["training"]
    sampler = TitleBalancedBatchSampler(
        train_set,
        int(cfg["titles_per_batch"]),
        int(cfg["samples_per_title"]),
        seed,
    )
    loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        collate_fn=_collate_latent,
        num_workers=int(cfg.get("workers", 4)),
        pin_memory=True,
        persistent_workers=int(cfg.get("workers", 4)) > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_latent_model(config).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.tessera_adapter.parameters(),
                "lr": float(cfg["learning_rate"]),
            },
            {
                "params": model.highres_adapter.parameters(),
                "lr": float(cfg["highres_learning_rate"]),
            },
            {
                "params": [model.logit_scale, model.fine_logit_scale],
                "lr": float(cfg["temperature_learning_rate"]),
                "weight_decay": 0.0,
            },
        ],
        weight_decay=float(cfg["weight_decay"]),
    )
    total_steps = len(loader) * int(cfg["epochs"])
    scheduler = _cosine_schedule(
        optimizer, int(total_steps * float(cfg["warmup_ratio"])), total_steps
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(cfg["mixed_precision"]) and device.type == "cuda"
    )
    output_dir = ensure_dir(cfg["output_dir"])
    best_path = output_dir / (
        f"best_{limit}.pt" if limit is not None else "best.pt"
    )
    best_metric, stale = -1.0, 0
    history = []
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    architecture = config["model"].get("architecture", "latent_v2")
    print(f"{architecture} trainable parameters: {trainable_parameters:,}")
    for epoch in range(int(cfg["epochs"])):
        model.train()
        sampler.set_epoch(epoch)
        totals = defaultdict(float)
        for batch in tqdm(loader, desc=f"train-{architecture}-{epoch + 1}"):
            descriptors_batch = batch["descriptors"].to(device, non_blocking=True)
            teacher_batch = torch.nn.functional.normalize(
                batch["highres"].to(device, non_blocking=True), dim=-1
            )
            token_batch = batch["highres_tokens"].to(device, non_blocking=True)
            text_batch = torch.nn.functional.normalize(
                batch["text"].to(device, non_blocking=True), dim=-1
            )
            title_ids = batch["title_ids"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=scaler.is_enabled(),
            ):
                tessera_batch = model.encode_tessera(descriptors_batch)
                highres_batch, latent_batch = model.encode_highres(
                    token_batch, teacher_batch
                )
                text_latent_batch = model.encode_text_latent(text_batch)
                text_gate_batch = model.encode_text_gate(text_batch)
                losses = latent_alignment_losses(
                    tessera_batch,
                    highres_batch,
                    latent_batch,
                    teacher_batch,
                    text_batch,
                    text_latent_batch,
                    text_gate_batch,
                    title_ids,
                    model.logit_scale,
                    model.fine_logit_scale,
                    cfg,
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["max_grad_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.clamp_temperature()
            for key, value in losses.items():
                totals[key] += float(value.detach())
        validation = _validation_metrics(
            model,
            val_set,
            device,
            int(cfg["max_validation_candidates"]),
            config,
        )
        metric = validation["primary_score"]
        record = {
            "epoch": epoch + 1,
            **validation,
            **{key: value / len(loader) for key, value in totals.items()},
        }
        history.append(record)
        print(record)
        checkpoint = {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch + 1,
            "metric": metric,
            "trainable_parameters": trainable_parameters,
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if metric > best_metric:
            best_metric, stale = metric, 0
            torch.save(checkpoint, best_path)
        else:
            stale += 1
            if stale >= int(cfg["early_stopping_patience"]):
                break
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"best validation primary score={best_metric:.6f}: {best_path}")
    return best_path
