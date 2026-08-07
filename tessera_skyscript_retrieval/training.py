from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from .config import ensure_dir
from .data import load_prepared
from .features import load_feature_arrays
from .losses import alignment_losses
from .metrics import semantic_retrieval_metrics
from .model import build_adapter


class AlignmentDataset(Dataset):
    def __init__(self, frame, descriptors, highres, text, split: str) -> None:
        selected = frame.index[frame["split"].eq(split)].to_numpy(np.int64)
        self.positions = selected
        self.frame = frame.iloc[selected].reset_index(drop=True)
        self.descriptors = descriptors
        self.highres = highres
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
            "text": np.asarray(self.text[title_id], dtype=np.float32),
            "title_id": title_id,
        }


class TitleBalancedBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: AlignmentDataset, titles_per_batch: int, samples_per_title: int, seed: int):
        self.dataset = dataset
        self.title_ids = sorted(dataset.by_title)
        self.titles_per_batch = min(titles_per_batch, len(self.title_ids))
        self.samples_per_title = samples_per_title
        self.batch_size = self.titles_per_batch * samples_per_title
        self.num_batches = math.ceil(len(dataset) / self.batch_size)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.num_batches):
            chosen = rng.sample(self.title_ids, self.titles_per_batch)
            batch: list[int] = []
            for title_id in chosen:
                candidates = self.dataset.by_title[title_id]
                if len(candidates) >= self.samples_per_title:
                    batch.extend(rng.sample(candidates, self.samples_per_title))
                else:
                    batch.extend(rng.choices(candidates, k=self.samples_per_title))
            rng.shuffle(batch)
            yield batch


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "descriptors": torch.from_numpy(np.stack([row["descriptor"] for row in batch])),
        "highres": torch.from_numpy(np.stack([row["highres"] for row in batch])),
        "text": torch.from_numpy(np.stack([row["text"] for row in batch])),
        "title_ids": torch.tensor([row["title_id"] for row in batch], dtype=torch.long),
    }


def _cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.inference_mode()
def _validation_metric(model, dataset: AlignmentDataset, device: torch.device, maximum: int) -> float:
    model.eval()
    count = min(len(dataset), maximum)
    indices = np.arange(count)
    descriptors = torch.from_numpy(np.asarray(dataset.descriptors[dataset.positions[indices]], dtype=np.float32)).to(device)
    features = []
    for start in range(0, count, 2048):
        features.append(model(descriptors[start:start + 2048]).cpu().numpy())
    candidates = np.concatenate(features)
    labels = dataset.frame["title_id"].iloc[:count].to_numpy()
    unique = np.unique(labels)
    queries = np.asarray(dataset.text[unique], dtype=np.float32)
    values = semantic_retrieval_metrics(queries, unique, candidates, labels, (1, 5, 10))
    return float(values["macro_mAP"])


def train_adapter(config: dict, limit: int | None = None) -> Path:
    if config.get("model", {}).get("architecture") == "tessera_box_mlp":
        from .tessera_box import train_tessera_box

        return train_tessera_box(config, limit)
    if config.get("model", {}).get("architecture") in {"latent_v2", "gated_coarse_v3", "anchored_gated_v4", "tri_modal_fusion_v1"}:
        from .latent_training import train_latent_model

        return train_latent_model(config, limit)
    seed = int(config["training"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    frame = load_prepared(config)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    descriptors, highres, text = load_feature_arrays(config, limit)
    train_set = AlignmentDataset(frame, descriptors, highres, text, "train")
    val_set = AlignmentDataset(frame, descriptors, highres, text, "val")
    if not len(train_set) or not len(val_set):
        raise ValueError(f"training and validation splits must be non-empty: {len(train_set)}, {len(val_set)}")
    cfg = config["training"]
    sampler = TitleBalancedBatchSampler(
        train_set, int(cfg["titles_per_batch"]), int(cfg["samples_per_title"]), seed
    )
    loader = DataLoader(
        train_set, batch_sampler=sampler, collate_fn=_collate,
        num_workers=int(cfg.get("workers", 4)), pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_adapter(config).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.projection.parameters(), "lr": float(cfg["learning_rate"])},
        {"params": [model.logit_scale], "lr": float(cfg["temperature_learning_rate"]), "weight_decay": 0.0},
    ], weight_decay=float(cfg["weight_decay"]))
    total_steps = len(loader) * int(cfg["epochs"])
    scheduler = _cosine_schedule(optimizer, int(total_steps * float(cfg["warmup_ratio"])), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["mixed_precision"]) and device.type == "cuda")
    output_dir = ensure_dir(cfg["output_dir"])
    best_path = output_dir / (f"best_{limit}.pt" if limit is not None else "best.pt")
    best_metric, stale = -1.0, 0
    history = []
    for epoch in range(int(cfg["epochs"])):
        model.train(); sampler.set_epoch(epoch)
        totals = defaultdict(float)
        for batch in tqdm(loader, desc=f"train-{epoch + 1}"):
            descriptors_batch = batch["descriptors"].to(device, non_blocking=True)
            highres_batch = torch.nn.functional.normalize(batch["highres"].to(device, non_blocking=True), dim=-1)
            text_batch = torch.nn.functional.normalize(batch["text"].to(device, non_blocking=True), dim=-1)
            title_ids = batch["title_ids"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=scaler.is_enabled()):
                tessera_batch = model(descriptors_batch)
                losses = alignment_losses(tessera_batch, highres_batch, text_batch, title_ids, model.logit_scale, cfg)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
            scaler.step(optimizer); scaler.update(); scheduler.step(); model.clamp_temperature()
            for key, value in losses.items(): totals[key] += float(value.detach())
        metric = _validation_metric(model, val_set, device, int(cfg["max_validation_candidates"]))
        record = {"epoch": epoch + 1, "val_macro_mAP": metric, **{key: value / len(loader) for key, value in totals.items()}}
        history.append(record); print(record)
        checkpoint = {"model_state": model.state_dict(), "config": config, "epoch": epoch + 1, "metric": metric}
        torch.save(checkpoint, output_dir / "latest.pt")
        if metric > best_metric:
            best_metric, stale = metric, 0
            torch.save(checkpoint, best_path)
        else:
            stale += 1
            if stale >= int(cfg["early_stopping_patience"]): break
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"best validation macro mAP={best_metric:.6f}: {best_path}")
    return best_path
