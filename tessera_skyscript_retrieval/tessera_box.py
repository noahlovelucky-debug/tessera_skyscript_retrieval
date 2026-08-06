"""Pure TESSERA-v1 + bbox baseline, without high-resolution image features."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import ensure_parent
from .data import load_prepared
from .features import load_feature_arrays
from .losses import symmetric_multi_positive
from .metrics import semantic_retrieval_metrics
from .model import build_tessera_box_adapter
from .training import TitleBalancedBatchSampler, _cosine_schedule


def bbox_features(frame) -> np.ndarray:
    """Position and physical footprint, with no signal derived from highres images."""
    lon = np.deg2rad(frame["center_lon"].to_numpy(np.float32))
    lat = np.deg2rad(frame["center_lat"].to_numpy(np.float32))
    width = (frame["bbox_east"] - frame["bbox_west"]).to_numpy(np.float32)
    height = (frame["bbox_north"] - frame["bbox_south"]).to_numpy(np.float32)
    width_m = np.maximum(width * 111_320.0 * np.cos(lat), 1.0)
    height_m = np.maximum(height * 110_574.0, 1.0)
    return np.stack(
        (np.sin(lon), np.cos(lon), np.sin(lat), np.cos(lat), np.log1p(width_m), np.log1p(height_m), np.log1p(width_m * height_m)),
        axis=1,
    ).astype(np.float32)


def normalize_boxes(frame, training_positions: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    values = bbox_features(frame)
    mean = values[training_positions].mean(axis=0)
    std = np.maximum(values[training_positions].std(axis=0), 1e-6)
    return ((values - mean) / std).astype(np.float32), {"mean": mean.tolist(), "std": std.tolist()}


class TesseraBoxDataset(Dataset):
    def __init__(self, frame, descriptors, boxes, text, split: str) -> None:
        self.positions = frame.index[frame["split"].eq(split)].to_numpy(np.int64)
        self.frame = frame.iloc[self.positions].reset_index(drop=True)
        self.descriptors, self.boxes, self.text = descriptors, boxes, text
        self.by_title: dict[int, list[int]] = defaultdict(list)
        for index, title_id in enumerate(self.frame["title_id"]):
            self.by_title[int(title_id)].append(index)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        position = int(self.positions[index])
        title_id = int(self.frame.iloc[index]["title_id"])
        return {"descriptor": np.asarray(self.descriptors[position], dtype=np.float32), "box": self.boxes[position], "text": np.asarray(self.text[title_id], dtype=np.float32), "title_id": title_id}


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "descriptors": torch.from_numpy(np.stack([row["descriptor"] for row in batch])),
        "boxes": torch.from_numpy(np.stack([row["box"] for row in batch])),
        "text": torch.from_numpy(np.stack([row["text"] for row in batch])),
        "title_ids": torch.tensor([row["title_id"] for row in batch], dtype=torch.long),
    }


@torch.inference_mode()
def encode_tessera_box(model, descriptors, boxes, device, batch_size: int = 2048) -> np.ndarray:
    values = np.empty((len(descriptors), 384), dtype=np.float32)
    model.eval()
    for start in range(0, len(descriptors), batch_size):
        stop = min(start + batch_size, len(descriptors))
        descriptor_batch = torch.from_numpy(np.asarray(descriptors[start:stop], dtype=np.float32)).to(device)
        box_batch = torch.from_numpy(np.asarray(boxes[start:stop], dtype=np.float32)).to(device)
        values[start:stop] = model.encode_image(descriptor_batch, box_batch).cpu().numpy()
    return values


@torch.inference_mode()
def encode_text_box(model, text, device, batch_size: int = 2048) -> np.ndarray:
    values = np.empty((len(text), 384), dtype=np.float32)
    model.eval()
    for start in range(0, len(text), batch_size):
        stop = min(start + batch_size, len(text))
        batch = torch.from_numpy(np.asarray(text[start:stop], dtype=np.float32)).to(device)
        values[start:stop] = model.encode_text(batch).cpu().numpy()
    return values


@torch.inference_mode()
def _validation_metric(model, dataset, device, maximum: int) -> float:
    count = min(len(dataset), maximum)
    positions = dataset.positions[:count]
    candidates = encode_tessera_box(model, dataset.descriptors[positions], dataset.boxes[positions], device)
    labels = dataset.frame["title_id"].iloc[:count].to_numpy(np.int64)
    unique = np.unique(labels)
    queries = encode_text_box(model, dataset.text[unique], device)
    return float(semantic_retrieval_metrics(queries, unique, candidates, labels, (10,))["macro_mAP"])


def train_tessera_box(config: dict, limit: int | None = None) -> Path:
    seed = int(config["training"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    frame = load_prepared(config)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    descriptors, _, text = load_feature_arrays(config, limit)
    train_positions = frame.index[frame["split"].eq("train")].to_numpy(np.int64)
    boxes, normalization = normalize_boxes(frame, train_positions)
    train_set = TesseraBoxDataset(frame, descriptors, boxes, text, "train")
    val_set = TesseraBoxDataset(frame, descriptors, boxes, text, "val")
    if not len(train_set) or not len(val_set):
        raise ValueError("training and validation splits must be non-empty")
    cfg = config["training"]
    sampler = TitleBalancedBatchSampler(train_set, int(cfg["titles_per_batch"]), int(cfg["samples_per_title"]), seed)
    loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=_collate, num_workers=int(cfg.get("workers", 4)), pin_memory=True, persistent_workers=int(cfg.get("workers", 4)) > 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tessera_box_adapter(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    total_steps = len(loader) * int(cfg["epochs"])
    scheduler = _cosine_schedule(optimizer, int(total_steps * float(cfg["warmup_ratio"])), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["mixed_precision"]) and device.type == "cuda")
    output_dir = Path(cfg["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "box_normalization.json").write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    best_path = output_dir / (f"best_{limit}.pt" if limit is not None else "best.pt")
    best_metric, stale, history = -1.0, 0, []
    for epoch in range(int(cfg["epochs"])):
        model.train(); sampler.set_epoch(epoch); total = 0.0
        for batch in tqdm(loader, desc=f"train-tessera-box-{epoch + 1}"):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=scaler.is_enabled()):
                image = model.encode_image(batch["descriptors"].to(device, non_blocking=True), batch["boxes"].to(device, non_blocking=True))
                query = model.encode_text(batch["text"].to(device, non_blocking=True))
                title_ids = batch["title_ids"].to(device, non_blocking=True)
                loss = symmetric_multi_positive(model.logit_scale.exp().clamp(max=100.0) * query @ image.T, title_ids[:, None].eq(title_ids[None, :]))
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
            scaler.step(optimizer); scaler.update(); scheduler.step(); model.clamp_temperature(); total += float(loss.detach())
        metric = _validation_metric(model, val_set, device, int(cfg["max_validation_candidates"]))
        record = {"epoch": epoch + 1, "val_macro_mAP": metric, "semantic": total / len(loader)}
        history.append(record); print(record, flush=True)
        checkpoint = {"model_state": model.state_dict(), "config": config, "epoch": epoch + 1, "metric": metric, "box_normalization": normalization, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        torch.save(checkpoint, output_dir / "latest.pt")
        if metric > best_metric:
            best_metric, stale = metric, 0; torch.save(checkpoint, best_path)
        else:
            stale += 1
            if stale >= int(cfg["early_stopping_patience"]):
                break
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return best_path


def evaluate_tessera_box(config: dict, limit: int | None = None, checkpoint_path: str | None = None) -> dict:
    frame = load_prepared(config)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    descriptors, _, text = load_feature_arrays(config, limit)
    checkpoint_value = Path(checkpoint_path or config["evaluation"]["checkpoint"])
    if limit is not None and checkpoint_path is None:
        checkpoint_value = Path(config["training"]["output_dir"]) / f"best_{limit}.pt"
    checkpoint = torch.load(checkpoint_value, map_location="cpu", weights_only=False)
    normal = checkpoint["box_normalization"]
    boxes = ((bbox_features(frame) - np.asarray(normal["mean"], dtype=np.float32)) / np.asarray(normal["std"], dtype=np.float32)).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tessera_box_adapter(config).to(device); model.load_state_dict(checkpoint["model_state"])
    report = {"checkpoint": str(checkpoint_value), "architecture": "tessera_v1_box_mlp_768_1024_384", "trainable_parameters": int(checkpoint["trainable_parameters"]), "input": {"tessera_descriptor": 2816, "bbox_features": 7, "label_supervision": "title_id multi-positive InfoNCE"}, "splits": {}}
    k_values = tuple(map(int, config["evaluation"]["k_values"]))
    for split in ("test", "oov_test"):
        positions = frame.index[frame["split"].eq(split)].to_numpy(np.int64)
        if len(positions):
            candidates = encode_tessera_box(model, descriptors[positions], boxes[positions], device)
            labels = frame["title_id"].iloc[positions].to_numpy(np.int64)
            unique = np.unique(labels)
            report["splits"][split] = {"text_to_tessera": semantic_retrieval_metrics(encode_text_box(model, text[unique], device), unique, candidates, labels, k_values)}
    report_path = ensure_parent(config["evaluation"]["report"])
    if limit is not None:
        report_path = report_path.with_name(f"{report_path.stem}_{limit}{report_path.suffix}")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
