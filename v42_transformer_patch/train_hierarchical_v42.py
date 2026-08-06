"""Train v4.2 hierarchical global-retrieval/local-localization model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .dataset import RetrievalRows, make_collate
from .dual_head_training import ConfusionAwareSampler, confusion_adjacency
from .hierarchical_v42_model import HierarchicalDualSpaceModel
from .hierarchical_v42_training import base_v42_metrics, hierarchical_v42_loss


def build_model(cfg: dict, num_classes: int):
    if cfg["model"].get("architecture") == "transformer_v42":
        from .transformer_v42_model import TransformerV42Model

        return TransformerV42Model(cfg, num_classes)
    return HierarchicalDualSpaceModel(cfg, num_classes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(
    cfg: dict,
    split: str,
    category_to_id: dict[str, int],
    train: bool,
) -> RetrievalRows:
    prepared = Path(cfg["data"]["prepared_dir"])
    return RetrievalRows(
        prepared / "splits.csv",
        split,
        category_to_id,
        "patch_vit",
        token_cache=Path(cfg["student"]["token_cache"]),
        tessera_cache=Path(cfg["tessera"]["feature_cache"]),
        preload_tokens=bool(cfg["student"].get("preload_token_cache", False)),
        train=train,
    )


def configure_text_phase(
    model: HierarchicalDualSpaceModel, cfg: dict, epoch: int
) -> str:
    unfreeze_epoch = int(cfg["training"]["text_encoder_unfreeze_epoch"])
    layers = int(cfg["training"].get("text_encoder_layers", 2))
    if epoch < unfreeze_epoch:
        model.text.freeze_encoder()
        model.text.encoder.eval()
        return "dual_adapter_warmup"
    model.text.unfreeze_last_layers(layers)
    model.text.encoder.train()
    return f"dual_adapter_plus_bge_last_{layers}"


def make_optimizer(
    model: HierarchicalDualSpaceModel, cfg: dict
) -> torch.optim.Optimizer:
    training = cfg["training"]
    if hasattr(model, "spatial_fusion"):
        image_parameters = list(model.spatial_fusion.parameters())
    else:
        image_parameters = list(model.global_image_head.parameters())
        image_parameters += list(model.local_image_projection.parameters())
    image_parameters += list(model.local_retrieval_adapter.parameters())
    text_parameters = list(model.text.global_projection.parameters())
    text_parameters += list(model.text.local_projection.parameters())
    head_parameters = list(model.localization_token_adapter.parameters())
    head_parameters += [model.global_prototypes, model.local_prototypes]
    return torch.optim.AdamW(
        [
            {
                "params": image_parameters,
                "lr": float(training["image_learning_rate"]),
            },
            {
                "params": text_parameters,
                "lr": float(training["text_adapter_learning_rate"]),
            },
            {
                "params": head_parameters,
                "lr": float(training["head_learning_rate"]),
            },
            {
                "params": model.text.encoder_parameters_for_last_layers(
                    int(training.get("text_encoder_layers", 2))
                ),
                "lr": float(training["text_learning_rate"]),
            },
            {
                "params": [
                    model.global_logit_scale,
                    model.local_logit_scale,
                    model.localization_logit_scale,
                ],
                "lr": float(training["temperature_learning_rate"]),
                "weight_decay": 0.0,
            },
        ],
        weight_decay=float(training["weight_decay"]),
    )


@torch.inference_mode()
def initialize_prototypes(
    model: HierarchicalDualSpaceModel,
    category_to_id: dict[str, int],
    device: torch.device,
    max_length: int,
) -> None:
    categories = [
        name for name, _ in sorted(category_to_id.items(), key=lambda item: item[1])
    ]
    tokens = model.text.tokenizer(
        [f"satellite image of {name}" for name in categories],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    text = model.encode_text(tokens["input_ids"], tokens["attention_mask"])
    model.initialize_prototypes(text["global"], text["local"])


@torch.inference_mode()
def encode_dataset(
    model: HierarchicalDualSpaceModel,
    dataset: RetrievalRows,
    device: torch.device,
    max_text_length: int,
    batch_size: int,
    workers: int,
) -> dict:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=make_collate(
            model.text.tokenizer, max_text_length, "patch_vit"
        ),
    )
    output = {
        "global_text": [],
        "local_text": [],
        "global_image": [],
        "local_image": [],
        "localization_tokens": [],
        "token_rows": [],
        "token_cols": [],
        "categories": [],
        "patch_ids": [],
        "poi_uids": [],
    }
    for batch in loader:
        text = model.encode_text(
            batch["input_ids_a"].to(device),
            batch["attention_mask_a"].to(device),
        )
        image = model.encode_images(
            batch["s2_tokens"].to(device),
            batch["s1_tokens"].to(device),
            batch["token_indices"].to(device),
        )
        output["global_text"].append(text["global"].cpu())
        output["local_text"].append(text["local"].cpu())
        output["global_image"].append(image["global"].cpu())
        output["local_image"].append(image["local"].cpu())
        output["localization_tokens"].append(
            image["localization_tokens"].to(torch.float16).cpu()
        )
        output["token_rows"].append(batch["token_rows"])
        output["token_cols"].append(batch["token_cols"])
        output["categories"].extend(batch["categories"])
        output["patch_ids"].extend(batch["patch_ids"])
        output["poi_uids"].extend(batch["poi_uids"])
    for key in (
        "global_text",
        "local_text",
        "global_image",
        "local_image",
        "localization_tokens",
        "token_rows",
        "token_cols",
    ):
        output[key] = torch.cat(output[key])
    return output


def harmonic_selection(metrics: dict[str, float], targets: dict) -> float:
    ratios = [
        metrics["global_macro_category_R@1"] / float(targets["macro_r1"]),
        metrics["global_category_R@1"] / float(targets["category_r1"]),
        metrics["global_mAP"] / float(targets["map"]),
        metrics["localization_exact_token_top1"]
        / float(targets["exact_top1"]),
    ]
    return float(len(ratios) / sum(1.0 / max(value, 1e-8) for value in ratios))


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["training"]["seed"])
    set_seed(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = Path(cfg["data"]["prepared_dir"])
    category_to_id = json.loads((prepared / "category_vocab.json").read_text())
    train_dataset = make_dataset(cfg, "train", category_to_id, train=True)
    validation_dataset = make_dataset(
        cfg, "validation", category_to_id, train=False
    )
    model = build_model(cfg, len(category_to_id)).to(device)
    text_phase = configure_text_phase(model, cfg, 0)
    optimizer = make_optimizer(model, cfg)
    start_epoch = 0
    best_score = -1.0
    best_retrieval = -1.0
    best_localization = -1.0
    stale = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_retrieval = float(checkpoint.get("best_retrieval", -1.0))
        best_localization = float(checkpoint.get("best_localization", -1.0))
    elif not cfg["model"].get("base_checkpoint"):
        initialize_prototypes(
            model,
            category_to_id,
            device,
            int(cfg["data"]["max_text_length"]),
        )

    output_dir = Path(cfg["output"]["v42_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    sampler = ConfusionAwareSampler(
        train_dataset,
        category_to_id,
        int(cfg["training"]["categories_per_batch"]),
        int(cfg["training"]["samples_per_category"]),
        seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=make_collate(
            model.text.tokenizer,
            int(cfg["data"]["max_text_length"]),
            "patch_vit",
        ),
        num_workers=int(cfg["training"]["workers"]),
        pin_memory=True,
        persistent_workers=int(cfg["training"]["workers"]) > 0,
    )
    adjacency = confusion_adjacency(category_to_id, device)
    mixed_precision = (
        bool(cfg["training"]["mixed_precision"]) and device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    print(
        json.dumps(
            {
                "architecture": cfg["model"].get("architecture", "hierarchical_dual_space_v4_2"),
                "global_image": "16x16x768 -> spatial Transformer -> CLS 384" if hasattr(model, "spatial_fusion") else "16x16x768 -> attention pool -> 768-1536-384",
                "local_image": "Transformer output tokens [16,16,384]" if hasattr(model, "spatial_fusion") else "16x16x768 -> per-token 768-1536-384",
                "text": "BGE384 -> global/local 384-1536-384",
                "train_rows": len(train_dataset),
                "validation_rows": len(validation_dataset),
                "categories": len(category_to_id),
                "parameters": sum(value.numel() for value in model.parameters()),
                "trainable_parameters": sum(
                    value.numel()
                    for value in model.parameters()
                    if value.requires_grad
                ),
                "initial_text_phase": text_phase,
                "bge_unfreeze_epoch": int(
                    cfg["training"]["text_encoder_unfreeze_epoch"]
                )
                + 1,
                "confusion_pairs": int(adjacency.sum()),
                "warm_start": cfg["model"].get("base_checkpoint"),
            },
            indent=2,
        ),
        flush=True,
    )

    history_path = output_dir / "metrics.jsonl"
    loss_keys = (
        "total",
        "global_retrieval",
        "local_retrieval",
        "prototype",
        "localization_exact",
        "same_image_hard",
        "global_confusion",
        "local_confusion",
        "global_consistency",
        "local_consistency",
        "distill_local",
    )
    for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
        sampler.set_epoch(epoch)
        model.train()
        text_phase = configure_text_phase(model, cfg, epoch)
        totals = {key: 0.0 for key in loss_keys}
        steps = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=mixed_precision
            ):
                text_a = model.encode_text(
                    batch["input_ids_a"].to(device),
                    batch["attention_mask_a"].to(device),
                )
                text_b = model.encode_text(
                    batch["input_ids_b"].to(device),
                    batch["attention_mask_b"].to(device),
                )
                image = model.encode_images(
                    batch["s2_tokens"].to(device),
                    batch["s1_tokens"].to(device),
                    batch["token_indices"].to(device),
                )
                localization_logits = model.localization_logits(
                    text_a["local"], image["localization_tokens"]
                )
                teacher = None
                if "teacher_features" in batch:
                    with torch.no_grad():
                        teacher = model.encode_teacher(
                            batch["teacher_features"].to(device)
                        )
                losses = hierarchical_v42_loss(
                    text_a,
                    text_b,
                    image,
                    localization_logits,
                    batch["category_ids"].to(device),
                    batch["patch_ids"],
                    model.global_prototypes,
                    model.local_prototypes,
                    adjacency,
                    model.global_logit_scale,
                    model.local_logit_scale,
                    batch["token_rows"].to(device),
                    batch["token_cols"].to(device),
                    cfg["loss"],
                    teacher_vectors=teacher,
                )
            if args.dry_run:
                print(
                    {key: float(value.detach()) for key, value in losses.items()},
                    flush=True,
                )
                print("dry-run successful", flush=True)
                return
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["training"]["max_grad_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            model.clamp_logit_scales()
            for key in loss_keys:
                totals[key] += float(losses[key].detach())
            steps += 1

        encoded = encode_dataset(
            model,
            validation_dataset,
            device,
            int(cfg["data"]["max_text_length"]),
            int(cfg["evaluation"]["batch_size"]),
            int(cfg["evaluation"]["workers"]),
        )
        metrics = base_v42_metrics(
            encoded, tuple(cfg["evaluation"]["k_values"])
        )
        selection_score = harmonic_selection(
            metrics, cfg["evaluation"]["acceptance_targets"]
        )
        record = {
            "epoch": epoch + 1,
            "text_phase": text_phase,
            "selection_score": selection_score,
            **{
                f"train_{key}": value / max(steps, 1)
                for key, value in totals.items()
            },
            **metrics,
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        retrieval_score = 0.5 * (
            metrics["global_macro_category_R@1"]
            + metrics["global_category_R@1"]
        )
        localization_score = metrics["localization_exact_token_top1"]
        checkpoint = {
            "epoch": epoch,
            "best_score": max(best_score, selection_score),
            "best_retrieval": max(best_retrieval, retrieval_score),
            "best_localization": max(best_localization, localization_score),
            "selection_score": selection_score,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "category_to_id": category_to_id,
            "metrics": metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if retrieval_score > best_retrieval:
            best_retrieval = retrieval_score
            checkpoint["best_retrieval"] = best_retrieval
            torch.save(checkpoint, output_dir / "best_retrieval.pt")
        if localization_score > best_localization:
            best_localization = localization_score
            checkpoint["best_localization"] = best_localization
            torch.save(checkpoint, output_dir / "best_localization.pt")
        if selection_score > best_score:
            best_score = selection_score
            stale = 0
            checkpoint["best_score"] = best_score
            torch.save(checkpoint, output_dir / "best.pt")
        elif epoch + 1 >= int(
            cfg["training"].get("early_stopping_start_epoch", 1)
        ):
            stale += 1
        if stale >= int(cfg["training"]["early_stopping_patience"]):
            print(f"early stopping after {stale} stale epochs", flush=True)
            break


if __name__ == "__main__":
    main()
