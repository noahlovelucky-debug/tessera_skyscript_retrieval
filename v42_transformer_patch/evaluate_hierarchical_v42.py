"""Locked-test and hierarchical reranking evaluation for v4.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from .hierarchical_v42_model import HierarchicalDualSpaceModel
from .hierarchical_v42_training import (
    base_v42_metrics,
    hierarchical_rerank_metrics,
)
from .metrics import (
    deduplicate_global_candidates,
    local_candidates,
    localization_metrics,
    retrieval_metrics,
)
from .train_hierarchical_v42 import encode_dataset, make_dataset
from .train_hierarchical_v42 import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hierarchical-top-m", type=int)
    parser.add_argument("--hierarchical-local-weight", type=float)
    return parser.parse_args()


@torch.inference_mode()
def prompt_ensemble(
    model: HierarchicalDualSpaceModel,
    category: str,
    templates: list[str],
    device: torch.device,
    max_length: int,
) -> dict[str, torch.Tensor]:
    tokens = model.text.tokenizer(
        [value.format(category=category) for value in templates],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    values = model.encode_text(tokens["input_ids"], tokens["attention_mask"])
    return {
        key: torch.nn.functional.normalize(value.mean(dim=0), dim=0).cpu()
        for key, value in values.items()
        if key in {"global", "local"}
    }


def oov_metrics(
    model: HierarchicalDualSpaceModel,
    test: dict,
    oov: dict,
    cfg: dict,
    device: torch.device,
) -> dict:
    categories = sorted(set(oov["categories"]))
    prompts = [
        prompt_ensemble(
            model,
            category,
            list(cfg["evaluation"]["oov_query_templates"]),
            device,
            int(cfg["data"]["max_text_length"]),
        )
        for category in categories
    ]
    global_queries = torch.stack([value["global"] for value in prompts])
    local_queries = torch.stack([value["local"] for value in prompts])
    combined_categories = test["categories"] + oov["categories"]
    combined_ids = test["patch_ids"] + oov["patch_ids"]
    global_candidates, global_ids, global_category_sets = (
        deduplicate_global_candidates(
            torch.cat([test["global_image"], oov["global_image"]]),
            combined_ids,
            combined_categories,
        )
    )
    global_values = retrieval_metrics(
        global_queries,
        categories,
        [f"oov-query:{value}" for value in categories],
        global_candidates,
        global_ids,
        global_category_sets,
        tuple(cfg["evaluation"]["k_values"]),
        prefix="oov_global",
    )
    local_candidates_values, local_ids, local_category_sets = local_candidates(
        torch.cat([test["local_image"], oov["local_image"]]),
        combined_ids,
        combined_categories,
    )
    local_values = retrieval_metrics(
        local_queries,
        categories,
        [f"oov-query:{value}" for value in categories],
        local_candidates_values,
        local_ids,
        local_category_sets,
        tuple(cfg["evaluation"]["k_values"]),
        prefix="oov_local",
    )
    values = {**global_values, **local_values}
    return {
        key: value for key, value in values.items() if "exact_patch" not in key
    }


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    output_dir = Path(cfg["output"]["v42_dir"])
    checkpoint_path = args.checkpoint or output_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    category_to_id = checkpoint["category_to_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, len(category_to_id))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    kwargs = {
        "device": device,
        "max_text_length": int(cfg["data"]["max_text_length"]),
        "batch_size": int(cfg["evaluation"]["batch_size"]),
        "workers": int(cfg["evaluation"]["workers"]),
    }
    test_dataset = make_dataset(cfg, "test", category_to_id, train=False)
    oov_dataset = make_dataset(cfg, "oov", category_to_id, train=False)
    test = encode_dataset(model, test_dataset, **kwargs)
    oov = encode_dataset(model, oov_dataset, **kwargs)
    closed = base_v42_metrics(test, tuple(cfg["evaluation"]["k_values"]))
    hierarchical_top_m = (
        args.hierarchical_top_m
        if args.hierarchical_top_m is not None
        else int(cfg["evaluation"]["hierarchical_top_m"])
    )
    hierarchical_local_weight = (
        args.hierarchical_local_weight
        if args.hierarchical_local_weight is not None
        else float(cfg["evaluation"]["hierarchical_local_weight"])
    )
    closed.update(
        hierarchical_rerank_metrics(
            test,
            hierarchical_top_m,
            hierarchical_local_weight,
            tuple(cfg["evaluation"]["k_values"]),
            device,
            int(cfg["evaluation"]["hierarchical_query_batch_size"]),
        )
    )
    oov_localization = localization_metrics(
        oov["local_text"],
        oov["localization_tokens"],
        oov["token_rows"],
        oov["token_cols"],
    )
    targets = cfg["evaluation"]["acceptance_targets"]
    global_acceptance = {
        "macro_r1": closed["global_macro_category_R@1"]
        >= float(targets["macro_r1"]),
        "category_r1": closed["global_category_R@1"]
        >= float(targets["category_r1"]),
        "exact_top1": closed["localization_exact_token_top1"]
        >= float(targets["exact_top1"]),
        "map": closed["global_mAP"] >= float(targets["map"]),
    }
    local_acceptance = {
        "macro_r1": closed["local_macro_category_R@1"]
        >= float(targets["macro_r1"]),
        "category_r1": closed["local_category_R@1"]
        >= float(targets["category_r1"]),
        "exact_top1": closed["localization_exact_token_top1"]
        >= float(targets["exact_top1"]),
        "map": closed["local_mAP"] >= float(targets["map"]),
    }
    result = {
        "model": cfg["model"].get("architecture", "hierarchical_dual_space_v4_2"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "test_rows": len(test_dataset),
        "oov_rows": len(oov_dataset),
        "closed_test": closed,
        "acceptance_targets": targets,
        "acceptance": local_acceptance,
        "acceptance_by_branch": {
            "global": global_acceptance,
            "local": local_acceptance,
        },
        "all_targets_met": all(local_acceptance.values()),
        "oov": oov_metrics(model, test, oov, cfg, device),
        "oov_localization": oov_localization,
    }
    output_path = args.output or output_dir / "final_test_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
