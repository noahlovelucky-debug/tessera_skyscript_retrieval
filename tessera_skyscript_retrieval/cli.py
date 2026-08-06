from __future__ import annotations

import argparse

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="TESSERA-SkyScript frozen-teacher retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "cache-tessera", "cache-skyclip", "train", "evaluate", "build-index"):
        item = sub.add_parser(name)
        item.add_argument("--config", default="configs/default.yaml")
        item.add_argument("--limit", type=int)
        if name in {"evaluate", "build-index"}:
            item.add_argument("--checkpoint")
    item = sub.add_parser("search")
    item.add_argument("--config", default="configs/default.yaml")
    item.add_argument("--query", required=True)
    item.add_argument("--modality", choices=("highres", "tessera", "both"), default="both")
    item.add_argument("--top-k", type=int, default=10)
    item.add_argument("--index-dir")
    item = sub.add_parser("visualize")
    item.add_argument("--config", default="configs/default.yaml")
    item.add_argument("--query", default="quarry area")
    item.add_argument("--top-k", type=int, default=5)
    item.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        from .data import prepare_manifest
        prepare_manifest(config, args.limit)
    elif args.command == "cache-tessera":
        from .pooling import cache_tessera
        cache_tessera(config, args.limit)
    elif args.command == "cache-skyclip":
        from .skyclip import cache_skyclip
        cache_skyclip(config, args.limit)
    elif args.command == "train":
        from .training import train_adapter
        train_adapter(config, args.limit)
    elif args.command == "evaluate":
        if config.get("model", {}).get("architecture") in {"latent_v2", "gated_coarse_v3", "anchored_gated_v4"}:
            from .latent_evaluation import evaluate_latent
            evaluate_latent(config, args.limit, args.checkpoint)
        elif config.get("model", {}).get("architecture") == "tessera_box_mlp":
            from .tessera_box import evaluate_tessera_box
            evaluate_tessera_box(config, args.limit, args.checkpoint)
        else:
            from .evaluation import evaluate
            evaluate(config, args.limit, args.checkpoint)
    elif args.command == "build-index":
        from .indexing import build_index
        build_index(config, args.limit, args.checkpoint)
    elif args.command == "search":
        from .indexing import search
        search(config, args.query, args.modality, args.top_k, args.index_dir)
    else:
        from .visualization import visualize_results
        visualize_results(config, args.query, args.top_k, args.output_dir)


if __name__ == "__main__":
    main()
