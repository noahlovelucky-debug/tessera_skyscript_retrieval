"""Measure local and Luna-confidence reranking on one Luna-labeled candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tessera_skyscript_retrieval.config import load_config
from tessera_skyscript_retrieval.data import load_prepared
from tessera_skyscript_retrieval.features import load_latent_feature_arrays
from tessera_skyscript_retrieval.model import build_latent_model
from tessera_skyscript_retrieval.skyclip import encode_query


def metrics(frame: pd.DataFrame, cutoffs: tuple[int, ...] = (10, 25, 50, 100)) -> dict:
    relevant = frame.sort_values("final_rank")["relevant"].astype(bool).to_numpy()
    total = int(relevant.sum())
    values = {"relevant_in_pool": total}
    for cutoff in cutoffs:
        selected = relevant[:cutoff]
        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
        ideal = discounts[: min(total, cutoff)].sum()
        values[f"precision_at_{cutoff}"] = float(selected.mean())
        # This is normalized against judged positives in the fixed Top-100 pool,
        # not against all relevant images in the full corpus.
        values[f"pool_ndcg_at_{cutoff}"] = float((selected * discounts).sum() / max(ideal, 1e-8))
    return values


def load_local_scorer(config: dict) -> tuple[dict[str, int], np.ndarray, np.ndarray, torch.nn.Module, torch.device]:
    frame = load_prepared(config)
    positions_by_id = pd.Series(frame.index.to_numpy(), index=frame["sample_id"]).to_dict()
    _, teacher, tokens, _ = load_latent_feature_arrays(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config["evaluation"]["checkpoint"], map_location="cpu", weights_only=False)
    model = build_latent_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return positions_by_id, teacher, tokens, model, device


@torch.inference_mode()
def local_scores(
    config: dict,
    candidates: pd.DataFrame,
    query: str,
    scorer: tuple[dict[str, int], np.ndarray, np.ndarray, torch.nn.Module, torch.device],
) -> tuple[np.ndarray, np.ndarray]:
    positions_by_id, teacher, tokens, model, device = scorer
    positions = np.array([positions_by_id[value] for value in candidates["sample_id"]], dtype=np.int64)
    text = encode_query(config, query, config["index"]["prompt_templates"])
    text_tensor = torch.from_numpy(text.astype(np.float32))[None].to(device)
    text_latent = model.encode_text_latent(text_tensor)
    teacher_batch = F.normalize(
        torch.from_numpy(np.asarray(teacher[positions], dtype=np.float32)).to(device), dim=-1
    )
    token_batch = torch.from_numpy(np.asarray(tokens[positions], dtype=np.float32)).to(device)
    global_features, image_latents = model.encode_highres(token_batch, teacher_batch)
    global_scores = (global_features @ text_tensor.T).squeeze(-1)
    local = torch.einsum("d,nkd->nk", text_latent[0], image_latents).amax(dim=-1)
    return global_scores.cpu().numpy(), local.cpu().numpy()


def ranked(frame: pd.DataFrame, method: str, score: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    result["method"] = method
    result["rerank_score"] = score
    result = result.sort_values(["rerank_score", "rank"], ascending=[False, True]).reset_index(drop=True)
    result["final_rank"] = np.arange(1, len(result) + 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gated_coarse_v3.yaml")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--queries", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-weight", type=float, default=0.65)
    args = parser.parse_args()
    config = load_config(args.config)
    candidates_all = pd.read_csv(args.candidates)
    queries = args.queries or candidates_all["query"].drop_duplicates().tolist()
    if not queries:
        raise ValueError("at least one query is required")
    local_weight = float(args.local_weight)
    records = []
    summaries = {}
    scorer = load_local_scorer(config)
    for query in queries:
        candidates = candidates_all[candidates_all["query"].eq(query)].sort_values("rank").reset_index(drop=True)
        if len(candidates) != 100:
            raise ValueError(f"expected exactly 100 candidates for {query!r}, got {len(candidates)}")
        global_scores, local = local_scores(config, candidates, query, scorer)
        ranked_methods = [
            ranked(candidates, "coarse", -candidates["rank"].to_numpy(np.float32)),
            ranked(candidates, "local_rerank", (1.0 - local_weight) * global_scores + local_weight * local),
            ranked(candidates, "luna_confidence_rerank", candidates["confidence"].to_numpy(np.float32)),
            # This uses the labels that also define the metric; retain solely as an upper bound.
            ranked(candidates, "luna_label_oracle_upper_bound", candidates["relevant"].astype(float).to_numpy() + candidates["confidence"].to_numpy() * 1e-3),
        ]
        records.extend(ranked_methods)
        summaries[query] = {item["method"].iloc[0]: metrics(item) for item in ranked_methods}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = pd.concat(records, ignore_index=True)
    all_rows.to_csv(output_dir / "ranked_candidates.csv", index=False)
    result = {
        "queries": queries,
        "candidate_pool": str(Path(args.candidates).resolve()),
        "evaluation": "Luna pixel relevance labels previously collected for the fixed candidate pool",
        "local_rule": f"{1.0 - local_weight:.2f} global + {local_weight:.2f} local MaxSim",
        "per_query_methods": summaries,
        "mean_methods": {
            method: {
                key: float(np.mean([summary[method][key] for summary in summaries.values()]))
                for key in next(iter(summaries.values()))[method]
            }
            for method in next(iter(summaries.values()))
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Multi-Class Rerank Comparison",
        "",
        "All methods reorder exactly the same coarse Top-100 candidates per query. Relevance labels are Luna pixel judgments collected before any reranking.",
        "",
        "| Method | Mean P@10 | Mean pool-nDCG@10 | Mean P@25 | Mean pool-nDCG@25 | Mean P@100 | Mean pool-nDCG@100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in result["mean_methods"].items():
        lines.append(
            f"| {name} | {value['precision_at_10']:.4f} | {value['pool_ndcg_at_10']:.4f} | "
            f"{value['precision_at_25']:.4f} | {value['pool_ndcg_at_25']:.4f} | "
            f"{value['precision_at_100']:.4f} | {value['pool_ndcg_at_100']:.4f} |"
        )
    lines += [
        "",
        "## Per-Query Detail",
        "",
        "| Query | Coarse P@10 | Local P@10 | Luna confidence P@10 | Coarse pool-nDCG@10 | Local pool-nDCG@10 | Luna confidence pool-nDCG@10 | Coarse P@100 | Local P@100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for query, value in summaries.items():
        coarse = value["coarse"]
        local = value["local_rerank"]
        luna = value["luna_confidence_rerank"]
        lines.append(
            f"| {query} | {coarse['precision_at_10']:.4f} | {local['precision_at_10']:.4f} | "
            f"{luna['precision_at_10']:.4f} | {coarse['pool_ndcg_at_10']:.4f} | "
            f"{local['pool_ndcg_at_10']:.4f} | {luna['pool_ndcg_at_10']:.4f} | "
            f"{coarse['precision_at_100']:.4f} | {local['precision_at_100']:.4f} |"
        )
    lines += [
        "",
        "`pool-nDCG` is normalized by the judged positives already present in each fixed Top-100 candidate pool. It is not a full-corpus nDCG or recall metric. `luna_label_oracle_upper_bound` is intentionally not a deployable result: it sorts by the same boolean labels used for evaluation. `luna_confidence_rerank` is the deployable-style proxy, but it is still self-evaluated by the same judge and needs a separate human or independent judge audit before any product claim.",
    ]
    (output_dir / "RERANK_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
