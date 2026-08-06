from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def semantic_retrieval_metrics(
    queries: np.ndarray,
    query_labels: np.ndarray,
    candidates: np.ndarray,
    candidate_labels: np.ndarray,
    k_values: tuple[int, ...],
) -> dict[str, float]:
    queries = _normalize(queries)
    candidates = _normalize(candidates)
    candidate_labels = np.asarray(candidate_labels)
    hits = {k: [] for k in k_values}
    precisions = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}
    average_precisions = []
    for query, label in zip(queries, np.asarray(query_labels)):
        positives = candidate_labels == label
        positive_count = int(positives.sum())
        if not positive_count:
            continue
        scores = candidates @ query
        order = np.argsort(scores)[::-1]
        ranked = positives[order]
        cumulative = np.cumsum(ranked)
        precision = cumulative / np.arange(1, len(ranked) + 1)
        average_precisions.append(float((precision * ranked).sum() / positive_count))
        for k in k_values:
            cutoff = min(k, len(ranked))
            selected = ranked[:cutoff]
            discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
            ideal_count = min(positive_count, cutoff)
            ideal_dcg = float(discounts[:ideal_count].sum())
            hits[k].append(float(selected.any()))
            precisions[k].append(float(selected.mean()))
            ndcgs[k].append(float((selected * discounts).sum() / max(ideal_dcg, 1e-8)))
    if not average_precisions:
        raise ValueError("no query has a positive candidate")
    result = {}
    for k in k_values:
        result[f"Hit@{k}"] = float(np.mean(hits[k]))
        result[f"Precision@{k}"] = float(np.mean(precisions[k]))
        result[f"nDCG@{k}"] = float(np.mean(ndcgs[k]))
    result["macro_mAP"] = float(np.mean(average_precisions))
    result["query_count"] = float(len(average_precisions))
    result["candidate_count"] = float(len(candidates))
    return result


def semantic_topk_metrics(
    top_indices: np.ndarray,
    query_labels: np.ndarray,
    candidate_labels: np.ndarray,
    k_values: tuple[int, ...],
) -> dict[str, float]:
    """Compute binary relevance metrics from pre-ranked candidate indices."""
    top_indices = np.asarray(top_indices, dtype=np.int64)
    query_labels = np.asarray(query_labels)
    candidate_labels = np.asarray(candidate_labels)
    hits = {k: [] for k in k_values}
    precisions = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}
    valid_queries = 0
    for row, label in enumerate(query_labels):
        positive_count = int((candidate_labels == label).sum())
        if not positive_count:
            continue
        valid_queries += 1
        ranked = candidate_labels[top_indices[row]] == label
        for k in k_values:
            cutoff = min(k, len(ranked))
            selected = ranked[:cutoff]
            discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
            ideal_count = min(positive_count, cutoff)
            ideal_dcg = float(discounts[:ideal_count].sum())
            hits[k].append(float(selected.any()))
            precisions[k].append(float(selected.mean()))
            ndcgs[k].append(float((selected * discounts).sum() / max(ideal_dcg, 1e-8)))
    if not valid_queries:
        raise ValueError("no query has a positive candidate")
    result = {}
    for k in k_values:
        result[f"Hit@{k}"] = float(np.mean(hits[k]))
        result[f"Precision@{k}"] = float(np.mean(precisions[k]))
        result[f"nDCG@{k}"] = float(np.mean(ndcgs[k]))
    result["query_count"] = float(valid_queries)
    result["candidate_count"] = float(len(candidate_labels))
    return result


@torch.inference_mode()
def late_interaction_topk(
    text_latents: np.ndarray,
    image_latents: np.ndarray,
    top_k: int,
    device: torch.device,
    query_batch_size: int = 64,
    candidate_chunk_size: int = 8192,
    text_global: np.ndarray | None = None,
    image_global: np.ndarray | None = None,
    fine_weight: float = 1.0,
) -> np.ndarray:
    """Rank a large latent index without materializing the full score matrix."""
    query_count = len(text_latents)
    candidate_count = len(image_latents)
    top_k = min(top_k, candidate_count)
    output = np.empty((query_count, top_k), dtype=np.int64)
    for query_start in range(0, query_count, query_batch_size):
        query_stop = min(query_start + query_batch_size, query_count)
        queries = F.normalize(
            torch.from_numpy(
                np.asarray(text_latents[query_start:query_stop], dtype=np.float32)
            ).to(device),
            dim=-1,
        )
        query_globals = None
        if text_global is not None:
            query_globals = F.normalize(
                torch.from_numpy(
                    np.asarray(text_global[query_start:query_stop], dtype=np.float32)
                ).to(device),
                dim=-1,
            )
        best_scores = torch.empty((len(queries), 0), device=device)
        best_indices = torch.empty((len(queries), 0), dtype=torch.long, device=device)
        for candidate_start in range(0, candidate_count, candidate_chunk_size):
            candidate_stop = min(candidate_start + candidate_chunk_size, candidate_count)
            candidates = F.normalize(
                torch.from_numpy(
                    np.asarray(
                        image_latents[candidate_start:candidate_stop], dtype=np.float32
                    )
                ).to(device),
                dim=-1,
            )
            scores = torch.einsum("qd,ckd->qck", queries, candidates).amax(dim=-1)
            if query_globals is not None and image_global is not None:
                candidate_globals = F.normalize(
                    torch.from_numpy(
                        np.asarray(
                            image_global[candidate_start:candidate_stop], dtype=np.float32
                        )
                    ).to(device),
                    dim=-1,
                )
                global_scores = query_globals @ candidate_globals.T
                scores = fine_weight * scores + (1.0 - fine_weight) * global_scores
            local_k = min(top_k, scores.shape[1])
            local_scores, local_indices = scores.topk(local_k, dim=1)
            local_indices += candidate_start
            combined_scores = torch.cat([best_scores, local_scores], dim=1)
            combined_indices = torch.cat([best_indices, local_indices], dim=1)
            keep = min(top_k, combined_scores.shape[1])
            best_scores, positions = combined_scores.topk(keep, dim=1)
            best_indices = combined_indices.gather(1, positions)
        output[query_start:query_stop] = best_indices.cpu().numpy()
    return output


@torch.inference_mode()
def gated_coarse_topk(
    text_global: np.ndarray,
    text_latents: np.ndarray,
    text_gates: np.ndarray,
    image_global: np.ndarray,
    image_latents: np.ndarray,
    top_k: int,
    device: torch.device,
    query_batch_size: int = 16,
    candidate_chunk_size: int = 8192,
) -> np.ndarray:
    """Exact full-index global/local mixture without a global prefilter.

    Each text query supplies its own global weight.  Local relevance is the
    MaxSim score against the best of an image's region tokens.
    """
    query_count = len(text_global)
    candidate_count = len(image_global)
    if not (len(text_latents) == len(text_gates) == query_count):
        raise ValueError("text global, latent, and gate arrays must have equal length")
    if len(image_latents) != candidate_count:
        raise ValueError("image global and latent arrays must have equal length")
    top_k = min(top_k, candidate_count)
    output = np.empty((query_count, top_k), dtype=np.int64)
    # Validation/evaluation candidate banks fit comfortably on the A100. Keep
    # them resident so a query batch does not repeatedly transfer the same
    # 8-token features over PCIe.
    candidate_globals = F.normalize(
        torch.from_numpy(np.asarray(image_global, dtype=np.float32)).to(device),
        dim=-1,
    )
    candidate_latents = F.normalize(
        torch.from_numpy(np.asarray(image_latents, dtype=np.float32)).to(device),
        dim=-1,
    )
    for query_start in range(0, query_count, query_batch_size):
        query_stop = min(query_start + query_batch_size, query_count)
        query_globals = F.normalize(torch.from_numpy(np.asarray(
            text_global[query_start:query_stop], dtype=np.float32
        )).to(device), dim=-1)
        queries = F.normalize(torch.from_numpy(np.asarray(
            text_latents[query_start:query_stop], dtype=np.float32
        )).to(device), dim=-1)
        gates = torch.from_numpy(np.asarray(
            text_gates[query_start:query_stop], dtype=np.float32
        )).to(device)[:, None]
        best_scores = torch.empty((len(queries), 0), device=device)
        best_indices = torch.empty((len(queries), 0), dtype=torch.long, device=device)
        for candidate_start in range(0, candidate_count, candidate_chunk_size):
            candidate_stop = min(candidate_start + candidate_chunk_size, candidate_count)
            global_scores = query_globals @ candidate_globals[candidate_start:candidate_stop].T
            local_scores = torch.einsum(
                "qd,ckd->qck",
                queries,
                candidate_latents[candidate_start:candidate_stop],
            ).amax(dim=-1)
            scores = gates * global_scores + (1.0 - gates) * local_scores
            local_k = min(top_k, scores.shape[1])
            local_scores, local_indices = scores.topk(local_k, dim=1)
            local_indices += candidate_start
            combined_scores = torch.cat([best_scores, local_scores], dim=1)
            combined_indices = torch.cat([best_indices, local_indices], dim=1)
            keep = min(top_k, combined_scores.shape[1])
            best_scores, positions = combined_scores.topk(keep, dim=1)
            best_indices = combined_indices.gather(1, positions)
        output[query_start:query_stop] = best_indices.cpu().numpy()
    return output


@torch.inference_mode()
def late_interaction_prefilter_topk(
    text_latents: np.ndarray,
    image_latents: np.ndarray,
    text_global: np.ndarray,
    image_global: np.ndarray,
    top_k: int,
    prefilter_k: int,
    fine_weight: float,
    device: torch.device,
    query_batch_size: int = 8,
) -> np.ndarray:
    """Reproduce deployed global prefiltering followed by latent MaxSim reranking."""
    candidate_count = len(image_latents)
    top_k = min(top_k, candidate_count)
    prefilter_k = min(max(top_k, prefilter_k), candidate_count)
    output = np.empty((len(text_latents), top_k), dtype=np.int64)
    candidate_globals = F.normalize(
        torch.from_numpy(np.asarray(image_global, dtype=np.float32)).to(device),
        dim=-1,
    )
    for start in range(0, len(text_latents), query_batch_size):
        stop = min(start + query_batch_size, len(text_latents))
        query_globals = F.normalize(
            torch.from_numpy(
                np.asarray(text_global[start:stop], dtype=np.float32)
            ).to(device),
            dim=-1,
        )
        coarse_scores, coarse_indices = (query_globals @ candidate_globals.T).topk(
            prefilter_k, dim=1
        )
        candidate_indices = coarse_indices.cpu().numpy()
        candidates = F.normalize(
            torch.from_numpy(
                np.asarray(image_latents[candidate_indices], dtype=np.float32)
            ).to(device),
            dim=-1,
        )
        queries = F.normalize(
            torch.from_numpy(
                np.asarray(text_latents[start:stop], dtype=np.float32)
            ).to(device),
            dim=-1,
        )
        fine_scores = torch.einsum("qd,qckd->qck", queries, candidates).amax(
            dim=-1
        )
        scores = fine_weight * fine_scores + (1.0 - fine_weight) * coarse_scores
        positions = scores.topk(top_k, dim=1).indices
        output[start:stop] = coarse_indices.gather(1, positions).cpu().numpy()
    return output


@torch.inference_mode()
def paired_recall(
    queries: np.ndarray,
    candidates: np.ndarray,
    k_values: tuple[int, ...],
    batch_size: int = 256,
) -> dict[str, float]:
    if len(queries) != len(candidates):
        raise ValueError("paired retrieval requires equal query and candidate counts")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_values = F.normalize(torch.from_numpy(np.asarray(candidates, dtype=np.float32)).to(device), dim=-1)
    hits = {k: 0 for k in k_values}
    for start in range(0, len(queries), batch_size):
        query = F.normalize(torch.from_numpy(np.asarray(queries[start:start + batch_size], dtype=np.float32)).to(device), dim=-1)
        scores = query @ candidate_values.T
        top = scores.topk(min(max(k_values), len(candidates)), dim=1).indices.cpu().numpy()
        truth = np.arange(start, start + len(query))[:, None]
        for k in k_values:
            hits[k] += int(np.any(top[:, :k] == truth, axis=1).sum())
    return {f"exact_R@{k}": hits[k] / max(1, len(queries)) for k in k_values}
