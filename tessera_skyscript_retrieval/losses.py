from __future__ import annotations

import torch
import torch.nn.functional as F


def directional_multi_positive(logits: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    return _directional_multi_positive_per_query(logits, positives).mean()


def _directional_multi_positive_per_query(
    logits: torch.Tensor, positives: torch.Tensor
) -> torch.Tensor:
    positives = positives.bool()
    if not bool(positives.any(dim=1).all()):
        raise ValueError("every anchor must have at least one positive")
    numerator = torch.logsumexp(logits.masked_fill(~positives, float("-inf")), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return denominator - numerator


def symmetric_multi_positive(logits: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        directional_multi_positive(logits, positives)
        + directional_multi_positive(logits.T, positives.T)
    )


def _distill_distribution(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    target = F.softmax(teacher.detach(), dim=-1)
    return F.kl_div(F.log_softmax(student, dim=-1), target, reduction="batchmean")


def alignment_losses(
    tessera: torch.Tensor,
    highres: torch.Tensor,
    text: torch.Tensor,
    title_ids: torch.Tensor,
    logit_scale: torch.Tensor,
    weights: dict,
) -> dict[str, torch.Tensor]:
    scale = logit_scale.exp().clamp(max=100.0)
    positives = title_ids[:, None].eq(title_ids[None, :])
    student_logits = scale * text @ tessera.T
    semantic = symmetric_multi_positive(student_logits, positives)
    pair_distill = (1.0 - F.cosine_similarity(tessera, highres, dim=-1)).mean()
    teacher_scale = 1.0 / 0.07
    teacher_logits = teacher_scale * text @ highres.T
    relation = 0.5 * (
        _distill_distribution(student_logits, teacher_logits)
        + _distill_distribution(student_logits.T, teacher_logits.T)
    )
    total = (
        float(weights["semantic_weight"]) * semantic
        + float(weights["pair_distill_weight"]) * pair_distill
        + float(weights["relation_distill_weight"]) * relation
    )
    return {
        "total": total,
        "semantic": semantic,
        "pair_distill": pair_distill,
        "relation_distill": relation,
    }


def late_interaction_scores(text: torch.Tensor, image_latents: torch.Tensor) -> torch.Tensor:
    """Score every text against every image using its best matching visual latent."""
    return torch.einsum("qd,ckd->qck", text, image_latents).amax(dim=-1)


def _latent_diversity(image_latents: torch.Tensor) -> torch.Tensor:
    similarities = image_latents @ image_latents.transpose(1, 2)
    count = similarities.shape[-1]
    mask = ~torch.eye(count, dtype=torch.bool, device=similarities.device)
    return similarities[:, mask].square().mean()


def latent_alignment_losses(
    tessera: torch.Tensor,
    highres_global: torch.Tensor,
    highres_latents: torch.Tensor,
    teacher_highres: torch.Tensor,
    text_global: torch.Tensor,
    text_latent: torch.Tensor,
    text_gate: torch.Tensor,
    fusion_weights: torch.Tensor | None,
    title_ids: torch.Tensor,
    logit_scale: torch.Tensor,
    fine_logit_scale: torch.Tensor,
    weights: dict,
) -> dict[str, torch.Tensor]:
    positives = title_ids[:, None].eq(title_ids[None, :])
    scale = logit_scale.exp().clamp(max=100.0)
    fine_scale = fine_logit_scale.exp().clamp(max=100.0)
    tessera_logits = scale * text_global @ tessera.T
    highres_logits = scale * text_global @ highres_global.T
    fine_logits = fine_scale * late_interaction_scores(text_latent, highres_latents)
    gated_logits = text_gate[:, None] * highres_logits + (1.0 - text_gate[:, None]) * fine_logits
    fusion_logits = None
    if fusion_weights is not None:
        fusion_logits = (
            fusion_weights[:, 0:1] * tessera_logits
            + fusion_weights[:, 1:2] * highres_logits
            + fusion_weights[:, 2:3] * fine_logits
        )
    tessera_semantic = symmetric_multi_positive(tessera_logits, positives)
    highres_semantic = symmetric_multi_positive(highres_logits, positives)
    fine_semantic = symmetric_multi_positive(fine_logits, positives)
    gated_semantic = symmetric_multi_positive(gated_logits, positives)
    fusion_semantic = (
        symmetric_multi_positive(fusion_logits, positives)
        if fusion_logits is not None else torch.zeros((), device=text_global.device)
    )
    pair_distill = (1.0 - F.cosine_similarity(tessera, highres_global, dim=-1)).mean()
    teacher_preservation = (
        1.0 - F.cosine_similarity(highres_global, teacher_highres, dim=-1)
    ).mean()
    teacher_logits = (1.0 / 0.07) * text_global @ teacher_highres.T
    relation = 0.5 * (
        _distill_distribution(tessera_logits, teacher_logits)
        + _distill_distribution(tessera_logits.T, teacher_logits.T)
    )
    latent_diversity = _latent_diversity(highres_latents)
    gate_base_weight = float(weights.get("gate_base_weight", weights.get("gate_target_mean", 0.5)))
    gate_max_delta = float(weights.get("gate_max_delta", 0.5))
    gate_balance = (text_gate.mean() - gate_base_weight).square()
    # The more discriminative branch for each text is a soft, detached routing
    # target. This prevents the gate from converging to a single fixed blend.
    route_temperature = float(weights.get("gate_route_temperature", 1.0))
    global_route_loss = _directional_multi_positive_per_query(highres_logits, positives)
    local_route_loss = _directional_multi_positive_per_query(fine_logits, positives)
    route_probability = torch.softmax(
        torch.stack([-global_route_loss, -local_route_loss], dim=-1) / route_temperature,
        dim=-1,
    )[:, 0].detach()
    # Anchored gates preserve a validated fixed blend unless branch evidence
    # supports a bounded text-specific adjustment.
    route_target = gate_base_weight + gate_max_delta * (2.0 * route_probability - 1.0)
    route_target = route_target.clamp(0.0, 1.0)
    gate_route = F.mse_loss(text_gate, route_target)
    fusion_route = torch.zeros((), device=text_global.device)
    if fusion_weights is not None:
        sentinel_route_loss = _directional_multi_positive_per_query(
            tessera_logits, positives
        )
        fusion_target = torch.softmax(
            torch.stack(
                [-sentinel_route_loss, -global_route_loss, -local_route_loss], dim=-1
            )
            / route_temperature,
            dim=-1,
        ).detach()
        fusion_route = F.kl_div(
            fusion_weights.clamp_min(1e-8).log(), fusion_target, reduction="batchmean"
        )
    total = (
        float(weights["tessera_semantic_weight"]) * tessera_semantic
        + float(weights["highres_semantic_weight"]) * highres_semantic
        + float(weights["fine_semantic_weight"]) * fine_semantic
        + float(weights.get("gated_semantic_weight", 0.0)) * gated_semantic
        + float(weights.get("fusion_semantic_weight", 0.0)) * fusion_semantic
        + float(weights["pair_distill_weight"]) * pair_distill
        + float(weights["relation_distill_weight"]) * relation
        + float(weights["teacher_preservation_weight"]) * teacher_preservation
        + float(weights["latent_diversity_weight"]) * latent_diversity
        + float(weights.get("gate_balance_weight", 0.0)) * gate_balance
        + float(weights.get("gate_route_weight", 0.0)) * gate_route
        + float(weights.get("fusion_route_weight", 0.0)) * fusion_route
    )
    return {
        "total": total,
        "tessera_semantic": tessera_semantic,
        "highres_semantic": highres_semantic,
        "fine_semantic": fine_semantic,
        "gated_semantic": gated_semantic,
        "fusion_semantic": fusion_semantic,
        "pair_distill": pair_distill,
        "relation_distill": relation,
        "teacher_preservation": teacher_preservation,
        "latent_diversity": latent_diversity,
        "gate_balance": gate_balance,
        "gate_mean": text_gate.mean(),
        "gate_std": text_gate.std(unbiased=False),
        "gate_route": gate_route,
        "gate_target_mean": route_target.mean(),
        "fusion_route": fusion_route,
    }
