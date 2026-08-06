from __future__ import annotations

import numpy as np
import torch

from tessera_skyscript_retrieval.data import normalize_title, stable_bucket
from tessera_skyscript_retrieval.losses import alignment_losses
from tessera_skyscript_retrieval.metrics import (
    gated_coarse_topk,
    late_interaction_prefilter_topk,
    late_interaction_topk,
    semantic_retrieval_metrics,
)
from tessera_skyscript_retrieval.model import LatentRetrievalModel, TesseraAdapter
from tessera_skyscript_retrieval.pooling import tessera_descriptor


def test_pyramid_descriptor_shape_and_finiteness():
    chip = np.arange(3 * 5 * 128, dtype=np.float32).reshape(3, 5, 128)
    descriptor = tessera_descriptor(chip, (1, 2, 4), True)
    assert descriptor.shape == (2816,)
    assert np.isfinite(descriptor).all()


def test_small_chip_can_be_adaptively_pooled():
    chip = np.ones((1, 1, 128), dtype=np.float16)
    descriptor = tessera_descriptor(chip, (1, 2, 4), True)
    assert descriptor.shape == (2816,)
    assert np.allclose(descriptor[: 21 * 128], 1.0)
    assert np.allclose(descriptor[-128:], 0.0)


def test_adapter_outputs_normalized_vectors_and_gradients():
    model = TesseraAdapter(2816, 64, 32, 0.0, 0.07)
    output = model(torch.randn(4, 2816))
    assert output.shape == (4, 32)
    assert torch.allclose(output.norm(dim=-1), torch.ones(4), atol=1e-5)
    output.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_multi_positive_loss_accepts_duplicate_titles():
    torch.manual_seed(1)
    tessera = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    highres = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    losses = alignment_losses(
        tessera, highres, text, torch.tensor([1, 1, 2, 2]),
        torch.tensor(2.0),
        {"semantic_weight": 1.0, "pair_distill_weight": 0.5, "relation_distill_weight": 0.25},
    )
    assert losses["total"].isfinite()


def test_hash_and_title_normalization_are_stable():
    assert stable_bucket("grid", 42) == stable_bucket("grid", 42)
    assert normalize_title("  An   aerial image.  ") == "An aerial image."


def test_precision_and_ndcg_respect_relevance_and_rank():
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    candidates = np.array([[1.0, 0.0], [0.9, 0.43589], [0.8, 0.6]], dtype=np.float32)
    result = semantic_retrieval_metrics(
        queries,
        np.array([1]),
        candidates,
        np.array([1, 0, 1]),
        (2, 3),
    )
    assert result["Hit@2"] == 1.0
    assert result["Precision@2"] == 0.5
    assert np.isclose(result["nDCG@2"], 1.0 / (1.0 + 1.0 / np.log2(3)))
    assert np.isclose(result["Precision@3"], 2.0 / 3.0)
    assert np.isclose(result["nDCG@3"], 1.5 / (1.0 + 1.0 / np.log2(3)))


def test_latent_v2_shapes_normalization_and_gradients():
    config = {
        "tessera": {"descriptor_dim": 32, "embedding_dim": 8},
        "model": {
            "common_dim": 16,
            "hidden_dim": 24,
            "latent_count": 2,
            "latent_dim": 8,
            "highres_layers": 1,
            "highres_heads": 2,
            "highres_feedforward_dim": 16,
            "tessera_width": 16,
            "tessera_layers": 1,
            "tessera_heads": 2,
            "tessera_feedforward_dim": 32,
            "dropout": 0.0,
            "temperature": 0.07,
        },
    }
    model = LatentRetrievalModel(config)
    tessera = model.encode_tessera(torch.randn(4, 32))
    highres, latents = model.encode_highres(
        torch.randn(4, 2, 8), torch.randn(4, 16)
    )
    text = model.encode_text_latent(torch.randn(4, 16))
    gates = model.encode_text_gate(torch.randn(4, 16))
    assert tessera.shape == (4, 16)
    assert highres.shape == (4, 16)
    assert latents.shape == (4, 2, 8)
    assert text.shape == (4, 8)
    assert gates.shape == (4,)
    assert torch.all((gates >= 0.0) & (gates <= 1.0))
    assert torch.allclose(tessera.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(latents.norm(dim=-1), torch.ones(4, 2), atol=1e-5)
    (tessera.sum() + highres.sum() + latents.sum() + text.sum() + gates.sum()).backward()
    assert any(
        parameter.grad is not None
        for parameter in model.tessera_adapter.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.highres_adapter.parameters()
    )


def test_anchored_gate_starts_at_fixed_blend_and_remains_bounded():
    config = {
        "tessera": {"descriptor_dim": 32, "embedding_dim": 8},
        "model": {
            "common_dim": 16,
            "hidden_dim": 24,
            "latent_count": 2,
            "latent_dim": 8,
            "highres_layers": 1,
            "highres_heads": 2,
            "highres_feedforward_dim": 16,
            "tessera_width": 16,
            "tessera_layers": 1,
            "tessera_heads": 2,
            "tessera_feedforward_dim": 32,
            "dropout": 0.0,
            "temperature": 0.07,
            "gate_mode": "anchored_tanh",
            "gate_base_weight": 0.35,
            "gate_max_delta": 0.20,
        },
    }
    model = LatentRetrievalModel(config)
    model.highres_adapter.initialize_anchor_gate()
    gates = model.encode_text_gate(torch.randn(32, 16))
    assert torch.allclose(gates, torch.full_like(gates, 0.35), atol=1e-6)
    with torch.no_grad():
        model.highres_adapter.gate_projection[-1].weight.fill_(10.0)
    gates = model.encode_text_gate(torch.randn(32, 16))
    assert torch.all((gates >= 0.15) & (gates <= 0.55))


def test_prefiltered_late_interaction_matches_full_ranking_when_prefilter_is_all():
    rng = np.random.default_rng(11)
    text_latents = rng.normal(size=(3, 8)).astype(np.float32)
    image_latents = rng.normal(size=(7, 2, 8)).astype(np.float32)
    text_global = rng.normal(size=(3, 4)).astype(np.float32)
    image_global = rng.normal(size=(7, 4)).astype(np.float32)
    full = late_interaction_topk(
        text_latents,
        image_latents,
        4,
        torch.device("cpu"),
        text_global=text_global,
        image_global=image_global,
        fine_weight=0.65,
    )
    prefiltered = late_interaction_prefilter_topk(
        text_latents,
        image_latents,
        text_global,
        image_global,
        4,
        7,
        0.65,
        torch.device("cpu"),
        query_batch_size=2,
    )
    assert np.array_equal(prefiltered, full)


def test_gated_coarse_matches_global_and_local_extremes():
    global_query = np.array([[1.0, 0.0]], dtype=np.float32)
    local_query = np.array([[1.0, 0.0]], dtype=np.float32)
    image_global = np.array(
        [[0.1, 1.0], [1.0, 0.0], [0.7, 0.7]], dtype=np.float32
    )
    image_latents = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.1, 1.0], [0.0, 1.0]],
            [[0.7, 0.7], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    global_only = gated_coarse_topk(
        global_query,
        local_query,
        np.array([1.0], dtype=np.float32),
        image_global,
        image_latents,
        3,
        torch.device("cpu"),
        query_batch_size=1,
        candidate_chunk_size=2,
    )
    local_only = gated_coarse_topk(
        global_query,
        local_query,
        np.array([0.0], dtype=np.float32),
        image_global,
        image_latents,
        3,
        torch.device("cpu"),
        query_batch_size=1,
        candidate_chunk_size=2,
    )
    assert global_only[0, 0] == 1
    assert local_only[0, 0] == 0
