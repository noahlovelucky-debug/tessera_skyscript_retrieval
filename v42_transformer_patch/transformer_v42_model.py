"""Transformer visual replacement for the v4.2 hierarchical retrieval model."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .hierarchical_v42_model import DualSpaceTextTower
from .model import ResidualMLP


class SpatialTransformerFusion(nn.Module):
    """Encode the complete 16x16 S1/S2 grid before producing scene and local tokens."""

    def __init__(self, layers: int, heads: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(768)
        self.input_projection = nn.Linear(768, 384)
        self.class_token = nn.Parameter(torch.randn(1, 1, 384) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(1, 257, 384) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=384,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(384)

    def forward(self, direct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if direct.shape[1:] != (16, 16, 768):
            raise ValueError(f"expected [B,16,16,768], got {tuple(direct.shape)}")
        tokens = self.input_projection(self.input_norm(direct.float())).flatten(1, 2)
        cls = self.class_token.expand(len(tokens), -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1) + self.position_embedding)
        global_vector = F.normalize(self.output_norm(encoded[:, 0]), dim=-1)
        local_tokens = F.normalize(self.output_norm(encoded[:, 1:]), dim=-1).reshape(-1, 16, 16, 384)
        return global_vector, local_tokens


class TransformerV42Model(nn.Module):
    """v4.2 objective with the image MLP paths replaced by a spatial Transformer."""

    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        hidden_dim = int(model_cfg.get("mlp_hidden_dim", 1536))
        dropout = float(model_cfg.get("dropout", 0.1))
        self.text = DualSpaceTextTower(model_cfg["text_model"], hidden_dim, dropout)
        self.spatial_fusion = SpatialTransformerFusion(
            int(model_cfg.get("transformer_layers", 4)),
            int(model_cfg.get("transformer_heads", 12)),
            int(model_cfg.get("transformer_feedforward_dim", 1536)),
            dropout,
        )
        self.local_retrieval_adapter = ResidualMLP(384, dropout)
        self.localization_token_adapter = ResidualMLP(384, dropout)
        self.local_sigma = float(model_cfg.get("local_sigma", 1.0))
        self.num_prototypes = int(model_cfg.get("prototypes_per_class", 3))
        self.global_prototypes = nn.Parameter(torch.empty(num_classes, self.num_prototypes, 384))
        self.local_prototypes = nn.Parameter(torch.empty(num_classes, self.num_prototypes, 384))
        nn.init.normal_(self.global_prototypes, std=0.02)
        nn.init.normal_(self.local_prototypes, std=0.02)
        temperature = float(model_cfg.get("temperature", 0.07))
        self.global_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.local_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.localization_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / float(model_cfg.get("localization_temperature", 0.07)))))
        self.teacher_image_adapter: ResidualMLP | None = None
        teacher_path = model_cfg.get("tessera_teacher_checkpoint")
        if teacher_path:
            checkpoint = torch.load(Path(teacher_path), map_location="cpu", weights_only=False)
            self.teacher_image_adapter = ResidualMLP(384, dropout)
            state = {key.removeprefix("image_adapter."): value for key, value in checkpoint["model_state"].items() if key.startswith("image_adapter.")}
            self.teacher_image_adapter.load_state_dict(state, strict=True)
            for parameter in self.teacher_image_adapter.parameters():
                parameter.requires_grad = False
        base_path = model_cfg.get("base_checkpoint")
        if base_path:
            self.initialize_from_v42(Path(base_path))

    @torch.no_grad()
    def initialize_from_v42(self, path: Path) -> None:
        """Reuse every compatible v4.2 component; the visual Transformer is new."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state"]
        def subset(prefix: str) -> dict[str, torch.Tensor]:
            return {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
        self.text.encoder.load_state_dict(subset("text.encoder."), strict=True)
        self.text.global_projection.load_state_dict(subset("text.global_projection."), strict=True)
        self.text.local_projection.load_state_dict(subset("text.local_projection."), strict=True)
        self.local_retrieval_adapter.load_state_dict(subset("local_retrieval_adapter."), strict=True)
        self.localization_token_adapter.load_state_dict(subset("localization_token_adapter."), strict=True)
        self.global_prototypes.copy_(state["global_prototypes"])
        self.local_prototypes.copy_(state["local_prototypes"])
        for name in ("global_logit_scale", "local_logit_scale", "localization_logit_scale"):
            getattr(self, name).copy_(state[name])

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.text(input_ids, attention_mask)

    def pool_3x3(self, tokens: torch.Tensor, token_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = tokens.flatten(1, 2)
        rows = token_indices // 16
        cols = token_indices % 16
        offsets = torch.arange(-1, 2, device=tokens.device)
        row_offsets, col_offsets = torch.meshgrid(offsets, offsets, indexing="ij")
        row_offsets, col_offsets = row_offsets.flatten(), col_offsets.flatten()
        target_rows, target_cols = rows[:, None] + row_offsets, cols[:, None] + col_offsets
        valid = (target_rows >= 0) & (target_rows < 16) & (target_cols >= 0) & (target_cols < 16)
        locations = target_rows.clamp(0, 15) * 16 + target_cols.clamp(0, 15)
        neighbors = flat.gather(1, locations.unsqueeze(-1).expand(-1, -1, 384))
        distance_sq = row_offsets.float().square() + col_offsets.float().square()
        weights = torch.exp(-distance_sq / (2.0 * self.local_sigma**2))[None] * valid.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        exact = flat[torch.arange(len(flat), device=tokens.device), token_indices]
        local = F.normalize(self.local_retrieval_adapter(exact + 0.5 * (neighbors * weights.unsqueeze(-1)).sum(dim=1)), dim=-1)
        return F.normalize(exact, dim=-1), local

    def encode_images(self, s2: torch.Tensor, s1: torch.Tensor, token_indices: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if s2.shape != s1.shape or s2.ndim != 4 or s2.shape[-1] != 384:
            raise ValueError(f"expected matching [B,16,16,384] inputs, got {s2.shape} and {s1.shape}")
        global_image, local_tokens = self.spatial_fusion(torch.cat((s2.float(), s1.float()), dim=-1))
        output = {"global": global_image, "local_tokens": local_tokens, "localization_tokens": F.normalize(self.localization_token_adapter(local_tokens.detach()), dim=-1)}
        if token_indices is not None:
            exact, local = self.pool_3x3(local_tokens, token_indices)
            output.update({"poi_patch": exact, "local": local})
        return output

    def localization_logits(self, local_text: torch.Tensor, localization_tokens: torch.Tensor) -> torch.Tensor:
        return self.localization_logit_scale.exp().clamp(max=100.0) * torch.einsum("bd,bhwd->bhw", local_text, localization_tokens)

    @torch.no_grad()
    def initialize_prototypes(self, global_vectors: torch.Tensor, local_vectors: torch.Tensor) -> None:
        for source, target in ((global_vectors, self.global_prototypes), (local_vectors, self.local_prototypes)):
            values = F.normalize(source.float(), dim=-1)[:, None, :].expand_as(target).clone()
            values.add_(0.01 * torch.randn_like(values))
            target.copy_(F.normalize(values, dim=-1))

    def encode_teacher(self, tessera_features: torch.Tensor) -> torch.Tensor:
        if self.teacher_image_adapter is None:
            raise RuntimeError("no TESSERA teacher adapter was configured")
        self.teacher_image_adapter.eval()
        return F.normalize(self.teacher_image_adapter(tessera_features), dim=-1)

    def clamp_logit_scales(self) -> None:
        with torch.no_grad():
            for value in (self.global_logit_scale, self.local_logit_scale, self.localization_logit_scale):
                value.clamp_(0.0, math.log(100.0))
