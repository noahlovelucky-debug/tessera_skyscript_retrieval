from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class TesseraAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, temperature: float):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(descriptors.float()), dim=-1)

    def clamp_temperature(self) -> None:
        with torch.no_grad():
            self.logit_scale.clamp_(0.0, math.log(100.0))


class TesseraBoxTextAdapter(nn.Module):
    """Pure TESSERA-v1 retrieval adapter with normalized bbox geometry.

    The requested 768->1024->384 MLP is preceded only by the necessary
    2816+7->768 input projection because TESSERA v1 descriptors are 2816D.
    """

    def __init__(self, descriptor_dim: int, box_dim: int, dropout: float, temperature: float) -> None:
        super().__init__()
        self.image_projection = nn.Sequential(
            nn.LayerNorm(descriptor_dim + box_dim),
            nn.Linear(descriptor_dim + box_dim, 768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 384),
        )
        self.text_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 384))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def encode_image(self, descriptors: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_projection(torch.cat((descriptors.float(), boxes.float()), dim=-1)), dim=-1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_projection(text.float()), dim=-1)

    def clamp_temperature(self) -> None:
        with torch.no_grad():
            self.logit_scale.clamp_(0.0, math.log(100.0))


def _transformer_encoder(
    width: int,
    heads: int,
    feedforward_dim: int,
    layers: int,
    dropout: float,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=feedforward_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class DeepTesseraAdapter(nn.Module):
    """Model the pooled TESSERA descriptor as spatial tokens before global pooling."""

    def __init__(
        self,
        descriptor_dim: int,
        token_dim: int,
        width: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if descriptor_dim % token_dim:
            raise ValueError(f"descriptor_dim={descriptor_dim} is not divisible by token_dim={token_dim}")
        self.descriptor_dim = descriptor_dim
        self.token_dim = token_dim
        self.token_count = descriptor_dim // token_dim
        self.input_norm = nn.LayerNorm(token_dim)
        self.input_projection = nn.Linear(token_dim, width)
        self.global_token = nn.Parameter(torch.randn(1, 1, width) * (width ** -0.5))
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.token_count + 1, width) * (width ** -0.5)
        )
        self.encoder = _transformer_encoder(
            width, heads, feedforward_dim, layers, dropout
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        values = descriptors.float().reshape(-1, self.token_count, self.token_dim)
        values = self.input_projection(self.input_norm(values))
        global_token = self.global_token.expand(len(values), -1, -1)
        values = torch.cat([global_token, values], dim=1) + self.position_embedding
        values = self.encoder(values)
        return F.normalize(self.output_projection(values[:, 0]), dim=-1)


class HighResLatentAdapter(nn.Module):
    """Refine frozen SkyCLIP region tokens and score their retrieval role from text."""

    def __init__(
        self,
        latent_count: int,
        latent_dim: int,
        text_dim: int,
        common_dim: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        gate_hidden_dim: int,
        gate_mode: str,
        gate_base_weight: float,
        gate_max_delta: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_count = latent_count
        self.latent_dim = latent_dim
        self.input_norm = nn.LayerNorm(latent_dim)
        self.slot_embedding = nn.Parameter(
            torch.randn(1, latent_count, latent_dim) * (latent_dim ** -0.5)
        )
        self.encoder = _transformer_encoder(
            latent_dim, heads, feedforward_dim, layers, dropout
        )
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, latent_dim),
        )
        self.gate_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )
        if gate_mode not in {"sigmoid", "anchored_tanh"}:
            raise ValueError(f"unsupported gate mode: {gate_mode}")
        if not 0.0 <= gate_base_weight <= 1.0:
            raise ValueError("gate_base_weight must be in [0, 1]")
        if gate_max_delta < 0.0 or gate_base_weight - gate_max_delta < 0.0 or gate_base_weight + gate_max_delta > 1.0:
            raise ValueError("gate_base_weight +/- gate_max_delta must remain in [0, 1]")
        self.gate_mode = gate_mode
        self.gate_base_weight = gate_base_weight
        self.gate_max_delta = gate_max_delta
        self.global_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, common_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def encode_image(
        self,
        region_tokens: torch.Tensor,
        teacher_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if region_tokens.shape[1:] != (self.latent_count, self.latent_dim):
            raise ValueError(
                f"expected Bx{self.latent_count}x{self.latent_dim} region tokens, "
                f"got {tuple(region_tokens.shape)}"
            )
        values = self.input_norm(region_tokens.float()) + self.slot_embedding
        values = self.encoder(values)
        latents = F.normalize(self.latent_norm(values), dim=-1)
        delta = self.global_projection(values.mean(dim=1))
        scale = torch.tanh(self.residual_scale)
        global_feature = F.normalize(teacher_global.float() + scale * delta, dim=-1)
        return global_feature, latents

    def encode_text(self, text_global: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_projection(text_global.float()), dim=-1)

    def encode_gate(self, text_global: torch.Tensor) -> torch.Tensor:
        """Return the text-conditioned global-score weight in [0, 1]."""
        logits = self.gate_projection(text_global.float()).squeeze(-1)
        if self.gate_mode == "anchored_tanh":
            return self.gate_base_weight + self.gate_max_delta * torch.tanh(logits)
        return torch.sigmoid(logits)

    def initialize_anchor_gate(self) -> None:
        """Make the anchored model start at the proven fixed global/local blend."""
        if self.gate_mode != "anchored_tanh":
            return
        final = self.gate_projection[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)


class LatentRetrievalModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        tessera_cfg = config["tessera"]
        self.tessera_adapter = DeepTesseraAdapter(
            descriptor_dim=int(tessera_cfg["descriptor_dim"]),
            token_dim=int(tessera_cfg["embedding_dim"]),
            width=int(model_cfg["tessera_width"]),
            layers=int(model_cfg["tessera_layers"]),
            heads=int(model_cfg["tessera_heads"]),
            feedforward_dim=int(model_cfg["tessera_feedforward_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            output_dim=int(model_cfg["common_dim"]),
            dropout=float(model_cfg["dropout"]),
        )
        self.highres_adapter = HighResLatentAdapter(
            latent_count=int(model_cfg["latent_count"]),
            latent_dim=int(model_cfg["latent_dim"]),
            text_dim=int(model_cfg["common_dim"]),
            common_dim=int(model_cfg["common_dim"]),
            layers=int(model_cfg["highres_layers"]),
            heads=int(model_cfg["highres_heads"]),
            feedforward_dim=int(model_cfg["highres_feedforward_dim"]),
            gate_hidden_dim=int(model_cfg.get("gate_hidden_dim", model_cfg["hidden_dim"])),
            gate_mode=str(model_cfg.get("gate_mode", "sigmoid")),
            gate_base_weight=float(model_cfg.get("gate_base_weight", 0.5)),
            gate_max_delta=float(model_cfg.get("gate_max_delta", 0.5)),
            dropout=float(model_cfg["dropout"]),
        )
        temperature = float(model_cfg["temperature"])
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.fine_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(int(model_cfg["common_dim"])),
            nn.Linear(int(model_cfg["common_dim"]), int(model_cfg.get("fusion_gate_hidden_dim", 512))),
            nn.GELU(),
            nn.Linear(int(model_cfg.get("fusion_gate_hidden_dim", 512)), 3),
        )
        self.initialize_fusion_gate()

    def encode_tessera(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.tessera_adapter(descriptors)

    def encode_highres(
        self,
        region_tokens: torch.Tensor,
        teacher_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.highres_adapter.encode_image(region_tokens, teacher_global)

    def encode_text_latent(self, text_global: torch.Tensor) -> torch.Tensor:
        return self.highres_adapter.encode_text(text_global)

    def encode_text_gate(self, text_global: torch.Tensor) -> torch.Tensor:
        return self.highres_adapter.encode_gate(text_global)

    def encode_fusion_weights(self, text_global: torch.Tensor) -> torch.Tensor:
        """Text-conditioned weights for Sentinel global, high-res global, and local tokens."""
        return torch.softmax(self.fusion_gate(text_global.float()), dim=-1)

    def initialize_fusion_gate(self) -> None:
        """Start a new fusion head from an unbiased three-way mixture."""
        final = self.fusion_gate[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        descriptors: torch.Tensor,
        region_tokens: torch.Tensor,
        teacher_global: torch.Tensor,
        text_global: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """DDP-safe joint encoding path used by tri-modal adapter training."""
        highres_global, highres_latents = self.encode_highres(
            region_tokens, teacher_global
        )
        return {
            "tessera": self.encode_tessera(descriptors),
            "highres_global": highres_global,
            "highres_latents": highres_latents,
            "text_latent": self.encode_text_latent(text_global),
            "text_gate": self.encode_text_gate(text_global),
            "fusion_weights": self.encode_fusion_weights(text_global),
        }

    def clamp_temperature(self) -> None:
        with torch.no_grad():
            self.logit_scale.clamp_(0.0, math.log(100.0))
            self.fine_logit_scale.clamp_(0.0, math.log(100.0))


def build_adapter(config: dict) -> TesseraAdapter:
    return TesseraAdapter(
        input_dim=int(config["tessera"]["descriptor_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        output_dim=int(config["model"]["common_dim"]),
        dropout=float(config["model"]["dropout"]),
        temperature=float(config["model"]["temperature"]),
    )


def build_tessera_box_adapter(config: dict) -> TesseraBoxTextAdapter:
    return TesseraBoxTextAdapter(
        descriptor_dim=int(config["tessera"]["descriptor_dim"]),
        box_dim=int(config["model"].get("box_feature_dim", 7)),
        dropout=float(config["model"]["dropout"]),
        temperature=float(config["model"]["temperature"]),
    )


def build_latent_model(config: dict) -> LatentRetrievalModel:
    return LatentRetrievalModel(config)
