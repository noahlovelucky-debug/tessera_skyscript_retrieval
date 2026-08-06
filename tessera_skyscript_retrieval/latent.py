from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .model import LatentRetrievalModel


@torch.inference_mode()
def encode_latent_features(
    model: LatentRetrievalModel,
    descriptors: np.ndarray,
    teacher_highres: np.ndarray,
    region_tokens: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(descriptors)
    common_dim = int(teacher_highres.shape[-1])
    latent_count = model.highres_adapter.latent_count
    latent_dim = model.highres_adapter.latent_dim
    tessera = np.empty((count, common_dim), dtype=np.float32)
    highres_global = np.empty((count, common_dim), dtype=np.float32)
    highres_latents = np.empty((count, latent_count, latent_dim), dtype=np.float16)
    model.eval()
    for start in tqdm(range(0, count, batch_size), desc="encode-latent-v2"):
        stop = min(start + batch_size, count)
        descriptor_batch = torch.from_numpy(
            np.asarray(descriptors[start:stop], dtype=np.float32)
        ).to(device)
        teacher_batch = torch.from_numpy(
            np.asarray(teacher_highres[start:stop], dtype=np.float32)
        ).to(device)
        token_batch = torch.from_numpy(
            np.asarray(region_tokens[start:stop], dtype=np.float32)
        ).to(device)
        tessera_batch = model.encode_tessera(descriptor_batch)
        highres_batch, latent_batch = model.encode_highres(token_batch, teacher_batch)
        tessera[start:stop] = tessera_batch.cpu().numpy()
        highres_global[start:stop] = highres_batch.cpu().numpy()
        highres_latents[start:stop] = latent_batch.cpu().numpy().astype(np.float16)
    return tessera, highres_global, highres_latents


@torch.inference_mode()
def encode_text_latents(
    model: LatentRetrievalModel,
    text_global: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    output = np.empty(
        (len(text_global), model.highres_adapter.latent_dim), dtype=np.float32
    )
    model.eval()
    for start in range(0, len(text_global), batch_size):
        stop = min(start + batch_size, len(text_global))
        values = torch.from_numpy(
            np.asarray(text_global[start:stop], dtype=np.float32)
        ).to(device)
        output[start:stop] = model.encode_text_latent(values).cpu().numpy()
    return output


@torch.inference_mode()
def encode_text_gates(
    model: LatentRetrievalModel,
    text_global: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    output = np.empty(len(text_global), dtype=np.float32)
    model.eval()
    for start in range(0, len(text_global), batch_size):
        stop = min(start + batch_size, len(text_global))
        values = torch.from_numpy(
            np.asarray(text_global[start:stop], dtype=np.float32)
        ).to(device)
        output[start:stop] = model.encode_text_gate(values).cpu().numpy()
    return output
