from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import ensure_dir, uses_latent_tokens
from .data import load_prepared


class ImageRows(Dataset):
    def __init__(self, paths: list[str], row_ids: np.ndarray, transform) -> None:
        self.paths = paths
        self.row_ids = row_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            value = self.transform(image.convert("RGB"))
        return value, int(self.row_ids[index])


def _resolve_checkpoint(config: dict[str, Any]) -> Path:
    configured = Path(config["skyclip"]["checkpoint"])
    if configured.is_file():
        return configured
    extracted = Path(config["skyclip"]["checkpoint_zip"]).with_suffix("")
    candidates = sorted(extracted.rglob("*.pt")) if extracted.exists() else []
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"SkyCLIP checkpoint is unavailable; run scripts/fetch_skyclip.sh: {configured}"
        )
    raise RuntimeError(f"multiple SkyCLIP checkpoints found under {extracted}: {candidates}")


def load_skyclip(config: dict[str, Any], device: torch.device):
    repository = Path(config["skyclip"]["repository"])
    if not (repository / "src/open_clip/factory.py").is_file():
        raise FileNotFoundError(f"SkyScript source is unavailable: {repository}")
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from src.open_clip.factory import create_model_and_transforms, get_tokenizer

    checkpoint = _resolve_checkpoint(config)
    # The official 2023 checkpoint includes NumPy metadata that predates the
    # weights_only=True default introduced by PyTorch 2.6.
    compatibility_key = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    previous_compatibility = os.environ.get(compatibility_key)
    os.environ[compatibility_key] = "1"
    try:
        model, _, preprocess = create_model_and_transforms(
            config["skyclip"]["model"],
            str(checkpoint),
            precision="fp32",
            device=device,
            force_quick_gelu=True,
            output_dict=True,
        )
    finally:
        if previous_compatibility is None:
            os.environ.pop(compatibility_key, None)
        else:
            os.environ[compatibility_key] = previous_compatibility
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model, preprocess, get_tokenizer(config["skyclip"]["model"]), checkpoint


def _distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        if device.type == "cuda":
            dist.init_process_group(backend="nccl", device_id=device)
        else:
            dist.init_process_group(backend="gloo")
    return rank, world_size, local_rank, device


def _suffix(limit: int | None) -> str:
    return f"_{limit}" if limit is not None else ""


def cache_skyclip(config: dict[str, Any], limit: int | None = None) -> None:
    rank, world_size, _, device = _distributed_context()
    frame = load_prepared(config)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    cfg = config["skyclip"]
    model_cfg = config.get("model", {})
    cache_tokens = uses_latent_tokens(config)
    latent_count = int(model_cfg.get("latent_count", 0))
    latent_dim = int(model_cfg.get("latent_dim", 0))
    pooling_grid = tuple(map(int, cfg.get("token_pooling_grid", (2, 4))))
    if cache_tokens and pooling_grid[0] * pooling_grid[1] != latent_count:
        raise ValueError(
            f"token_pooling_grid={pooling_grid} does not produce latent_count={latent_count}"
        )
    output_dir = ensure_dir(cfg["cache_dir"])
    suffix = _suffix(limit)
    model, preprocess, tokenizer, checkpoint = load_skyclip(config, device)
    local_positions = np.arange(rank, len(frame), world_size, dtype=np.int64)
    local_frame = frame.iloc[local_positions]
    dataset = ImageRows(
        local_frame["image_path"].tolist(),
        local_frame["row_id"].to_numpy(np.int64),
        preprocess,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["image_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["workers"]),
        pin_memory=True,
        persistent_workers=int(cfg["workers"]) > 0,
    )
    local_features = open_memmap(
        output_dir / f"highres_rank{rank}{suffix}.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(dataset), int(cfg["embedding_dim"])),
    )
    local_tokens = None
    if cache_tokens:
        local_tokens = open_memmap(
            output_dir / f"highres_tokens_rank{rank}{suffix}.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(dataset), latent_count, latent_dim),
        )
        model.visual.output_tokens = True
    local_row_ids = np.empty(len(dataset), dtype=np.int64)
    offset = 0
    with torch.inference_mode():
        for images, row_ids in tqdm(loader, desc=f"skyclip-images-r{rank}", disable=rank != 0):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if cache_tokens:
                    pooled, patch_tokens = model.visual(images)
                    features = F.normalize(pooled, dim=-1)
                    patch_tokens = model.visual.ln_post(patch_tokens)
                    grid_height, grid_width = model.visual.grid_size
                    if patch_tokens.shape[1:] != (grid_height * grid_width, latent_dim):
                        raise ValueError(
                            f"expected ViT tokens Bx{grid_height * grid_width}x{latent_dim}, "
                            f"got {tuple(patch_tokens.shape)}"
                        )
                    patch_tokens = patch_tokens.reshape(
                        len(images), grid_height, grid_width, latent_dim
                    ).permute(0, 3, 1, 2)
                    region_tokens = F.adaptive_avg_pool2d(
                        patch_tokens, pooling_grid
                    ).flatten(2).transpose(1, 2)
                else:
                    features = model.encode_image(images, normalize=True)
            count = len(images)
            local_features[offset:offset + count] = features.float().cpu().numpy().astype(np.float16)
            if local_tokens is not None:
                local_tokens[offset:offset + count] = (
                    region_tokens.float().cpu().numpy().astype(np.float16)
                )
            local_row_ids[offset:offset + count] = row_ids.numpy()
            offset += count
    local_features.flush()
    if local_tokens is not None:
        local_tokens.flush()
    np.save(output_dir / f"highres_row_ids_rank{rank}{suffix}.npy", local_row_ids)
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        merged = open_memmap(
            output_dir / f"highres_features{suffix}.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(frame), int(cfg["embedding_dim"])),
        )
        row_to_position = {int(row_id): index for index, row_id in enumerate(frame["row_id"])}
        for source_rank in range(world_size):
            features = np.load(output_dir / f"highres_rank{source_rank}{suffix}.npy", mmap_mode="r")
            row_ids = np.load(output_dir / f"highres_row_ids_rank{source_rank}{suffix}.npy")
            positions = np.fromiter((row_to_position[int(value)] for value in row_ids), dtype=np.int64)
            merged[positions] = features
        merged.flush()

        if cache_tokens:
            merged_tokens = open_memmap(
                output_dir / f"highres_tokens{suffix}.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(frame), latent_count, latent_dim),
            )
            for source_rank in range(world_size):
                tokens = np.load(
                    output_dir / f"highres_tokens_rank{source_rank}{suffix}.npy",
                    mmap_mode="r",
                )
                row_ids = np.load(
                    output_dir / f"highres_row_ids_rank{source_rank}{suffix}.npy"
                )
                positions = np.fromiter(
                    (row_to_position[int(value)] for value in row_ids), dtype=np.int64
                )
                merged_tokens[positions] = tokens
            merged_tokens.flush()

        titles = frame[["title_id", "title"]].drop_duplicates("title_id").sort_values("title_id")
        title_feature_count = int(titles["title_id"].max()) + 1
        text_features = open_memmap(
            output_dir / f"text_features{suffix}.npy",
            mode="w+",
            dtype=np.float16,
            shape=(title_feature_count, int(cfg["embedding_dim"])),
        )
        text_features[:] = 0
        batch_size = max(1, int(cfg["image_batch_size"]) * 2)
        with torch.inference_mode():
            for start in tqdm(range(0, len(titles), batch_size), desc="skyclip-text"):
                texts = titles["title"].iloc[start:start + batch_size].tolist()
                tokens = tokenizer(texts).to(device)
                values = model.encode_text(tokens, normalize=True)
                title_ids = titles["title_id"].iloc[start:start + len(texts)].to_numpy(np.int64)
                text_features[title_ids] = values.float().cpu().numpy().astype(np.float16)
        text_features.flush()
        titles.to_parquet(output_dir / f"titles{suffix}.parquet", index=False)
        (output_dir / f"checkpoint{suffix}.txt").write_text(str(checkpoint) + "\n", encoding="utf-8")
        print(f"cached {len(frame)} high-resolution features and {len(titles)} title features")
        if cache_tokens:
            print(
                f"cached high-resolution region tokens with shape "
                f"({len(frame)}, {latent_count}, {latent_dim})"
            )
        if not bool(cfg.get("keep_rank_shards", True)):
            for source_rank in range(world_size):
                (output_dir / f"highres_rank{source_rank}{suffix}.npy").unlink(missing_ok=True)
                (output_dir / f"highres_row_ids_rank{source_rank}{suffix}.npy").unlink(
                    missing_ok=True
                )
                if cache_tokens:
                    (output_dir / f"highres_tokens_rank{source_rank}{suffix}.npy").unlink(
                        missing_ok=True
                    )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def encode_query(config: dict[str, Any], query: str, templates: list[str] | None = None) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, tokenizer, _ = load_skyclip(config, device)
    prompts = [query] if templates is None else [template.format(query=query) for template in templates]
    with torch.inference_mode():
        tokens = tokenizer(prompts).to(device)
        values = model.encode_text(tokens, normalize=True)
        value = F.normalize(values.float().mean(dim=0), dim=-1)
    return value.cpu().numpy()
