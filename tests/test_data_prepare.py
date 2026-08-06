from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from tessera_skyscript_retrieval.data import prepare_manifest


def test_prepare_joins_image_chip_and_pickle(tmp_path: Path):
    image_root = tmp_path / "image-emb-text"
    metadata_root = tmp_path / "tessera_v2"
    (image_root / "images2").mkdir(parents=True)
    (image_root / "tessera_chips/npy").mkdir(parents=True)
    (metadata_root / "meta2").mkdir(parents=True)
    rows = []
    for index in range(12):
        sample_id = f"a{index}_CH_21"
        Image.new("RGB", (16, 12), color=(index, 0, 0)).save(image_root / f"images2/{sample_id}.jpg")
        np.save(image_root / f"tessera_chips/npy/{sample_id}.npy", np.ones((2, 3, 128), np.float16))
        with (metadata_root / f"meta2/{sample_id}.pickle").open("wb") as stream:
            pickle.dump({"bbox": (7.0 + index * 0.2, 46.0, 7.001 + index * 0.2, 46.001)}, stream)
        rows.append({"sample_id": sample_id, "filepath": f"images2/{sample_id}.jpg", "title": "  Quarry   area. "})
    source = image_root / "manifest.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    config = {
        "data": {
            "source_manifest": str(source), "image_root": str(image_root),
            "chip_root": str(image_root / "tessera_chips/npy"), "metadata_root": str(metadata_root),
            "prepared_manifest": str(tmp_path / "prepared.parquet"), "allowed_sources": ["CH"],
            "split_seed": 42, "train_percent": 80, "val_percent": 10,
            "oov_min_samples": 100, "oov_percent": 10,
        },
        "tessera": {"workers": 2},
    }
    frame = prepare_manifest(config)
    assert len(frame) == 12
    assert frame["title"].eq("Quarry area.").all()
    assert frame["chip_h"].eq(2).all()
    assert frame["chip_w"].eq(3).all()
    assert (tmp_path / "prepared.parquet").is_file()
