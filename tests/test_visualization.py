from pathlib import Path

from PIL import Image

from tessera_skyscript_retrieval.visualization import _draw_retrievals, _plot_metrics


def _report() -> dict:
    values = {
        "Precision@10": 0.25,
        "Precision@100": 0.10,
        "nDCG@10": 0.50,
        "nDCG@100": 0.75,
    }
    block = {
        "text_to_highres": values,
        "text_to_tessera": values,
        "highres_to_tessera_semantic": values,
    }
    return {"splits": {"test": block, "oov_test": block}}


def test_visualization_outputs_are_nonempty(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.png"
    _plot_metrics(_report(), metrics)
    with Image.open(metrics) as image:
        assert image.width > 1000
        assert image.height > 500

    source = tmp_path / "source.jpg"
    Image.new("RGB", (120, 80), "#3f7652").save(source)
    row = {
        "rank": 1,
        "score": 0.5,
        "source": "US",
        "year": 2021,
        "title": "An aerial image. It shows: Quarry area.",
        "center_lon": -75.0,
        "center_lat": 41.0,
        "image_path": str(source),
    }
    retrieval = tmp_path / "retrieval.png"
    _draw_retrievals("quarry area", {"highres": [row], "tessera": [row]}, retrieval)
    with Image.open(retrieval) as image:
        assert image.width >= 300
        assert image.height >= 700
