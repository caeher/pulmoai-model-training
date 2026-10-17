import json

import numpy as np
import pytest
import torch

from pulmoai.config import load_config
from pulmoai.datasets.synthetic import create_fixture
from pulmoai.evaluation.runner import evaluate
from pulmoai.inference.detector import Detector, predict_heatmap
from pulmoai.preprocessing.volume import Geometry, Volume
from pulmoai.training.engine import export_model, train
from pulmoai.utils.io import read_json, write_json


def test_train_evaluate_infer_export_and_reproducibility(tmp_path, config_dir):
    config = load_config(config_dir, "smoke")
    manifest = create_fixture(tmp_path / "data", config)
    checkpoint = train(config, manifest, tmp_path / "run")
    first = torch.load(checkpoint, weights_only=True)
    assert first["epoch"] >= 0
    assert np.isfinite(first["best_loss"])
    record = next((tmp_path / "run" / "tracking").rglob("run.json"))
    tracked = read_json(record)
    assert tracked["status"] == "FINISHED"
    assert tracked["artifacts"]["dataset"]["synthetic"] is True
    assert len(tracked["metrics"]["train_loss"]) == 2
    assert "synthetic-patient" not in json.dumps(tracked)
    metrics = evaluate(checkpoint, manifest, tmp_path / "eval", device="cpu")
    assert metrics["scan_count"] == 2 and metrics["nodule_count"] == 1
    assert metrics["synthetic"] is True
    exported = export_model(checkpoint, tmp_path / "export" / "model.pt")
    assert "optimizer" not in torch.load(exported, weights_only=True)
    image = manifest.parent / "synthetic_4.npz"
    original = Detector.from_checkpoint(checkpoint, "cpu").predict(image)
    result = Detector.from_checkpoint(exported, "cpu").predict(image, tmp_path / "infer.json")
    assert original["candidates"] == result["candidates"]
    assert result["coordinate_system"] == "LPS"
    assert "not malignancy" in result["score_meaning"]
    second = train(config, manifest, tmp_path / "repeat")
    repeated = torch.load(second, weights_only=True)
    for key, tensor in first["model"].items():
        assert torch.equal(tensor, repeated["model"][key])
    with pytest.raises(ValueError, match="stopping"):
        train(config, manifest, tmp_path / "run", tmp_path / "run" / "last.pt")


def test_sliding_windows_cover_edges_and_small_volumes(config_dir):
    config = load_config(config_dir, "smoke")

    class Constant(torch.nn.Module):
        def forward(self, image):
            return torch.ones_like(image) * 2

    for shape in ((5, 7, 9), (21, 25, 30)):
        volume = Volume(np.full(shape, -800, dtype=np.float32), Geometry(shape))
        heatmap = predict_heatmap(Constant(), volume, config, torch.device("cpu"))
        assert heatmap.shape == shape
        assert np.allclose(heatmap, 1 / (1 + np.exp(-2)))

    class Invalid(torch.nn.Module):
        def forward(self, image):
            return torch.full_like(image, float("nan"))

    with pytest.raises(RuntimeError, match="nonfinite"):
        predict_heatmap(Invalid(), volume, config, torch.device("cpu"))


def test_incompatible_artifact_rejected(tmp_path):
    path = tmp_path / "cxr.pt"
    torch.save({"format_version": 1, "task": "cxr_classification"}, path)
    with pytest.raises(ValueError, match="compatible"):
        Detector.from_checkpoint(path)


def test_interrupted_epoch_boundary_resume_matches_uninterrupted(tmp_path, config_dir, monkeypatch):
    from pulmoai.training import engine

    config = load_config(config_dir, "smoke")
    manifest = create_fixture(tmp_path / "data", config)
    train(config, manifest, tmp_path / "full")
    save = engine._atomic_save

    def interrupt_after_saved_epoch(payload, path):
        save(payload, path)
        if path.name == "last.pt" and payload["epoch"] == 0:
            raise RuntimeError("Synthetic interruption after committed epoch")

    monkeypatch.setattr(engine, "_atomic_save", interrupt_after_saved_epoch)
    with pytest.raises(RuntimeError, match="interruption"):
        train(config, manifest, tmp_path / "resumed")
    monkeypatch.setattr(engine, "_atomic_save", save)
    record = next((tmp_path / "resumed" / "tracking").rglob("run.json"))
    partial = read_json(record)
    partial["metrics"]["train_loss"].append({"step": 1, "value": 999})
    write_json(record, partial)
    train(config, manifest, tmp_path / "resumed", tmp_path / "resumed" / "last.pt")
    full = torch.load(tmp_path / "full" / "last.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)
    assert full["epoch"] == resumed["epoch"] == 1
    for key in full["model"]:
        assert torch.equal(full["model"][key], resumed["model"][key])
    record = next((tmp_path / "resumed" / "tracking").rglob("run.json"))
    assert read_json(record)["resume_count"] == 1
    assert [v["step"] for v in read_json(record)["metrics"]["train_loss"]] == [0, 1]
    assert "error" not in read_json(record)
