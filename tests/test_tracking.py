import json
import socket

import numpy as np
import pytest

from respiratory_xray.config import load_config
from respiratory_xray.tracking import REDACTED, Tracker, TrackingError, redact


@pytest.fixture
def cfg(config_dir):
    return load_config(config_dir)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def guard(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", guard)


def _synthetic_auroc(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 200)
    s = y * 0.6 + rng.random(200) * 0.8
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean())


def test_synthetic_run_logs_version_seed_config(cfg, tmp_path):
    t = Tracker(cfg, root=tmp_path, backend="json")
    with t.run("synthetic"):
        for epoch in range(3):
            t.log_metric("val_auroc", _synthetic_auroc(epoch), step=epoch)
        t.log_artifact("summary", {"auroc_mean": 0.8, "per_class": {"Edema": 0.81}})
        t.log_selection("keep", "best val_auroc", "val_auroc", 0.8)
    rec = t.load()
    assert rec["status"] == "FINISHED"
    assert rec["seed"] == 42
    assert rec["package_version"]
    assert rec["git_commit"]
    assert rec["environment"]["python"]
    assert rec["config_fingerprint"] == cfg.fingerprint()
    assert rec["preprocessing_signature"]["classes"] == list(cfg.data.classes)
    assert len(rec["metrics"]["val_auroc"]) == 3
    assert rec["wandb_enabled"] is False
    assert rec["wall_time_s"] >= 0
    assert "peak_vram_mb" in rec


def test_hashes_logged_without_paths(cfg, tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("split,label\ntrain,1\n")
    t = Tracker(cfg, root=tmp_path / "runs", backend="json")
    with t.run():
        digest = t.log_hash("split_manifest", manifest)
    rec = t.load()
    assert rec["hashes"]["split_manifest"] == digest and len(digest) == 64
    assert str(manifest) not in json.dumps(rec)


def test_resume_appends_and_counts(cfg, tmp_path):
    t = Tracker(cfg, root=tmp_path, backend="json")
    with t.run():
        t.log_metric("loss", 1.0, step=0)
    run_id = t.run_id
    t2 = Tracker(cfg, root=tmp_path, backend="json")
    with t2.run(resume_id=run_id):
        t2.log_metric("loss", 0.5, step=1)
    rec = t2.load()
    assert rec["resume_count"] == 1
    assert [m["value"] for m in rec["metrics"]["loss"]] == [1.0, 0.5]


def test_resume_with_different_config_rejected(cfg, config_dir, tmp_path):
    t = Tracker(cfg, root=tmp_path, backend="json")
    with t.run():
        pass
    other = load_config(config_dir, overrides={"seeds": {"global_seed": 7}})
    with pytest.raises(TrackingError):
        Tracker(other, root=tmp_path, backend="json").start(resume_id=t.run_id)


def test_failed_run_recorded(cfg, tmp_path):
    t = Tracker(cfg, root=tmp_path, backend="json")
    with pytest.raises(RuntimeError):
        with t.run():
            t.log_metric("loss", 1.0)
            raise RuntimeError("boom reading /data/patient_001.png")
    rec = t.load()
    assert rec["status"] == "FAILED"
    assert rec["error"]["type"] == "RuntimeError"
    assert "patient_001" not in json.dumps(rec)


def test_nan_metric_and_unknown_resume_error(cfg, tmp_path):
    t = Tracker(cfg, root=tmp_path, backend="json")
    t.start()
    with pytest.raises(TrackingError):
        t.log_metric("loss", float("nan"))
    with pytest.raises(TrackingError):
        t.log_metric("loss", float("inf"))
    with pytest.raises(TrackingError):
        Tracker(cfg, root=tmp_path, backend="json").start(resume_id="missing")


def test_sensitive_keys_and_images_refused(cfg, tmp_path):
    t = Tracker(cfg, root=tmp_path, backend="json")
    t.start()
    for key in ("patient_id_auc", "hf_token", "image_path", "seriesuid_auc", "volume_path"):
        with pytest.raises(TrackingError):
            t.log_metric(key, 1.0)
    with pytest.raises(TrackingError):
        t.log_artifact("preds", {"rows": [{"patient_id": "p1", "score": 0.3}]})
    assert redact({"api_key": "x", "img": "a/b.dcm", "ok": 1}) == {
        "api_key": REDACTED,
        "img": REDACTED,
        "ok": 1,
    }


def test_wandb_disabled_by_default_and_gated(cfg, config_dir, tmp_path, monkeypatch):
    assert cfg.tracking.wandb_enabled is False
    monkeypatch.delenv("RXDL_ALLOW_WANDB", raising=False)
    wb = load_config(config_dir, overrides={"tracking": {"wandb_enabled": True}})
    with pytest.raises(TrackingError):
        Tracker(wb, root=tmp_path, backend="json")


def test_mlflow_backend_local_only(cfg, tmp_path):
    pytest.importorskip("mlflow")
    t = Tracker(cfg, root=tmp_path, backend="mlflow")
    with t.run("mlflow-synthetic"):
        t.log_metric("val_auroc", 0.75, step=0)
    assert (tmp_path / "mlflow").exists()
    assert t.load()["mlflow_run_id"]
