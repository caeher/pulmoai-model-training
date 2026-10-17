from __future__ import annotations

import csv
from pathlib import Path

import torch

from pulmoai.datasets.manifest import load_manifest
from pulmoai.evaluation.metrics import bootstrap_cpm, detection_metrics
from pulmoai.inference.detector import Detector, candidates, predict_heatmap
from pulmoai.preprocessing.volume import Geometry, read_volume
from pulmoai.tracking import environment_info, sha256_file
from pulmoai.utils.io import write_json


def evaluate(
    checkpoint: str | Path,
    manifest_path: str | Path,
    output: str | Path,
    split: str = "test",
    device: str = "auto",
) -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation accepts only validation or independent test scans")
    detector = Detector.from_checkpoint(checkpoint, device)
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = load_manifest(manifest_path, detector.config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("Use the frozen training manifest for this internal holdout evaluation")
    scans = [s for s in manifest["scans"] if s["split"] == split]
    predictions = {}
    for scan in scans:
        volume = read_volume(manifest_path.parent / scan["volume"])
        heatmap = predict_heatmap(detector.model, volume, detector.config, detector.device)
        predictions[scan["seriesuid"]] = candidates(
            heatmap, volume.geometry, Geometry.from_dict(scan["original_geometry"]), detector.config
        )
    e = detector.config.evaluation
    metrics = detection_metrics(scans, predictions, e.fp_rates, e.operating_threshold)
    metrics["cpm_bootstrap"] = bootstrap_cpm(
        scans, predictions, e.fp_rates, e.bootstrap_samples, detector.config.seeds.global_seed
    )
    metrics.update(
        {
            "split": split,
            "synthetic": manifest["synthetic"],
            "dataset": manifest["dataset"],
            "dataset_version": manifest["dataset_version"],
            "manifest_sha256": sha256_file(manifest_path),
            "model_sha256": detector.model_hash,
            "config_fingerprint": detector.config.fingerprint(),
            "candidate_policy": {
                "score_threshold": e.score_threshold,
                "nms_distance_mm": e.nms_distance_mm,
                "max_candidates": e.max_candidates,
            },
            "excluded_annotations_supplied": manifest["excluded_annotations_supplied"],
            "environment": environment_info(),
            "research_only": True,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    # Per-scan results are private local outputs, not aggregate tracker artifacts.
    write_json(output / "predictions.json", {"schema_version": 1, "predictions": predictions})
    with open(output / "predictions.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seriesuid", "coordX", "coordY", "coordZ", "probability"])
        for uid, found in predictions.items():
            for candidate in found:
                writer.writerow([uid, *candidate["center_xyz_mm"], candidate["score"]])
    return metrics
