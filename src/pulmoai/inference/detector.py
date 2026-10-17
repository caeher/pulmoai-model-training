from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label, maximum_filter

from pulmoai import __version__
from pulmoai.config import ExperimentConfig, config_from_dict
from pulmoai.models.unet import SmallUNet3D
from pulmoai.preprocessing.volume import (
    Volume,
    normalize_hu,
    read_volume,
    resample,
)
from pulmoai.tracking import sha256_file
from pulmoai.utils.io import write_json
from pulmoai.utils.runtime import configure_runtime


def _starts(length: int, patch: int, stride: int) -> list[int]:
    end = max(0, length - patch)
    return sorted(set([*range(0, end + 1, stride), end]))


@torch.inference_mode()
def predict_heatmap(model, volume: Volume, config: ExperimentConfig, device) -> np.ndarray:
    """Overlap-average logits on CPU; only one tile resides on the accelerator."""
    model.eval()
    size = config.data.patch_size
    image = normalize_hu(volume.hu, config.data.hu_window)
    padded = np.pad(image, [(0, max(0, size - n)) for n in image.shape])
    total = np.zeros(padded.shape, dtype=np.float32)
    count = np.zeros(padded.shape, dtype=np.float32)
    stride = max(1, int(size * (1 - config.evaluation.overlap)))
    axes = [_starts(n, size, stride) for n in padded.shape]
    for start in itertools.product(*axes):
        region = tuple(slice(s, s + size) for s in start)
        tensor = torch.from_numpy(padded[region].copy()[None, None]).to(device)
        logits = model(tensor)[0, 0].float().cpu().numpy()
        if not np.isfinite(logits).all():
            raise RuntimeError("Detector produced nonfinite logits")
        total[region] += logits
        count[region] += 1
    logits = total / count
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return probabilities[tuple(slice(0, n) for n in image.shape)]


def candidates(
    heatmap: np.ndarray, geometry, original_geometry, config: ExperimentConfig
) -> list[dict]:
    """Extract local maxima, collapse plateaus, then apply physical-distance NMS."""
    e = config.evaluation
    peaks = (heatmap >= e.score_threshold) & (heatmap == maximum_filter(heatmap, size=3))
    regions, number = label(peaks)
    maxima = []
    # Extract one representative per connected maximum plateau without repeated volume scans.
    points = np.argwhere(peaks)
    if number:
        ids = regions[tuple(points.T)]
        order = np.lexsort((-heatmap[tuple(points.T)], ids))
        _, offsets = np.unique(ids[order], return_index=True)
        maxima = points[order[offsets]]
    ranked = sorted(maxima, key=lambda p: -float(heatmap[tuple(p)]))
    accepted = []
    for point in ranked:
        center = geometry.world(point[::-1])
        original_index = original_geometry.index(center)
        if np.any(original_index < -0.5) or np.any(
            original_index > np.array(original_geometry.shape_zyx[::-1]) - 0.5
        ):
            continue
        if any(
            np.linalg.norm(center - np.array(c["center_xyz_mm"])) < e.nms_distance_mm
            for c in accepted
        ):
            continue
        accepted.append(
            {
                "center_xyz_mm": center.tolist(),
                "center_voxel_xyz": original_index.tolist(),
                "score": float(heatmap[tuple(point)]),
            }
        )
        if len(accepted) >= e.max_candidates:
            break
    return accepted


class Detector:
    """Load a local trained artifact and return CT nodule candidates for an external API."""

    def __init__(self, model, config, device, model_hash: str):
        self.model, self.config, self.device, self.model_hash = model, config, device, model_hash

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> Detector:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("format_version") != 1 or artifact.get("task") != "ct_nodule_candidates":
            raise ValueError("Artifact is not a compatible PulmoAI CT detector")
        config = config_from_dict(artifact["config"])
        if config.fingerprint() != artifact["config_fingerprint"]:
            raise ValueError("Artifact configuration fingerprint mismatch")
        runtime = config.to_dict()
        runtime["train"]["device"] = device
        target = configure_runtime(config_from_dict(runtime))
        model = SmallUNet3D(config.model.base_channels).to(target)
        model.load_state_dict(artifact["model"])
        return cls(model, config, target, sha256_file(path))

    def predict_volume(self, volume: Volume) -> dict:
        prepared = resample(volume, self.config.data.spacing_xyz)
        heatmap = predict_heatmap(self.model, prepared, self.config, self.device)
        found = candidates(heatmap, prepared.geometry, volume.geometry, self.config)
        return {
            "schema_version": 1,
            "package_version": __version__,
            "task": "ct_nodule_candidates",
            "modality": "CT",
            "coordinate_system": "LPS",
            "coordinate_units": "mm",
            "voxel_order": "xyz",
            "voxel_index_base": 0,
            "model_sha256": self.model_hash,
            "config_fingerprint": self.config.fingerprint(),
            "original_geometry": volume.geometry.to_dict(),
            "processed_geometry": prepared.geometry.to_dict(),
            "preprocessing": self.config.preprocessing_signature(),
            "candidate_policy": {
                "score_threshold": self.config.evaluation.score_threshold,
                "nms_distance_mm": self.config.evaluation.nms_distance_mm,
                "max_candidates": self.config.evaluation.max_candidates,
            },
            "score_meaning": "uncalibrated nodule candidate score, not malignancy probability",
            "research_only": True,
            "candidates": found,
        }

    def predict(self, image: str | Path, output: str | Path | None = None) -> dict:
        result = self.predict_volume(read_volume(image))
        if output:
            write_json(output, result)
        return result
