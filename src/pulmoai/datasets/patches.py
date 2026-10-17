from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pulmoai.config import ExperimentConfig
from pulmoai.preprocessing.volume import normalize_hu, read_volume


def sphere_mask(shape, start_zyx, geometry, nodules) -> np.ndarray:
    grid = np.ogrid[
        tuple(slice(start, start + size) for start, size in zip(start_zyx, shape, strict=True))
    ]
    mask = np.zeros(shape, dtype=np.float32)
    for nodule in nodules:
        center = geometry.index(nodule["center_xyz_mm"])[::-1]
        distances = sum(
            ((axis - c) * spacing) ** 2
            for axis, c, spacing in zip(grid, center, geometry.spacing_xyz[::-1], strict=True)
        )
        mask[distances <= (nodule["diameter_mm"] / 2) ** 2] = 1.0
    return mask


class PatchDataset(Dataset):
    """Deterministic scan/epoch patch sampling; no whole-cohort memory cache."""

    def __init__(self, manifest: dict, root: Path, split: str, config: ExperimentConfig):
        self.scans = [s for s in manifest["scans"] if s["split"] == split]
        self.root, self.split, self.config, self.epoch = root, split, config, 0
        if not self.scans:
            raise ValueError(f"No scans in {split} split")

    def __len__(self):
        return len(self.scans) * self.config.data.patches_per_scan

    def __getitem__(self, index):
        cfg = self.config
        scan = self.scans[index // cfg.data.patches_per_scan]
        volume = read_volume(self.root / scan["volume"])
        epoch = self.epoch if self.split == "train" else 0
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seeds.global_seed, epoch, index]))
        size = cfg.data.patch_size
        shape = np.asarray(volume.hu.shape)
        center = np.asarray([rng.integers(0, s) for s in shape])
        if scan["nodules"] and rng.random() < cfg.data.positive_fraction:
            nodule = scan["nodules"][rng.integers(len(scan["nodules"]))]
            center = np.rint(volume.geometry.index(nodule["center_xyz_mm"])[::-1]).astype(int)
            center += rng.integers(-size // 4, size // 4 + 1, size=3)
        start = np.clip(center - size // 2, 0, np.maximum(shape - size, 0))
        crop = tuple(
            slice(int(s), int(min(s + size, n))) for s, n in zip(start, shape, strict=True)
        )
        hu = volume.hu[crop]
        padding = [(0, size - n) for n in hu.shape]
        hu = np.pad(hu, padding, constant_values=cfg.data.hu_window[0])
        target = sphere_mask((size,) * 3, start, volume.geometry, scan["nodules"])
        valid = 1 - sphere_mask((size,) * 3, start, volume.geometry, scan["excluded"])
        valid[target > 0] = 1
        real = np.zeros_like(valid)
        real[tuple(slice(0, int(min(size, n - s))) for n, s in zip(shape, start, strict=True))] = 1
        valid *= real
        image = normalize_hu(hu, cfg.data.hu_window)
        if self.split == "train":
            for axis in range(3):
                if rng.random() < cfg.data.flip_probability:
                    image, target, valid = (np.flip(a, axis) for a in (image, target, valid))
        return tuple(torch.from_numpy(a.copy()[None]) for a in (image, target, valid))
