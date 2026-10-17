"""Small explicitly synthetic CT fixture generator for software verification only."""

from pathlib import Path

import numpy as np

from pulmoai.config import ExperimentConfig
from pulmoai.preprocessing.volume import Geometry, Volume, save_volume, voxel_digest
from pulmoai.tracking import sha256_file
from pulmoai.utils.io import write_json


def create_fixture(output: str | Path, config: ExperimentConfig) -> Path:
    if config.data.dataset != "synthetic":
        raise ValueError("Fixture generation requires data.dataset=synthetic")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Synthetic fixture output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seeds.global_seed)
    scans = []
    size = config.data.patch_size
    shape = (size,) * 3
    grid = np.indices(shape)
    for i, split in enumerate(("train", "train", "validation", "validation", "test", "test")):
        geometry = Geometry(shape, (-30.0, -20.0, -10.0), config.data.spacing_xyz)
        hu = rng.normal(-850, 20, shape).astype(np.float32)
        nodules = []
        if i % 2 == 0:
            center = np.array([size // 2] * 3)
            mask = (
                sum(((grid[a] - center[a]) * geometry.spacing_xyz[::-1][a]) ** 2 for a in range(3))
                <= 3**2
            )
            hu[mask] = 100
            nodules = [{"center_xyz_mm": geometry.world(center[::-1]).tolist(), "diameter_mm": 6.0}]
        filename = f"synthetic_{i}.npz"
        save_volume(output / filename, Volume(hu, geometry))
        scans.append(
            {
                "seriesuid": f"synthetic-series-{i}",
                "patient_id": f"synthetic-patient-{i}",
                "split": split,
                "volume": filename,
                "sha256": sha256_file(output / filename),
                "voxel_sha256": voxel_digest(Volume(hu, geometry)),
                "original_geometry": geometry.to_dict(),
                "geometry": geometry.to_dict(),
                "nodules": nodules,
                "excluded": [],
            }
        )
    target = output / "manifest.json"
    write_json(
        target,
        {
            "schema_version": 1,
            "dataset": "synthetic",
            "dataset_version": config.data.version,
            "synthetic": True,
            "preprocessing_signature": config.preprocessing_signature(),
            "excluded_annotations_supplied": False,
            "scans": scans,
        },
    )
    return target
