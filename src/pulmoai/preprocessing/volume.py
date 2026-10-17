"""Scalar CT volumes: arrays use ZYX, physical coordinates use XYZ millimeters."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform


@dataclass(frozen=True)
class Geometry:
    shape_zyx: tuple[int, int, int]
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    direction: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.shape_zyx) != 3 or any(type(v) is not int or v < 1 for v in self.shape_zyx):
            raise ValueError("CT geometry requires three positive integer dimensions")
        if len(self.origin_xyz) != 3 or len(self.spacing_xyz) != 3 or len(self.direction) != 9:
            raise ValueError("Invalid CT geometry dimensions")
        if not np.isfinite([*self.origin_xyz, *self.spacing_xyz, *self.direction]).all():
            raise ValueError("CT geometry must be finite")
        matrix = np.asarray(self.direction).reshape(3, 3)
        if any(s <= 0 for s in self.spacing_xyz) or not np.allclose(matrix.T @ matrix, np.eye(3)):
            raise ValueError("Spacing must be positive and direction must be orthonormal")

    def world(self, index_xyz) -> np.ndarray:
        return np.asarray(self.origin_xyz) + np.asarray(self.direction).reshape(3, 3) @ (
            np.asarray(index_xyz) * self.spacing_xyz
        )

    def index(self, world_xyz) -> np.ndarray:
        return (
            np.asarray(self.direction).reshape(3, 3).T @ (np.asarray(world_xyz) - self.origin_xyz)
        ) / self.spacing_xyz

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> Geometry:
        return cls(**{key: tuple(item) for key, item in value.items()})


@dataclass
class Volume:
    hu: np.ndarray
    geometry: Geometry

    def __post_init__(self) -> None:
        if self.hu.shape != self.geometry.shape_zyx or not np.isfinite(self.hu).all():
            raise ValueError("CT array shape/values do not match finite scalar volume geometry")


def voxel_digest(volume: Volume) -> str:
    """Hash decoded voxel values and shape independently of container metadata."""
    digest = hashlib.sha256(np.asarray(volume.hu.shape, dtype="<i8").tobytes())
    values = np.ascontiguousarray(volume.hu, dtype="<f4")
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def read_volume(path: str | Path) -> Volume:
    """Read LUNA16 HU MHD/RAW or the documented internal HU NPZ format."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if str(data["units"].item()) != "HU":
                raise ValueError("NPZ CT intensities must explicitly declare units=HU")
            hu = data["hu"].astype(np.float32)
            geometry = Geometry(
                tuple(hu.shape),
                tuple(data["origin_xyz"]),
                tuple(data["spacing_xyz"]),
                tuple(data["direction"]),
            )
        return Volume(hu, geometry)
    if path.suffix.lower() != ".mhd":
        raise ValueError("Supported inputs: LUNA16 .mhd/.raw and internal HU .npz; CT only")
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Expected a scalar three-dimensional CT image in HU")
    hu = sitk.GetArrayFromImage(image).astype(np.float32)
    return Volume(
        hu, Geometry(tuple(hu.shape), image.GetOrigin(), image.GetSpacing(), image.GetDirection())
    )


def save_volume(path: str | Path, volume: Volume) -> None:
    np.savez_compressed(
        path,
        hu=volume.hu.astype(np.float32),
        units="HU",
        origin_xyz=volume.geometry.origin_xyz,
        spacing_xyz=volume.geometry.spacing_xyz,
        direction=volume.geometry.direction,
    )


def resample(volume: Volume, spacing_xyz: tuple[float, float, float]) -> Volume:
    old = volume.geometry
    extent = (np.asarray(old.shape_zyx[::-1]) - 1) * old.spacing_xyz
    size_xyz = np.ceil(extent / spacing_xyz).astype(int) + 1
    geometry = Geometry(
        tuple(int(v) for v in size_xyz[::-1]), old.origin_xyz, tuple(spacing_xyz), old.direction
    )
    # Same origin and direction: output index maps to input index by the spacing ratio.
    ratio = (np.asarray(spacing_xyz) / old.spacing_xyz)[::-1]
    hu = affine_transform(
        volume.hu,
        np.diag(ratio),
        output_shape=geometry.shape_zyx,
        order=1,
        mode="constant",
        cval=-1000.0,
        prefilter=False,
    )
    return Volume(hu.astype(np.float32), geometry)


def normalize_hu(hu: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    low, high = window
    return ((np.clip(hu, low, high) - low) / (high - low)).astype(np.float32)
