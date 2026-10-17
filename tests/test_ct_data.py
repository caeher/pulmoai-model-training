import csv

import numpy as np
import pytest

from pulmoai.config import config_from_dict
from pulmoai.datasets.manifest import load_manifest, patient_splits, prepare_luna
from pulmoai.datasets.patches import PatchDataset
from pulmoai.datasets.synthetic import create_fixture
from pulmoai.preprocessing.volume import (
    Geometry,
    Volume,
    normalize_hu,
    read_volume,
    resample,
    save_volume,
    voxel_digest,
)
from pulmoai.utils.io import read_json, write_json


def test_geometry_rotated_anisotropic_roundtrip():
    geometry = Geometry((7, 8, 9), (20, -10, 7), (0.7, 1.2, 2.5), (0, -1, 0, 1, 0, 0, 0, 0, -1))
    point = np.array([3.25, 4.1, 2.5])
    assert np.allclose(geometry.index(geometry.world(point)), point)
    volume = Volume(np.arange(7 * 8 * 9).reshape(7, 8, 9).astype(np.float32), geometry)
    processed = resample(volume, (0.5, 0.5, 1.0))
    assert processed.geometry.direction == geometry.direction
    assert np.allclose(
        processed.geometry.world(processed.geometry.index(geometry.world(point))),
        geometry.world(point),
    )
    # Linear interpolation of a known physical ramp verifies actual array-to-world mapping.
    q = np.array([2, 2, 2])
    old = geometry.index(processed.geometry.world(q))
    expected = old[2] * 8 * 9 + old[1] * 9 + old[0]
    assert processed.hu[tuple(q[::-1])] == pytest.approx(expected, abs=1e-4)


def test_hu_and_invalid_images(tmp_path):
    assert np.allclose(
        normalize_hu(np.array([-2000, -1000, -300, 400, 900]), (-1000, 400)), [0, 0, 0.5, 1, 1]
    )
    with pytest.raises(ValueError, match="Supported"):
        read_volume(tmp_path / "image.jpg")
    with pytest.raises(ValueError):
        Geometry((8, 8, 8), spacing_xyz=(0, 1, 1))
    with pytest.raises(ValueError):
        Volume(np.full((2, 2, 2), np.nan), Geometry((2, 2, 2)))
    path = tmp_path / "test.npz"
    np.savez(path, hu=np.ones((8, 8, 8)), units="arbitrary")
    with pytest.raises(ValueError, match="HU"):
        read_volume(path)


def test_patient_splits_group_multiple_series():
    ids = ["a", "a", "b", "c", "d", "e", "f"]
    first = patient_splits(ids, (0.7, 0.15, 0.15), 42)
    assert first == patient_splits(reversed(ids), (0.7, 0.15, 0.15), 42)
    assert set(first.values()) == {"train", "validation", "test"}
    with pytest.raises(ValueError):
        patient_splits(["a", "b"], (0.7, 0.15, 0.15), 42)


def _synthetic_config(**extra):
    return config_from_dict(
        {"data": {"dataset": "synthetic", "version": "test-v1", "patch_size": 16, **extra}}
    )


def test_manifest_rejects_leakage_hash_and_geometry_tampering(tmp_path):
    config = _synthetic_config()
    path = create_fixture(tmp_path / "fixture", config)
    original = read_json(path)
    modified = read_json(path)
    modified["scans"][-1]["patient_id"] = modified["scans"][0]["patient_id"]
    write_json(path, modified)
    with pytest.raises(ValueError, match="Patient leakage"):
        load_manifest(path, config)
    write_json(path, original)
    volume_path = path.parent / original["scans"][0]["volume"]
    volume = read_volume(volume_path)
    volume.hu += 1
    save_volume(volume_path, volume)
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(path, config)


def test_patch_sampling_alignment_and_fixed_validation(tmp_path):
    config = _synthetic_config(positive_fraction=1.0, flip_probability=1.0)
    path = create_fixture(tmp_path / "fixture", config)
    manifest = load_manifest(path, config)
    dataset = PatchDataset(manifest, path.parent, "train", config)
    image, mask, valid = dataset[0]
    assert image.shape == mask.shape == valid.shape == (1, 16, 16, 16)
    assert mask.sum() > 0
    assert np.all(image.numpy()[mask.numpy() == 1] > 0.7)
    assert all(
        np.array_equal(a.numpy(), b.numpy()) for a, b in zip(dataset[0], dataset[0], strict=True)
    )
    validation = PatchDataset(manifest, path.parent, "validation", config)
    first = validation[0]
    validation.epoch = 9
    assert all(
        np.array_equal(a.numpy(), b.numpy()) for a, b in zip(first, validation[0], strict=True)
    )


def test_prepare_real_mhd_format_on_synthetic_pixels(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(4):
        image = sitk.GetImageFromArray(np.full((9, 8, 7), -900 + i, np.int16))
        image.SetSpacing((0.8, 1.1, 2.0))
        image.SetOrigin((-30.0, 10.0, -20.0))
        image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
        sitk.WriteImage(image, str(raw / f"s{i}.mhd"))
    patients = tmp_path / "patients.csv"
    with patients.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seriesuid", "patient_id"])
        writer.writerows([[f"s{i}", "p0" if i < 2 else f"p{i}"] for i in range(4)])
    annotations = tmp_path / "annotations.csv"
    annotations.write_text("seriesuid,coordX,coordY,coordZ,diameter_mm\ns0,-32.2,12.4,-14,4\n")
    config = config_from_dict({"data": {"version": "mhd-fixture-v1"}})
    path = prepare_luna(raw, annotations, patients, tmp_path / "processed", config)
    manifest = load_manifest(path, config)
    assert len(manifest["scans"]) == 4
    assert manifest["scans"][0]["split"] == manifest["scans"][1]["split"]
    assert sum(len(s["nodules"]) for s in manifest["scans"]) == 1
    assert read_volume(path.parent / "volume_00000.npz").geometry.spacing_xyz == (1, 1, 1)


def test_duplicate_voxels_rejected_even_with_different_container_metadata(tmp_path):
    config = _synthetic_config()
    path = create_fixture(tmp_path / "fixture", config)
    manifest = read_json(path)
    first, other = manifest["scans"][0], manifest["scans"][-1]
    volume = read_volume(path.parent / first["volume"])
    volume.geometry = Geometry(volume.geometry.shape_zyx, origin_xyz=(100, 100, 100))
    target = path.parent / other["volume"]
    save_volume(target, volume)
    from pulmoai.tracking import sha256_file

    other.update(
        {
            "sha256": sha256_file(target),
            "voxel_sha256": voxel_digest(volume),
            "geometry": volume.geometry.to_dict(),
            "original_geometry": volume.geometry.to_dict(),
        }
    )
    write_json(path, manifest)
    with pytest.raises(ValueError, match="Duplicate volume leakage"):
        load_manifest(path, config)
