"""Private, versioned manifests; patient identities never enter aggregate tracking."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pulmoai.config import ExperimentConfig
from pulmoai.preprocessing.volume import Geometry, read_volume, resample, save_volume, voxel_digest
from pulmoai.tracking import sha256_file
from pulmoai.utils.io import read_json, write_json

SPLITS = ("train", "validation", "test")


def patient_splits(patient_ids, fractions, seed: int) -> dict[str, str]:
    patients = sorted(set(patient_ids))
    if len(patients) < 3:
        raise ValueError("At least three distinct patients are required for independent splits")
    np.random.default_rng(seed).shuffle(patients)
    n = len(patients)
    n_train = min(max(1, int(n * fractions[0])), n - 2)
    n_val = min(max(1, int(n * fractions[1])), n - n_train - 1)
    return {
        p: SPLITS[0 if i < n_train else 1 if i < n_train + n_val else 2]
        for i, p in enumerate(patients)
    }


def _rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _nodules(rows: list[dict], known: set[str]) -> dict[str, list[dict]]:
    result = {uid: [] for uid in known}
    for row in rows:
        uid = row["seriesuid"].strip()
        if uid not in known:
            raise ValueError("Annotation references a series missing from the scan inventory")
        center = [float(row[key]) for key in ("coordX", "coordY", "coordZ")]
        diameter = float(row["diameter_mm"])
        if not np.isfinite([*center, diameter]).all() or diameter <= 0:
            raise ValueError("Nodule coordinates must be finite and diameter_mm positive")
        result[uid].append({"center_xyz_mm": center, "diameter_mm": diameter})
    return result


def prepare_luna(
    scans: str | Path,
    annotations: str | Path,
    patients: str | Path,
    output: str | Path,
    config: ExperimentConfig,
    excluded: str | Path | None = None,
) -> Path:
    """Prepare already acquired LUNA16 MHD volumes; never downloads anything."""
    if config.data.dataset != "luna16":
        raise ValueError("LUNA preparation requires data.dataset=luna16")
    files = sorted(Path(scans).rglob("*.mhd"))
    inventory = {p.stem: p for p in files}
    if not inventory or len(inventory) != len(files):
        raise ValueError("Expected a nonempty inventory with unique MHD series names")
    mapping = {}
    for row in _rows(patients):
        uid, patient = row["seriesuid"].strip(), row["patient_id"].strip()
        if not patient or uid in mapping:
            raise ValueError("Patient mapping requires nonempty IDs and unique series")
        mapping[uid] = patient
    if set(mapping) != set(inventory):
        raise ValueError("Patient mapping must cover exactly the selected scan inventory")
    included = _nodules(_rows(annotations), set(inventory))
    ignored = (
        _nodules(_rows(excluded), set(inventory)) if excluded else {uid: [] for uid in inventory}
    )
    assignments = patient_splits(
        mapping.values(), config.data.split_fractions, config.seeds.split_seed
    )
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preparation output must be new or empty; use a new dataset version")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (uid, path) in enumerate(inventory.items()):
        original = read_volume(path)
        for annotation in included[uid] + ignored[uid]:
            index = original.geometry.index(annotation["center_xyz_mm"])
            if np.any(index < -0.5) or np.any(index > np.array(original.hu.shape[::-1]) - 0.5):
                raise ValueError("Annotation center lies outside its CT volume")
        prepared = resample(original, config.data.spacing_xyz)
        filename = f"volume_{i:05d}.npz"
        save_volume(output / filename, prepared)
        records.append(
            {
                "seriesuid": uid,
                "patient_id": mapping[uid],
                "split": assignments[mapping[uid]],
                "volume": filename,
                "sha256": sha256_file(output / filename),
                "voxel_sha256": voxel_digest(prepared),
                "original_geometry": original.geometry.to_dict(),
                "geometry": prepared.geometry.to_dict(),
                "nodules": included[uid],
                "excluded": ignored[uid],
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": config.data.dataset,
        "dataset_version": config.data.version,
        "synthetic": False,
        "preprocessing_signature": config.preprocessing_signature(),
        "split_seed": config.seeds.split_seed,
        "source_hashes": {
            "annotations": sha256_file(annotations),
            "patient_mapping": sha256_file(patients),
        },
        "excluded_annotations_supplied": excluded is not None,
        "scans": records,
    }
    if excluded:
        manifest["source_hashes"]["excluded"] = sha256_file(excluded)
    target = output / "manifest.json"
    write_json(target, manifest)
    load_manifest(target, config)
    return target


def load_manifest(
    path: str | Path, config: ExperimentConfig | None = None, verify_hashes: bool = True
) -> dict:
    path = Path(path)
    manifest = read_json(path)
    if manifest.get("schema_version") != 1 or not manifest.get("scans"):
        raise ValueError("Unsupported or empty CT manifest")
    if config and (
        manifest["dataset"] != config.data.dataset
        or manifest["dataset_version"] != config.data.version
        or manifest["preprocessing_signature"] != config.preprocessing_signature()
    ):
        raise ValueError("Manifest dataset/version/preprocessing differs from experiment")
    seen, groups, hashes = set(), {}, {}
    for scan in manifest["scans"]:
        uid, patient, split = scan["seriesuid"], scan["patient_id"], scan["split"]
        if not uid or uid in seen or not patient or split not in SPLITS:
            raise ValueError("Invalid/duplicate series, patient, or split in manifest")
        seen.add(uid)
        if patient in groups and groups[patient] != split:
            raise ValueError("Patient leakage across dataset splits")
        groups[patient] = split
        relative = Path(scan["volume"])
        volume_path = (path.parent / relative).resolve()
        if relative.is_absolute() or not volume_path.is_relative_to(path.parent.resolve()):
            raise ValueError("Cached volume must remain inside the manifest directory")
        if verify_hashes and sha256_file(volume_path) != scan["sha256"]:
            raise ValueError("Cached CT volume checksum differs from manifest")
        Geometry.from_dict(scan["original_geometry"])
        geometry = Geometry.from_dict(scan["geometry"])
        if config and not np.allclose(geometry.spacing_xyz, config.data.spacing_xyz):
            raise ValueError("Cached spacing differs from experiment")
        if verify_hashes:
            volume = read_volume(volume_path)
            if volume.geometry != geometry:
                raise ValueError("Cached volume geometry differs from manifest")
            digest = voxel_digest(volume)
            if scan.get("voxel_sha256", digest) != digest:
                raise ValueError("Decoded voxel checksum differs from manifest")
        else:
            digest = scan.get("voxel_sha256", scan["sha256"])
        if digest in hashes and hashes[digest] != split:
            raise ValueError("Duplicate volume leakage across splits")
        hashes[digest] = split
        for nodule in scan["nodules"] + scan["excluded"]:
            center, diameter = nodule["center_xyz_mm"], nodule["diameter_mm"]
            if len(center) != 3 or not np.isfinite([*center, diameter]).all() or diameter <= 0:
                raise ValueError("Invalid nodule geometry")
    if set(groups.values()) != set(SPLITS):
        raise ValueError("Manifest must contain nonempty train, validation, and test splits")
    return manifest
