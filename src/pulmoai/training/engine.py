from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pulmoai import __version__
from pulmoai.config import ExperimentConfig
from pulmoai.datasets.manifest import load_manifest
from pulmoai.datasets.patches import PatchDataset
from pulmoai.models.unet import SmallUNet3D
from pulmoai.tracking import Tracker, sha256_file
from pulmoai.utils.io import write_json
from pulmoai.utils.runtime import configure_runtime


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(
    config: ExperimentConfig,
    manifest_path: str | Path,
    output: str | Path,
    resume: str | Path | None = None,
) -> Path:
    """Train/validate patches; select checkpoints by fixed validation proxy loss."""
    device = configure_runtime(config)
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = load_manifest(manifest_path, config)
    manifest_hash = sha256_file(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise ValueError("Training output must be new or empty; resume explicitly or use a new run")
    model = SmallUNet3D(config.model.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    amp = config.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    datasets = {
        s: PatchDataset(manifest, manifest_path.parent, s, config) for s in ("train", "validation")
    }
    tracker = Tracker(config, root=output / config.tracking.output_dir)
    first, best, bad_epochs, resume_id = 0, float("inf"), 0, None
    best_artifact = None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            state.get("format_version") != 1
            or state.get("task") != "ct_nodule_candidates"
            or state["config_fingerprint"] != config.fingerprint()
            or state["manifest_sha256"] != manifest_hash
        ):
            raise ValueError("Cannot resume: checkpoint configuration or manifest differs")
        if state["device_type"] != device.type:
            raise ValueError("Exact training resume requires the same device type")
        if not tracker.root.exists() or "best_artifact" not in state:
            raise ValueError(
                "Resume requires last.pt and tracking in the original output directory"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        first, best, bad_epochs = state["epoch"] + 1, state["best_loss"], state["bad_epochs"]
        resume_id = state["run_id"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if first >= config.train.epochs or bad_epochs >= config.train.patience:
            raise ValueError("Checkpoint has already reached the configured stopping condition")
        best_artifact = state["best_artifact"]
        # JSON is authoritative: discard metrics from an epoch interrupted before checkpoint commit.
        record = tracker.store.read(resume_id)
        if record["config_fingerprint"] != config.fingerprint():
            raise ValueError("Resume tracking configuration differs from checkpoint")
        _atomic_save(best_artifact, output / "best.pt")
        for key in ("metrics", "selection_decisions"):
            record[key] = state["tracking_record"][key]
        tracker.store.write(resume_id, record)
    write_json(output / "config.json", config.to_dict())
    with tracker.run("ct-candidate-training", resume_id=resume_id):
        tracker.log_hash("manifest", manifest_path)
        tracker.log_artifact(
            "dataset",
            {
                "name": manifest["dataset"],
                "version": manifest["dataset_version"],
                "synthetic": manifest["synthetic"],
            },
        )
        for epoch in range(first, config.train.epochs):
            losses = {}
            for split, dataset in datasets.items():
                training = split == "train"
                dataset.epoch = epoch
                generator = torch.Generator().manual_seed(config.seeds.global_seed + epoch)
                loader = DataLoader(
                    dataset,
                    batch_size=config.train.batch_size,
                    shuffle=training,
                    num_workers=config.train.num_workers,
                    generator=generator,
                    pin_memory=device.type == "cuda",
                )
                model.train(training)
                total_loss, total_weight = 0.0, 0
                optimizer.zero_grad(set_to_none=True)
                for step, (image, target, valid) in enumerate(loader):
                    image, target, valid = (v.to(device) for v in (image, target, valid))
                    if not valid.any():
                        raise ValueError(
                            "Patch batch has no supervised voxels; review excluded regions"
                        )
                    with (
                        torch.set_grad_enabled(training),
                        torch.autocast(device_type=device.type, enabled=amp),
                    ):
                        logits = model(image)
                        voxel_loss = F.binary_cross_entropy_with_logits(
                            logits,
                            target,
                            reduction="none",
                            pos_weight=torch.tensor(config.train.positive_weight, device=device),
                        )
                        loss = (voxel_loss * valid).sum() / valid.sum().clamp_min(1)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training/validation loss")
                    if training:
                        accumulation = config.train.grad_accum_steps
                        group_start = (step // accumulation) * accumulation
                        group_size = min(accumulation, len(loader) - group_start)
                        scaler.scale(loss / group_size).backward()
                        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                    weight = image.shape[0]
                    total_loss += float(loss.detach()) * weight
                    total_weight += weight
                losses[f"{split}_loss"] = total_loss / total_weight
            tracker.log_metrics(losses, step=epoch)
            improved = losses["validation_loss"] < best
            best = min(best, losses["validation_loss"])
            bad_epochs = 0 if improved else bad_epochs + 1
            if improved:
                tracker.log_selection(
                    "keep", "lowest validation proxy loss", "validation_loss", best
                )
            payload = {
                "format_version": 1,
                "package_version": __version__,
                "task": "ct_nodule_candidates",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_loss": best,
                "bad_epochs": bad_epochs,
                "config": config.to_dict(),
                "config_fingerprint": config.fingerprint(),
                "manifest_sha256": manifest_hash,
                "run_id": tracker.run_id,
                "device_type": device.type,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            }
            if improved:
                best_artifact = {
                    key: copy.deepcopy(payload[key])
                    for key in (
                        "format_version",
                        "package_version",
                        "task",
                        "config",
                        "config_fingerprint",
                        "manifest_sha256",
                        "epoch",
                        "best_loss",
                    )
                }
                best_artifact["model"] = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            payload["best_artifact"] = best_artifact
            payload["tracking_record"] = {
                key: copy.deepcopy(tracker.record[key])
                for key in ("metrics", "selection_decisions")
            }
            _atomic_save(payload, output / "last.pt")
            if improved:
                _atomic_save(best_artifact, output / "best.pt")
            tracker.log_hash("last_checkpoint", output / "last.pt")
            if bad_epochs >= config.train.patience:
                break
        tracker.log_hash("best_checkpoint", output / "best.pt")
    return output / "best.pt"


def export_model(checkpoint: str | Path, destination: str | Path) -> Path:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1 or state.get("task") != "ct_nodule_candidates":
        raise ValueError("Expected a compatible PulmoAI CT checkpoint")
    state.setdefault("package_version", "unknown")
    destination = Path(destination)
    if destination.suffix not in {".pt", ".pth"}:
        raise ValueError("Export destination must use .pt or .pth")
    if (
        destination.resolve() == Path(checkpoint).resolve()
        or destination.exists()
        or destination.with_suffix(".json").exists()
    ):
        raise ValueError("Export destination and its JSON metadata must be new files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = (
        "format_version",
        "package_version",
        "task",
        "model",
        "config",
        "config_fingerprint",
        "manifest_sha256",
        "epoch",
    )
    _atomic_save({key: state[key] for key in keys}, destination)
    write_json(
        destination.with_suffix(".json"),
        {
            "format_version": 1,
            "task": state["task"],
            "sha256": sha256_file(destination),
            "config": state["config"],
            "training_manifest_sha256": state["manifest_sha256"],
            "research_only": True,
            "score_meaning": "nodule candidate score, not malignancy",
        },
    )
    return destination
