"""Explicit offline commands; heavy dependencies load only for the selected task."""

from __future__ import annotations

import argparse
import json

from pulmoai.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PulmoAI CT nodule research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-config", "prepare", "synthetic", "train"):
        command = commands.add_parser(name)
        command.add_argument("--config-dir", default="configs")
        command.add_argument("--config", default="experiment")
        if name != "validate-config":
            command.add_argument("--output", required=True)
        if name == "prepare":
            command.add_argument("--scans", required=True)
            command.add_argument("--annotations", required=True)
            command.add_argument("--patients", required=True)
            command.add_argument("--excluded")
        if name == "train":
            command.add_argument("--manifest", required=True)
            command.add_argument("--resume")
    validate = commands.add_parser("validate-data")
    validate.add_argument("--manifest", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    infer = commands.add_parser("infer")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--image", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--preview", help="Optional local PNG slice contact sheet")
    infer.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config_dir, args.config) if hasattr(args, "config") else None
        if args.command == "validate-config":
            print(json.dumps(config.to_dict(), indent=2))
        elif args.command == "prepare":
            from pulmoai.datasets.manifest import prepare_luna

            print(
                prepare_luna(
                    args.scans, args.annotations, args.patients, args.output, config, args.excluded
                )
            )
        elif args.command == "validate-data":
            from pulmoai.datasets.manifest import load_manifest

            manifest = load_manifest(args.manifest)
            print(f"Validated {len(manifest['scans'])} scans with patient-disjoint splits")
        elif args.command == "synthetic":
            from pulmoai.datasets.synthetic import create_fixture

            print(create_fixture(args.output, config))
        elif args.command == "train":
            from pulmoai.training.engine import train

            print(train(config, args.manifest, args.output, args.resume))
        elif args.command == "evaluate":
            from pulmoai.evaluation.runner import evaluate

            metrics = evaluate(args.checkpoint, args.manifest, args.output, args.split, args.device)
            print(json.dumps({k: metrics[k] for k in ("split", "synthetic", "cpm", "protocol")}))
        elif args.command == "infer":
            from pulmoai.inference.detector import Detector

            result = Detector.from_checkpoint(args.checkpoint, args.device).predict(
                args.image, args.output
            )
            if args.preview:
                from pulmoai.inference.preview import save_preview

                save_preview(args.image, result, args.preview)
            print(f"Wrote {len(result['candidates'])} research candidates")
        elif args.command == "export":
            from pulmoai.training.engine import export_model

            print(export_model(args.checkpoint, args.output))
    except (ValueError, OSError, KeyError, ImportError, RuntimeError) as error:
        parser.exit(2, f"PulmoAI error: {error}\n")
    return 0
