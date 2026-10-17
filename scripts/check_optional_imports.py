"""Report optional ML and tracking package versions and import status."""

from __future__ import annotations

import importlib
from importlib import metadata

PACKAGES = {
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "SimpleITK": "SimpleITK",
    "pillow": "PIL",
    "mlflow": "mlflow",
    "torchxrayvision": "torchxrayvision",
    "timm": "timm",
    "tensorboard": "tensorboard",
    "wandb": "wandb",
}


def main() -> int:
    failed = False
    for distribution, module_name in PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            failed = True
            print(f"{distribution}: IMPORT FAIL ({type(exc).__name__}: {exc})")
        else:
            module_version = getattr(module, "__version__", "version metadata")
            installed_version = metadata.version(distribution)
            print(f"{distribution} {installed_version}: IMPORT PASS ({module_version})")

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
