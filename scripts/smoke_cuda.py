"""Exercise PyTorch CUDA, a CUDA operation, and autocast without model assets."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment diagnostic
        print(f"FAIL: PyTorch could not be imported: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print(
            "FAIL: CUDA is required for this smoke test; refusing to fall back to CPU.",
            file=sys.stderr,
        )
        return 2

    try:
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        left = torch.randn((256, 256), device=device)
        right = torch.randn((256, 256), device=device)
        matrix_result = left @ right

        image = torch.randn((2, 3, 64, 64), device=device)
        kernel = torch.randn((8, 3, 3, 3), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            conv_result = torch.nn.functional.conv2d(image, kernel, padding=1)
            amp_matmul = left @ right

        torch.cuda.synchronize(device)
        for label, tensor in (
            ("matmul", matrix_result),
            ("conv2d", conv_result),
            ("autocast matmul", amp_matmul),
        ):
            if tensor.device.type != "cuda":
                raise RuntimeError(f"{label} unexpectedly ran on {tensor.device}")

        print(f"Device: {properties.name}")
        print(f"VRAM: {properties.total_memory / (1024**3):.2f} GiB")
        print(f"Compute capability: {properties.major}.{properties.minor}")
        print(f"CUDA matmul: PASS ({matrix_result.shape}, {matrix_result.device})")
        print(f"CUDA conv2d: PASS ({conv_result.shape}, {conv_result.device})")
        print(f"CUDA autocast: PASS (dtype={amp_matmul.dtype}, {amp_matmul.device})")
    except Exception as exc:  # pragma: no cover - environment diagnostic
        print(f"FAIL: CUDA operation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    print("SMOKE CUDA: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
