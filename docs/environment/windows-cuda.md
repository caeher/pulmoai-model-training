# Windows CUDA environment: historical record

The original repository reported a CUDA diagnostic on 2026-10-15. Those recorded
measurements are preserved below; they were not reproduced in the current CT
implementation session. Current validation uses Python 3.13.14, PyTorch 2.14.0+cpu
and no available CUDA device. A successful old tensor test does not verify the CT
training pipeline on that GPU.

| Original recorded item | Historical value |
| --- | --- |
| OS | Windows 11 Pro 26H2, build 26300.9711, x86-64 |
| Python | 3.13.15, 64 bit |
| GPU | NVIDIA RTX A1000 Laptop, 4096 MiB, compute capability 8.6 |
| Driver | 597.16; nvidia-smi reported CUDA capability 13.2 |
| PyTorch / torchvision / torchaudio | 2.7.1+cu128 / 0.22.1+cu128 / 2.7.1+cu128 |
| PyTorch CUDA runtime | 12.8; CUDA available=True |
| CPU / RAM | 24 logical processors / 32 GB |
| Exact recorded GPU memory | 4,294,443,008 bytes |

The historical test reported CUDA matmul, conv2d and autocast passing, including
float16 autocast. Historical optional imports were torchxrayvision 1.5.5, timm
1.0.30, tensorboard 2.21.0 and wandb 0.30.0; the old report recorded pip check as
passing. These packages are not requirements for the new CT MVP.

For a fresh installation select wheels for the actual OS, interpreter and driver
from the [official PyTorch selector](https://pytorch.org/get-started/locally/) or
[previous versions](https://pytorch.org/get-started/previous-versions/). The old
CUDA 12.8 wheel combination is historical, not a newly verified dependency lock.
The CUDA capability reported by the driver is distinct from torch.version.cuda.
Pip wheels supply their runtime dependencies; a separate full CUDA Toolkit is not
required for normal wheel execution.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,ct]"
python -m pip check
python scripts/smoke_cuda.py
```

Choose/install the correct CUDA PyTorch wheel before the CT extra if using a GPU.
The preserved smoke script creates synthetic tensors only. Without CUDA it exits
2 instead of silently falling back to CPU; operation failure exits 3. The training
CLI independently supports CPU and auto device selection. No training capacity is
implied by the smoke test.

For a 4 GB GPU start with small batches and patches, close competing GPU workloads,
and measure peak memory. Reduce batch/patch size on OOM and record changes in a
new experiment config. Emptying CUDA cache does not free live tensors or repair
an oversized workload. For DLL errors confirm the active Python/wheel, driver,
and Microsoft Visual C++ runtime; do not mix unrelated CUDA DLL paths.

Windows DataLoader uses spawn: keep execution under `if __name__ == "__main__"`
and begin with num_workers=0. The package CLI follows this convention. Confirm
actual installed versions and CUDA status rather than assuming this old report
matches the current machine. No full CT CUDA training or AMP run has been verified.
