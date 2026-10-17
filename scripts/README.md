# Scripts

The preferred offline interface is `python -m pulmoai --help` (or `pulmoai`).
`prepare_data.py` delegates to `pulmoai prepare` for previously acquired LUNA16
files. `acquire_data.py` remains an explicitly nonfunctional acquisition guard:
it never downloads data, even after network opt-in. Importing code never runs a
script or loads credentials.

`smoke_cuda.py` preserves the existing synthetic CUDA matmul, convolution and
autocast diagnostic. It fails clearly if CUDA is unavailable. It does not train a
model or load data/weights. `check_optional_imports.py` reports the current CT and
optional tracking imports; missing optional packages are reported with exit code 1.
