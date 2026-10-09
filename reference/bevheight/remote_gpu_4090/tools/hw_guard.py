"""4090 runtime guards. Import after torch is available."""

from __future__ import annotations

import os


def apply_4090_runtime(strict: bool = True) -> dict:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    if not torch.cuda.is_available():
        raise SystemExit("remote_gpu_4090: CUDA not available")
    cap = tuple(torch.cuda.get_device_capability())
    name = torch.cuda.get_device_name(0)
    info = {
        "device": name,
        "capability": list(cap),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tf32": False,
        "flavor": "4090_sm89",
        "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", "8.6+PTX"),
    }
    print(
        f"gpu={name} capability={cap} torch={torch.__version__} "
        f"cuda={torch.version.cuda} tf32=off",
        flush=True,
    )
    if strict and cap != (8, 9):
        raise SystemExit(
            f"remote_gpu_4090 expects sm_89 (RTX 4090 / Ada). Got {cap} ({name}). "
            "This reference was measured on a 4090. Do not mix in 3090 timings."
        )
    return info
