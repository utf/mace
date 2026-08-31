"""Report which GPUs can actually take an allocation, one HEALTHY line per usable device.

nvidia-smi is NOT a health check. When b3's GPU3 (PCI 0000:1E:00.0) failed, nvidia-smi still
listed the seven surviving cards with sane memory figures while cudaGetDeviceCount returned
999 and every training run died at init -- including runs pinned to other cards, and including
CUDA_VISIBLE_DEVICES set by UUID rather than index. Only a real allocation separates a device
that is enumerable from one that is usable.

Indices are printed under CUDA_DEVICE_ORDER=PCI_BUS_ID so they mean the same thing here as in
the CUDA_VISIBLE_DEVICES the launcher sets. Callers must set that variable for both.
"""

import sys

try:
    import torch
except Exception as exc:  # noqa: BLE001 - any import failure means "not usable"
    print(f"NO_TORCH {type(exc).__name__}")
    sys.exit(0)

try:
    count = torch.cuda.device_count()
except Exception as exc:  # noqa: BLE001
    print(f"DRIVER_WEDGED {type(exc).__name__}")
    sys.exit(0)

if count == 0:
    # This is the wedged signature seen on b3: no exception, just nothing to use.
    print("NO_DEVICES")
    sys.exit(0)

for i in range(count):
    try:
        device = torch.device(f"cuda:{i}")
        x = torch.randn(1024, 1024, device=device)
        value = float((x @ x).sum().item())
        torch.cuda.synchronize(device)
        del x
        torch.cuda.empty_cache()
        if value != value:  # NaN: the card computes, but not correctly
            print(f"BAD {i} NaN")
        else:
            print(f"HEALTHY {i}")
    except Exception as exc:  # noqa: BLE001
        print(f"BAD {i} {type(exc).__name__}")
