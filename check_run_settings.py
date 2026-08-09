"""What was each run actually configured with? Read it off the saved model, not the log."""

import glob

import torch

for name in ("beta_seed_off", "beta_seed_size", "cal_off", "cal_w2"):
    paths = glob.glob(f"/home/alex/runs/{name}/{name}.model")
    if not paths:
        print(f"{name}: no model")
        continue
    model = torch.load(paths[0], map_location="cpu", weights_only=False)
    settings = getattr(model, "size_extensivity_settings", "ABSENT")
    seed = getattr(model, "logit_seed", "ABSENT")
    gamma = getattr(model, "logit_seed_gamma", None)
    print(
        f"{name:16s} size_weight={settings.get('size_weight') if isinstance(settings, dict) else settings}"
        f"  logit_seed={seed}"
        f"  gamma={None if gamma is None else [round(float(v), 4) for v in gamma]}"
    )
