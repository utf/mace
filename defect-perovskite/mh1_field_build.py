"""MACE-MH-1-Field, step 1: build the field-aware MH-1 (plan §1). Instantiate `MACEField` with
MH-1's own hyperparameters (extract_config_mace_model), copy every MH-1 tensor, zero-initialise the
field-coupling parameters (field_feats / field_linear; the only new tensors), freeze everything
else, cast to float32 (plan §3), save the initial model and the parameter counts.

  python defect-perovskite/mh1_field_build.py --out ~/runs/mh1field/MACE-MH-1-Field-init.model
"""
import argparse, json, sys, time
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace.tools.scripts_utils import extract_config_mace_model   # noqa: E402
from mace.modules.extensions import MACEField                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mh1", default="/home/alex/.cache/mace/macemh1model")
ap.add_argument("--out", required=True); ap.add_argument("--json", default=None)
args = ap.parse_args()
t0 = time.time()
mh1 = torch.load(args.mh1, map_location="cpu", weights_only=False)
cfg = extract_config_mace_model(mh1)
model = MACEField(**cfg)
missing, unexpected = model.load_state_dict(mh1.state_dict(), strict=False)
assert all(k.startswith("field_") for k in missing), missing
# the unexpected keys are e3nn buffers of MH-1's gated nonlinearity (`mul.weight` of size 0 and
# its output mask) that this MACE version's block does not register; the exactness test decides
print("missing (new field tensors):", missing); print("unexpected (buffers, not parameters):", unexpected)
with torch.no_grad():
    for n, p in model.named_parameters():
        if n.startswith("field_"):
            p.zero_()
for n, p in model.named_parameters():
    p.requires_grad_(n.startswith("field_"))
model = model.float()
n_total = sum(p.numel() for p in model.parameters()); n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_mh1 = sum(p.numel() for p in mh1.parameters())
rec = {"mh1": args.mh1, "mh1_params": n_mh1, "total_params": n_total, "trainable_field_params": n_train,
       "field_tensors": [(n, list(p.shape)) for n, p in model.named_parameters() if p.requires_grad],
       "heads": list(model.heads), "n_elements": int(model.atomic_numbers.numel()), "num_interactions": int(model.num_interactions),
       "field_injection_layers": len(model.field_feats), "hidden_irreps": str(cfg["hidden_irreps"]), "dtype": "float32",
       "missing_keys": missing, "unexpected_keys": unexpected}
model.field_config = {"injection_layers": "after every interaction layer t < T (T = %d): %d layer(s)" % (model.num_interactions, len(model.field_feats)),
                      "tensor_product": "FullyConnectedTensorProduct(hidden x 1o -> hidden) + Linear(hidden -> hidden), residual subtracted", "init": "zero", "frozen": "all MH-1 tensors"}
torch.save(model, args.out)
json.dump(rec, open(args.json or (args.out + ".json"), "w"), indent=1)
print(f"MH-1 params {n_mh1}; MH-1-Field total {n_total}; trainable (field) {n_train}; saved {args.out} ({time.time()-t0:.0f}s)")
