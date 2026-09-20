"""MACE-MH-1-Field, two-model plan, §3 construction (shared by Model A and Model B).

`MACEField` from MH-1's own config with the head list extended by `mp-dielectric` and
`mp-ferroelectric` (appended at indices 6 and 7 so MH-1's six heads keep their indices). Every
MH-1 tensor is copied; the two new heads are exact copies of `omat_pbe` (E0 row, scale/shift
entry, readout column, and the 16-channel block of the non-linear readout). Field layers as the
Foundation script leaves them (MACEField defaults: one injection after layer 0, tensor product
hidden x 1o -> hidden + Linear hidden -> hidden), `field_linear` zero-initialised, `field_feats`
e3nn default. float64 throughout. Verified on a batch: heads 0-5 equal MH-1, heads 6/7 equal
omat_pbe to 1e-12, field residual at E = 0 exactly zero.

  python defect-perovskite/mh1f2_build.py --out ~/runs/mh1f2/MACE-MH-1-Field-init.model
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase.io import read
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools                          # noqa: E402
from mace.tools import torch_geometric as tg                       # noqa: E402
from mace.tools.scripts_utils import extract_config_mace_model     # noqa: E402
from mace.modules.extensions import MACEField                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mh1", default="/home/alex/.cache/mace/macemh1model")
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu"); ap.add_argument("--check_frames", default="/home/alex/src/2025-04-mace-field/scripts/Dielectrics/MP-Dielectrics-filtered-valid.xyz")
args = ap.parse_args()
torch.set_default_dtype(torch.float64); torch.manual_seed(args.seed); t0 = time.time()
NEW_HEADS = ["mp-dielectric", "mp-ferroelectric"]

mh1 = torch.load(args.mh1, map_location="cpu", weights_only=False).double()
cfg = extract_config_mace_model(mh1)
heads = list(cfg["heads"]); i_om = heads.index("omat_pbe"); n_old = len(heads)
cfg["heads"] = heads + NEW_HEADS; n_new = len(cfg["heads"])
ae = np.asarray(cfg["atomic_energies"]); cfg["atomic_energies"] = np.concatenate([ae] + [ae[i_om:i_om + 1]] * len(NEW_HEADS), 0)
for k in ("atomic_inter_scale", "atomic_inter_shift"):
    v = np.asarray(cfg[k]).reshape(-1); cfg[k] = np.concatenate([v] + [v[i_om:i_om + 1]] * len(NEW_HEADS))
model = MACEField(**cfg)
sd_old, sd_new = mh1.state_dict(), model.state_dict()
copied, extended, skipped = [], [], []
with torch.no_grad():
    for k, v in sd_old.items():
        if k not in sd_new:
            skipped.append(k); continue
        if sd_new[k].shape == v.shape:
            sd_new[k].copy_(v); copied.append(k); continue
        if k in ("atomic_energies_fn.atomic_energies", "scale_shift.scale", "scale_shift.shift"):   # per-head buffers already set from the extended config
            assert torch.equal(sd_new[k][:n_old], v) and all(torch.equal(sd_new[k][n_old + j], v[i_om]) for j in range(len(NEW_HEADS))), k
            extended.append(k); continue
        # head-dependent readout weights (e3nn Linear, one 0e->0e instruction: flat (mul_in, mul_out))
        if k == "readouts.0.linear.weight":                      # 512x0e+512x1o -> n_heads x0e
            w = v.view(-1, n_old); nw = sd_new[k].view(-1, n_new)
            nw[:, :n_old] = w; nw[:, n_old:] = w[:, i_om:i_om + 1]
        elif k == "readouts.1.linear_1.weight":                  # 512x0e -> (c * n_heads) x0e, head-major blocks of c channels
            c = mh1.readouts[1].hidden_irreps.dim // n_old
            w = v.view(-1, n_old, c); nw = sd_new[k].view(-1, n_new, c)
            nw[:, :n_old] = w; nw[:, n_old:] = w[:, i_om:i_om + 1]
        elif k == "readouts.1.linear_2.weight":                  # (c * n_heads) x0e -> n_heads x0e
            c = mh1.readouts[1].hidden_irreps.dim // n_old
            # e3nn Linear normalises each path by 1/sqrt(mul_in): the input multiplicity grows from
            # c*n_old to c*n_new, so the copied weights are rescaled by sqrt(n_new/n_old) to keep the map
            w = v.view(n_old, c, n_old) * (n_new / n_old) ** 0.5; nw = sd_new[k].view(n_new, c, n_new); nw.zero_()
            nw[:n_old, :, :n_old] = w
            for j in range(len(NEW_HEADS)):
                nw[n_old + j, :, n_old + j] = w[i_om, :, i_om]
        elif k.endswith("output_mask") and sd_new[k].numel() == n_new * (v.numel() // n_old):   # e3nn Linear output masks: per-head layout, new model's own values are right
            assert torch.equal(sd_new[k].view(n_new, -1)[:n_old].reshape(-1), v.view(n_old, -1).reshape(-1)), k
        else:
            raise RuntimeError(f"unexpected shape mismatch for {k}: {tuple(v.shape)} vs {tuple(sd_new[k].shape)}")
        extended.append(k)
    model.load_state_dict(sd_new)
    for n, p in model.named_parameters():
        if n.startswith("field_linear"):
            p.zero_()
missing = [k for k in sd_new if k not in sd_old and not k.startswith("field_")]
assert not missing, missing
model = model.double().eval().to(args.device); mh1 = mh1.to(args.device)

# ---- verification on real structures: heads 0-5 = MH-1, heads 6/7 = omat_pbe, field residual 0
frames = read(args.check_frames, ":")[:12]
zt = tools.AtomicNumberTable([int(z) for z in mh1.atomic_numbers]); ks = mace_data.KeySpecification.from_defaults()
def batch(m, head):
    ads = [mace_data.AtomicData.from_config(mace_data.config_from_atoms(a, key_specification=ks, head_name=head), z_table=zt, cutoff=float(m.r_max), heads=list(m.heads)) for a in frames]
    return next(iter(tg.dataloader.DataLoader(ads, batch_size=len(ads)))).to(args.device).to_dict()
n_at = torch.tensor([len(a) for a in frames], dtype=torch.float64, device=args.device)
worst_old = 0.0; worst_new = 0.0
for h in heads:
    o_ref = mh1(batch(mh1, h), training=False, compute_force=True, compute_stress=True)
    o_new = model(batch(model, h), training=False, compute_force=True, compute_stress=True)
    worst_old = max(worst_old, float(((o_new["energy"] - o_ref["energy"]) / n_at).abs().max()), float((o_new["forces"] - o_ref["forces"]).abs().max()))
o_om = model(batch(model, "omat_pbe"), training=False, compute_force=True, compute_stress=True)
for h in NEW_HEADS:
    o_h = model(batch(model, h), training=False, compute_force=True, compute_stress=True)
    worst_new = max(worst_new, float((o_h["energy"] - o_om["energy"]).abs().max()), float((o_h["forces"] - o_om["forces"]).abs().max()), float((o_h["stress"] - o_om["stress"]).abs().max()))
feats = torch.randn(int(n_at.sum()), model.field_feats[0].irreps_in1.dim, device=args.device); zero_field = torch.zeros(int(n_at.sum()), 3, device=args.device)
resid = float(model.field_linear[0](model.field_feats[0](feats, zero_field)).abs().max())
print(f"old heads vs MH-1 (12 frames, float64, meV/atom or meV/A): max {1e3*worst_old:.3e}; new heads vs omat_pbe: max {worst_new:.3e}; field residual at E=0: {resid:.1e}")
assert worst_new < 1e-12 and resid == 0.0 and worst_old < 1e-6

n_total = sum(p.numel() for p in model.parameters()); n_field = sum(p.numel() for n, p in model.named_parameters() if n.startswith("field_"))
model.field_config = {"injection_layers": f"after every interaction layer t < T (T = {int(model.num_interactions)}): {len(model.field_feats)} layer(s)",
                      "tensor_product": "FullyConnectedTensorProduct(hidden x 1o -> hidden) + Linear(hidden -> hidden), residual subtracted (MACEField defaults, as the Foundation script)",
                      "init": "field_feats e3nn default (seed %d), field_linear zero" % args.seed, "dtype": "float64"}
model.build_record = {"mh1": args.mh1, "heads": list(model.heads), "new_heads_copied_from": "omat_pbe", "mh1_params": sum(p.numel() for p in mh1.parameters()),
                      "total_params": n_total, "field_params": n_field, "extended_tensors": extended, "skipped_mh1_buffers": skipped,
                      "check": {"old_heads_max_dev": worst_old, "new_heads_vs_omat_max_dev": worst_new, "field_residual_E0": resid}}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(model.cpu(), args.out); Path(args.out + ".json").write_text(json.dumps(model.build_record, indent=1))
print(f"heads {list(model.heads)}; params total {n_total} (MH-1 {model.build_record['mh1_params']}, field {n_field}); saved {args.out} ({time.time()-t0:.0f}s)")
