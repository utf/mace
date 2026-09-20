"""Two-model plan §3: one float64 check on one batch that the loss gradient w.r.t. the field maps
under cuequivariance matches plain e3nn to 1e-6 (relative, per tensor, and max absolute);
otherwise no cueq. Uses MACE's own converter (run_e3nn_to_cueq) on the init model.

  python defect-perovskite/mh1f2_cueq_check.py --init ~/runs/mh1f2/MACE-MH-1-Field-init.model --out ~/runs/mh1f2/cueq_check.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase.io import read
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools          # noqa: E402
from mace.tools import torch_geometric as tg       # noqa: E402
from mace.modules.loss import UniversalFieldLoss   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
ap.add_argument("--n_frames", type=int, default=6)
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device; t0 = time.time()
D = "/home/alex/runs/mh1f2/data"
model = torch.load(args.init, map_location=dev, weights_only=False).to(dev).double().train()
ks = mace_data.KeySpecification.from_defaults()
ks.info_keys.update(energy="REF_energy", stress="REF_stress", polarization="REF_polarization", polarizability="REF_polarizability", electric_field="REF_electric_field")
ks.arrays_keys.update(forces="REF_forces", becs="REF_becs")
zt = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
fr = [a for a in read(f"{D}/dielectric_valid_seed123.xyz", ":") if len(a) <= 20][: args.n_frames // 2] + [a for a in read(f"{D}/ferroelectric_valid_seed123.xyz", ":") if len(a) <= 30][: args.n_frames - args.n_frames // 2]
for a in fr: a.info["head"] = "mp-dielectric" if "REF_becs" in a.arrays else "mp-ferroelectric"
ads = [mace_data.AtomicData.from_config(mace_data.config_from_atoms(a, key_specification=ks, head_name=a.info["head"]), z_table=zt, cutoff=float(model.r_max), heads=list(model.heads)) for a in fr]
b = next(iter(tg.dataloader.DataLoader(ads, batch_size=len(ads)))).to(dev)
loss_fn = UniversalFieldLoss(energy_weight=1.0, forces_weight=100.0, stress_weight=1.0, polarization_weight=100.0, becs_weight=100.0, polarizability_weight=100.0).to(dev)
# the field maps must be non-zero for the gradient to be informative: perturb field_linear away from zero (both models identically)
with torch.no_grad():
    for n, p in model.named_parameters():
        if n.startswith("field_linear"): p.normal_(0.0, 1e-2)


def grads(m):
    for p in m.parameters(): p.requires_grad_(True)
    m.zero_grad(set_to_none=True)
    out = m(b.to_dict(), training=True, compute_force=True, compute_stress=False, compute_polarization=True, compute_becs=True, compute_polarizability=True)
    loss = loss_fn(pred=out, ref=b); loss.backward()
    return float(loss), {n: p.grad.detach().clone() for n, p in m.named_parameters() if n.startswith("field_")}, {k: out[k].detach().clone() for k in ("energy", "polarization", "becs", "polarizability")}


loss_e, g_e, o_e = grads(model)
res = {"loss_e3nn": loss_e, "n_frames": len(fr), "n_atoms": int(b.positions.shape[0])}
try:
    from mace.cli.convert_e3nn_cueq import run as to_cueq
    mc = to_cueq(model.cpu(), device=dev, return_model=True).to(dev).double().train()
    missing = [k for k in model.state_dict() if k.startswith("field_") and k not in mc.state_dict()]
    assert not missing, ("field tensors not transferred by the converter", missing)
    for k in model.state_dict():
        if k.startswith("field_"): assert torch.equal(model.state_dict()[k].to(dev), mc.state_dict()[k].to(dev)), k
    model = model.to(dev)
    loss_c, g_c, o_c = grads(mc)
    res["loss_cueq"] = loss_c; res["loss_rel_diff"] = abs(loss_c - loss_e) / max(abs(loss_e), 1e-30)
    res["outputs_max_abs_diff"] = {k: float((o_c[k] - o_e[k]).abs().max()) for k in o_e}
    res["field_grad"] = {n: {"max_abs_diff": float((g_c[n] - g_e[n]).abs().max()), "rel_norm_diff": float((g_c[n] - g_e[n]).norm() / max(float(g_e[n].norm()), 1e-30)), "e3nn_norm": float(g_e[n].norm())} for n in g_e}
    res["pass_1e-6"] = all(v["rel_norm_diff"] <= 1e-6 and v["max_abs_diff"] <= 1e-6 * max(v["e3nn_norm"], 1.0) for v in res["field_grad"].values())
    res["cueq"] = "converted"
except Exception as e:                       # pylint: disable=broad-except
    import traceback; res["cueq"] = "conversion failed: " + repr(e); res["traceback"] = traceback.format_exc()[-2000:]; res["pass_1e-6"] = False
Path(args.out).write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "traceback"}, indent=1)); print(f"done {time.time()-t0:.0f}s")
