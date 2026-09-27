"""Deliverables: convert a trained (cueq) MACE-MH-1-Field checkpoint back to plain e3nn ops so it
loads without cuequivariance, copying the field layers' state across MACE's converter (which
re-instantiates them) and checking the two models agree on a batch (energy, forces, Z*, α, P).

  python defect-perovskite/mh1f2_to_e3nn.py --cueq deliver/X_cueq.model --out deliver/X.model
"""
import argparse, json, sys
from pathlib import Path
import torch
from ase.io import read
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools          # noqa: E402
from mace.tools import torch_geometric as tg       # noqa: E402
from mace.cli.convert_cueq_e3nn import run as to_e3nn   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cueq", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
ap.add_argument("--frames", default="/home/alex/runs/mh1f2/data_mh0/dielectric_valid_seed123.xyz")
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device
cu = torch.load(args.cueq, map_location="cpu", weights_only=False).double()
e3 = to_e3nn(cu, device="cpu", return_model=True).double()
sd_c, sd_e = cu.state_dict(), e3.state_dict()
e3.load_state_dict({k: v.reshape(sd_e[k].shape) for k, v in sd_c.items() if k.startswith("field_") and k in sd_e}, strict=False)
for attr in ("field_config", "build_record", "training_record"):
    if hasattr(cu, attr): setattr(e3, attr, getattr(cu, attr))
cu = cu.to(dev).eval(); e3 = e3.to(dev).eval()
zt = tools.AtomicNumberTable([int(z) for z in cu.atomic_numbers]); ks = mace_data.KeySpecification.from_defaults()
fr = [a for a in read(args.frames, ":") if len(a) <= 10][:3]
ads = [mace_data.AtomicData.from_config(mace_data.config_from_atoms(a, key_specification=ks, head_name="mp-dielectric"), z_table=zt, cutoff=float(cu.r_max), heads=list(cu.heads)) for a in fr]
b = next(iter(tg.dataloader.DataLoader(ads, batch_size=len(ads)))).to(dev)
kw = dict(training=True, compute_force=True, compute_stress=False, compute_polarization=True, compute_becs=True, compute_polarizability=True)
oc = cu(b.to_dict(), **kw); oe = e3(b.to_dict(), **kw)
diff = {k: float((oc[k].detach() - oe[k].detach()).abs().max()) for k in ("energy", "forces", "polarization", "becs", "polarizability")}
ok = diff["energy"] < 1e-6 and diff["forces"] < 1e-6 and diff["becs"] < 1e-6 and diff["polarizability"] < 1e-6
e3.conversion_check = {"cueq_source": args.cueq, "max_abs_diff": diff, "pass": ok}
torch.save(e3.cpu(), args.out); print(json.dumps(e3.conversion_check)); assert ok
