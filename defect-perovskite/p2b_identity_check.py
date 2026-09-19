"""Plan P Phase 2b §1: the S == 1 identity on a REAL arm B checkpoint (the one permitted use of
the arm B checkpoints): with r_c = 0 the switched model (B_s and B_0) must reproduce arm B's
energy and forces to 1e-9 on a charged 79-atom training frame; with r_c = 8 / delta = 2 the
E_far / E_near decomposition is reported on the same frame.

  python defect-perovskite/p2b_identity_check.py --run ~/runs/dscc/dscc_p2b_s0 --out ~/runs/dscc/p2b/identity_s0.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mace import tools                                     # noqa: E402
from mace.modules.dscc import data as dd                   # noqa: E402
from mace.modules.dscc.train import load_frames            # noqa: E402
from mace.tools import torch_geometric                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dataset", default=str(HERE / "dataset_pbe")); ap.add_argument("--device", default="cuda")
ap.add_argument("--frame", type=int, default=-1, help="index into the charged training frames (-1 = first)")
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
model = torch.load(Path(args.run) / "model.pt", weights_only=False, map_location=args.device).to(args.device).eval()
# The identity is read with the base in float64: the float32 base path (cueq/cuBLAS reductions)
# has a run-to-run noise of ~3e-5 eV on this frame, far above the 1e-9 the plan asks for.
model.base.double()
frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
metas = [dd.frame_meta(i, a, "CsPbCl3", pristine_atoms=80) for i, a in enumerate(frames)]
charged = [m.index for m in metas if m.state.Q != 0 and m.n_atoms == 79]
idx = charged[0] if args.frame < 0 else charged[args.frame]
z_table = tools.AtomicNumberTable(model.atomic_numbers)
r_cut = float(model.r_cut)
b = next(iter(torch_geometric.dataloader.DataLoader(dd.atomic_data([frames[idx]], z_table, r_cut), batch_size=1))).to(args.device).to_dict()
def run(near, r_c, delta):
    model.near_field, model.r_c, model.delta = near, float(r_c), float(delta)
    out = model(dict(b), training=False, compute_force=True)
    return {"energy": float(out["energy"]), "forces": out["forces"].detach().cpu().numpy(),
            "e_host": out["diagnostics"]["e_host"][0], "e_far": out["diagnostics"].get("e_far", [None])[0],
            "e_near": out["diagnostics"].get("e_near", [None])[0], "dq_sum": float(out["dq"].sum())}
ref = run("off", 8.0, 2.0)
res = {"run": args.run, "frame": int(idx), "n_atoms": int(len(frames[idx])), "r_cut": r_cut, "arm_B": {k: v for k, v in ref.items() if k != "forces"}}
for near in ("mulliken", "none"):
    ident = run(near, 0.0, 2.0)
    res[f"{near}_S1"] = {"dE": abs(ident["energy"] - ref["energy"]), "dF_max": float(np.abs(ident["forces"] - ref["forces"]).max()),
                        "pass": abs(ident["energy"] - ref["energy"]) < 1e-9 and float(np.abs(ident["forces"] - ref["forces"]).max()) < 1e-9}
    sw = run(near, 8.0, 2.0)
    res[f"{near}_r8"] = {"energy": sw["energy"], "e_far": sw["e_far"], "e_near": sw["e_near"], "e_cP_armB": ref["e_host"],
                         "F_rms_diff_vs_armB_meV_A": 1e3 * float(np.sqrt(((sw["forces"] - ref["forces"]) ** 2).mean())), "dq_sum": sw["dq_sum"]}
    print(f"{near}: S==1 identity dE {res[f'{near}_S1']['dE']:.2e} eV, dF_max {res[f'{near}_S1']['dF_max']:.2e} eV/A -> {'PASS' if res[f'{near}_S1']['pass'] else 'FAIL'}; "
          f"r_c 8: E_far {sw['e_far']:+.4f} eV, E_near {sw['e_near']:+.4f} eV (arm B E_cP {ref['e_host']:+.4f}), force change rms {res[f'{near}_r8']['F_rms_diff_vs_armB_meV_A']:.2f} meV/A")
json.dump(res, open(args.out, "w"), indent=1); print("saved", args.out)
