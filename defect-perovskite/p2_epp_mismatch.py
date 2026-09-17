"""Plan P Phase 2 entry measurement (registered with A3): base + E_PP against base alone on the
neutral validation. With the same base on both sides the difference is the E_PP force itself, so
the number is the per-component RMS of the E_PP forces from the trained base's own dipoles at the
Phase 0 r_split (4.5 A) and eps_inf (4.0), plus the species-signed mean. Threshold tau = 1.7 meV/A.
PASS -> Phase 2 opens with the head adding E_PP; EXCEEDS -> registered remedy (readout-only refit
with E_PP on the target frames alone, p2_readout_refit.py).

    python defect-perovskite/p2_epp_mismatch.py --base RUN/base.pt --out RUN/P2_EPP.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from ase.io import read
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mace import tools                                       # noqa: E402
from mace.modules.dscc import data as dd                     # noqa: E402
from mace.modules.dscc.field import load_field_provider      # noqa: E402
from mace.modules.dscc.polarisation import dipole_dipole_energy   # noqa: E402
from mace.tools import torch_geometric                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--head", default="Default")
ap.add_argument("--frames", default="/home/alex/runs/dscc/p1v2/valid_v2.xyz")
ap.add_argument("--r_split", type=float, default=4.5); ap.add_argument("--eps_inf", type=float, default=4.0)
ap.add_argument("--tau", type=float, default=1.7); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device
prov = load_field_provider(args.base, head=args.head, device=dev)
frames = read(args.frames, index=":"); z3 = tools.AtomicNumberTable([17, 55, 82])
sq = 0.0; n = 0; signed = {17: [], 55: [], 82: []}; per_frame = []
for a in frames:
    a = a.copy(); a.info.setdefault("carrier_counts", np.zeros(4))
    b = next(iter(torch_geometric.dataloader.DataLoader(dd.atomic_data([a], z3, 6.0), batch_size=1))).to(dev).to_dict()
    pos = b["positions"].clone().requires_grad_(True); cell = b["cell"].view(3, 3)
    shifts = torch.einsum("ei,eij->ej", b["unit_shifts"], cell.expand(b["edge_index"].shape[1], 3, 3))
    numbers = torch.tensor([[17, 55, 82][k] for k in b["node_attrs"].argmax(-1).tolist()], device=dev)
    p = prov.node_dipoles(pos, cell, b["batch"], b["ptr"], b["edge_index"], shifts, b["unit_shifts"], numbers)
    e = dipole_dipole_energy(pos, cell, p, args.r_split, args.eps_inf)
    (g,) = torch.autograd.grad(e, pos)
    f = -g.detach().cpu().numpy(); sq += float((f ** 2).sum()); n += f.size
    for z in signed:
        m = numbers.cpu().numpy() == z
        if m.any(): signed[z].append(f[m].mean(0))
    per_frame.append(float(np.sqrt((f ** 2).mean())) * 1e3)
rms = 1e3 * float(np.sqrt(sq / n))
res = {"base": args.base, "frames": args.frames, "n_frames": len(frames), "r_split": args.r_split, "eps_inf": args.eps_inf, "tau_meV_per_A": args.tau,
       "rms_F_PP_meV_per_A": rms, "per_frame_rms_max": max(per_frame), "species_signed_mean_meV_per_A": {str(z): (1e3 * np.mean(v, axis=0)).tolist() for z, v in signed.items() if v},
       "verdict": "PASS" if rms <= args.tau else "EXCEEDS"}
json.dump(res, open(args.out, "w"), indent=1)
print(f"E_PP mismatch on {len(frames)} neutral frames: rms {rms:.3f} meV/A (tau {args.tau}) -> {res['verdict']}; per-frame max {max(per_frame):.2f}")
print("species signed mean (meV/A):", {k: np.round(v, 3).tolist() for k, v in res["species_signed_mean_meV_per_A"].items()})
