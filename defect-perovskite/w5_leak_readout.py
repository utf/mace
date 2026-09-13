"""W5's registered leak readout: the HEAD's own axial force on the flanking Pb pair as a
function of the collective coordinate `d`, fitted separately at 79 and 159 atoms.

Why this quantity. `d` is the flanking Pb-Pb minimum-image distance (`collective_coordinate`),
the coordinate the whole programme reads defect physics against, and the axial hub force is
fitted on every charged frame at both sizes -- unlike the energy residual, which is a
by-product of a forces-only loss. A head whose correction leaks across the cell shows a
size-dependent `d`-slope: the 159-atom cell puts the periodic image twice as far away, so a
local head must give the same slope at both sizes and a leaking one must not. The label slope
(`F_DFT - F_base`, same projection) is the primary guard: the head is asked to reproduce the
labels' own `d`-trend, not to be flat.

    python defect-perovskite/w5_leak_readout.py --models /home/alex/runs/dscc/dscc_w6_s*/model.pt \
        --out /home/alex/runs/dscc/leak_w6.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mace import tools                                                     # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.admission import collective_coordinate              # noqa: E402
from mace.modules.dscc.kernels import minimum_image_distances              # noqa: E402
from mace.modules.dscc.train import load_frames, vacancy_centre_full       # noqa: E402
from mace.tools import torch_geometric                                     # noqa: E402


def axial(forces: torch.Tensor, pos: torch.Tensor, cell: torch.Tensor, flank: torch.Tensor) -> float:
    """The hub component: `1/2 (F_i - F_j) . u`, `u` the unit vector along the minimum-image
    flanking Pb pair. Positive = the pair is pushed apart."""
    i, j = int(flank[0]), int(flank[1])
    dv = pos[i] - pos[j]
    dv = dv - torch.round(dv @ torch.linalg.inv(cell)) @ cell
    u = dv / dv.norm()
    return float(0.5 * ((forces[i] - forces[j]) @ u))


def fit(d: np.ndarray, f: np.ndarray):
    """Slope of `f` on `d` with its standard error (meV/A per A when f is in meV/A)."""
    if d.size < 3:
        return None
    A = np.stack([np.ones_like(d), d], axis=1)
    beta, *_ = np.linalg.lstsq(A, f, rcond=None)
    resid = f - A @ beta
    dof = max(d.size - 2, 1)
    cov = np.linalg.inv(A.T @ A) * float(resid @ resid) / dof
    return {"n": int(d.size), "slope": float(beta[1]), "slope_se": float(np.sqrt(cov[1, 1])),
            "intercept": float(beta[0]), "d_range": [float(d.min()), float(d.max())]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dataset", default=str(HERE / "dataset_pbe"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_79", type=int, default=0, help="cap the 79-atom set (debug)")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
    metas = [dd.frame_meta(i, a, "CsPbCl3", pristine_atoms=80) for i, a in enumerate(frames)]
    charged = [m.index for m in metas if m.state.Q != 0]
    if args.max_79:
        small = [i for i in charged if len(frames[i]) == 79][: args.max_79]
        charged = small + [i for i in charged if len(frames[i]) != 79]
    z_table = tools.AtomicNumberTable([17, 55, 82])
    report = {}
    for path in args.models:
        name = Path(path).parent.name
        model = torch.load(path, weights_only=False, map_location=args.device).to(args.device).eval()
        rows = []
        for idx in charged:
            atoms = frames[idx]
            ds = dd.atomic_data([atoms], z_table, model.r_cut)
            batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(args.device).to_dict()
            pos = batch["positions"].detach()
            cell = batch["cell"].view(3, 3).detach()
            numbers = atoms.get_atomic_numbers()
            found = vacancy_centre_full(pos, cell, numbers)
            d = collective_coordinate(pos, cell, numbers)
            if found is None or d is None:
                continue
            out = model(batch, compute_force=True)
            base = model.base_forward(model._trunk_data(dict(batch)), compute_force=True)
            f_head = (out["forces"].detach() - base["forces"].detach())
            f_label = (batch["forces"].detach() - base["forces"].detach())
            rows.append({"n": len(atoms), "d": float(d),
                         "head": 1000 * axial(f_head, pos, cell, found[1]),
                         "label": 1000 * axial(f_label, pos, cell, found[1])})
        entry = {}
        for size in sorted({r["n"] for r in rows}):
            sub = [r for r in rows if r["n"] == size]
            dd_ = np.array([r["d"] for r in sub])
            entry[str(size)] = {"head": fit(dd_, np.array([r["head"] for r in sub])),
                                "label": fit(dd_, np.array([r["label"] for r in sub]))}
        if "79" in entry and "159" in entry and entry["79"]["head"] and entry["159"]["head"]:
            a, b = entry["79"]["head"], entry["159"]["head"]
            entry["head_79_minus_159"] = {
                "difference": a["slope"] - b["slope"],
                "se": float(np.hypot(a["slope_se"], b["slope_se"]))}
            la, lb = entry["79"]["label"], entry["159"]["label"]
            entry["label_79_minus_159"] = {"difference": la["slope"] - lb["slope"],
                                           "se": float(np.hypot(la["slope_se"], lb["slope_se"]))}
        report[name] = entry
        print(name, json.dumps(entry.get("head_79_minus_159", {})), flush=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
