"""Entry gates for Arm 2+3 on the Arm-1 winners (v4.2): the bound-state precondition on
every neutral-vacancy frame, and the local-neutrality tiling-ladder gate (cubic CsPbCl3
ladder 2/3/4, fixed carrier on a flanking Pb, the model's own q0 pattern vs the species
pattern) with the Madelung coefficient of the cubic convention.
    python defect-perovskite/dscc_arm1_gates.py --winners /home/alex/runs/dscc/dscc_arm1_full_s* --out gates.json
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

from ase import Atoms                                                     # noqa: E402
from mace import tools                                                    # noqa: E402
from mace.modules.dscc import data as dd, ladder, precondition             # noqa: E402
from mace.modules.dscc.train import load_frames                           # noqa: E402
from mace.tools import torch_geometric                                    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winners", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=str(HERE / "dataset_pbe"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--no_ladder", action="store_true", help="precondition only (the Route B' ladder gate is closed)")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
    metas = [dd.frame_meta(i, a, "CsPbCl3", 80) for i, a in enumerate(frames)]
    neutral_vac = [m.index for m in metas if m.state.Q == 0 and m.n_atoms != 80]
    if args.max_frames:
        neutral_vac = neutral_vac[: args.max_frames]
    z_table = tools.AtomicNumberTable([17, 55, 82])
    unit = Atoms("CsPbCl3", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
                 cell=np.eye(3) * 5.6, pbc=True)

    def batch_fn(atoms_list):
        ds = dd.atomic_data(atoms_list, z_table, 10.0)
        return next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=len(ds)))).to(args.device).to_dict()

    report = {}
    for w in args.winners:
        ckpt = Path(w) / "model_dscc.pt" if (Path(w) / "model_dscc.pt").exists() else Path(w) / "model.pt"
        model = torch.load(ckpt, weights_only=False, map_location=args.device).to(args.device).eval()
        batches = []
        ds = dd.atomic_data([frames[i] for i in neutral_vac], z_table, 10.0)
        for b in torch_geometric.dataloader.DataLoader(ds, batch_size=8):
            batches.append(b.to(args.device).to_dict())
        pre = precondition.bound_state_precondition(model, batches)
        # KEEP the per-frame records. They were being dropped, so every change to `Delta_c`
        # cost a fresh 1191-frame sweep per model (~6 min of CPU each); with the separation
        # and n_eff of every frame saved, re-thresholding is arithmetic. Two floats a frame.
        # (`bound_state_precondition` already returns them as dicts.)
        print(Path(w).name, "precondition", pre["passed"], f"{pre['pass_fraction']:.3f}",
              f"sep p50 {1000*pre['separation_p50']:.0f} meV, N_eff p50 {pre['n_eff_p50']:.2f}", flush=True)
        if args.no_ladder:
            report[Path(w).name] = {"precondition": pre}
            continue
        model.route_b = True                      # the pattern needs r_split/q0 machinery only
        lad = ladder.local_neutrality_gate(model, unit, [(2, 2, 2), (3, 3, 3), (4, 4, 4)], z_table, 10.0, batch_fn)
        lad.pop("values", None)
        report[Path(w).name] = {"precondition": pre, "ladder": lad}
        print(Path(w).name, "ladder q0 rel err", f"{lad['relative_error']['q0']:.3f}", "passed", lad["passed"],
              "negative test", lad["negative_test_passed"], flush=True)
    json.dump(report, open(args.out, "w"), indent=1, default=str)
    print("saved", args.out)


if __name__ == "__main__":
    main()
