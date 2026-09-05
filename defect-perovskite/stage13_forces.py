#!/usr/bin/env python3
"""Plan v8 Stage 1.3: the per-arm numbers the gate queue does not produce -- the 79-atom
force loss on the charged validation frames, the participation (N_eff = 1 / sum alpha^2)
on charged 79- and 159-atom frames, the frontier term and its w, and the learned static
charges Z -- for a list of model files, one JSON.

    python defect-perovskite/stage13_forces.py --models ~/runs/s13a_models/*.model \\
        --out ~/runs/s13a_forces.json --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401
from ase.io import read

from mace import data as mace_data
from mace import tools
from mace.data.defects import prepare_defect_configurations
from mace.modules.defect_context import ForwardContext
from mace.tools import torch_geometric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_gates import KEYSPEC  # noqa: E402

HERE = Path(__file__).resolve().parent


def batches(atoms_list, z_table, cutoff, batch_size=4):
    configs = [mace_data.config_from_atoms(a, key_specification=KEYSPEC) for a in atoms_list]
    prepare_defect_configurations(configs)
    ds = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=cutoff) for c in configs]
    loader = torch_geometric.dataloader.DataLoader(ds, batch_size=batch_size, shuffle=False)
    return list(loader)


def score(model, ctx, frames, natoms, device):
    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    sel = [a for a in frames if len(a) == natoms
           and int(np.asarray(a.info["carrier_counts"]).sum()) != 0]
    sq, n, neff, phi, w = [], 0, [], [], []
    for b in batches(sel, z_table, ctx.cutoff):
        b = b.to(device)
        d = ctx.forward_dict(b, requires_grad=True)
        out = model(d, training=False, compute_force=True)
        f_ref = b.forces
        sq.append(float(((out["forces"] - f_ref) ** 2).sum()))
        n += int(f_ref.numel())
        alpha = out["carrier_alpha"][:, 0].detach()
        for g in range(int(b.num_graphs)):
            a = alpha[b.batch == g]
            neff.append(float(1.0 / (a ** 2).sum().clamp_min(1e-30)))
        phi.extend(out["frontier_energy"].detach().cpu().tolist())
        w.extend(out["frontier_w_ref"].detach().cpu().tolist())
    return dict(n_frames=len(sel), force_rmse_meV_A=1000.0 * float(np.sqrt(sum(sq) / max(n, 1))),
                n_eff_mean=float(np.mean(neff)) if neff else None,
                n_eff_median=float(np.median(neff)) if neff else None,
                phi_ff_mean=float(np.mean(phi)) if phi else None,
                w_ref_mean=float(np.mean(w)) if w else None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", type=Path, nargs="+", required=True)
    p.add_argument("--data", type=Path, default=HERE / "dataset_pbe" / "valid.xyz")
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    torch.set_default_dtype(torch.float64)
    frames = read(str(args.data), index=":")
    charged = [a for a in frames if int(np.asarray(a.info["carrier_counts"]).sum()) != 0]
    frames = charged[: args.limit] if args.limit else charged
    results = {}
    for path in args.models:
        model = torch.load(path, map_location="cpu", weights_only=False).to(args.device).eval()
        ctx = ForwardContext.production(model, device=args.device)
        row = {"madelung_range": getattr(model, "madelung_range", "full"),
               "z": [float(v) for v in model.madelung.z.detach().cpu()]
               if getattr(model, "madelung", None) is not None else None}
        for natoms in (79, 159):
            row[str(natoms)] = score(model, ctx, frames, natoms, args.device)
        results[path.stem] = row
        r79, r159 = row["79"], row["159"]
        print(f"{path.stem}: [{row['madelung_range']}] 79: F {r79['force_rmse_meV_A']:.1f} meV/A "
              f"N_eff {r79['n_eff_mean']:.1f} Phi {r79['phi_ff_mean']:+.3f} w_ref {r79['w_ref_mean']:.2f} | "
              f"159: F {r159['force_rmse_meV_A']:.1f} N_eff {r159['n_eff_mean']:.1f} "
              f"Phi {r159['phi_ff_mean']:+.3f} w_ref {r159['w_ref_mean']:.2f} | Z {np.round(row['z'], 3).tolist()}",
              flush=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
