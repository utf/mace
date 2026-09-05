"""Energy and force errors per (size, charge) class on the validation set, for a list of
models.

Per class: n frames, RMSE of the energy per atom, its mean signed error (a constant offset
is a calibration of `c`, not a model error), the RMSE after removing that offset, and the
force RMSE. `--frontier off` evaluates with the frontier term removed (what a Stage B head's
forces were, since its retired E_LR was detached; its energies then lack E_LR).

Usage: python stage13_errors.py --models ~/runs/s13ra_s*/s13ra_s*.model --device cuda:6 \
           --out ~/runs/s13ra_errors.json
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arm_gates import KEYSPEC  # noqa: E402


def batches(atoms_list, z_table, cutoff, batch_size=4):
    configs = [mace_data.config_from_atoms(a, key_specification=KEYSPEC) for a in atoms_list]
    prepare_defect_configurations(configs)
    ds = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=cutoff) for c in configs]
    loader = torch_geometric.dataloader.DataLoader(ds, batch_size=batch_size, shuffle=False)
    return list(loader)


def evaluate(model, ctx, frames, device):
    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    de, fsq, fn = [], [], []
    for b in batches(frames, z_table, ctx.cutoff):
        b = b.to(device)
        d = ctx.forward_dict(b, requires_grad=True)
        out = model(d, training=False, compute_force=True)
        e_pred = out["energy"].detach().cpu().numpy()
        e_ref = b.energy.detach().cpu().numpy()
        n_at = torch.bincount(b.batch, minlength=int(b.num_graphs)).cpu().numpy()
        de.extend(((e_pred - e_ref) / n_at).tolist())
        df = (out["forces"].detach() - b.forces) ** 2
        for g in range(int(b.num_graphs)):
            m = b.batch == g
            fsq.append(float(df[m].sum()))
            fn.append(int(m.sum()) * 3)
    return np.asarray(de), np.asarray(fsq), np.asarray(fn)


def summarise(de, fsq, fn):
    de_mev = 1000.0 * de
    return dict(n=int(len(de)),
                rmse_E_meV_atom=float(np.sqrt((de_mev ** 2).mean())),
                bias_E_meV_atom=float(de_mev.mean()),
                rmse_E_debiased=float(de_mev.std()),
                rmse_F_meV_A=1000.0 * float(np.sqrt(fsq.sum() / max(fn.sum(), 1))))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", type=Path, nargs="+", required=True)
    p.add_argument("--data", type=Path, default=HERE / "dataset_pbe" / "valid.xyz")
    p.add_argument("--device", default="cuda")
    p.add_argument("--frontier", choices=["on", "off"], default="on")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    torch.set_default_dtype(torch.float64)
    frames = read(args.data, ":")
    charged = np.array([int(np.asarray(a.info["carrier_counts"]).sum()) != 0 for a in frames])
    sizes = np.array([len(a) for a in frames])
    groups = {}
    for n in sorted(set(sizes.tolist())):
        for q, name in ((False, "neutral"), (True, "charged")):
            idx = np.where((sizes == n) & (charged == q))[0]
            if len(idx):
                groups[f"{name}_{n}"] = idx
    results = {}
    for path in args.models:
        model = torch.load(path, map_location=args.device, weights_only=False).to(args.device)
        model.eval()
        if args.frontier == "off":
            model.frontier_ewald = None
        ctx = ForwardContext.production(model, device=args.device)
        de, fsq, fn = evaluate(model, ctx, frames, args.device)
        row = {"all": summarise(de, fsq, fn)}
        for name, idx in groups.items():
            row[name] = summarise(de[idx], fsq[idx], fn[idx])
        results[path.stem] = row
        parts = [f"{k}: E {v['rmse_E_meV_atom']:.1f} (bias {v['bias_E_meV_atom']:+.1f}, "
                 f"debiased {v['rmse_E_debiased']:.1f}) F {v['rmse_F_meV_A']:.1f}"
                 for k, v in row.items()]
        print(f"{path.stem} [frontier {args.frontier}] " + " | ".join(parts), flush=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
