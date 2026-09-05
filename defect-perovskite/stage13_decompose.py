"""Energy decomposition of the charged frames: where a per-class energy offset lives.

Per (size, charged) class and model, the mean over frames of `E_pred - E_ref`, the trunk's
own error `base_trunk_energy - E_ref`, the head correction `correction_energy` (band term
plus c, plus the frontier term), and the frontier term alone. Frames come from the
validation set and, with `--train_per_class`, a fixed sample of the training set, so a bias
present on both is a property of the fit, not of generalisation.

Usage: python stage13_decompose.py --models A.model B.model --device cuda:6 --out d.json
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

from mace.modules.defect_context import ForwardContext

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage13_errors import batches  # noqa: E402
from mace import tools  # noqa: E402

KEYS = ("energy", "base_trunk_energy", "correction_energy", "frontier_energy")


def run(model, ctx, frames, device):
    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cols = {k: [] for k in KEYS}
    refs, nat = [], []
    for b in batches(frames, z_table, ctx.cutoff):
        b = b.to(device)
        d = ctx.forward_dict(b, requires_grad=True)
        out = model(d, training=False, compute_force=False)
        for k in KEYS:
            v = out.get(k)
            cols[k].extend((v.detach().cpu().numpy().tolist() if v is not None
                            else [float("nan")] * int(b.num_graphs)))
        refs.extend(b.energy.detach().cpu().numpy().tolist())
        nat.extend(torch.bincount(b.batch, minlength=int(b.num_graphs)).cpu().numpy().tolist())
    c = {k: np.asarray(v) for k, v in cols.items()}
    refs, nat = np.asarray(refs), np.asarray(nat)
    return dict(n=int(len(refs)),
                dE=float((c["energy"] - refs).mean()),
                dE_sd=float((c["energy"] - refs).std()),
                trunk_dE=float((c["base_trunk_energy"] - refs).mean()),
                correction=float(c["correction_energy"].mean()),
                frontier=float(np.nan_to_num(c["frontier_energy"]).mean()),
                band_plus_c=float((c["correction_energy"]
                                   - np.nan_to_num(c["frontier_energy"])).mean()),
                per_atom_bias_meV=float(1000.0 * ((c["energy"] - refs) / nat).mean()))


def classes(frames):
    out = {}
    for a in frames:
        q = int(np.asarray(a.info["carrier_counts"]).sum()) != 0
        out.setdefault(f"{'charged' if q else 'neutral'}_{len(a)}", []).append(a)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", type=Path, nargs="+", required=True)
    p.add_argument("--valid", type=Path, default=HERE / "dataset_pbe" / "valid.xyz")
    p.add_argument("--train", type=Path, default=HERE / "dataset_pbe" / "train.xyz")
    p.add_argument("--train_per_class", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--frontier", choices=["on", "off"], default="on")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    torch.set_default_dtype(torch.float64)
    sets = {"valid": classes(read(args.valid, ":"))}
    if args.train_per_class:
        rng = np.random.default_rng(0)
        tr = classes(read(args.train, ":"))
        sets["train"] = {k: [v[i] for i in rng.choice(len(v), min(len(v), args.train_per_class),
                                                       replace=False)] for k, v in tr.items()}
    results = {}
    for path in args.models:
        model = torch.load(path, map_location=args.device, weights_only=False).to(args.device)
        model.eval()
        if args.frontier == "off":
            model.frontier_ewald = None
        ctx = ForwardContext.production(model, device=args.device)
        row = {}
        for sname, cl in sets.items():
            for cname, fr in cl.items():
                if not cname.startswith("charged"):
                    continue
                r = run(model, ctx, fr, args.device)
                row[f"{sname}/{cname}"] = r
                print(f"{path.stem} [{args.frontier}] {sname}/{cname} n={r['n']}: dE {r['dE']:+.3f} "
                      f"± {r['dE_sd']:.3f} eV ({r['per_atom_bias_meV']:+.1f} meV/atom); trunk "
                      f"{r['trunk_dE']:+.3f}; correction {r['correction']:+.3f} = band+c "
                      f"{r['band_plus_c']:+.3f} + frontier {r['frontier']:+.3f}", flush=True)
        results[path.stem] = row
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
