"""v8.1 addendum section 8, audit items 5-6: analytic nuisance intercepts c_g* of saved seeds.

For each model, every charged TRAINING frame's unpaired total-energy residual before any
constant is read as the residual of the saved checkpoint (the checkpoint's own c is part of
its head and is reported), grouped into strata keyed by (label provenance = unpaired, host,
formal charge, composition key, cell convention, size class). Per stratum: the raw weighted
residual mean, the profiled intercept c_g* = -mean, the profiled mean (zero to the analytic
floor), the within-stratum RMS, and, across size strata of the same charge, the intercept
difference. Held-out (validation) frames are then centred with the TRAINING intercepts and
reported as a shape-only diagnostic. Forces are re-read after the profile and asserted
bit-identical (the profile touches no model quantity).

Usage: python stage1_recalibrate.py --models ~/runs/s13ra_s*/s13ra_s*.model --device cuda:5 \
           --out ~/runs/stage1_recalibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401
from ase.io import read

from mace import tools
from mace.modules.defect_context import ForwardContext

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage13_errors import batches  # noqa: E402


def stratum_key(atoms, size_of_pristine: int) -> str:
    counts = np.asarray(atoms.info["carrier_counts"]).reshape(-1)
    q_formal = int(counts[2] + counts[3] - counts[0] - counts[1])   # holes minus electrons
    comp = Counter(atoms.get_chemical_symbols())
    comp_key = ",".join(f"{k}{comp[k]}" for k in sorted(comp))
    size_cls = int(round(len(atoms) / size_of_pristine))
    return f"unpaired|CsPbCl3|Q{q_formal:+d}|{comp_key}|pbc|{size_cls}x"


def residuals(model, ctx, frames, device):
    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    res, forces = [], []
    for b in batches(frames, z_table, ctx.cutoff):
        b = b.to(device)
        d = ctx.forward_dict(b, requires_grad=True)
        out = model(d, training=False, compute_force=True)
        res.extend((out["energy"].detach() - b.energy).cpu().tolist())
        forces.append(out["forces"].detach().cpu())
    return np.asarray(res), torch.cat(forces)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", type=Path, nargs="+", required=True)
    p.add_argument("--train", type=Path, default=HERE / "dataset_pbe" / "train.xyz")
    p.add_argument("--valid", type=Path, default=HERE / "dataset_pbe" / "valid.xyz")
    p.add_argument("--pristine_atoms", type=int, default=80)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    torch.set_default_dtype(torch.float64)
    sets = {}
    for name, path in (("train", args.train), ("valid", args.valid)):
        fr = [a for a in read(path, ":")
              if int(np.asarray(a.info["carrier_counts"]).sum()) != 0]
        keys = [stratum_key(a, args.pristine_atoms) for a in fr]
        sets[name] = (fr, keys)
    results = {}
    for path in args.models:
        model = torch.load(path, map_location=args.device, weights_only=False).to(args.device)
        model.eval()
        ctx = ForwardContext.production(model, device=args.device)
        head = model.spectral
        row = {"c_table_saved": head.c_shift_table.detach().cpu().tolist(),
               "c_shift_saved": float(head.c_shift.detach()), "strata": {}}
        r_tr, f_tr = residuals(model, ctx, sets["train"][0], args.device)
        keys_tr = np.asarray(sets["train"][1])
        intercepts = {}
        for g in sorted(set(keys_tr)):
            v = r_tr[keys_tr == g]
            c_star = -float(v.mean())
            intercepts[g] = c_star
            row["strata"][g] = dict(n_train=int(v.size), raw_mean=float(v.mean()),
                                    c_star=c_star, profiled_mean=float((v + c_star).mean()),
                                    within_rms=float(np.sqrt(((v + c_star) ** 2).mean())),
                                    within_sd=float(v.std(ddof=1)) if v.size > 1 else None)
        r_va, _ = residuals(model, ctx, sets["valid"][0], args.device)
        keys_va = np.asarray(sets["valid"][1])
        for g in sorted(set(keys_va)):
            v = r_va[keys_va == g]
            c_star = intercepts.get(g)
            row["strata"].setdefault(g, {}).update(dict(
                n_valid=int(v.size), valid_raw_mean=float(v.mean()),
                valid_centred_by_train_mean=(None if c_star is None else float((v + c_star).mean())),
                valid_within_rms=(None if c_star is None
                                  else float(np.sqrt(((v + c_star) ** 2).mean())))))
        # between-size intercept difference per charge
        by_q = {}
        for g, c in intercepts.items():
            q = g.split("|")[2]
            by_q.setdefault(q, {})[g.split("|")[-1]] = c
        row["between_size_intercept_difference"] = {
            q: (None if len(d) < 2 else float(max(d.values()) - min(d.values())))
            for q, d in by_q.items()}
        # forces after the profile: the profile is post hoc on energies, nothing else moves
        _, f_again = residuals(model, ctx, sets["train"][0][:8], args.device)
        row["forces_bit_identical"] = bool(torch.equal(f_again, f_tr[: f_again.shape[0]]))
        results[path.stem] = row
        print(path.stem, json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                         for kk, vv in v.items()}
                                     for k, v in row["strata"].items()}),
              "between-size", row["between_size_intercept_difference"],
              "forces identical", row["forces_bit_identical"], flush=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
