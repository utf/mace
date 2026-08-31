"""R1 -- can any head variant fit the hub better than the cage?

DIAGNOSTIC ONLY. Uses the vacancy assignment to build the masks. Never ships; production
configs must refuse `clamp_mask`.

The question this cycle turns on is site SELECTION, not compactness. Every compact failure so
far sat inside the vacancy neighbourhood and simply chose the wrong atoms in it -- E1 put 7 of
8 seeds on the ligand cage, and its one near-correct seed fitted forces WORSE than the cage
seeds. A spoke carrier reproduces a Pb-dominated force pattern automatically, with five times
the fitting freedom of a two-atom state, so no generic push toward localisation will select
the hub until the head prefers it.

So force the answer and read off the fit. Clamp the carrier onto each candidate site set, train
the head briefly against a frozen Stage-A base, and compare:

    hub   the two vacancy-sharing Pb            (the physically correct answer)
    cage  their ten ligand Cl                   (what the free head keeps choosing)
    nbhd  hub + cage                            (both available: where does it put weight?)
    free  unclamped                             (today's behaviour)

PASS for a variant: hub force RMSE at least 10% below cage, AND the Pb fraction of alpha under
`nbhd` at least 0.5. The first variant that passes advances to R2.

Only the head trains: the base is frozen, so this measures the head's representational
preference and not the trunk's ability to compensate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401
from ase.io import read

from mace import data as mace_data
from mace import tools
from mace.data.defects import prepare_defect_configurations
from mace.modules.defect_stage import is_correction_param
from mace.tools import torch_geometric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_gates import KEYSPEC  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402

MASKS = ("hub", "cage", "nbhd", "free")


def masks_for(atoms):
    """Per-atom boolean masks for the candidate site sets, or None if unlocatable."""
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return None
    if site.cage is None or len(site.cage) == 0:
        return None
    n = len(atoms)
    hub = np.zeros(n, dtype=bool)
    hub[np.asarray(site.shell, dtype=int)] = True
    cage = np.zeros(n, dtype=bool)
    cage[np.asarray(site.cage, dtype=int)] = True
    return dict(hub=hub, cage=cage, nbhd=hub | cage, free=np.ones(n, dtype=bool),
                is_pb=np.array(atoms.get_chemical_symbols()) == "Pb", hub_idx=site.shell)


def make_batches(frames, z_table, cutoff, batch_size, device):
    configs = [mace_data.config_from_atoms(a, key_specification=KEYSPEC) for a in frames]
    prepare_defect_configurations(configs)
    atomic = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=cutoff)
              for c in configs]
    loader = torch_geometric.dataloader.DataLoader(atomic, batch_size=batch_size,
                                                   shuffle=False)
    out = []
    start = 0
    for batch in loader:
        n = int(batch.num_graphs)
        out.append((batch.to(device), frames[start:start + n]))
        start += n
    return out


def stacked_mask(frame_masks, which, device):
    return torch.tensor(np.concatenate([m[which] for m in frame_masks]),
                        dtype=torch.bool, device=device)


def run_one(model_path, mask_name, batches, frame_masks, device, epochs, lr):
    """Train the head only, with the carrier clamped to `mask_name`, and report the fit."""
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    model.train()

    # Head only. The base is frozen so this measures the HEAD's preference; if the trunk could
    # move, it would simply absorb whichever choice the head made, which is the laundering
    # that made the free-attention arms uninformative in the first place.
    head_params = [p for n, p in model.named_parameters() if is_correction_param(n)]
    for n, p in model.named_parameters():
        p.requires_grad_(is_correction_param(n))
    opt = torch.optim.AdamW(head_params, lr=lr)

    for _ in range(epochs):
        for batch, frames in batches:
            fm = [frame_masks[id(a)] for a in frames]
            m = None if mask_name == "free" else stacked_mask(fm, mask_name, device)
            d = batch.to_dict()
            d["_clamp_mask"] = m
            out = model(d, training=True, compute_force=True)
            loss = (out["forces"] - batch.forces).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    # Evaluate: force RMSE, residual on the hub, |eps| range, and -- for nbhd -- where the
    # head actually put the weight when both site sets were available.
    model.eval()
    sq, cnt, hub_res, pb_frac, eps_max = 0.0, 0, [], [], 0.0
    for batch, frames in batches:
        fm = [frame_masks[id(a)] for a in frames]
        m = None if mask_name == "free" else stacked_mask(fm, mask_name, device)
        d = batch.to_dict()
        d["_clamp_mask"] = m
        out = model(d, training=False, compute_force=True)
        err = (out["forces"] - batch.forces).detach().cpu().numpy()
        sq += float((err ** 2).sum())
        cnt += err.size
        eps_max = max(eps_max, float(out["carrier_readouts"].abs().max()))

        idx = batch.batch.detach().cpu().numpy()
        alpha = out["carrier_alpha"].detach().cpu().numpy()
        for k, a in enumerate(frames):
            sel = idx == k
            mm = fm[k]
            hub_res.append(float(np.linalg.norm(err[sel][mm["hub"]], axis=1).mean()))
            act = alpha[sel][:, 2]
            tot = act.sum()
            if tot > 0:
                pb_frac.append(float(act[mm["is_pb"] & mm["nbhd"]].sum() / tot))

    return dict(force_rmse=float(np.sqrt(sq / cnt)) * 1000.0,
                hub_residual=float(np.mean(hub_res)) * 1000.0,
                pb_fraction=float(np.mean(pb_frac)) if pb_frac else float("nan"),
                eps_max=eps_max)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", type=Path, nargs="+", required=True,
                    help="one model file per head variant (H0, H1, H2, ...)")
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "valid.xyz")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    frames = [a for a in read(args.data, ":")
              if np.asarray([int(x) for x in a.info["carrier_counts"].split()]
                            if isinstance(a.info["carrier_counts"], str)
                            else a.info["carrier_counts"]).any()]
    keep, frame_masks = [], {}
    for a in frames:
        m = masks_for(a)
        if m is not None:
            keep.append(a)
            frame_masks[id(a)] = m
        if len(keep) >= args.frames:
            break
    print(f"R1 on {len(keep)} charged frames, {args.epochs} epochs, head only\n")

    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    results = {}
    for path in args.variants:
        cutoff = float(torch.load(path, map_location="cpu",
                                  weights_only=False).spectral_r_cut)
        batches = make_batches(keep, z_table, cutoff, args.batch_size, args.device)
        row = {}
        for mask in MASKS:
            t0 = time.perf_counter()
            row[mask] = run_one(path, mask, batches, frame_masks, args.device,
                                args.epochs, args.lr)
            row[mask]["seconds"] = time.perf_counter() - t0
        results[path.stem] = row

        hub, cage = row["hub"]["force_rmse"], row["cage"]["force_rmse"]
        gain = (cage - hub) / max(cage, 1e-12)
        pbf = row["nbhd"]["pb_fraction"]
        verdict = "PASS" if (gain >= 0.10 and pbf >= 0.5) else "FAIL"
        row["verdict"] = dict(verdict=verdict, hub_gain=gain, nbhd_pb_fraction=pbf)

        print(f"=== {path.stem} ===")
        print(f"  {'mask':<6} {'F RMSE':>9} {'hub resid':>10} {'nbhd Pb':>8} {'|eps|max':>9}")
        for mask in MASKS:
            r = row[mask]
            print(f"  {mask:<6} {r['force_rmse']:>9.1f} {r['hub_residual']:>10.1f} "
                  f"{r['pb_fraction']:>8.3f} {r['eps_max']:>9.2f}")
        print(f"  hub vs cage: {gain:+.1%} force RMSE   nbhd Pb fraction {pbf:.3f}"
              f"   -> {verdict}\n")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2, default=float))
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
