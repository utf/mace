#!/usr/bin/env python3
"""D-1: where in the cell does H actually depend on the geometry?

PRE-REGISTERED PREDICTION, AND IT MAY BE NULL. D1 measured the on-site leak at 1-2% and the
r = 0.995 fit ran through missing-neighbour hoppings at physical scales, so the head may
already be as local as it looks. A null result here does NOT gate Edit 3: bounded elements
are adopted on structural grounds (they close the V1 spectrum-shift and the superatom escape
classes), not on this measurement. Recording the prediction before the run is the point.

WHAT IS MEASURED. For every atom j,

    ||dH/dR_j||_F^2 = sum_ab sum_c (dH_ab / dR_jc)^2

by Hutchinson probe: for Rademacher G, grad_j (sum_ab G_ab H_ab) has expected squared norm
exactly the above. K backward passes instead of n^2.

DEFECT-NEAR VS BULK-LIKE IS DEFINED WITHOUT THE VACANCY. An atom is bulk-like when its
block-1 descriptor sits within the pristine spread of its own species -- the reference is the
per-species median over pristine frames and the tolerance is the pristine 95th percentile of
distance to it. That is a statement about features, not coordinates, so it neither uses nor
leaks the defect assignment. The geometric classification is computed too, but only as a
cross-check reported alongside; it is never the definition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from ase.io import read

from mace import tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo, make_batch  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402

N_PROBES = 8


def select_pristine(frames, limit):
    """Stoichiometric frames only, chosen by COMPOSITION -- n(Cl) == 3 n(Pb).

    Not by atom count: the largest cells in this training set are 159-atom DEFECT
    supercells, so "the biggest frames are the pristine ones" silently picks defect cells
    and the bulk-like reference is then built from the very structures it is meant to
    distinguish. Composition is a property of the formula unit, not of the defect
    assignment, so this stays label-free.
    """
    out = []
    for a in frames:
        z = a.get_atomic_numbers()
        if int((z == 17).sum()) == 3 * int((z == 82).sum()):
            out.append(a)
        if len(out) >= limit:
            break
    if not out:
        raise ValueError("no stoichiometric frames in the reference file")
    return out


def hooked(model):
    """Capture the head's internals AND the block-1 descriptor it was handed."""
    grabbed: dict = {}
    head = model.spectral
    original = head.forward

    def wrapped(node_feats, *args, **kwargs):
        grabbed["node_feats"] = node_feats
        kwargs["internals"] = grabbed
        return original(node_feats, *args, **kwargs)

    head.forward = wrapped
    return grabbed, (lambda: setattr(head, "forward", original))


def descriptors(model, atoms, z, cutoff, device):
    """Block-1 descriptor per atom, detached, plus the species vector."""
    batch = make_batch([atoms], z, cutoff, device)
    d = batch.to_dict()
    g, restore = hooked(model)
    try:
        with torch.no_grad():
            model(d, training=False, compute_force=False)
    finally:
        restore()
    feats = g["node_feats"].detach().cpu().numpy()
    species = d["node_attrs"].argmax(dim=-1).detach().cpu().numpy()
    return feats, species


def pristine_reference(model, frames, z, cutoff, device, num_elements):
    """Per-species median block-1 descriptor and the pristine spread around it."""
    per_species = {s: [] for s in range(num_elements)}
    for a in frames:
        feats, species = descriptors(model, a, z, cutoff, device)
        for s in range(num_elements):
            sel = species == s
            if sel.any():
                per_species[s].append(feats[sel])
    ref, tol = {}, {}
    for s, chunks in per_species.items():
        if not chunks:
            continue
        stack = np.concatenate(chunks, axis=0)
        med = np.median(stack, axis=0)
        dist = np.linalg.norm(stack - med, axis=1)
        ref[s] = med
        # The pristine frames carry thermal disorder, so this tolerance already contains the
        # variation a bulk atom shows for reasons that have nothing to do with the defect.
        tol[s] = float(np.percentile(dist, 95))
    return ref, tol


def sensitivity(model, atoms, z, cutoff, device, n_probes=N_PROBES, seed=0):
    """||dH/dR_j|| per atom, plus N_eff and the descriptor distances."""
    batch = make_batch([atoms], z, cutoff, device)
    d = batch.to_dict()
    d["positions"] = d["positions"].detach().requires_grad_(True)
    pos = d["positions"]
    g, restore = hooked(model)
    try:
        out = model(d, training=False, compute_force=False)
    finally:
        restore()

    n = len(atoms)
    slots = g["local"][:n].long()
    ch = int(torch.argmax(d["carrier_counts"].reshape(-1)).item())
    H = g["H"][0, ch][slots][:, slots]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    acc = torch.zeros(n, device=pos.device, dtype=pos.dtype)
    for k in range(n_probes):
        probe = (torch.randint(0, 2, (n, n), generator=gen).to(pos.dtype) * 2 - 1).to(
            pos.device)
        s = (probe * H).sum()
        grad = torch.autograd.grad(s, pos, retain_graph=(k < n_probes - 1),
                                   allow_unused=True)[0]
        if grad is None:
            return None
        acc = acc + (grad ** 2).sum(dim=-1)
    norm = torch.sqrt(acc / n_probes).detach().cpu().numpy()

    alpha = out["carrier_alpha"][:, ch].detach().cpu().numpy()
    s2 = float((alpha ** 2).sum())
    neff = 1.0 / s2 if s2 > 0 else float("nan")
    feats = g["node_feats"].detach().cpu().numpy()
    species = d["node_attrs"].argmax(dim=-1).detach().cpu().numpy()
    return dict(norm=norm, neff=neff, feats=feats, species=species)


def geometric_near(atoms, radius=5.0):
    """Cross-check only, never the definition: atoms within `radius` of the vacancy."""
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return None
    from ase.geometry import get_distances
    pos = atoms.get_positions()
    centre = pos[list(site.shell)].mean(axis=0)[None]
    _, dd = get_distances(centre, pos, cell=atoms.get_cell(), pbc=atoms.pbc)
    return dd[0] <= radius


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path.home() / "runs" / "s1_charged.xyz")
    ap.add_argument("--pristine", type=Path,
                    default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--n-pristine", type=int, default=12)
    ap.add_argument("--probes", type=int, default=N_PROBES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    frames = read(str(args.data), ":")[: args.limit]
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))

    pristine = select_pristine(read(str(args.pristine), ":"), args.n_pristine)
    print(f"pristine reference: {len(pristine)} stoichiometric frames of "
          f"{len(pristine[0])} atoms", flush=True)

    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            print(f"  MISSING {mp}", flush=True)
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        ref, tol = pristine_reference(model, pristine, z, cutoff, args.device,
                                      int(model.atomic_numbers.numel()))

        near_v, bulk_v, agree, n_bulk, n_near, neffs = [], [], [], 0, 0, []
        rank_near_v, rank_bulk_v = [], []
        for atoms in frames:
            r = sensitivity(model, atoms, z, cutoff, args.device, args.probes)
            if r is None:
                continue
            neffs.append(r["neff"])
            dist = np.array([
                np.linalg.norm(r["feats"][i] - ref[int(r["species"][i])])
                if int(r["species"][i]) in ref else np.inf
                for i in range(len(atoms))])
            thresh = np.array([tol.get(int(s), 0.0) for s in r["species"]])
            is_bulk = dist <= thresh
            # Within-frame ranking, reported alongside. The absolute tolerance is calibrated
            # on 80-atom stoichiometric frames and applied to 79-atom charged ones; if those
            # are rattled harder, every atom drifts out together and the absolute split
            # inflates. A per-frame rank split cancels any such uniform offset, so where the
            # two disagree the rank version is the safer read.
            order = np.argsort(dist)
            k_near = max(1, int(round(0.15 * len(atoms))))
            rank_near = np.zeros(len(atoms), dtype=bool)
            rank_near[order[-k_near:]] = True
            rank_bulk = np.zeros(len(atoms), dtype=bool)
            rank_bulk[order[: len(atoms) // 2]] = True
            rank_near_v += list(r["norm"][rank_near])
            rank_bulk_v += list(r["norm"][rank_bulk])
            near_v += list(r["norm"][~is_bulk])
            bulk_v += list(r["norm"][is_bulk])
            n_bulk += int(is_bulk.sum())
            n_near += int((~is_bulk).sum())
            geo = geometric_near(atoms)
            if geo is not None:
                agree.append(float((is_bulk == ~geo).mean()))

        if not near_v and not bulk_v:
            continue
        nv = np.array(near_v) if near_v else np.array([np.nan])
        bv = np.array(bulk_v) if bulk_v else np.array([np.nan])
        rnv = np.array(rank_near_v) if rank_near_v else np.array([np.nan])
        rbv = np.array(rank_bulk_v) if rank_bulk_v else np.array([np.nan])
        row = dict(model=Path(mp).name, n_frames=len(neffs),
                   neff=float(np.mean(neffs)),
                   near_mean=float(np.mean(nv)), bulk_mean=float(np.mean(bv)),
                   near_median=float(np.median(nv)), bulk_median=float(np.median(bv)),
                   ratio=float(np.mean(nv) / max(np.mean(bv), 1e-30)),
                   rank_ratio=float(np.mean(rnv) / max(np.mean(rbv), 1e-30)),
                   rank_near_mean=float(np.mean(rnv)), rank_bulk_mean=float(np.mean(rbv)),
                   frac_near=float(n_near / max(n_near + n_bulk, 1)),
                   geometric_agreement=float(np.mean(agree)) if agree else float("nan"))
        rows.append(row)
        print(f"  {row['model']:34s} N_eff {row['neff']:6.2f}  "
              f"|dH/dR| near {row['near_mean']:.4g}  bulk {row['bulk_mean']:.4g}  "
              f"ratio {row['ratio']:6.2f} (rank {row['rank_ratio']:5.2f})  "
              f"near-frac {row['frac_near']:.2f}  "
              f"geo-agree {row['geometric_agreement']:.2f}", flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    if rows:
        loc = [r for r in rows if r["neff"] <= 8.0]
        del_ = [r for r in rows if r["neff"] > 8.0]
        for name, grp in (("localised (N_eff<=8)", loc), ("delocalised", del_)):
            if grp:
                print(f"\n  {name}: n={len(grp)}  ratio "
                      f"{np.mean([r['ratio'] for r in grp]):.2f}  rank-ratio "
                      f"{np.mean([r['rank_ratio'] for r in grp]):.2f}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
