"""Test 1: does the head bind the carrier at SHORT d(Pb-Pb), even though it does not on average?

R2 scored localisation over the whole charged ensemble and got 0/20. But the charged ensemble
has median d(Pb-Pb) = 5.49 A (undimerised) while the neutral ensemble sits at 4.72 A (a
contracted dimer), and the underlying physics is a dynamic vacancy level that moves in and out
of the gap as the pair breathes. An ensemble-averaged "not localised" is then consistent with
a head that binds at short d and not at long d -- in which case the 0/20 was a criterion
error, not a model failure.

So: every metric, resolved by d rather than pooled.

  binds at short d : N_eff <= 8 and Delta >> 20 meV in the 4.8-5.1 A bins, rising to band-like
                     above ~5.4 A -> the head learned a dynamic level
  flat             : N_eff > 30 in every bin -> the head never binds

No autograd here (nothing is differentiated), so the whole charged set is affordable.

Test 1b evaluates the same head with n = (0,0,1,0) on NEUTRAL V_Cl0 frames, which sit at
d ~ 4.7 A and are dimerised. That is off-distribution for the counters and therefore
diagnostic only -- it asks "does a level split off when the geometry is a dimer?", not "what
is the energy".

The vacancy assignment supplies d, the hub pair and the hub mass. It never enters the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from ase.geometry import get_distances
from ase.io import read

from mace import tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo, make_batch  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402

SPECIES = {17: "Cl", 55: "Cs", 82: "Pb"}


def counters_of(atoms):
    c = atoms.info.get("carrier_counts")
    if isinstance(c, str):
        c = [int(x) for x in c.split()]
    return np.asarray(c, dtype=int)


def frame_metrics(model, atoms, z_table, cutoff, device, force_counts=None):
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return None
    a, b = int(site.shell[0]), int(site.shell[1])
    pos = atoms.get_positions()
    _, dist = get_distances(pos[a][None], pos[b][None], cell=atoms.get_cell(), pbc=atoms.pbc)
    d_hub = float(dist[0, 0])

    batch = make_batch([atoms], z_table, cutoff, device)
    data = batch.to_dict()
    if force_counts is not None:
        # Test 1b: ask the hole channel what it would do at a dimerised neutral geometry.
        data["carrier_counts"] = torch.tensor(
            [force_counts], dtype=data["carrier_counts"].dtype, device=device)

    grabbed: dict = {}
    head = model.spectral
    original = head.forward

    def wrapped(*args, **kwargs):
        kwargs["internals"] = grabbed
        return original(*args, **kwargs)

    head.forward = wrapped
    try:
        with torch.no_grad():
            out = model(data, training=False, compute_force=False)
    finally:
        head.forward = original

    counts = data["carrier_counts"].reshape(-1)
    c = int(torch.argmax(counts).item())
    n = len(atoms)

    alpha = out["carrier_alpha"] if "carrier_alpha" in out else None
    if alpha is None:
        # alpha is exposed on the spectral output; rebuild from psi/w if the model does not
        # surface it under that key.
        psi, w = grabbed["psi"], grabbed["w"]
        slots = grabbed["local"][:n].long()
        alpha_full = (psi[0][:, slots, :].pow(2) * w[0].unsqueeze(1)).sum(-1).T  # [n, C]
    else:
        alpha_full = alpha[:n]

    def n_eff_of(vec):
        v = vec / max(float(vec.sum()), 1e-30)
        return float(1.0 / float(v.pow(2).sum()))

    n_eff_active = n_eff_of(alpha_full[:, c])
    nulls = [i for i in range(alpha_full.shape[1]) if i != c]
    n_eff_null = float(np.mean([n_eff_of(alpha_full[:, i]) for i in nulls]))

    lam = grabbed["lam"][0, c]
    physical = lam < 500.0
    lam_p = lam[physical]
    delta = float(lam_p[1] - lam_p[0]) if lam_p.numel() > 1 else float("nan")

    slots = grabbed["local"][:n].long()
    H_c = grabbed["H"][0, c][slots][:, slots]
    z = np.array(atoms.get_atomic_numbers())
    av = alpha_full[:, c].detach().cpu().numpy()
    tot = max(av.sum(), 1e-30)

    return dict(
        d=d_hub, n_atoms=n,
        n_eff=n_eff_active, n_eff_null=n_eff_null,
        null_ratio=n_eff_active / max(n_eff_null, 1e-30),
        delta=delta,
        hub_mass=float(av[[a, b]].sum() / tot),
        mass={SPECIES.get(int(k), str(k)): float(av[z == k].sum() / tot)
              for k in sorted(set(z.tolist()))},
        hub_hop=float(abs(H_c[a, b])),
    )


def binned_report(rows, edges, label):
    print(f"\n=== {label} ===")
    print(f"  {'d bin (A)':>12} {'n':>5} {'N_eff':>8} {'ratio':>7} {'Delta(eV)':>10} "
          f"{'hub':>6} {'|H_ab|':>8} {'Pb':>5} {'Cl':>5}")
    out = []
    d = np.array([r["d"] for r in rows])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if not m.any():
            continue
        sel = [r for r, k in zip(rows, m) if k]
        rec = dict(lo=float(lo), hi=float(hi), n=int(m.sum()),
                   n_eff=float(np.median([r["n_eff"] for r in sel])),
                   ratio=float(np.median([r["null_ratio"] for r in sel])),
                   delta=float(np.median([r["delta"] for r in sel])),
                   hub=float(np.median([r["hub_mass"] for r in sel])),
                   hop=float(np.median([r["hub_hop"] for r in sel])),
                   pb=float(np.median([r["mass"].get("Pb", 0.0) for r in sel])),
                   cl=float(np.median([r["mass"].get("Cl", 0.0) for r in sel])))
        out.append(rec)
        print(f"  {lo:5.2f}-{hi:5.2f} {rec['n']:5d} {rec['n_eff']:8.1f} {rec['ratio']:7.3f} "
              f"{rec['delta']:10.4f} {rec['hub']:6.3f} {rec['hop']:8.4f} "
              f"{rec['pb']:5.2f} {rec['cl']:5.2f}")
    return out



def graph_cutoff_for(model, override=None):
    """Neighbour-list radius the head was TRAINED with.

    run_train builds the graph at the spectral Hamiltonian's range (10 A here) and filters the
    trunk back to r_max inside the model. An analysis that rebuilds batches at r_max instead
    silently drops every 5-10 A edge, so the head is handed a truncated H: the hub pair at a
    median 5.43 A then has no edge at all and reports |H_ab| = 0 with no hopping term, which
    is an artefact of the analysis and not a property of the model. Derived from the model so
    it cannot drift from what training used.
    """
    if override:
        return float(override)
    return max(float(model.r_max), float(getattr(model, "spectral_r_cut", 0.0) or 0.0))


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cutoff", type=float, default=None,
                    help="override; default = the model's own graph cutoff")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--label", default="")
    ap.add_argument("--neutral", action="store_true", help="Test 1b: hole counters on V_Cl0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    _assert_repo()
    model = torch.load(args.model, map_location=args.device,
                       weights_only=False).to(args.device).eval()
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    cutoff = graph_cutoff_for(model, args.cutoff)
    print(f"  graph cutoff {cutoff:.1f} A (model r_max {float(model.r_max):.1f})")

    frames = read(args.data / "train.xyz", ":") + read(args.data / "valid.xyz", ":")
    if args.neutral:
        pool = [a for a in frames if not counters_of(a).any() and len(a) != 80]
        forced = [0, 0, 1, 0]
    else:
        pool = [a for a in frames if counters_of(a).any()]
        forced = None
    if args.limit:
        pool = pool[: args.limit]

    rows = [r for r in (frame_metrics(model, a, z_table, cutoff, args.device, forced)
                        for a in pool) if r is not None]
    if not rows:
        raise SystemExit("no frames analysable")

    edges = np.round(np.arange(4.6, 7.4, 0.2), 2)
    label = args.label or args.model.stem
    tag = "Test 1b (hole counters on NEUTRAL dimerised frames -- diagnostic only)" \
        if args.neutral else "Test 1 (charged frames)"
    small = binned_report([r for r in rows if r["n_atoms"] < 100], edges,
                          f"{label}  {tag}  [79-atom]")
    big = [r for r in rows if r["n_atoms"] >= 100]
    big_rep = binned_report(big, edges, f"{label}  {tag}  [159-atom]") if big else []

    allr = [r["n_eff"] for r in rows]
    print(f"\n  pooled N_eff median {np.median(allr):.1f}   frames {len(rows)}")
    print("  gate: binds if N_eff <= 8 and Delta >> 0.020 eV at short d; "
          "flat if N_eff > 30 everywhere")

    if args.out:
        args.out.write_text(json.dumps(
            dict(label=label, neutral=bool(args.neutral), n=len(rows),
                 bins_79=small, bins_159=big_rep, rows=rows), indent=2, default=float))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
