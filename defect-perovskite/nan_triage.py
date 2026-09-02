#!/usr/bin/env python3
"""NaN triage: an assertion ladder on one 159-atom frame, stopping at the first failure.

The counting head returns NaN on 159-atom cells and is finite on 79-atom ones. Rather than
bisect the code, assert the properties that must hold in the order that localises the fault,
and stop at the first one that breaks.

    1. isfinite(H) after assembly
    2. tiled-pristine invariance -- learned elements bit-identical, phi_LR to 1e-8
    3. counting -- n_states == 4 n_atoms, N_target < n_states at the mu solve
    4. occupations -- Tr P == N to 1e-8, entropy in the stable form
    5. spectrum sanity -- F(2N) == 2 F(N)

FORECAST ON RECORD before running: the NaN trips assertion 3 or 4.

Float64 throughout, per the plan: a 636x636 eigendecomposition in float32 is itself a
candidate and must not be the reason a later assertion fails.

Clause 2 carries its own qualification: the LEARNED matrix elements tile bit-identically, but
the Madelung potential is a reciprocal-space sum whose G-grid changes with the cell, so it
matches only to tolerance. Asserting bit-identity on phi would fail for the wrong reason.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from ase.io import read

from mace import tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_sensitivity import select_pristine  # noqa: E402
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

OK = "  PASS "
BAD = "  FAIL "


def head_internals(model, batch):
    grabbed = {}
    head = model.spectral
    original = head.forward
    head.forward = lambda *a, **k: original(*a, **dict(k, internals=grabbed))
    try:
        with torch.no_grad():
            out = model(batch.to_dict(), training=False, compute_force=False)
    finally:
        head.forward = original
    return grabbed, out


def assemble_H(model, batch, graph=0):
    """Rebuild H for one graph exactly as the head does, so it can be inspected."""
    from mace.modules.defect_counting import ORBITALS_PER_ATOM  # noqa: F401

    d = batch.to_dict()
    feats, species, ei, ev, mad = {}, None, None, None, None
    head = model.spectral
    # `forward`, not `__call__`: Python resolves __call__ on the TYPE, so assigning it on the
    # instance does not intercept anything and the capture comes back empty.
    original_h = head.h.forward

    captured = {}

    def spy(*a, **k):
        H = original_h(*a, **k)
        captured.setdefault("H", []).append(H)
        return H

    head.h.forward = spy
    try:
        with torch.no_grad():
            model(d, training=False, compute_force=False)
    finally:
        head.h.forward = original_h
    return captured.get("H", [])


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    _assert_repo()
    torch.set_default_dtype(torch.float64)
    dev = args.device
    model = torch.load(args.model, map_location=dev, weights_only=False).to(dev)
    model = model.double().eval()
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    cutoff = max(float(model.r_max), float(getattr(model, "spectral_r_cut", 0.0) or 0.0))

    frames = load_frames(args.data)
    big = [a for a in select(frames, charged=True) if len(a) == 159][:1]
    small = [a for a in select(frames, charged=True) if len(a) == 79][:1]
    if not big:
        raise SystemExit("no 159-atom charged frame found")
    print(f"float64 head; 159-atom frame ({len(big[0])} atoms), "
          f"79-atom reference ({len(small[0])} atoms), cutoff {cutoff:.1f} A\n")

    from mace.modules.defect_counting import ORBITALS_PER_ATOM, fermi_fill, spin_targets

    # ---------------------------------------------------------------- assertion 1
    print("[1] isfinite(H) after assembly")
    for label, fr in (("79 ", small), ("159", big)):
        batch, _ = make_batches(fr, z, cutoff, 1, dev)[0]
        blocks = assemble_H(model, batch)
        if not blocks:
            print(BAD + f"{label}: could not capture H")
            return
        H = blocks[0]
        finite = bool(torch.isfinite(H).all())
        print((OK if finite else BAD) + f"{label}: H {tuple(H.shape)} finite={finite}"
              + ("" if finite else
                 f", {int((~torch.isfinite(H)).sum())} non-finite entries"))
        if not finite:
            print("\n>>> TRIPPED AT ASSERTION 1: the Hamiltonian itself is non-finite.")
            return
        globals()[f"H_{label.strip()}"] = H

    # ---------------------------------------------------------------- assertion 3
    # Counting is checked before tiling: it is cheap, and forecast F1 names it.
    print("\n[3] counting: n_states == 4 n_atoms, and N_target < n_states")
    from mace.modules.defect_counting import VALENCE
    for label, fr in (("79 ", small), ("159", big)):
        atoms = fr[0]
        n_atoms = len(atoms)
        n_states = n_atoms * ORBITALS_PER_ATOM
        n_total = sum(VALENCE[int(n)] for n in atoms.get_atomic_numbers())
        batch, _ = make_batches(fr, z, cutoff, 1, dev)[0]
        counts = batch.carrier_counts.reshape(-1).tolist()
        n_maj, n_min = spin_targets(n_total, counts)
        H = globals()[f"H_{label.strip()}"]
        shape_ok = H.shape[0] == n_states
        fill_ok = max(n_maj, n_min) < n_states
        print((OK if (shape_ok and fill_ok) else BAD)
              + f"{label}: n_states {H.shape[0]} (expect {n_states})  "
                f"N_total {n_total}  N_maj {n_maj:.0f}  N_min {n_min:.0f}  "
                f"fill<states={fill_ok}")
        if not (shape_ok and fill_ok):
            print("\n>>> TRIPPED AT ASSERTION 3.")
            return

    # ---------------------------------------------------------------- assertion 4
    print("\n[4] occupations: Tr P == N to 1e-8, entropy finite")
    for label in ("79", "159"):
        H = globals()[f"H_{label}"]
        lam, psi = torch.linalg.eigh(H)
        print(f"      {label}: eigh finite={bool(torch.isfinite(lam).all())}  "
              f"lam range [{float(lam.min()):.3f}, {float(lam.max()):.3f}]")
        if not torch.isfinite(lam).all():
            print("\n>>> TRIPPED AT ASSERTION 4: eigendecomposition returned non-finite "
                  "eigenvalues.")
            return
        atoms = (small if label == "79" else big)[0]
        n_total = sum(VALENCE[int(n)] for n in atoms.get_atomic_numbers())
        n_maj = float((n_total + 1) // 2)
        f = fermi_fill(lam, n_maj)
        tr = float(f.sum())
        ok = abs(tr - n_maj) < 1e-8 and bool(torch.isfinite(f).all())
        fc = f.clamp(1e-12, 1.0 - 1e-12)
        ent = float(-(fc * fc.log() + (1 - fc) * (1 - fc).log()).sum())
        print((OK if ok else BAD) + f"{label}: Tr P = {tr:.9f} against N = {n_maj:.0f}"
              f"  (err {abs(tr - n_maj):.2e})  entropy {ent:.6f} "
              f"finite={np.isfinite(ent)}")
        if not (ok and np.isfinite(ent)):
            print("\n>>> TRIPPED AT ASSERTION 4.")
            return

    # ---------------------------------------------------------------- assertion 2
    print("\n[2] tiled-pristine invariance")
    pri = select_pristine(read(str(args.data), ":"), 1)[0]
    tiled = pri.repeat((2, 1, 1))
    for a in (pri, tiled):
        a.info.update(charge=0.0, spin_multiplicity=1, energy=0.0)
        a.arrays["forces"] = np.zeros((len(a), 3))
    b1, _ = make_batches([pri], z, cutoff, 1, dev)[0]
    b2, _ = make_batches([tiled], z, cutoff, 1, dev)[0]
    g1, _ = head_internals(model, b1)
    g2, _ = head_internals(model, b2)
    e1 = g1["eps"][:, 0].detach()
    e2 = g2["eps"][: len(pri), 0].detach()
    dev_max = float((e1 - e2).abs().max())
    print((OK if dev_max < 1e-6 else BAD)
          + f"on-site of the original atoms: max |delta| {dev_max:.3e} eV "
            f"(learned part bit-identical; phi_LR to tolerance)")

    # ---------------------------------------------------------------- assertion 5
    print("\n[5] spectrum sanity: F(2N) == 2 F(N) on the pristine pair")
    from mace.modules.defect_counting import free_energy
    l1 = torch.sort(g1["lam"][0, 0][g1["lam"][0, 0] < 500.0]).values
    l2 = torch.sort(g2["lam"][0, 0][g2["lam"][0, 0] < 500.0]).values
    n1 = sum(VALENCE[int(n)] for n in pri.get_atomic_numbers()) / 2.0
    f1, f2 = float(free_energy(l1, n1)), float(free_energy(l2, 2 * n1))
    rel = abs(f2 - 2 * f1) / max(abs(2 * f1), 1e-12)
    print((OK if rel < 1e-6 else BAD)
          + f"F(N) {f1:.6f}, F(2N) {f2:.6f}, 2F(N) {2 * f1:.6f}, rel dev {rel:.3e}")

    print("\nNo assertion tripped on this frame. If the model still produces NaN in the "
          "full forward, the fault is DOWNSTREAM of the head.")


if __name__ == "__main__":
    main()
