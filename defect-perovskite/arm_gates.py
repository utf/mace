"""Score a trained arm against the section 9 acceptance gates.

The headline number for every arm is a RATE: how many seeds localised the carrier correctly.
That question has to be asked the same way of every arm, or the comparison between the
production head and the spectral head measures the scorer rather than the models.

Gate 1, localisation (primary, and the only one that decides PASS):

    shell mass  >= 0.70      alpha within 4.0 A of the vacancy, on the active channel
    Cs mass     <  0.05      the sublattice failing models collapse onto
    N_eff       <= 8         1 / sum(alpha^2); the correct answer is ~2
    site spread >  0         within-species spread of the site energy; zero means the head
                             cannot tell one Pb from another, which is the observed
                             sublattice-level failure mode

    FAIL overrides everything if the state is sublattice-uniform, i.e. the mass on some
    species is proportional to its site count. That is a band state wearing a high species
    mass, and it is the failure the plan singles out.

Gate 3, accuracy: a CONSTRAINT, never a selector. RMSE has been actively misleading here --
a wrong-physics seed posted the best forces of its group, and in a later controlled test the
best-RMSE arm localised worst. It is reported so a model that localised by wrecking the fit
can be spotted, and for no other purpose. Nothing in this file ranks by it.

The vacancy position is evaluation-only machinery. It is used to define "within 4.0 A" and
for nothing else; the model is never told where the defect is.
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

from mace import data as mace_data
from mace import tools
from mace.data.defects import prepare_defect_configurations
from mace.tools import torch_geometric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vacancy_site import distance_to_vacancy, locate_vacancy  # noqa: E402

SHELL_RADIUS = 4.0
GATES = dict(shell_mass=0.70, cs_mass=0.05, n_eff=8.0)

KEYSPEC = mace_data.KeySpecification(
    info_keys={"energy": "REF_energy", "carrier_counts": "carrier_counts",
               "multiplicity": "multiplicity", "host": "host",
               "m_s_ref_doubled": "m_s_ref_doubled", "cell_charge": "cell_charge",
               "e_cbm_cell": "e_cbm_cell", "e_vbm_cell": "e_vbm_cell"},
    arrays_keys={"forces": "REF_forces"})


def evaluate(model, frames, z_table, cutoff, device, batch_size=4):
    """Per-frame alpha, site energy, and the energy/force error."""
    rows = []
    for s in range(0, len(frames), batch_size):
        chunk = frames[s:s + batch_size]
        configs = [mace_data.config_from_atoms(a, key_specification=KEYSPEC) for a in chunk]
        prepare_defect_configurations(configs)
        atomic = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=cutoff)
                  for c in configs]
        batch = next(iter(torch_geometric.dataloader.DataLoader(
            atomic, batch_size=len(atomic), shuffle=False))).to(device)
        out = model(batch.to_dict(), training=False, compute_force=True)

        alpha = out["carrier_alpha"].detach().cpu().numpy()
        site = out.get("carrier_site_energy", out.get("carrier_readouts"))
        site = site.detach().cpu().numpy() if site is not None else None
        counts = batch.carrier_counts.detach().cpu().numpy().reshape(-1, 4)
        idx = batch.batch.detach().cpu().numpy()
        f_pred = out["forces"].detach().cpu().numpy()
        e_pred = out["energy"].detach().cpu().numpy().reshape(-1)

        for k, atoms in enumerate(chunk):
            m = idx == k
            rows.append(dict(
                atoms=atoms, alpha=alpha[m],
                site=None if site is None else site[m],
                counts=counts[k],
                f_err=atoms.arrays["REF_forces"] - f_pred[m],
                e_err=(atoms.info["REF_energy"] - float(e_pred[k])) / len(atoms)))
    return rows


def score_frame(row):
    atoms = row["atoms"]
    active = int(np.argmax(row["counts"]))          # V_Cl+ is (0,0,1,0): the h_maj channel
    if row["counts"][active] == 0:
        return None
    a = row["alpha"][:, active]
    site = None if row["site"] is None else row["site"][:, active]

    try:
        vac = locate_vacancy(atoms)
    except ValueError:
        return None
    d = distance_to_vacancy(atoms, vac)
    sym = np.array(atoms.get_chemical_symbols())

    total = a.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    a = a / total                                   # guard against any un-normalised head

    per_species = {s: float(a[sym == s].sum()) for s in ("Cs", "Pb", "Cl")}
    counts_species = {s: int((sym == s).sum()) for s in ("Cs", "Pb", "Cl")}

    # Sublattice-uniform: the mass on a species matches what an even spread over that species
    # would give. A band state can post a high species mass, so this is checked separately
    # from the species mass itself.
    uniform = False
    for s, mass in per_species.items():
        if mass > 0.5:
            spread_evenly = a[sym == s]
            if spread_evenly.size:
                rel = spread_evenly.std() / max(spread_evenly.mean(), 1e-12)
                if rel < 0.5:                       # nearly flat across the whole sublattice
                    uniform = True

    return dict(
        shell_mass=float(a[d <= SHELL_RADIUS].sum()),
        cs_mass=per_species["Cs"],
        pb_mass=per_species["Pb"],
        cl_mass=per_species["Cl"],
        n_eff=float(1.0 / np.square(a).sum()),
        site_spread_pb=(float(site[sym == "Pb"].std()) if site is not None else np.nan),
        sublattice_uniform=uniform,
        n_species={k: counts_species[k] for k in counts_species},
        e_err=row["e_err"], f_err=float(np.sqrt((row["f_err"] ** 2).mean())),
    )


def training_rmse(model_path):
    """Read the final validation error table from the run's log.

    The per-atom energy error cannot be recomputed here from `out["energy"]` and REF_energy:
    the labels are on a referenced scale, and reconstructing that referencing outside the
    training loop reproduces it only approximately -- measured against a known model, this
    file's own arithmetic gave 16.5 meV/atom where training reported 3.1. Forces need no
    referencing and are computed directly. Rather than publish a number that is wrong by an
    unknown offset, take the energy from the code that computes it correctly.
    """
    log = Path.home() / "runs" / f"{Path(model_path).stem}.log"
    if not log.exists():
        return None, None
    e = f = None
    for line in log.read_text(errors="ignore").splitlines():
        if "valid_Default" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                try:
                    e, f = float(parts[1]), float(parts[2])
                except ValueError:
                    pass
    return e, f


def verdict(agg):
    """PASS / FAIL / AMBIGUOUS from gate 1 alone. Accuracy is never consulted here."""
    if agg["sublattice_uniform_frac"] > 0.5:
        return "FAIL", "sublattice-uniform (band state)"
    ok = (agg["shell_mass"] >= GATES["shell_mass"]
          and agg["cs_mass"] < GATES["cs_mass"]
          and agg["n_eff"] <= GATES["n_eff"])
    if ok and (np.isnan(agg["site_spread_pb"]) or agg["site_spread_pb"] > 0):
        return "PASS", ""
    hard_fail = (agg["shell_mass"] < 0.3 or agg["cs_mass"] > 0.5
                 or agg["n_eff"] > 16)
    if hard_fail:
        why = []
        if agg["shell_mass"] < 0.3:
            why.append(f"shell mass {agg['shell_mass']:.2f}")
        if agg["cs_mass"] > 0.5:
            why.append(f"Cs mass {agg['cs_mass']:.2f}")
        if agg["n_eff"] > 16:
            why.append(f"N_eff {agg['n_eff']:.1f}")
        return "FAIL", ", ".join(why)
    return "AMBIGUOUS", "between the pass and fail bands; dump alpha and site maps"


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "valid.xyz")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    frames = [a for a in read(args.data, ":")
              if np.asarray([int(x) for x in a.info["carrier_counts"].split()]
                            if isinstance(a.info["carrier_counts"], str)
                            else a.info["carrier_counts"]).any()]
    frames = frames[: args.limit]
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    print(f"scoring on {len(frames)} charged validation frames "
          f"(shell = within {SHELL_RADIUS} A of the vacancy)\n")

    header = (f"{'model':<20} {'verdict':<10} {'shell':>6} {'Cs':>6} {'Pb':>6} "
              f"{'N_eff':>7} {'eps sd':>7} {'E':>6} {'F':>6}")
    print(header)
    print("-" * len(header))

    results = {}
    for path in args.models:
        model = torch.load(path, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        rows = [score_frame(r) for r in
                evaluate(model, frames, z_table, args.cutoff, args.device)]
        rows = [r for r in rows if r is not None]
        if not rows:
            print(f"{path.stem:<20} no scorable frames")
            continue

        agg = {k: float(np.median([r[k] for r in rows]))
               for k in ("shell_mass", "cs_mass", "pb_mass", "cl_mass", "n_eff",
                         "site_spread_pb")}
        agg["sublattice_uniform_frac"] = float(
            np.mean([r["sublattice_uniform"] for r in rows]))
        log_e, log_f = training_rmse(path)
        agg["e_rmse"] = log_e if log_e is not None else float("nan")
        agg["f_rmse"] = (log_f if log_f is not None
                         else float(np.sqrt(np.mean([r["f_err"] ** 2 for r in rows]))) * 1000)

        v, why = verdict(agg)
        results[path.stem] = dict(verdict=v, reason=why, **agg)
        print(f"{path.stem:<20} {v:<10} {agg['shell_mass']:6.3f} {agg['cs_mass']:6.3f} "
              f"{agg['pb_mass']:6.3f} {agg['n_eff']:7.2f} {agg['site_spread_pb']:7.3f} "
              f"{agg['e_rmse']:6.1f} {agg['f_rmse']:6.1f}"
              + (f"   {why}" if why else ""))

    passed = sum(1 for r in results.values() if r["verdict"] == "PASS")
    print(f"\nrate: {passed} of {len(results)} PASS")
    print("E / F are meV/atom and meV/A, reported as a CONSTRAINT (parity band 3.5 / 13.0). "
          "They never decide the verdict: a wrong-physics seed has posted the best forces "
          "of its group before.")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
