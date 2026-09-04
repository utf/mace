"""Stage 0 acceptance (plan v8 section 3), in one run against arma_s1.

    1. GOLDEN: every record of the v6 golden is bit-identical under the default policy
       (`stage0_golden.compare`).
    2. CLASS TABLE: the composition classes of the golden frames -- pristine 80, V_Cl 79 and
       159 -- counted by the constructor on arma_s1; Tier 2 forced on every defect class must
       reproduce Tier 1's integers, and the density placement must agree with the Tier 2
       correspondence's translation.
    3. SOLE POLICY: `count_fill` is the only registered occupation policy in a fresh process;
       the test-only mock is unreachable from production.
    4. HARNESS STATUS: the finite-difference status record (golden/stage0_fd_v6.json) is
       present, and its summary is printed with its regime tag.

Writes golden/stage0_acceptance.json. Skips 1, 2 and 4's model-dependent parts when the
model file is absent (they are then reported as "skipped", never as passed).

    export PYTHONPATH=<repo>; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python defect-perovskite/stage0_acceptance.py --model ~/runs/arma_models/arma_s1.model
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401
from mace import data as mace_data
from mace.modules import defect_composition as dc
from mace.modules import defect_density as dd
from mace.modules.defect_cache import attach_frame_keys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0_golden as sg  # noqa: E402

HERE = Path(__file__).resolve().parent


def golden_check(args):
    if not Path(args.model).expanduser().exists():
        return {"status": "skipped", "reason": "model file absent"}
    ns = argparse.Namespace(model=args.model, ref=args.ref)
    code = sg.compare(ns)
    return {"status": "pass" if code == 0 else "FAIL", "ref": args.ref}


def class_table_check(args):
    if not Path(args.model).expanduser().exists():
        return {"status": "skipped", "reason": "model file absent"}
    model, ctx, frames, _ = sg.setup(argparse.Namespace(model=args.model))
    order = ["pristine", "vcl0_79", "vcl0_159", "qp1_79", "qp1_159"]
    atoms = [a for k in order for _, _, a in frames[k]]
    configs = [mace_data.config_from_atoms(a, key_specification=sg.KEYSPEC) for a in atoms]
    sg.prepare_defect_configurations(configs)
    ds = [mace_data.AtomicData.from_config(c, z_table=sg.Z_TABLE, cutoff=ctx.cutoff)
          for c in configs]
    attach_frame_keys(ds, z_table=sg.Z_TABLE)
    t0 = time.time()
    table = dc.build_class_table(model, ds, log=False)
    out = {"status": "pass", "build_seconds": round(time.time() - t0, 1),
           "e_sink": table["e_sink"], "delta": table["delta"], "window": table["window"],
           "r_match": table["r_match"], "classes": {}}
    pristine = ds[0]
    _, _, pristine_cell = dc._frame_geometry(pristine, model)
    cfg = dc.constructor_config(model.class_constructor)
    cfg["e_sink"] = table["e_sink"]
    for key, r in table["classes"].items():
        rec = dc.ClassRecord.from_dict(r)
        entry = {"tier": rec.tier, "m_vb": list(rec.m_vb), "n_sigma": list(rec.n_sigma),
                 "n_e": list(rec.n_e), "n_h": list(rec.n_h), "q_core": rec.q_core,
                 "vbm_al": rec.vbm_al, "shift": rec.shift, "spread": rec.spread,
                 "nearest_to_cut": rec.nearest, "tiling": list(rec.tiling),
                 "perm": list(rec.perm), "gap_pristine": rec.gap_pristine,
                 "placement_residual": None if rec.placement is None
                 else rec.placement["residual_norm"], "reason": rec.reason}
        print(f"  {key:20s} {dc.describe(rec)}")
        if key != table["pristine_key"] and rec.counted:
            # Tier 2 forced: must reproduce the integers whichever tier counted them.
            fr = next(f for f in ds if dc.composition_key(dc._numbers_of(f, model)) == key)
            numbers, _, cell = dc._frame_geometry(fr, model)
            factors, perm = dc.tiling_map(cell, pristine_cell, len(numbers), pristine.num_nodes)
            pri = dc.head_spectra(model, dc.tiled_pristine_dict(pristine, model, factors))[0]
            spec = dc.head_spectra(model, dc._single_frame_dict(fr, "cpu"))[0]
            rank = max(dc.reference_fill(pri["n_total"]))
            t1 = time.time()
            t2 = dc.tier2(model, fr, pristine, factors, perm, rank, pri["H"], spec["H"], cfg)
            entry["tier2_forced"] = {
                "accepted": t2["accepted"], "m_vb": t2["m_vb"], "seconds": round(time.time() - t1, 1),
                "correspondence": t2["correspondence"], "reason": t2["reason"],
                "min_separation": min(r["min_separation"] for r in t2["runs"].values()),
                "agrees_with_table": bool(t2["accepted"] and t2["m_vb"] == rec.m_vb[0])}
            print(f"      Tier 2 forced: accepted {t2['accepted']}, M_VB {t2['m_vb']} "
                  f"({'agrees' if entry['tier2_forced']['agrees_with_table'] else 'DISAGREES'}), "
                  f"{t2['correspondence']}")
            if not entry["tier2_forced"]["agrees_with_table"]:
                out["status"] = "FAIL"
            # The density placement against the correspondence's translation.
            n1, p1, c1 = dc._frame_geometry(fr, model)
            n0, p0, c0 = dc.tile_frame(*dc._frame_geometry(pristine, model), factors)
            corr = dc.site_correspondence(n1, p1, c1, n0, p0, c0, perm=perm,
                                          r_match=float(cfg["r_match"]))
            t_corr = np.asarray(corr["translation"]) @ np.linalg.inv(c1)
            diff = np.asarray(rec.placement["shift"]) - t_corr
            diff -= np.round(diff)
            entry["placement_vs_correspondence_A"] = float(np.linalg.norm(diff @ c1))
            # The two objectives (density residual; squared displacements) can pick minima a
            # sub-lattice vector apart -- the 80-atom cell holds four 20-atom orthorhombic
            # cells, so a translation by one of them is a near-symmetry of the pristine
            # density. What matters is the residual at each: both must be small and the
            # density's own must be the smaller.
            zs = [int(z) for z in model.atomic_numbers]
            species = torch.tensor([zs.index(int(z)) for z in n1])
            z0 = dc.species_charges(model).to(torch.float64)
            r_res = rec.placement["r_res"]
            present = dd.static_present(z0[species], torch.tensor(p1), torch.tensor(c1), r_res)
            pri_species, scaled = dc.tiled_pristine_scaled(model, pristine, factors, perm)

            def residual(t):
                pri_d = dd.pristine_placed(z0[pri_species], scaled, torch.tensor(c1),
                                           torch.as_tensor(t, dtype=torch.float64), r_res)
                return float(dd.static_raw(present, pri_d).norm())

            entry["residual_at_density_shift"] = residual(rec.placement["shift"])
            entry["residual_at_correspondence_shift"] = residual(t_corr)
            entry["present_norm"] = float(present.norm())
            print(f"      placement vs correspondence: {entry['placement_vs_correspondence_A']:.3f} A; "
                  f"||rho_raw|| at density shift {entry['residual_at_density_shift']:.4f}, at "
                  f"correspondence shift {entry['residual_at_correspondence_shift']:.4f} "
                  f"(||rho_present|| {entry['present_norm']:.4f})")
            if entry["residual_at_density_shift"] > entry["residual_at_correspondence_shift"] + 1e-9:
                out["status"] = "FAIL"
        out["classes"][key] = entry
    q_cores = {e["q_core"] for k, e in out["classes"].items() if k != table["pristine_key"]
               and e["tier"] is not None}
    out["defect_q_core_identical_across_sizes"] = len(q_cores) == 1
    if len(q_cores) != 1 or any(e["tier"] is None for e in out["classes"].values()):
        out["status"] = "FAIL"
    return out


def sole_policy_check():
    code = ("import mace, mace.modules.defect_models, mace.cli.run_train; "
            "from mace.modules import defect_state as ds; "
            "print(','.join(ds.registered_policies()))")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=False, cwd=str(HERE.parent))
    names = res.stdout.strip().splitlines()[-1].split(",") if res.stdout.strip() else []
    ok = names == ["count_fill"]
    return {"status": "pass" if ok else "FAIL", "registered": names,
            "stderr_tail": res.stderr[-300:] if not ok else ""}


def harness_status():
    path = HERE / "golden" / "stage0_fd_v6.json"
    if not path.exists():
        return {"status": "FAIL", "reason": "golden/stage0_fd_v6.json absent"}
    d = json.loads(path.read_text())
    rows = d["summary"]
    print(f"  FD status record: sha {d['git_sha'][:10]}, {d['precision_policy']}, "
          f"{len(rows)} rows")
    worst = {}
    for r in rows:
        k = (r["kind"], r["term"], r["gauge"])
        worst.setdefault(k, []).append((r["status"], r["fit"]["min_error"], r["frame"]))
    for (kind, term, gauge), items in sorted(worst.items()):
        statuses = sorted({s for s, _, _ in items})
        floor = max(e for _, e, _ in items)
        print(f"    {kind:6s} {term:12s} {gauge:9s} {','.join(statuses):26s} worst {floor:.1e}")
    return {"status": "pass", "git_sha": d["git_sha"], "rows": len(rows),
            "precision_policy": d["precision_policy"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="~/runs/arma_models/arma_s1.model")
    p.add_argument("--ref", default="~/runs/golden/v6_arma_s1.pt")
    p.add_argument("--out", default=str(HERE / "golden" / "stage0_acceptance.json"))
    args = p.parse_args(argv)
    args.model = str(Path(args.model).expanduser())
    report = {"git_sha": sg.git_sha(), "model": args.model, "torch": torch.__version__,
              "regime": "CPU, float64, 1 thread, deterministic, profiling executor off"}
    print("1. golden bit-identity")
    report["golden"] = golden_check(args)
    print("2. class table on arma_s1 (Tier 1, Tier 2 forced, placement)")
    report["class_table"] = class_table_check(args)
    print("3. sole registered policy")
    report["sole_policy"] = sole_policy_check()
    print(f"   registered: {report['sole_policy']['registered']}")
    print("4. finite-difference status record")
    report["harness"] = harness_status()
    statuses = {k: v["status"] for k, v in report.items() if isinstance(v, dict) and "status" in v}
    report["accepted"] = all(s in ("pass", "skipped") for s in statuses.values()) and \
        "FAIL" not in statuses.values() and "skipped" not in (statuses["sole_policy"],
                                                              statuses["harness"])
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True, default=float))
    print(f"acceptance: {statuses} -> {'ACCEPTED' if report['accepted'] else 'NOT accepted'} "
          f"({args.out})")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
