"""Plan P Phase 3a §2.4: the frozen-core ladder (vertical; energies only).

The §2.1 core (p0_ladder core.json) embedded within r_core in tilings of the base-v8-relaxed pristine cell; at fixed
geometry E(+1) - E(0), E(0) = base + E_PP (the checkpoint at zero carriers). The model's charged energy already carries
the exact eps_inf monopole E_M = Q^2 xi(h)/(2 eps_inf) (diagnostic `e_madelung`), so the remainder after the monopole is
dE itself; dE_raw = dE - E_M is recorded alongside. Fit remainder = E_inf + c/L^3 on the registered sizes with 639 held
out as the check; a forced 1/L term (fit E_inf + b/L + c/L^3) must be below 5 meV across the range. Also sum dq within
3/5/8/12 A and N_eff at every size (the §0 compactness gate at 639 and 2159). Resumable (rows appended per tiling).

    python defect-perovskite/p3a_frozen_ladder.py --model <ckpt> --core ~/runs/dscc/p3a/ladder/core.json --cell ~/runs/dscc/p3a/pristine_relaxed_v8.json \
        --tilings 2,2,2 2,2,3 --device cuda --out ~/runs/dscc/p3a/frozen_ladder.json
    OMP_NUM_THREADS=48 python ... --tilings 3,3,3 3,3,4 --device cpu --out (same file)
    python ... --fit_only 1 --out (same file)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace import tools                                                     # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.ewald import COULOMB, madelung_self                 # noqa: E402
from mace.tools import torch_geometric as tg                               # noqa: E402
from dscc_embed_ladder import KEY_TOL, embed, relative, vacancy_site        # noqa: E402
from p3a_common import load_cell                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--core", required=True); ap.add_argument("--cell", required=True)
ap.add_argument("--tilings", nargs="+", default=["2,2,2", "2,2,3", "3,3,3", "3,3,4"]); ap.add_argument("--r_core", type=float, default=8.0)
ap.add_argument("--fit_sizes", type=int, nargs="+", default=[959, 2159, 2879]); ap.add_argument("--heldout", type=int, default=639)
ap.add_argument("--device", default="cpu"); ap.add_argument("--fit_only", type=int, default=0); ap.add_argument("--out", required=True)
args = ap.parse_args(); torch.set_default_dtype(torch.float64); dev = args.device
out = Path(args.out).expanduser(); rows = json.load(open(out))["rows"] if out.exists() else []
unit = load_cell(args.cell)
if not args.fit_only:
    src = json.load(open(Path(args.core).expanduser())); vac0, cell0 = vacancy_site(unit, tuple(src["tiling"]))
    rel0 = np.array(src["rel_positions"]); u = np.array(src["displacements"])
    core_disp = {tuple(np.round(rel0[i] / KEY_TOL).astype(np.int64)): u[i] for i in range(len(u))}
    model = torch.load(Path(args.model).expanduser(), weights_only=False, map_location=dev).to(dev).eval(); model.base = model.base.double()
    z3 = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    def run(atoms, counts, charge):
        a = atoms.copy(); a.info.update({"carrier_counts": np.array(counts, float), "cell_charge": charge, "config_type": "fl", "source_dir": "fl"})
        b = next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, model.r_cut), batch_size=1))).to(dev).to_dict()
        with torch.no_grad():
            o = model(b, compute_force=False)
        diag = {k: (float(v.reshape(-1)[0]) if torch.is_tensor(v) else (v[0] if isinstance(v, list) else v)) for k, v in o.get("diagnostics", {}).items() if k in ("e_host", "e_madelung", "dq_sum", "q0_sum", "e_pp")}
        return float(o["energy"][0]), (o["dq"].detach().cpu().numpy() if "dq" in o else None), diag, (float(o["e_pp"][0]) if "e_pp" in o else None)

    done = {r["n_atoms"] for r in rows}
    for t in args.tilings:
        tt = tuple(int(x) for x in t.split(",")); t0 = time.time()
        atoms, moved, missing, inside = embed(unit, tt, core_disp, (vac0, cell0), args.r_core)
        if len(atoms) in done:
            continue
        cell = np.array(atoms.get_cell()); L = float(abs(np.linalg.det(cell))) ** (1 / 3)
        al = -float(madelung_self(torch.tensor(cell, dtype=torch.float64))) * L / COULOMB
        vac, _ = vacancy_site(unit, tt); r = np.linalg.norm(relative(np.array(atoms.get_positions()), vac, cell), axis=1)
        e1, dq, diag1, epp1 = run(atoms, [0, 0, 1, 0], 1); e0, _, diag0, epp0 = run(atoms, [0, 0, 0, 0], 0)
        row = {"tiling": list(tt), "n_atoms": len(atoms), "L": L, "alpha_cell": al, "E0": e0, "E1": e1, "dE": e1 - e0, "E_PP_neutral": epp0, "E_PP_charged": epp1,
               "e_madelung": diag1.get("e_madelung"), "e_host": diag1.get("e_host"), "core_moved": moved, "core_missing": missing, "core_inside": inside,
               "dq_sum": float(dq.sum()) if dq is not None else None, "n_eff": float(1.0 / (dq ** 2).sum()) if dq is not None else None,
               "sum_dq_within": ({f"{R}": float(dq[r <= R].sum()) for R in (3, 5, 8, 12)} if dq is not None else None), "device": dev, "seconds": time.time() - t0}
        row["dE_raw_without_EM"] = row["dE"] - row["e_madelung"] if row["e_madelung"] is not None else None
        rows.append(row); json.dump({"rows": rows}, open(out, "w"), indent=1)
        print(f"{tt}: n={len(atoms)} L={L:.2f} dE {row['dE']:.4f} eV (E_M {row['e_madelung']}), N_eff {row['n_eff']}, sum dq <3/5/8/12 {row['sum_dq_within']}, {row['seconds']:.0f}s", flush=True)

# fit
rows.sort(key=lambda r: r["n_atoms"]); fit = {}
pts = [r for r in rows if r["n_atoms"] in args.fit_sizes]
if len(pts) >= 2:
    L = np.array([r["L"] for r in pts]); y = np.array([r["dE"] for r in pts])
    A3 = np.stack([np.ones_like(L), L ** -3], 1); (e_inf, c3), *_ = np.linalg.lstsq(A3, y, rcond=None)
    fit["E_inf_frozen"] = float(e_inf); fit["c_L3_eV_A3"] = float(c3); fit["residual_meV"] = (1e3 * (y - A3 @ [e_inf, c3])).tolist()
    held = [r for r in rows if r["n_atoms"] == args.heldout]
    if held:
        pred = e_inf + c3 * held[0]["L"] ** -3; fit["heldout_639_meV"] = {"dE": held[0]["dE"], "predicted": float(pred), "miss_meV": 1e3 * float(held[0]["dE"] - pred)}
    if len(pts) >= 3:
        A13 = np.stack([np.ones_like(L), 1 / L, L ** -3], 1); (e1, b1, c1), *_ = np.linalg.lstsq(A13, y, rcond=None)
        fit["forced_1_over_L"] = {"b_eV_A": float(b1), "span_meV_over_range": 1e3 * float(abs(b1) * (1 / L.min() - 1 / L.max())), "E_inf": float(e1), "c_L3": float(c1),
                                  "pass_below_5meV": bool(1e3 * abs(b1) * (1 / L.min() - 1 / L.max()) < 5.0)}
    fit["flat_to_5meV"] = bool((y.max() - y.min()) * 1e3 <= 5.0); fit["range_meV"] = 1e3 * float(y.max() - y.min())
json.dump({"rows": rows, "fit": fit}, open(out, "w"), indent=1); print("fit:", json.dumps(fit)); print("saved", out)
