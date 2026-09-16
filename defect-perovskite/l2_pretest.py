"""L2 pre-test (LES head line, L0 registration): the frozen-core ladder at KERNEL level, with
the base's latent charges `q_LES` as the static host pattern and a FIXED localised carrier `dq`.

For each pristine tiling with one vacancy and the relaxed core embedded rigidly (R_core), the
converted MACELES base gives `q_LES` on the actual cell (energies only, no gradients); the same
tiling without the vacancy gives the pristine pattern `q_pri`, so `delta_q = q_LES - q_pri` on the
surviving sites is the base's response to the defect. The carrier is a Gaussian-weighted unit
charge on the atoms within `carrier_r` of the vacancy site, identical in every cell because the
core is rigid. With `K_LR = gamma_lr(r_g, r_split, eps_inf)` (reciprocal, k = 0 dropped -- the
L2 kernel):

    E_self  = 1/2 dq^T K_LR dq          the carrier's own monopole: slope in x = alpha_cell / L
                                        must be EXACTLY -C / (2 eps_inf) (the registered check)
    E_cross = dq^T K_LR q_LES           L2's E_cross at s = 1: its 1/L slope is the RESIDUAL the
                                        plan asks for; it is what a non-neutral delta_q costs
    E_host  = dq^T K_LR q_pri           the part a pristine pattern (W6's E_host) would supply
    E_delta = dq^T K_LR delta_q         the defect response's part

and the local-neutrality reading: sum of delta_q within R of the vacancy for several R, and its
cell total Q_delta. The 1/L slope of E_cross implies a net charge `b / (-C / eps_inf)` to compare
with Q_delta. The global sign of q_LES is free (energy quadratic in q); `--sign` applies the
Born-fixed sign, default +1, recorded in the output.

    python defect-perovskite/l2_pretest.py --base ~/runs/base_v4_lesmh_prod/base_v4_lesmh_prod_base.pt \
        --core /home/alex/runs/dscc/relax159/relaxed_structures.json --tilings 2,2,2 2,2,3 3,3,3 3,3,4 --out l2_pretest.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np, torch
from ase import Atoms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from mace import tools                                                   # noqa: E402
from mace.modules.dscc import data as dd                                 # noqa: E402
from mace.modules.dscc.ewald import COULOMB, madelung_self               # noqa: E402
from mace.modules.dscc.kernels import gamma_lr                           # noqa: E402
from mace.modules.dscc.model import R_SPLIT_DEFAULT                      # noqa: E402
from mace.tools import torch_geometric                                   # noqa: E402
from dscc_embed_ladder import KEY_TOL, embed, relative, vacancy_site     # noqa: E402
from mace.modules.dscc.ladder import tiling_ladder                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="converted single-head MACELES base (.pt)")
    ap.add_argument("--core", default="/home/alex/runs/dscc/relax159/relaxed_structures.json")
    ap.add_argument("--core_key", default="2,2,2")
    ap.add_argument("--r_core", type=float, default=8.0)
    ap.add_argument("--tilings", nargs="+", default=["2,2,2", "2,2,3", "3,3,3", "3,3,4"])
    ap.add_argument("--static_cell", default=str(HERE / "static_pristine_cell.json"))
    ap.add_argument("--r_g", type=float, default=1.0)
    ap.add_argument("--r_split", type=float, default=R_SPLIT_DEFAULT)
    ap.add_argument("--eps_inf", type=float, default=4.0)
    ap.add_argument("--carrier_r", type=float, default=4.0, help="carrier support radius, A")
    ap.add_argument("--carrier_sigma", type=float, default=2.0, help="carrier Gaussian width, A")
    ap.add_argument("--sign", type=float, default=1.0, help="Born-fixed global sign of q_LES")
    ap.add_argument("--neutral_radii", nargs="+", type=float, default=[4.0, 6.0, 8.0, 12.0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base_float32", type=int, default=1,
                    help="run the base in float32 (its training precision; the 2879-atom float64 forward "
                         "does not fit a 16 GB card). Latent charges are returned in float64.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    doc = json.load(open(args.static_cell))
    unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
    src = json.load(open(args.core))[args.core_key]
    src_t = tuple(int(x) for x in args.core_key.split(","))
    pristine = tiling_ladder(unit, [src_t], vacancy_species=17, vacancy_index=0)[0]
    vac0, cell0 = vacancy_site(unit, src_t)
    p0 = np.array(pristine.get_positions()); p1 = np.array(src["positions"])
    u = p1 - p0; u -= np.round(u @ np.linalg.inv(cell0)) @ cell0
    rel0 = relative(p0, vac0, cell0)
    core_disp = {tuple(np.round(rel0[i] / KEY_TOL).astype(np.int64)): u[i] for i in range(len(p0))}

    base = torch.load(os.path.expanduser(args.base), map_location="cpu", weights_only=False)
    base = (base.float() if args.base_float32 else base.double()).to(args.device).eval()
    bdt = next(base.parameters()).dtype
    assert type(base).__name__ == "MACELES", type(base).__name__
    z_table = tools.AtomicNumberTable([int(z) for z in base.atomic_numbers])

    def latent(atoms):
        a = atoms.copy()
        a.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0, "config_type": "l2", "source_dir": "l2"})
        ds = dd.atomic_data([a], z_table, float(base.r_max))
        batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(args.device).to_dict()
        batch = {k: (v.to(bdt) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in batch.items()}
        with torch.no_grad():
            out = base(batch, compute_force=False)
        torch.cuda.empty_cache()
        return (args.sign * out["latent_charges"].detach().reshape(-1).double(), float(out["les_energy"]),
                float(out["energy"]))

    rows = []
    for t in args.tilings:
        tt = tuple(int(x) for x in t.split(",")); t0 = time.time()
        atoms, moved, missing, inside = embed(unit, tt, core_disp, (vac0, cell0), args.r_core)
        vac, cell_np = vacancy_site(unit, tt)
        cell = torch.tensor(cell_np, dtype=torch.float64)
        L = float(abs(np.linalg.det(cell_np))) ** (1 / 3)
        alpha = -float(madelung_self(cell)) * L / COULOMB
        x = alpha / L
        # the pristine pattern on the same lattice: the un-displaced full tiling, the vacancy site dropped
        full = unit.repeat(tt)
        q_full, _, _ = latent(full)
        sites = [i for i, z in enumerate(full.get_atomic_numbers()) if int(z) == 17]
        keep = [i for i in range(len(full)) if i != sites[0]]
        q_pri = q_full[keep]
        q_site = float(q_full[sites[0]])                # the pristine charge of the removed site
        q_les, e_les, e_base = latent(atoms)
        dq_les = q_les - q_pri
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=args.device)
        rel = torch.tensor(relative(np.array(atoms.get_positions()), vac, cell_np), dtype=torch.float64, device=args.device)
        r = rel.norm(dim=1)
        # the fixed localised carrier: Gaussian weights on the atoms within carrier_r, unit total
        w = torch.exp(-0.5 * (r / args.carrier_sigma) ** 2) * (r <= args.carrier_r)
        dq = w / w.sum()
        K = gamma_lr(pos, cell.to(args.device), args.r_g, args.r_split, args.eps_inf)
        e_self = 0.5 * float(dq @ K @ dq)
        e_cross = float(dq @ K @ q_les)
        e_host = float(dq @ K @ q_pri)
        e_delta = float(dq @ K @ dq_les)
        neutral = {f"{R:g}": float(dq_les[r <= R].sum()) for R in args.neutral_radii}
        row = {"tiling": list(tt), "n_atoms": len(atoms), "L": L, "alpha_cell": alpha, "x": x,
               "E_self": e_self, "E_cross": e_cross, "E_host": e_host, "E_delta": e_delta,
               "Q_les_cell": float(q_les.sum()), "Q_pri_cell": float(q_pri.sum()), "Q_delta_cell": float(dq_les.sum()),
               "q_site_removed": q_site, "Q_defect": float(dq_les.sum()) - q_site,
               "xi_over_eps": float(madelung_self(cell)) / args.eps_inf,
               "delta_q_within_R": neutral, "abs_delta_q_max": float(dq_les.abs().max()),
               "e_les_meV_per_atom": 1e3 * e_les / len(atoms), "carrier_atoms": int((w > 0).sum()),
               "core_moved": moved, "core_missing": missing, "seconds": time.time() - t0}
        rows.append(row)
        print("%s n=%5d L=%6.2f x=%.5f  E_self=%.5f  E_cross=%.5f (host %.5f, delta %.5f)  Q_delta=%.4f  "
              "dq(R<=4/8)=%.4f/%.4f  [%.0f s]" % (t, len(atoms), L, x, e_self, e_cross, e_host, e_delta,
              row["Q_delta_cell"], neutral.get("4", float("nan")), neutral.get("8", float("nan")), row["seconds"]), flush=True)

    # fits in x = alpha / L: E = a + b x  (b in eV A); the exact carrier monopole is b = -C / (2 eps_inf)
    # Two fits in x = alpha / L: linear (a + b x) and with the Makov-Payne cubic (a + b x + c x^3),
    # which a Gaussian carrier of width carrier_sigma in a 28 A cell needs before its monopole
    # slope reads exactly; b is in eV A and the exact carrier monopole is b = -C / (2 eps_inf).
    fit = {}
    if len(rows) >= 3:
        b_exact = -COULOMB / (2.0 * args.eps_inf)
        for form, cols in (("linear", lambda x: [1.0, x]), ("cubic", lambda x: [1.0, x, x ** 3])):
            X = np.array([cols(rr["x"]) for rr in rows])
            if X.shape[0] <= X.shape[1] - 1:
                continue
            f = {}
            for key in ("E_self", "E_cross", "E_host", "E_delta"):
                y = np.array([rr[key] for rr in rows])
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                f[key] = {"a": float(coef[0]), "b": float(coef[1]), "max_resid": float(np.abs(X @ coef - y).max())}
                if form == "cubic":
                    f[key]["c"] = float(coef[2])
            f["E_self"]["b_exact"] = b_exact
            f["E_self"]["b_over_exact"] = f["E_self"]["b"] / b_exact
            # A localised net charge Q_p in the pattern interacts with the unit carrier as
            # Q_p xi(L) / eps_inf + O(1), i.e. slope -Q_p C / eps_inf in x. The pattern's net
            # defect charge is Q_delta (the base's response) minus the removed site's charge.
            f["E_cross"]["implied_net_charge"] = f["E_cross"]["b"] / (-COULOMB / args.eps_inf)
            f["E_cross"]["b_expected_from_Q_defect"] = -rows[-1]["Q_defect"] * COULOMB / args.eps_inf
            f["E_host"]["b_expected_from_q_site"] = rows[-1]["q_site_removed"] * COULOMB / args.eps_inf
            f["E_delta"]["b_expected_from_Q_delta"] = -rows[-1]["Q_delta_cell"] * COULOMB / args.eps_inf
            f["E_cross"]["residual_meV_at_79"] = 1e3 * f["E_cross"]["b"] * rows[0]["x"]
            fit[form] = f
        # THE MONOPOLE TEST proper: subtract the exact monopole 1/2 xi(L) / eps_inf from E_self; what
        # is left must be a + c / V (the second moments of carrier and kernel), with no 1/L part.
        # The size of the 1/V fit's residual against the monopole span is the reading.
        V = np.array([rr["L"] ** 3 for rr in rows]); y = np.array([rr["E_self"] - 0.5 * rr["xi_over_eps"] for rr in rows])
        X = np.stack([np.ones_like(V), 1.0 / V], axis=1); coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        span = float(np.ptp([0.5 * rr["xi_over_eps"] for rr in rows]))
        fit["monopole_test"] = {"E_self_minus_monopole_fit_a": float(coef[0]), "c_over_V": float(coef[1]),
                                "max_resid_eV": float(np.abs(X @ coef - y).max()), "monopole_span_eV": span,
                                "resid_over_span": float(np.abs(X @ coef - y).max() / span)}
    out = {"args": vars(args), "rows": rows, "fit": fit}
    json.dump(out, open(os.path.expanduser(args.out), "w"), indent=1)
    for form, f in fit.items():
        if form == "monopole_test":
            print("monopole test (E_self - 1/2 xi/eps_inf fitted as a + c/V):", json.dumps({k: round(v, 6) for k, v in f.items()}))
        else:
            print(f"fit {form} (x = alpha/L):", json.dumps({k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in f.items()}))
    print("saved", args.out)


if __name__ == "__main__":
    main()
