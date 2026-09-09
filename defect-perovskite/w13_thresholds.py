"""W1.3 stage 1b: turn the out-of-fold neutral rows into the registered W0.6/W0.6a thresholds
and the shell floor table. Writes `w1_proxy_thresholds.json` BEFORE the charged window opens.

  u_F(size) = 95th percentile over atoms in the 2-4 A band of (|F_ft - F_found| + sd_folds)
  u_E(size) = 95th percentile over frames of (|E_ft - E_found_aligned| + sd_folds(E)) / N
with the foundation energies aligned by OLS on the three species counts (W0.6a item 3).
"""
import argparse, json, os
import numpy as np

SPECIES = ["17", "55", "82"]
SHELLS = ["2-4", "4-8", "8-10", "10-12", "12+"]
NEAR = ["pb_flank", "cl_first", "other"]


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = json.load(open(os.path.expanduser(args.rows)))

    # --- energy alignment (W0.6a.3): E_found_aligned = E_found + sum_s n_s a_s, OLS over all
    # neutral out-of-fold frames; the fit costs three degrees of freedom on the same frames.
    A = np.array([[r["counts"][s] for s in SPECIES] for r in rows], dtype=np.float64)
    y = np.array([r["e_ft"] - r["e_found"] for r in rows], dtype=np.float64)
    a_s, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid_E = y - A @ a_s                      # aligned energy disagreement, eV per frame

    out = {"alignment": {"a_Cl": float(a_s[0]), "a_Cs": float(a_s[1]), "a_Pb": float(a_s[2]),
                         "n_frames": len(rows), "rms_before_meV_per_atom": float(1000 * np.sqrt((y ** 2).mean()) / 79),
                         "rms_after_meV_per_atom": float(1000 * np.sqrt((resid_E ** 2).mean()) / 79)},
           "sizes": {}}
    for size in sorted({r["n"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["n"] == size]
        sub = [rows[i] for i in idx]
        pf = np.concatenate([np.asarray(r["proxy_f"]) for r in sub]) if sub else np.array([])
        eu = np.array([(abs(resid_E[i]) + sub_j["e_sd_folds"]) / sub_j["n"] for i, sub_j in zip(idx, sub)])
        rec = {"n_frames": len(sub), "u_F_meV_per_A": float(1000 * np.percentile(pf, 95)) if pf.size else None,
               "u_E_meV_per_atom": float(1000 * np.percentile(eu, 95)) if eu.size else None,
               "u_F_terms": {"dis_p95": med([r["proxy_f_terms"]["dis_p95"] for r in sub]),
                             "sd_p95": med([r["proxy_f_terms"]["sd_p95"] for r in sub])},
               "force_rmse_median_meV_per_A": 1000 * med([r["force_rmse"] for r in sub]),
               "energy_resid_median_meV_per_atom": 1000 * med([abs(r["e_resid"]) / r["n"] for r in sub]),
               "shells_meV_per_A": {}, "shell_atoms": {}, "near_meV_per_A": {}, "near_atoms": {},
               "by_fold": {}}
        for k in SHELLS:
            v = med([r["shells"][k]["rmse"] for r in sub if r["shells"].get(k)])
            rec["shells_meV_per_A"][k] = 1000 * v if v is not None else None
            rec["shell_atoms"][k] = med([r["shells"][k]["n"] for r in sub if r["shells"].get(k)])
        for k in NEAR:
            v = med([r["near"][k]["rmse"] for r in sub if r["near"].get(k)])
            rec["near_meV_per_A"][k] = 1000 * v if v is not None else None
            rec["near_atoms"][k] = med([r["near"][k]["n"] for r in sub if r["near"].get(k)])
        for f in sorted({r["fold"] for r in sub}):
            ff = [r for r in sub if r["fold"] == f]
            rec["by_fold"][str(f)] = {"n_frames": len(ff),
                                      "force_rmse_median_meV_per_A": 1000 * med([r["force_rmse"] for r in ff]),
                                      "2-4_meV_per_A": (lambda v: 1000 * v if v is not None else None)(
                                          med([r["shells"]["2-4"]["rmse"] for r in ff if r["shells"].get("2-4")])),
                                      "energy_resid_median_meV_per_atom": 1000 * med([abs(r["e_resid"]) / r["n"] for r in ff])}
        rec["frames_without_flanking_pair"] = sum(1 for r in sub if not r["shells"])
        out["sizes"][str(size)] = rec
    json.dump(out, open(os.path.expanduser(args.out), "w"), indent=1)
    print(json.dumps(out["alignment"], indent=1))
    for size, rec in out["sizes"].items():
        print(f"n={size}: frames {rec['n_frames']}  F {rec['force_rmse_median_meV_per_A']:.2f}  "
              f"u_F {rec['u_F_meV_per_A']}  u_E {rec['u_E_meV_per_atom']}")


if __name__ == "__main__":
    main()
