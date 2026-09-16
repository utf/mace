"""Plan P Decision 0 (amendment A1): variant eps0_ladder >= 8 and >= 2 x W6's, and the LO-TO
splitting of E_PP within 10 % of the smoothed expectation (g^2 f^2 x unsmoothed). Reads the JSON
records, writes DECISION0_STATUS (PASS / FAIL / INCOMPLETE) and the numbers.

    python defect-perovskite/p0_decision.py --w6 ~/runs/dscc/p0/ladder2_w6 ~/runs/dscc/p0/ladder2_w6_large \
        --var ~/runs/dscc/p0/ladder3_var ~/runs/dscc/p0/ladder3_var_large --loto ~/runs/dscc/p0/p0_loto_r45.json \
        --out ~/runs/dscc/p0/DECISION0_STATUS
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mace.modules.dscc.ewald import COULOMB

ap = argparse.ArgumentParser()
ap.add_argument("--w6", nargs="+", required=True); ap.add_argument("--var", nargs="+", required=True)
ap.add_argument("--loto", required=True); ap.add_argument("--eps_inf", type=float, default=4.0)
ap.add_argument("--min_rungs", type=int, default=4); ap.add_argument("--out", required=True)
args = ap.parse_args()


def rows_of(dirs):
    rows = []
    for d in dirs:
        for name in ("ladder_small.json", "ladder.json"):
            f = Path(d).expanduser() / name
            if f.exists():
                rows += json.load(open(f))
    seen = {}
    for r in rows:
        seen[r["n_atoms"]] = r
    return [seen[k] for k in sorted(seen)]


def fit(rows):
    x = np.array([r["x"] for r in rows]); y = np.array([r["dE_relax"] for r in rows])
    A = np.stack([np.ones_like(x), x], 1); (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    inv = 1 / args.eps_inf - b / (COULOMB / 2)
    loo = []
    for k in range(len(rows)):
        if len(rows) < 3: break
        m = np.ones(len(rows), bool); m[k] = False
        (_, bk), *_ = np.linalg.lstsq(A[m], y[m], rcond=None); ik = 1 / args.eps_inf - bk / (COULOMB / 2)
        loo.append(1 / ik if ik > 0 else float("inf"))
    return {"n": [r["n_atoms"] for r in rows], "dE_relax": [r["dE_relax"] for r in rows], "dE_inf": a, "slope": b,
            "eps0_ladder": (1 / inv if inv > 0 else float("inf")), "leave_one_out": loo}


w6, var = rows_of(args.w6), rows_of(args.var)
out = {"w6": fit(w6) if len(w6) >= 2 else None, "var": fit(var) if len(var) >= 2 else None}
lo = json.load(open(Path(args.loto).expanduser())) if Path(args.loto).expanduser().exists() else None
if lo is not None and not lo.get("nac_pm_q_fixed"):      # records written before the +-q correction
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "p0_loto_fix2.py"), str(Path(args.loto).expanduser())], check=True)
    lo = json.load(open(Path(args.loto).expanduser()))
status = "INCOMPLETE"
if lo is not None:
    b_ = lo["branches"]; split_pp = b_["LO"]["k_pp"] - b_["TO"]["k_pp"]
    expect = b_["LO"].get("k_nac_smoothed_x_ff2", b_["LO"]["k_nac_smoothed"]) - b_["TO"].get("k_nac_smoothed_x_ff2", 0.0)
    out["loto"] = {"split_e_pp": split_pp, "expected_smoothed_ff2": expect, "rel_dev": abs(split_pp - expect) / abs(expect) if expect else None}
if out["w6"] and out["var"] and lo is not None and len(w6) >= args.min_rungs and len(var) >= args.min_rungs:
    e_var, e_w6 = out["var"]["eps0_ladder"], out["w6"]["eps0_ladder"]
    c1 = e_var >= 8.0; c2 = e_var >= 2.0 * e_w6; c3 = out["loto"]["rel_dev"] is not None and out["loto"]["rel_dev"] <= 0.10
    out["criteria"] = {"eps0_var_ge_8": c1, "eps0_var_ge_2x_w6": c2, "loto_within_10pct": c3}
    status = "PASS" if (c1 and c2 and c3) else "FAIL"
out["status"] = status
Path(args.out).expanduser().write_text(status + "\n" + json.dumps(out, indent=1) + "\n")
print(status, json.dumps({k: out[k] for k in out if k != "status"}, default=str)[:1500])
