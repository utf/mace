"""W1.3 stage 2 (runs only after the thresholds of stage 1b are written): the charged d-window.

For every charged frame (`eval_qp1.xyz`, neutral-state predictions at charged geometries):
  * `d`, the flanking Pb-Pb distance, and the energy residual E_label(V+) - E_base for `sQ`;
  * the W0.6/W0.6a proxy, `F_ft` = the PRODUCTION base (the model W3 wraps; no base saw a
    charged frame, so all of them are out-of-fold here) and the cross-fit spread over the four
    fold bases; the energy proxy uses the species offsets fitted in stage 1b;
  * the fraction of frames/atoms inside the registered `u_F`, `u_E`.
Then `admission_table` (coverage bins, `s0 +- SE` from the stage-1 neutral rows, `sQ`) per size.

  python w13_charged.py --rows w1_oof_stage1.json --thresholds w1_proxy_thresholds.json --out w1_charged.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from ase.io import read
from mace.modules.dscc.admission import AdmissionConfig, admission_table, table_record
from mace.tools import AtomicNumberTable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w13_eval import load_base, forces_energies, per_component, shell_masks, SPECIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--charged", default=f"{W}/defect-perovskite/dataset_cf/eval_qp1.xyz")
    ap.add_argument("--folds", default="0,1,2,3")
    ap.add_argument("--models", default="/home/alex/runs/base_v2_f{fold}/base_v2_f{fold}.model")
    ap.add_argument("--prod", default="/home/alex/runs/base_v2_prod/base_v2_prod.model")
    ap.add_argument("--foundation", default="~/.cache/mace/macemh1model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--r_max", type=float, default=6.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--s_tol", type=float, default=0.018, help="W0.6: the v4 tau_noise_shape margin")
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",")]
    torch.set_default_dtype(torch.float64)
    t0 = time.time()

    rows = json.load(open(os.path.expanduser(args.rows)))
    thr = json.load(open(os.path.expanduser(args.thresholds)))
    a_s = np.array([thr["alignment"][k] for k in ("a_Cl", "a_Cs", "a_Pb")])

    bases = {k: load_base(args.models.format(fold=k), "Default").to(args.device) for k in folds}
    prod = load_base(args.prod, "Default").to(args.device)
    found_model = load_base(args.foundation, "omat_pbe").to(args.device)
    ztable = AtomicNumberTable(SPECIES)
    frames = read(os.path.expanduser(args.charged), ":")
    print(f"{len(frames)} charged frames, models loaded [{time.time()-t0:.0f}s]", flush=True)

    pF, pE = forces_energies(prod, frames, ztable, args.r_max, args.device, args.batch_size)
    fF, fE = forces_energies(found_model, frames, ztable, args.r_max, args.device, args.batch_size)
    kF = {k: forces_energies(bases[k], frames, ztable, args.r_max, args.device, args.batch_size) for k in folds}
    print(f"predictions done [{time.time()-t0:.0f}s]", flush=True)

    out_rows = []
    for i, a in enumerate(frames):
        n = len(a)
        counts = np.array([int((a.get_atomic_numbers() == z).sum()) for z in SPECIES], dtype=np.float64)
        rad, cats, d = shell_masks(a)
        stack = np.stack([kF[k][0][i] for k in folds])
        sd = per_component(stack.std(axis=0, ddof=1))
        dis = per_component(pF[i] - fF[i])
        band = ((rad >= 2.0) & (rad < 4.0)) if rad is not None else np.ones(n, dtype=bool)
        proxy_f = (dis + sd)[band]
        e_sd = float(np.std([kF[k][1][i] for k in folds], ddof=1))
        e_proxy = (abs(float(pE[i]) - (float(fE[i]) + counts @ a_s)) + e_sd) / n
        ref_e = float(a.info.get("REF_energy", np.nan))
        out_rows.append({"n": n, "id": a.info.get("id", ""), "d": d,
                         "e_resid_label_minus_base": ref_e - float(pE[i]),
                         "proxy_f_p95_meV_per_A": float(1000 * np.percentile(proxy_f, 95)) if proxy_f.size else None,
                         "proxy_f_median_meV_per_A": float(1000 * np.median(proxy_f)) if proxy_f.size else None,
                         "proxy_e_meV_per_atom": float(1000 * e_proxy),
                         "band_atoms": int(band.sum()),
                         "dis_p95": float(1000 * np.percentile(dis[band], 95)) if band.sum() else None,
                         "sd_p95": float(1000 * np.percentile(sd[band], 95)) if band.sum() else None})
    print(f"rows built [{time.time()-t0:.0f}s]", flush=True)

    # --- coverage of the registered thresholds, per size
    report = {"thresholds": {k: {"u_F_meV_per_A": v["u_F_meV_per_A"], "u_E_meV_per_atom": v["u_E_meV_per_atom"]}
                             for k, v in thr["sizes"].items()},
              "sizes": {}, "admission": {}}
    for size in sorted({r["n"] for r in out_rows}):
        sub = [r for r in out_rows if r["n"] == size]
        key = str(size)
        uF = (thr["sizes"].get(key) or {}).get("u_F_meV_per_A")
        uE = (thr["sizes"].get(key) or {}).get("u_E_meV_per_atom")
        inside_f = [r for r in sub if uF is not None and r["proxy_f_p95_meV_per_A"] is not None and r["proxy_f_p95_meV_per_A"] <= uF]
        inside_e = [r for r in sub if uE is not None and r["proxy_e_meV_per_atom"] <= uE]
        report["sizes"][key] = {
            "n_charged": len(sub),
            "d_window": [float(min(r["d"] for r in sub if r["d"])), float(max(r["d"] for r in sub if r["d"]))],
            "proxy_f_p95_median_meV_per_A": float(np.median([r["proxy_f_p95_meV_per_A"] for r in sub if r["proxy_f_p95_meV_per_A"] is not None])),
            "proxy_e_median_meV_per_atom": float(np.median([r["proxy_e_meV_per_atom"] for r in sub])),
            "frames_within_u_F": len(inside_f), "frames_within_u_E": len(inside_e),
            "fraction_within_u_F": len(inside_f) / max(len(sub), 1),
            "fraction_within_u_E": len(inside_e) / max(len(sub), 1)}

    # --- admission per size (s0 from the out-of-fold NEUTRAL rows; sQ from the charged rows)
    cfg = AdmissionConfig(s_tol=args.s_tol, z=args.z, base_protocol="base_v2_cf_4fold")
    charged_d, charged_r, neutral_d, neutral_r = {}, {}, {}, {}
    for size in sorted({r["n"] for r in out_rows}):
        sub = [r for r in out_rows if r["n"] == size and r["d"]]
        charged_d[size] = [r["d"] for r in sub]; charged_r[size] = [r["e_resid_label_minus_base"] for r in sub]
        nsub = [r for r in rows if r["n"] == size and r.get("d")]
        neutral_d[size] = [r["d"] for r in nsub]
        neutral_r[size] = [-r["e_resid"] for r in nsub]        # E_label - E_base = -(E_base - E_label)
    table = admission_table(charged_d, charged_r, neutral_d, neutral_r, cfg)
    report["admission"] = table_record(table, cfg, fold=-1)     # -1: the pooled cross-fit reading
    report["rows"] = out_rows
    json.dump(report, open(os.path.expanduser(args.out), "w"), indent=1, default=str)
    for size, rec in report["sizes"].items():
        t = table.get(int(size))
        print(f"n={size}: {rec['n_charged']} charged, d {rec['d_window'][0]:.2f}-{rec['d_window'][1]:.2f} A, "
              f"proxy_F p95 median {rec['proxy_f_p95_median_meV_per_A']:.1f} (within u_F {100*rec['fraction_within_u_F']:.0f} %), "
              f"proxy_E median {rec['proxy_e_median_meV_per_atom']:.2f} (within u_E {100*rec['fraction_within_u_E']:.0f} %)")
        if t:
            print(f"        s0 {t.s0:.4f} +- {t.s0_se:.4f} eV/A, sQ {t.sQ:.4f} +- {t.sQ_se:.4f}, coverage {t.coverage}, admitted {t.admitted} ({t.reason})")


if __name__ == "__main__":
    main()
