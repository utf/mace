"""L0 gate reader (registered thresholds in DSCC_V5_IMPLEMENTATION_SPEC.md, "L0 registration").
Reads the run log (G-L0.1), the base_compare_n.py JSON (G-L0.2), the les_gates.py JSON (G-L0.3)
and the converter's liveness (G-L0.4, asserted at conversion; recorded here), writes PASS/FAIL.
    python l0_gates.py --log ~/runs/base_v4_lesmh_prod.log --compare ~/runs/base_v4_lesmh_prod/compare_v2.json \
        --label v4 --born ~/runs/base_v4_lesmh_prod/les_gates.json --out ~/runs/base_v4_lesmh_prod/L0_GATES.json
"""
import argparse, json, os, re
ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True); ap.add_argument("--compare", required=True); ap.add_argument("--label", default="v4")
ap.add_argument("--born", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
# G-L0.1: base v2 prod epoch 29 (MACE table, per component)
G1_F, G1_E = float(os.environ.get("GATE_F", 10.23)), float(os.environ.get("GATE_E", 1.16))   # L0: base v2; D2 (G-D2.1): 1.2x -> 12.3 / 1.4
# G-L0.2: base v2 on the charged 79-atom set, per-component meV/A (the LES-report table)
G2 = {"total": 33.22, "2-4": 74.9, "4-8": 30.4, "8-10": 15.3, "10-12": 8.9, "12+": 8.0}
rows = [l for l in open(a.log, errors="replace") if "Epoch 29: head: Default" in l]
res = {}
if rows:
    m = re.search(r"RMSE_E_per_atom=\s*([\d.]+) meV, RMSE_F=\s*([\d.]+) meV", rows[-1])
    e29, f29 = float(m.group(1)), float(m.group(2))
    res["G-L0.1"] = {"epoch29_E_meV_per_atom": e29, "epoch29_F_meV_per_A": f29, "thresholds": [G1_E, G1_F],
                     "pass": e29 <= G1_E and f29 <= G1_F}
else:
    res["G-L0.1"] = {"pass": False, "error": "no epoch-29 Default row"}
c = json.load(open(a.compare))["categories"]
ch = c["charged vacancy 79"][a.label]; v2 = c["charged vacancy 79"].get("v2")
got = {"total": ch["force_rmse_meV_per_A"], **ch["shells_meV_per_A"]}
res["G-L0.2"] = {"charged79": got, "thresholds_base_v2_registered": G2,
                 "same_run_v2": ({"total": v2["force_rmse_meV_per_A"], **v2["shells_meV_per_A"]} if v2 else None),
                 "fails": [k for k in G2 if got[k] > G2[k]], "n_frames": c["charged vacancy 79"]["n_frames"]}
res["G-L0.2"]["pass"] = not res["G-L0.2"]["fails"]
nv = c["neutral vacancy 79"][a.label]
res["neutral79_reported"] = {"total": nv["force_rmse_meV_per_A"], **nv["shells_meV_per_A"],
                             "base_v2_floor": {"total": 10.96, "2-4": 20.0, "4-8": 10.0, "8-10": 8.8, "10-12": 7.1, "12+": 6.5}}
b = json.load(open(a.born))
res["G-L0.3"] = {"Z_star_signfixed": b["Z_star_iso_signfixed"], "abs_Pb_over_Cs": b["abs_Pb_over_Cs"],
                 "sum_Z_star": b["sum_Z_star_per_cell_signfixed"], "sign_fix": b["sign_fix"], "gate": b["gate"],
                 "pass": b["G-L0.3"] == "PASS"}
res["recorded"] = {"net_latent_charge_per_cell_raw": b["net_latent_charge_per_cell_raw"],
                   "q_species_mean_raw": b["q_species_mean_raw"], "abs_q_mean": b["abs_q_mean"],
                   "e_les_meV_per_atom": b["e_les_meV_per_atom"]}
res["L0"] = "PASS" if all(res[k]["pass"] for k in ("G-L0.1", "G-L0.2", "G-L0.3")) else "FAIL"
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1)); print("L0:", res["L0"])
