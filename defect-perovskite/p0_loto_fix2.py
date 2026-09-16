"""Apply the +-q factor of 2 to the nonanalytic reference fields of a p0_loto JSON written before
the fix (the model-side numbers are untouched)."""
import json, sys
for path in sys.argv[1:]:
    d = json.load(open(path))
    if d.get("nac_pm_q_fixed"):
        continue
    for br in d["branches"].values():
        for k in ("k_nac_raw", "k_nac_smoothed", "k_nac_smoothed_x_ff2"):
            if k in br: br[k] *= 2.0
    for k in ("nac_raw", "nac_smoothed"):
        d["splitting"][k] *= 2.0
    d["nac_pm_q_fixed"] = True
    json.dump(d, open(path, "w"), indent=1); print("fixed", path, d["splitting"])
