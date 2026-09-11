"""Re-read C5 at any threshold from saved per-frame records -- no re-sweep.

    python c5_rethreshold.py --gates ~/runs/dscc/c5_*.json --delta_c 0.20
"""
import argparse
import glob
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gates", nargs="+", required=True)
ap.add_argument("--delta_c", type=float, default=0.20)
ap.add_argument("--n_loc", type=float, default=4.0)
ap.add_argument("--fraction", type=float, default=0.95)
args = ap.parse_args()

print("Delta_c %.3f eV   N_loc %.1f   need %.0f%% of frames" % (args.delta_c, args.n_loc, 100 * args.fraction))
print("%-28s %8s %8s   %s" % ("head", "pass", "sep p50", "verdict"))
for pattern in args.gates:
    for path in sorted(glob.glob(pattern)):
        for name, entry in json.load(open(path)).items():
            pre = entry.get("precondition", entry)
            recs = pre.get("records") or []
            if not recs:
                print("%-28s %8s   (no per-frame records saved)" % (name[:28], "-"))
                continue
            ok = [r for r in recs if r["separation"] >= args.delta_c and r["n_eff"] <= args.n_loc]
            frac = len(ok) / len(recs)
            seps = sorted(r["separation"] for r in recs)
            print("%-28s %8.3f %8.0f   %s" % (name.replace("dscc_", "")[:28], frac,
                                              1000 * seps[len(seps) // 2],
                                              "PASS" if frac >= args.fraction else "fail"))
