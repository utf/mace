"""Period-2 statistic of the Gamma-gap term from a per-step record (spike investigation II): per epoch, the fraction of
consecutive steps on which (gap - 2.40) changes sign, the lag-1 autocorrelation, the median |gap - 2.40|, the epoch-end
gap, the loss, and the largest l_gap. Usage: python p2c_gap_stats.py steps.jsonl [steps.jsonl ...] [--every 5]"""
import json, sys, numpy as np
from collections import defaultdict
every = 5
args = sys.argv[1:]
if "--every" in args:
    i = args.index("--every"); every = int(args[i + 1]); del args[i:i + 2]
paths = args
for p in paths:
    rows = [json.loads(l) for l in open(p)]
    by = defaultdict(list)
    for r in rows: by[r["epoch"]].append(r)
    print(p); print("epoch   n  |err| p50  alternation  lag-1 ac   gap(end)   loss mean   l_gap max   n(l_gap>1e-3)")
    for e in sorted(by):
        rs = by[e]; x = np.array([r["gap"] - 2.40 for r in rs]); s = np.sign(x)
        alt = float(np.mean(s[1:] * s[:-1] < 0)); ac = float(np.corrcoef(x[1:], x[:-1])[0, 1]) if x.std() > 0 else float("nan")
        lg = np.array([r["l_gap"] for r in rs])
        if e % every == 0 or e == max(by) or lg.max() > 1e-3:
            print(f"{e:4d} {len(rs):4d}   {np.median(np.abs(x)):.4f}      {alt:.2f}      {ac:+.2f}    {rs[-1]['gap']:.4f}    {np.mean([r['loss'] for r in rs]):.3g}    {lg.max():.3g}    {(lg > 1e-3).sum()}")
