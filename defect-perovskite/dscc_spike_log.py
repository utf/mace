"""v4.3: log the loss-spike events of a set of runs (epoch, loss, held-out RMSE before/after)
from their train.log — recorded, not acted on. A spike is an epoch >= 20 whose training loss
exceeds 3x the median loss of epochs 30+ (the Arm-1 watch item's definition).
    python defect-perovskite/dscc_spike_log.py --runs ~/runs/dscc/dscc_arm23_* --out spikes.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def spikes_of(run: Path, factor: float = 3.0, from_epoch: int = 20):
    lines = (run / "train.log").read_text().splitlines()
    ep = [(int(m.group(1)), float(m.group(2)), m.group(3)) for l in lines
          for m in [re.search(r"epoch (\d+) loss ([0-9.e+-]+) .* held (\S+)", l)] if m]
    if not ep:
        return {"epochs": 0, "spikes": []}
    loss = np.array([x[1] for x in ep])
    held = {e: float(h) for e, _, h in ep if h != "None"}
    median = float(np.median(loss[30:])) if len(loss) > 30 else float(np.median(loss))
    out = []
    for e, l, _ in ep:
        if e >= from_epoch and l > factor * median:
            before = [h for k, h in sorted(held.items()) if k < e]
            after = [h for k, h in sorted(held.items()) if k >= e]
            out.append({"epoch": e, "loss": l, "loss_over_median": l / median,
                        "held_before": before[-1] if before else None, "held_after": after[0] if after else None})
    return {"epochs": len(ep), "median_loss_30plus": median, "spikes": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {Path(r).name: spikes_of(Path(r)) for r in args.runs}
    json.dump(report, open(args.out, "w"), indent=1)
    for name, rec in report.items():
        print(name, f"{rec['epochs']} epochs,", len(rec["spikes"]), "spikes",
              [(s["epoch"], f"{s['loss_over_median']:.0f}x") for s in rec["spikes"]][:6])


if __name__ == "__main__":
    main()
