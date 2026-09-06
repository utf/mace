"""Arm 4 queue (plan 2.9 / section 7): the two F-SCC comparators on the selected D-SCC
configuration (same seeds, protocol and Arm-1 initialisation), plus the matched-kernel
comparator frozen at the D-SCC lambda_dir / U_eff values (reported in addition).
    python defect-perovskite/dscc_make_arm4_queue.py --selected B_A_full --out ~/runs/arm4_queue.txt
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--selected", required=True, help="the Arm-2+3 selection, e.g. B_A_full or B_Bp_lambda1")
ap.add_argument("--winners", default="/home/alex/runs/dscc")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--seeds", type=int, default=6)
args = ap.parse_args()
regime, route, mode = args.selected.split("_", 2)
rb = 1 if route == "Bp" else 0
lines = []
for seed in range(args.seeds):
    init = f"{args.winners}/dscc_arm1_full_s{seed}/h0_state.pt"
    common = (f"--seed {seed} --fold {seed % 4} --epochs {args.epochs} --directional 1 --regime {regime} "
              f"--init_from {init} --coupling 1 --coupling_mode {mode} --route_b 0")
    lines.append(f"dscc_arm4_matched_s{seed}|{common} --fscc matched")
    lines.append(f"dscc_arm4_fullk_s{seed}|{common} --fscc full")
Path(args.out).write_text("\n".join(lines) + "\n")
print(f"{len(lines)} runs written to {args.out} (Route B has no F-SCC analogue: host coupling is intrinsic)")
