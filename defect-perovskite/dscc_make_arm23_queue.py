"""Write the Arm 2+3 queue file (v4.1 layout): regime B x route {A, B'} x coupling
{lr_only, lr_u, full, lambda1} + Phi = 0 per route = 10 configurations x 6 seeds, each
starting from the Arm-1 full-H0 winner of the same seed; plus the regime-A ablation (full
coupling x 2 routes x 6 seeds), excluded from selection.
    python defect-perovskite/dscc_make_arm23_queue.py --winners /home/alex/runs/dscc --out ~/runs/arm23_queue.txt
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--winners", default="/home/alex/runs/dscc", help="directory holding dscc_arm1_full_s{seed}/model.pt")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--ablation", type=int, default=1)
ap.add_argument("--routes", default="A", help="A, Bp or A,Bp (Route B' only after its ladder gate passes)")
ap.add_argument("--skip_seeds", default="", help="seeds whose Arm-1 H0 failed the bound-state precondition")
ap.add_argument("--seed_list", default="", help="explicit seeds (e.g. 6,7,8,9,10,11 for the route-C repeat's winners); overrides --seeds")
args = ap.parse_args()
routes = tuple(args.routes.split(","))
skip = {int(x) for x in args.skip_seeds.split(",") if x}
seeds = [int(x) for x in args.seed_list.split(",") if x] or list(range(args.seeds))
lines = []
for seed in seeds:
    if seed in skip:
        continue
    init = f"{args.winners}/dscc_arm1_full_s{seed}/h0_state.pt"      # the converted H0 state (sweep-safe)
    common = f"--seed {seed} --fold {seed % 4} --epochs {args.epochs} --directional 1 --regime B --init_from {init}"
    for route in routes:
        rb = 1 if route == "Bp" else 0
        lines.append(f"dscc_arm23_B_{route}_phi0_s{seed}|{common} --coupling 0 --route_b {rb}")
        for mode in ("lr_only", "lr_u", "full", "lambda1"):
            lines.append(f"dscc_arm23_B_{route}_{mode}_s{seed}|{common} --coupling 1 --coupling_mode {mode} --route_b {rb}")
    if args.ablation:
        for route in routes:
            rb = 1 if route == "Bp" else 0
            lines.append(f"dscc_arm23_A_{route}_full_s{seed}|{common.replace('--regime B', '--regime A')} --coupling 1 --coupling_mode full --route_b {rb}")
Path(args.out).write_text("\n".join(lines) + "\n")
print(f"{len(lines)} runs written to {args.out}")
