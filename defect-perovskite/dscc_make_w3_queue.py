"""Write the W3 queue file: three arms (Phi = 0, LR-only, B' LR-only) x six seeds on base v2,
folds by the W0.3 map, the registered optimiser (60 epochs, lr 2e-3, batch 4), the W0.4 epoch
average, and the setup batching that lets three runs share a 24 GB card (W3 registration).
No `--init_from`: the Arm-1 H0 states were fitted to the old base's feature width (W1.2).

v5 amendment A1: the head is the reference-free form. Run names carry `w3a1` so nothing
collides with the archived pre-A1 runs under `~/runs/dscc/pre_a1/`, and the A1 numbers are
written on every command line rather than left to the defaults, so the queue file is itself
the record of what was launched.

    python defect-perovskite/dscc_make_w3_queue.py --out ~/runs/w3_queue.txt
"""
import argparse

FOLD_OF = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 6: 2}          # W0.3: seed -> outer fold
ARMS = [("phi0", "--coupling 0 --route_b 0"),
        ("lr_only", "--coupling 1 --coupling_mode lr_only --route_b 0"),
        ("bp_lr_only", "--coupling 1 --coupling_mode lr_only --route_b 1")]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--base", default="/home/alex/runs/base_v2_prod/base_v2_prod_base.pt")
ap.add_argument("--setup_batch_size", type=int, default=2)
ap.add_argument("--base_cache_batch_size", type=int, default=2)
ap.add_argument("--base_float32", type=int, default=1,
                help="frozen base in float32, head in float64 (registered for W3): the training "
                     "peak falls from 11.6 to 7.2 GB, so two runs share a 23.5 GB card")
ap.add_argument("--gpu_memory_fraction", type=float, default=0.42,
                help="two runs per 24 GB card with the float32 base (7.2 GB peak, 7.9 reserved)")
ap.add_argument("--eta", type=float, default=0.5, help="A1 registered SK modulation bound")
ap.add_argument("--beta_b", type=float, default=0.0, help="A1: rank-2 environment bound; 0 in W3")
ap.add_argument("--beta_a", type=float, default=0.0, help="A1: rank-1 environment bound; 0 in W3")
ap.add_argument("--delta_frac", type=float, default=0.40, help="A1: Delta_Z / baseline spread")
ap.add_argument("--l2_weight", type=float, default=1.8e-6, help="A1: weak L2 per tanh term")
ap.add_argument("--skip", default="", help="run names already launched, comma separated")
ap.add_argument("--arms", default="", help="restrict to these arms (phi0, lr_only, bp_lr_only), comma separated")
args = ap.parse_args()

lines = []
wanted = {x for x in args.arms.split(",") if x}
for arm, flags in ARMS:
    if wanted and arm not in wanted:
        continue
    for seed, fold in FOLD_OF.items():
        name = f"dscc_w3a1_{arm}_s{seed}"
        if name in {x for x in args.skip.split(",") if x}:
            continue
        lines.append(f"{name}|--seed {seed} --fold {fold} --epochs {args.epochs} --directional 1 "
                     f"--regime B --base {args.base} {flags} "
                     f"--setup_batch_size {args.setup_batch_size} "
                     f"--base_cache_batch_size {args.base_cache_batch_size} "
                     f"--gpu_memory_fraction {args.gpu_memory_fraction} "
                     f"--base_float32 {args.base_float32} "
                     f"--eta {args.eta} --beta_b {args.beta_b} --beta_a {args.beta_a} "
                     f"--delta_frac {args.delta_frac} --l2_weight {args.l2_weight}")
open(args.out, "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} runs written to {args.out}")
