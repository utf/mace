"""W4 factorial queue: the bounded environment-dependent rank-2 / rank-1 coefficients.

Variants (v5 plan W4, restated by amendment A1 in its reference-free form):
  scalar  directional off                      -- the scalar-only control
  spec    directional on, beta_b = beta_a = 0  -- species coefficients; THIS IS WHAT W3 RAN,
                                                 so the W3 arm is reused rather than repeated
  rank2   b_i = b_Z (1 + beta tanh g_Z(h_i))
  rank1   a_i = a_Z (1 + beta tanh f_Z(h_i))
  full    both

    python dscc_make_w4_queue.py --out ~/runs/w4_queue.txt --arm bp_lr_only --variants rank2
"""
import argparse

FOLD_OF = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 6: 2}          # W0.3: seed -> outer fold
ARMS = {"phi0": "--coupling 0 --route_b 0",
        "lr_only": "--coupling 1 --coupling_mode lr_only --route_b 0",
        "bp_lr_only": "--coupling 1 --coupling_mode lr_only --route_b 1"}
VARIANTS = {"scalar": "--directional 0 --beta_b 0.0 --beta_a 0.0",
            "spec":   "--directional 1 --beta_b 0.0 --beta_a 0.0",
            "rank2":  "--directional 1 --beta_b 0.5 --beta_a 0.0",
            "rank1":  "--directional 1 --beta_b 0.0 --beta_a 0.5",
            "full":   "--directional 1 --beta_b 0.5 --beta_a 0.5"}

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--arm", default="bp_lr_only", choices=sorted(ARMS))
ap.add_argument("--variants", default="rank2", help="comma separated; see VARIANTS")
ap.add_argument("--seeds", default="0,1,2,3,4,6")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--base", default="/home/alex/runs/base_v2_prod/base_v2_prod_base.pt")
ap.add_argument("--setup_batch_size", type=int, default=4)
ap.add_argument("--base_cache_batch_size", type=int, default=4)
ap.add_argument("--gpu_memory_fraction", type=float, default=0.0)
args = ap.parse_args()

lines = []
for variant in args.variants.split(","):
    for seed in (int(x) for x in args.seeds.split(",")):
        name = f"dscc_w4_{variant}_{args.arm}_s{seed}"
        lines.append(
            f"{name}|--seed {seed} --fold {FOLD_OF[seed]} --epochs {args.epochs} --regime B "
            f"--base {args.base} {ARMS[args.arm]} {VARIANTS[variant]} "
            f"--setup_batch_size {args.setup_batch_size} "
            f"--base_cache_batch_size {args.base_cache_batch_size} "
            f"--gpu_memory_fraction {args.gpu_memory_fraction} --base_float32 1")
open(args.out, "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} runs written to {args.out}")
