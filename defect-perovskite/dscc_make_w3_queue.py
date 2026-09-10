"""Write the W3 queue file: three arms (Phi = 0, LR-only, B' LR-only) x six seeds on base v2,
folds by the W0.3 map, the registered optimiser (60 epochs, lr 2e-3, batch 4), the W0.4 epoch
average, and the setup batching that lets three runs share a 24 GB card (W3 registration).
No `--init_from`: the Arm-1 H0 states were fitted to the old base's feature width (W1.2).

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
ap.add_argument("--gpu_memory_fraction", type=float, default=0.30,
                help="three runs per 24 GB card; the cap bounds the caching allocator")
ap.add_argument("--skip", default="", help="run names already launched, comma separated")
args = ap.parse_args()

lines = []
for arm, flags in ARMS:
    for seed, fold in FOLD_OF.items():
        name = f"dscc_w3_{arm}_s{seed}"
        if name in {x for x in args.skip.split(",") if x}:
            continue
        lines.append(f"{name}|--seed {seed} --fold {fold} --epochs {args.epochs} --directional 1 "
                     f"--regime B --base {args.base} {flags} "
                     f"--setup_batch_size {args.setup_batch_size} "
                     f"--base_cache_batch_size {args.base_cache_batch_size} "
                     f"--gpu_memory_fraction {args.gpu_memory_fraction}")
open(args.out, "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} runs written to {args.out}")
