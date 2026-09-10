"""D-SCC arm launcher (plan section 7). One configuration, one seed, one outer fold.

    python defect-perovskite/dscc_train.py --name arm1_full_s0 --seed 0 --fold 0 \
        --directional 1 --coupling 0 --epochs 60 --device cuda --run_dir /home/alex/runs/dscc/arm1_full_s0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mace.modules.dscc import data as dd                      # noqa: E402
from mace.modules.dscc.hamiltonian import DELTA_FRACTION_DEFAULT, ETA_DEFAULT   # noqa: E402
from mace.modules.dscc.kernels import KernelConfig            # noqa: E402
from mace.modules.dscc.model import MACEDSCC                  # noqa: E402
from mace.modules.dscc.scf import ScfOptions                  # noqa: E402
from mace.modules.dscc.train import TrainConfig, Trainer, load_frames, outer_folds   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--directional", type=int, default=1)
    ap.add_argument("--coupling", type=int, default=0)
    ap.add_argument("--route_b", type=int, default=0)
    ap.add_argument("--regime", default="B")
    ap.add_argument("--gap_weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base", default="/home/alex/runs/aprime_prod/aprime_prod_base.pt",
                    help="the frozen base as a plain ScaleShiftMACE (re-saved from aprime_prod.model, equal to 1e-16)")
    ap.add_argument("--dataset", default=str(HERE / "dataset_pbe"))
    ap.add_argument("--cf_dir", default=str(HERE / "dataset_cf"))
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--subset", type=int, default=0, help="debug: use only this many charged frames")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--coupling_mode", default="full", help="lr_only | lr_u | full | lambda1 (Arm 2+3)")
    ap.add_argument("--init_from", default="", help="Arm-1 winner checkpoint (model.pt) to start H0 from")
    ap.add_argument("--n_max", type=int, default=100)
    ap.add_argument("--base_float32", type=int, default=0,
                    help="run the FROZEN base in float32 while the head stays float64. The base "
                         "was fine-tuned in float32, so this costs no fidelity to the trained "
                         "model; it halves the activation memory of the base forward/backward, "
                         "which is the run's high-water mark on base v2.")
    ap.add_argument("--gpu_memory_fraction", type=float, default=0.0,
                    help="cap this process at a fraction of the GPU so several runs share a card; "
                         "0 disables. The cap bounds the caching allocator, which otherwise grows "
                         "to fill an idle card and starves the next process.")
    ap.add_argument("--setup_batch_size", type=int, default=16, help="pristine-reference pass; peak memory only")
    # v5 amendment A1 (registered 2026-09-10). Defaults are the registered W3 values.
    ap.add_argument("--eta", type=float, default=ETA_DEFAULT,
                    help="SK modulation bound, exp(eta tanh m); A1 registers 0.5 (pre-A1 ran ln 3)")
    ap.add_argument("--beta_b", type=float, default=0.0,
                    help="rank-2 environment bound; 0 in W3 (species b_Z), moved by the W4 factorial")
    ap.add_argument("--beta_a", type=float, default=0.0, help="rank-1 environment bound; 0 in W3")
    ap.add_argument("--delta_frac", type=float, default=DELTA_FRACTION_DEFAULT,
                    help="Delta_Z as a fraction of the spread of the species onsite baselines")
    ap.add_argument("--l2_weight", type=float, default=1.8e-6,
                    help="A1's weak L2 on the tanh outputs, per term")
    ap.add_argument("--base_cache_batch_size", type=int, default=8, help="E_base/F_base cache; peak memory only")
    ap.add_argument("--fscc", default="", help="Arm 4 comparator: matched | full (empty: D-SCC)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")])
    torch.set_default_dtype(torch.float64)
    if args.gpu_memory_fraction > 0 and args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        logging.info("GPU memory capped at fraction %.2f (%.1f GB)", args.gpu_memory_fraction,
                     args.gpu_memory_fraction * torch.cuda.get_device_properties(0).total_memory / 1e9)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = TrainConfig(name=args.name, seed=args.seed, fold=args.fold, epochs=args.epochs, lr=args.lr,
                      batch_size=args.batch_size, coupling=bool(args.coupling), coupling_mode=args.coupling_mode, route_b=bool(args.route_b),
                      directional=bool(args.directional), regime=args.regime, gap_weight=args.gap_weight,
                      device=args.device, eval_every=args.eval_every,
                      setup_batch_size=args.setup_batch_size, base_cache_batch_size=args.base_cache_batch_size,
                      l2_weight=args.l2_weight,
                      static_cell_path=str(HERE / "static_pristine_cell.json"))
    base = torch.load(args.base, weights_only=False, map_location="cpu")
    base = base.float() if args.base_float32 else base.double()
    logging.info("base precision: %s (head float64)", next(base.parameters()).dtype)
    model = MACEDSCC(base, r_cut=cfg.r_cut, directional=cfg.directional, coupling=cfg.coupling,
                     route_b=cfg.route_b, kernel=KernelConfig(regime=cfg.regime),
                     eta=args.eta, beta_b=args.beta_b, beta_a=args.beta_a,
                     delta_frac=args.delta_frac,
                     scf=ScfOptions(n_max=args.n_max)).to(args.device)
    logging.info("A1: eta %.4f beta_b %.2f beta_a %.2f delta_Z %.3f eV l2 %.2e",
                 args.eta, args.beta_b, args.beta_a, float(model.h0.sk.delta[0, 0]), args.l2_weight)
    if args.init_from:
        model.load_h0_from(args.init_from)
        logging.info("H0 initialised from %s", args.init_from)
    if args.fscc:
        model.fscc = args.fscc
        logging.info("F-SCC comparator: %s kernel", args.fscc)
    if cfg.coupling:
        model.set_coupling_mode(args.coupling_mode)
        logging.info("coupling mode %s (lambda_fixed %s, u_zero %s)", args.coupling_mode, model.lambda_fixed, model.u_zero)
    frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
    metas = [dd.frame_meta(i, a, "CsPbCl3", pristine_atoms=80) for i, a in enumerate(frames)]
    fold_of = outer_folds(frames, metas, args.cf_dir, cfg.n_folds, cfg.seed)
    pristine = [m.index for m in metas if m.n_atoms == 80 and m.state.Q == 0]
    trainer = Trainer(model, cfg, frames, metas, fold_of, str(run_dir))
    if args.subset:
        rng = np.random.default_rng(args.seed)
        trainer.train_idx = sorted(rng.choice(trainer.train_idx, size=min(args.subset, len(trainer.train_idx)), replace=False).tolist())
        trainer.held_idx = trainer.held_idx[: max(4, args.subset // 4)]
        pristine = pristine[:32]
    logging.info("config %s", json.dumps(cfg.__dict__, default=str))
    logging.info("train %d charged frames, held %d, pristine %d", len(trainer.train_idx), len(trainer.held_idx), len(pristine))
    trainer.prepare(pristine)
    if args.init_from:
        model.load_h0_from(args.init_from)     # the centre is part of the winner's H0
    result = trainer.fit()
    logging.info("done: held-out force RMSE %.4f eV/A", result["held_final"]["force_rmse"])


if __name__ == "__main__":
    main()
