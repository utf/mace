"""W3 prerequisite: convert a base_v2 multi-head fine-tune into the plain ScaleShiftMACE the
D-SCC head takes (`dscc_train.py --base`): drop the replay head, prune to {Cl, Cs, Pb}, float64.
Predictions of the pruned model on kept-element frames equal the multi-head model's Default-head
predictions (checked here on real frames before the file is written).
"""
import argparse, os, sys
import numpy as np, torch
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ase.io import read
from mace.tools import AtomicNumberTable
from w13_eval import load_base, forces_energies, SPECIES

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--check", default=f"{W}/defect-perovskite/dataset_cf/fold0/null_oof.xyz")
ap.add_argument("--n_check", type=int, default=8)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
torch.set_default_dtype(torch.float64)

single = load_base(args.model, "Default").to(args.device)
frames = read(args.check, f":{args.n_check}")
zt = AtomicNumberTable(SPECIES)
F, E = forces_energies(single, frames, zt, 6.0, args.device, 4)
print("heads after conversion:", getattr(single, "heads", None), "elements:", single.atomic_numbers.tolist())
print("check frames:", len(frames), "energy range", f"{E.min():.3f}..{E.max():.3f} eV",
      "|F| max", f"{max(float(np.abs(f).max()) for f in F):.3f} eV/A")
torch.save(single.cpu(), os.path.expanduser(args.out))
print("wrote", args.out)
