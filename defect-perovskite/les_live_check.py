"""The gate that was missing last time: are the latent charges actually LIVE?"""
import sys, os, glob
import numpy as np, torch
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
path = sys.argv[1]
m = torch.load(path, weights_only=False, map_location="cpu")
tot = 0.0
for i, r in enumerate(m.les_readouts):
    final = getattr(r, "linear_2", None) or getattr(r, "linear", None)
    n = sum(float((p ** 2).sum()) for p in final.parameters()) ** 0.5
    tot += n
    print("  les_readouts[%d] final-layer ||W|| = %.6e" % (i, n))
print("LATENT CHARGES:", "LIVE" if tot > 0 else "*** PINNED AT ZERO -- the run is a no-op ***")
