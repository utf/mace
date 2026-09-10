"""Label-free size-extensivity test. Tile a 79-atom frame 1x1x2 along its short axis. For a
size-extensive model E(tiled) = 2 E(single) exactly and the forces repeat. A model whose
receptive field exceeds the short axis sees its own periodic image in the single cell but not
in the tiled one, and the identity breaks.
"""
import sys
import numpy as np, torch
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from mace.modules.dscc.train import load_frames
from mace.modules.dscc import data as dd
from mace.tools import torch_geometric as tg, AtomicNumberTable
from mace.modules.models import ScaleShiftMACE
torch.set_default_dtype(torch.float64)
HERE = W + "/defect-perovskite"
frames = load_frames(f"{HERE}/dataset_pbe/train.xyz", f"{HERE}/dataset_pbe/valid.xyz")
singles = [a for a in frames if len(a) == 79 and int(a.info.get("cell_charge", 0)) == 0][:6]

def tile_z(a):
    b = a.repeat((1, 1, 2))
    return b

def energy_forces(m, atoms):
    z = AtomicNumberTable([int(x) for x in m.atomic_numbers])
    b = next(iter(tg.dataloader.DataLoader(dd.atomic_data([atoms], z, float(m.r_max)), batch_size=1))).to("cuda").to_dict()
    o = ScaleShiftMACE.forward(m, b, training=False, compute_force=True)
    return float(o["energy"][0]), o["forces"].detach().cpu().numpy()

for tag, path in (("old base (r_f 10.0 A)", "/home/alex/runs/aprime_prod/aprime_prod_base.pt"),
                  ("base v2 (r_f 12.0 A)", "/home/alex/runs/base_v2_prod/base_v2_prod_base.pt")):
    m = torch.load(path, weights_only=False, map_location="cpu").double().to("cuda").eval()
    dE, dF = [], []
    for a in singles:
        e1, f1 = energy_forces(m, a)
        e2, f2 = energy_forces(m, tile_z(a))
        dE.append(abs(e2 - 2 * e1) / len(a))                 # eV per atom of the single cell
        dF.append(np.abs(f2[:len(a)] - f1).max())
    print(f"{tag}: |E(2x) - 2E(1x)| median {1000*np.median(dE):9.4f} meV/atom, max {1000*np.max(dE):9.4f}; "
          f"|dF| median {1000*np.median(dF):8.4f} meV/A, max {1000*np.max(dF):8.4f}")
    del m; torch.cuda.empty_cache()
