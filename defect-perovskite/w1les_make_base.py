"""L0 converter (G-L0.5): a `base_v4_lesmh` multi-head MACELES fine-tune -> the single-head,
{Cl, Cs, Pb}-only float64 MACELES the D-SCC head takes. Unlike `w3_make_base.py` this ASSERTS
the conversion on real frames rather than printing ranges: the pruned model must reproduce the
multi-head model's Default-head `energy`, `forces`, `latent_charges` and `les_energy` (1e-5), and
the rebuilt `les` module must carry the same conventions (sigma, dl, self-interaction, ...) --
`remove_pt_head` rebuilds through `extract_config_mace_model`, which carries no `les_arguments`,
so the rebuilt module gets the library defaults and the two are compared explicitly.
Also G-L0.4: the latent-charge readouts are LIVE on both heads and on the converted head.
"""
import argparse, os, sys
import numpy as np, torch
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from ase.io import read
from mace.modules.dscc import data as dd
from mace.modules.dscc.prune import prune_elements
from mace.tools import AtomicNumberTable, torch_geometric as tg
from mace.tools.scripts_utils import remove_pt_head

SPECIES = [17, 55, 82]
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--check", default=f"{W}/defect-perovskite/dataset_cf/fold0/null_oof.xyz")
ap.add_argument("--charged", default=f"{W}/defect-perovskite/dataset_cf/eval_qp1.xyz")
ap.add_argument("--n_check", type=int, default=6)
ap.add_argument("--tol", type=float, default=1e-5)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
torch.set_default_dtype(torch.float64)


def readout_norms(model):
    out = []
    for r in model.les_readouts:
        final = getattr(r, "linear_2", None) or getattr(r, "linear", None)
        out.append(sum(float((p ** 2).sum()) for p in final.parameters()) ** 0.5)
    return out


def predict(model, frames, head_index, device):
    zt = AtomicNumberTable([int(z) for z in model.atomic_numbers.tolist()])   # the model's OWN table (86 elements before pruning)
    loader = tg.dataloader.DataLoader(dd.atomic_data(frames, zt, float(model.r_max)), batch_size=3)
    E, F, Q, EL = [], [], [], []
    for b in loader:
        batch = b.to(device).to_dict()
        batch["head"] = torch.full((int(batch["ptr"].numel()) - 1,), head_index, dtype=torch.long, device=device)
        out = model(batch, training=False, compute_force=True)
        E.append(out["energy"].detach().cpu()); F.append(out["forces"].detach().cpu())
        Q.append(out["latent_charges"].detach().cpu()); EL.append(out["les_energy"].detach().cpu())
    return torch.cat(E), torch.cat(F), torch.cat(Q), torch.cat(EL)


multi = torch.load(os.path.expanduser(args.model), map_location="cpu", weights_only=False).double().eval()
assert type(multi).__name__ == "MACELES", type(multi).__name__
heads = list(multi.heads)
print("multi-head model:", heads, "elements", len(multi.atomic_numbers))
norms = readout_norms(multi)
print("latent-charge readout final-layer ||W|| (all heads, concatenated):", ["%.4e" % n for n in norms])
assert all(n > 0 for n in norms), "G-L0.4: a latent-charge readout is at zero -- PINNED"
# per-head liveness: slice each head out and read its own readouts
for h in heads:
    single_h = remove_pt_head(multi, h)
    nh = readout_norms(single_h)
    print(f"  head {h!r}: les_readouts final ||W|| = {['%.4e' % n for n in nh]}")
    assert all(n > 0 for n in nh), f"G-L0.4: head {h} latent-charge readout at zero"

single = remove_pt_head(multi, "Default")
if [int(z) for z in single.atomic_numbers.tolist()] != SPECIES:      # a D1 student is already {Cl, Cs, Pb}
    single = prune_elements(single, SPECIES)
single = single.double().eval()
assert list(single.heads) == ["Default"] and single.atomic_numbers.tolist() == SPECIES

# the LES module must carry the same conventions (remove_pt_head rebuilds it from defaults)
for obj in ("les", "les.ewald", "les.bec"):
    a = eval("multi." + obj); b = eval("single." + obj)
    va = {k: v for k, v in vars(a).items() if not k.startswith("_") and k != "training"}
    vb = {k: v for k, v in vars(b).items() if not k.startswith("_") and k != "training"}
    assert va == vb, f"{obj}: {va} != {vb}"
print("les conventions identical:", {k: v for k, v in vars(single.les).items()
                                     if k in ("sigma", "dl", "remove_self_interaction", "remove_mean", "epsilon_factor")})

frames = read(args.check, f":{args.n_check}") + read(args.charged, f":{args.n_check}")
multi, single = multi.to(args.device), single.to(args.device)
Em, Fm, Qm, ELm = predict(multi, frames, heads.index("Default"), args.device)
Es, Fs, Qs, ELs = predict(single, frames, 0, args.device)
d = {"energy eV": float((Em - Es).abs().max()), "forces eV/A": float((Fm - Fs).abs().max()),
     "latent_charges e": float((Qm - Qs).abs().max()), "les_energy eV": float((ELm - ELs).abs().max())}
print("max |multi(Default) - single| over %d frames (%d neutral OOF + %d charged):" % (len(frames), args.n_check, args.n_check), d)
assert all(v < args.tol for v in d.values()), f"G-L0.5 FAILED: {d}"
print("E range %.3f..%.3f eV; |F| max %.3f; |q| mean %.4f e; E_LES/atom %.2f meV" % (
    float(Es.min()), float(Es.max()), float(Fs.abs().max()), float(Qs.abs().mean()),
    1e3 * float(ELs.sum()) / Fs.shape[0]))
torch.save(single.cpu(), os.path.expanduser(args.out))
print("wrote", args.out)
