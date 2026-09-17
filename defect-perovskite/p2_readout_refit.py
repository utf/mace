"""Plan P registered remedy (A3): readout-only refit with E_PP in the forward on the target
frames alone (never replay). Loads the pruned Phase 1 base, switches its own E_PP on
(pp_r_split / pp_eps_inf), freezes everything except the energy readouts, trains a few epochs on
the neutral training frames with the Phase 1 loss weights, validates on the neutral validation and
keeps the best-force refit epoch (the pre-refit state is not a candidate). Acceptance (registered):
after the refit, validation F with E_PP <= base-alone F + 0.2 meV/A and E <= base-alone E + 0.2
meV/atom, i.e. E_PP absorbed with no degradation beyond noise; written to the JSON as accepted.

    python defect-perovskite/p2_readout_refit.py --base RUN/base.pt --out RUN/base_epp.pt --json RUN/P2_REFIT.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase.io import read
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools            # noqa: E402
from mace.modules.loss import UniversalFieldLoss       # noqa: E402
from mace.tools import torch_geometric as tg           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--out", required=True); ap.add_argument("--json", required=True)
ap.add_argument("--train", default="/home/alex/runs/dscc/p1v2/train_v2_iso.xyz"); ap.add_argument("--valid", default="/home/alex/runs/dscc/p1v2/valid_v2.xyz")
ap.add_argument("--r_split", type=float, default=4.5); ap.add_argument("--eps_inf", type=float, default=4.0)
ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--batch_size", type=int, default=2)
ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
torch.manual_seed(args.seed); torch.set_default_dtype(torch.float32); dev = torch.device(args.device)
m = torch.load(args.base, map_location="cpu", weights_only=False).to(dev).float()
head = list(m.heads)[-1]; zt = tools.AtomicNumberTable([int(z) for z in m.atomic_numbers])
ks = mace_data.KeySpecification.from_defaults()
ks.info_keys.update(energy="REF_energy", stress="REF_stress", polarization="REF_polarization", polarizability="REF_polarizability")
ks.arrays_keys.update(forces="REF_forces", becs="REF_becs")
def dataset(path):
    ads = []
    for a in read(path, ":"):
        if len(a) == 1: continue
        cfg = mace_data.config_from_atoms(a, key_specification=ks, head_name=head)
        ads.append(mace_data.AtomicData.from_config(cfg, z_table=zt, cutoff=float(m.r_max), heads=list(m.heads)))
    return ads
train, valid = dataset(args.train), dataset(args.valid)
tl = tg.dataloader.DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(args.seed))
vl = tg.dataloader.DataLoader(valid, batch_size=1, shuffle=False)
loss_fn = UniversalFieldLoss(energy_weight=10, forces_weight=10, stress_weight=1, polarization_weight=10, becs_weight=1, polarizability_weight=1).to(dev)
KW = dict(training=True, compute_force=True, compute_stress=True, compute_polarization=True, compute_becs=True, compute_polarizability=True)
def evaluate():
    # energies and forces only (no response derivatives): cheap, and E_PP is in the forward when on
    m.eval(); se = 0.0; sf = 0.0; nf = 0; ne = 0
    for b in vl:
        b = b.to(dev); o = m(b.to_dict(), training=False, compute_force=True, compute_stress=False)
        n_at = (b.ptr[1:] - b.ptr[:-1]).float()
        se += float((((o["energy"] - b.energy) / n_at) ** 2).sum()); ne += int(b.num_graphs)
        sf += float(((o["forces"] - b.forces) ** 2).sum()); nf += int(b.forces.numel()); del o
    m.train(); return 1e3 * np.sqrt(se / ne), 1e3 * np.sqrt(sf / nf)
e0, f0 = evaluate(); print(f"base alone: valid E {e0:.2f} meV/atom, F {f0:.2f} meV/A", flush=True)
m.pp_r_split = float(args.r_split); m.pp_eps_inf = float(args.eps_inf)
e1, f1 = evaluate(); print(f"base + E_PP (r_split {args.r_split}), before refit: E {e1:.2f}, F {f1:.2f}", flush=True)
for p in m.parameters(): p.requires_grad_(False)
params = [p for r in m.readouts for p in r.parameters()]
for p in params: p.requires_grad_(True)
print("trainable parameters (readouts):", sum(p.numel() for p in params), flush=True)
opt = torch.optim.Adam(params, lr=args.lr)
best = (float("inf"), 0); best_state = None; hist = [{"epoch": -1, "E": e1, "F": f1}]
t0 = time.time()
for ep in range(args.epochs):
    for b in tl:
        b = b.to(dev); opt.zero_grad(set_to_none=True); o = m(b.to_dict(), **KW); l = loss_fn(pred=o, ref=b); l.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); del o, l
    e, f = evaluate(); hist.append({"epoch": ep, "E": e, "F": f}); print(f"epoch {ep}: valid E {e:.2f} meV/atom, F {f:.2f} meV/A  ({time.time()-t0:.0f}s)", flush=True)
    if f < best[0]: best = (f, ep + 1); best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
m.load_state_dict(best_state); ef, ff = evaluate()
accepted = (ff <= f0 + 0.2) and (ef <= e0 + 0.2)
torch.save(m.cpu(), args.out)
json.dump({"base": args.base, "out": args.out, "r_split": args.r_split, "eps_inf": args.eps_inf, "base_alone": {"E": e0, "F": f0}, "with_epp_before": {"E": e1, "F": f1},
           "with_epp_after": {"E": ef, "F": ff}, "best_epoch": best[1], "history": hist, "acceptance_F_max": f0 + 0.2, "acceptance_E_max": e0 + 0.2, "accepted": bool(accepted)}, open(args.json, "w"), indent=1)
print(f"refit: E {ef:.2f}, F {ff:.2f} (acceptance F <= {f0+0.2:.2f}, E <= {e0+0.2:.2f}) -> {'ACCEPTED' if accepted else 'REJECTED'}; saved {args.out}")
