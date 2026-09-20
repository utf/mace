"""Benchmark for the replay-drift numbers: the MH-0 MACE-Field's own error against MH-1's labels
on the same 10 % replay hold-out the unfrozen runs were read on (omat_pbe pseudolabels of MH-1).
Forces are compared directly (per component); energies after a per-element offset fit on the
hold-out (the two models' E0 conventions differ), reported per atom. Frames with elements outside
MH-0's table are skipped and counted.

  python defect-perovskite/mh1_field_replay_bench.py --out ~/runs/mh1field/replay_bench_mh0.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from ase.io import read
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools          # noqa: E402
from mace.tools import torch_geometric as tg       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mh0", default="/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model")
ap.add_argument("--heads", default="pt_head,mp-dielectric"); ap.add_argument("--replay", default="/home/alex/.cache/mace/replay_mh1_omat_pbe_sel10k.xyz")
ap.add_argument("--max_atoms", type=int, default=120); ap.add_argument("--seed", type=int, default=124); ap.add_argument("--fraction", type=float, default=0.1)
ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_default_dtype(torch.float32); dev = args.device
m = torch.load(args.mh0, map_location=dev, weights_only=False).to(dev).float().eval()
rp = [a for a in read(args.replay, ":") if len(a) <= args.max_atoms]
perm = np.random.default_rng(args.seed).permutation(len(rp)); hold = [rp[i] for i in perm[: int(args.fraction * len(rp))]]
table = set(int(z) for z in m.atomic_numbers); ok = [a for a in hold if set(a.numbers.tolist()) <= table]
print(f"replay hold-out {len(hold)} frames; {len(ok)} within MH-0's {len(table)}-element table", flush=True)
ks = mace_data.KeySpecification.from_defaults(); ks.info_keys.update(energy="REF_energy", stress="REF_stress"); ks.arrays_keys.update(forces="REF_forces")
zt = tools.AtomicNumberTable(sorted(table))
res = {"n_holdout": len(hold), "n_evaluated": len(ok), "heads": {}}
for head in args.heads.split(","):
    E_pred, E_ref, counts, sf, nf = [], [], [], 0.0, 0
    for k in range(0, len(ok), 8):
        fs = ok[k:k + 8]
        ads = [mace_data.AtomicData.from_config(mace_data.config_from_atoms(a, key_specification=ks, head_name=head), z_table=zt, cutoff=float(m.r_max), heads=list(m.heads)) for a in fs]
        b = next(iter(tg.dataloader.DataLoader(ads, batch_size=len(ads)))).to(dev)
        o = m(b.to_dict(), training=False, compute_force=True, compute_stress=False)
        E_pred += o["energy"].detach().cpu().tolist(); E_ref += b.energy.cpu().tolist()
        sf += float(((o["forces"] - b.forces) ** 2).sum()); nf += int(b.forces.numel())
        for a in fs:
            c = np.zeros(len(zt.zs)); 
            for z in a.numbers: c[zt.z_to_index(int(z))] += 1
            counts.append(c)
    C = np.array(counts); dE = np.array(E_pred) - np.array(E_ref); n_at = C.sum(1)
    raw = 1e3 * float(np.sqrt(np.mean((dE / n_at) ** 2)))
    off, *_ = np.linalg.lstsq(C / n_at[:, None], dE / n_at, rcond=None); resid = dE - C @ off   # fit in the per-atom metric
    fitted = 1e3 * float(np.sqrt(np.mean((resid / n_at) ** 2)))
    res["heads"][head] = {"rmse_E_raw_meV_atom": raw, "rmse_E_after_per_element_offset_meV_atom": fitted, "rmse_F_meV_A": 1e3 * float(np.sqrt(sf / nf)), "dE_eV": dE.tolist(), "n_atoms": n_at.tolist()}
    print(f"MH-0 {head:14s} vs MH-1 labels: E raw {raw:.1f} meV/atom, after per-element offsets {fitted:.2f} meV/atom; F {1e3*np.sqrt(sf/nf):.1f} meV/A", flush=True)
Path(args.out).write_text(json.dumps(res, indent=1)); print("saved", args.out)
