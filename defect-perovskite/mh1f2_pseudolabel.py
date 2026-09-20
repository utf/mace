"""Two-model plan §6: the OMAT-PBE replay for Model B. The published 10 000-frame FPS subset
(`subselected-replay-data-mh-0-omat-pbe.xyz`) is split by the published validation indices
(`MACEField-omat-dielectric_valid_indices_123.txt`, 1000 frames): the 9000 training frames are
pseudolabelled by MH-1's `omat_pbe` head (E, F, sigma; the published `--pseudolabel_replay`),
the 1000 hold-out frames keep their DFT labels (never trained on) and also record MH-1's
predictions as `MH1_*`, so MH-1's own E/F/sigma RMSE against DFT on the hold-out (the §6 pass
lines: E and F within 5 %, sigma within 10 %) comes out of the same file. float64.

  python defect-perovskite/mh1f2_pseudolabel.py --out_dir ~/runs/mh1f2/data
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase.io import read, write
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools          # noqa: E402
from mace.tools import torch_geometric as tg       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mh1", default="/home/alex/.cache/mace/macemh1model"); ap.add_argument("--head", default="omat_pbe")
ap.add_argument("--replay", default="/home/alex/src/2025-04-mace-field/scripts/Foundation/data/subselected-replay-data-mh-0-omat-pbe.xyz")
ap.add_argument("--valid_indices", default="/home/alex/src/2025-04-mace-field/scripts/Foundation/MACEField-omat-dielectric_valid_indices_123.txt")
ap.add_argument("--out_dir", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device; t0 = time.time()
out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
m = torch.load(args.mh1, map_location=dev, weights_only=False).to(dev).double().eval()
frames = read(args.replay, ":"); hold = sorted(int(x) for x in open(args.valid_indices).read().split())
assert len(hold) == len(set(hold)) and max(hold) < len(frames)
hold_set = set(hold)
zt = tools.AtomicNumberTable([int(z) for z in m.atomic_numbers]); ks = mace_data.KeySpecification.from_defaults()
ks.info_keys.update(energy="REF_energy", stress="REF_stress"); ks.arrays_keys.update(forces="REF_forces")
E, F, S = {}, {}, {}
idx_all = list(range(len(frames)))
for k in range(0, len(idx_all), args.batch):
    ids = idx_all[k:k + args.batch]
    ads = [mace_data.AtomicData.from_config(mace_data.config_from_atoms(frames[i], key_specification=ks, head_name=args.head), z_table=zt, cutoff=float(m.r_max), heads=list(m.heads)) for i in ids]
    b = next(iter(tg.dataloader.DataLoader(ads, batch_size=len(ads)))).to(dev)
    o = m(b.to_dict(), training=False, compute_force=True, compute_stress=True)
    ptr = b.ptr.tolist(); f = o["forces"].detach().cpu().numpy()
    for j, i in enumerate(ids):
        E[i] = float(o["energy"][j]); F[i] = f[ptr[j]:ptr[j + 1]]; S[i] = o["stress"][j].detach().cpu().numpy().reshape(9)
train_out, hold_out = [], []
dE, dF, dS, nF = [], 0.0, [], 0
for i, a in enumerate(frames):
    b = a.copy(); b.info["head"] = args.head
    if i in hold_set:
        b.info["MH1_energy"] = E[i]; b.arrays["MH1_forces"] = F[i]; b.info["MH1_stress"] = S[i]
        dE.append((E[i] - float(a.info["REF_energy"])) / len(a)); dF += float(((F[i] - np.asarray(a.arrays["REF_forces"])) ** 2).sum()); nF += 3 * len(a)
        dS.append(S[i] - np.asarray(a.info["REF_stress"]).reshape(9))
        hold_out.append(b)
    else:
        b.info["DFT_energy"] = float(a.info["REF_energy"]); b.arrays["DFT_forces"] = np.asarray(a.arrays["REF_forces"]); b.info["DFT_stress"] = np.asarray(a.info["REF_stress"]).reshape(9)
        b.info["REF_energy"] = E[i]; b.arrays["REF_forces"] = F[i]; b.info["REF_stress"] = S[i]
        train_out.append(b)
write(str(out / "replay_train_pseudolabel_mh1.xyz"), train_out); write(str(out / "replay_holdout_dft.xyz"), hold_out)
rec = {"replay": args.replay, "valid_indices": args.valid_indices, "n_train": len(train_out), "n_holdout": len(hold_out), "head": args.head, "dtype": "float64",
       "mh1_vs_dft_holdout": {"rmse_e_meV_atom": 1e3 * float(np.sqrt(np.mean(np.square(dE)))), "rmse_f_meV_A": 1e3 * float(np.sqrt(dF / nF)),
                              "rmse_stress_meV_A3": 1e3 * float(np.sqrt(np.mean(np.square(np.array(dS)))))},
       "pass_lines_model_B": {"rmse_e_meV_atom": 1.05 * 1e3 * float(np.sqrt(np.mean(np.square(dE)))), "rmse_f_meV_A": 1.05 * 1e3 * float(np.sqrt(dF / nF)),
                              "rmse_stress_meV_A3": 1.10 * 1e3 * float(np.sqrt(np.mean(np.square(np.array(dS)))))}}
(out / "replay_record.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1)); print(f"done {time.time()-t0:.0f}s")
