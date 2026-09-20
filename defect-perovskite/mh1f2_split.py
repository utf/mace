"""Two-model plan §1: the foundation recipe's data. The authors' description (relayed by the user):
the foundation fine-tune used the FULL MP-Dielectric file and the FULL MP-Ferroelectric file from the
release zips, with MACE's own random 10 % validation split (seed 123). This regenerates that split
with MACE's rule (np.random.default_rng(seed).shuffle over the file order; the last 10 % are
validation) on each file and writes train/valid xyz files. No test set exists under this recipe.
Recorded deviation: the published log names data/dielectric_corrected_asr.xyz with 5501 frames;
the released full file has 5274 frames, so the regenerated dielectric split is the recipe applied
to the released file, not the run's own frame list.

  python defect-perovskite/mh1f2_split.py --out_dir ~/runs/mh1f2/data
"""
import argparse, json
from pathlib import Path
import numpy as np
from ase.io import read, write
S = "/home/alex/src/2025-04-mace-field/scripts"
ap = argparse.ArgumentParser(); ap.add_argument("--out_dir", required=True); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--valid_fraction", type=float, default=0.1)
args = ap.parse_args(); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True); rec = {"seed": args.seed, "valid_fraction": args.valid_fraction, "rule": "mace.data.utils.random_train_valid_split", "sets": {}}
for tag, path, head in (("dielectric", f"{S}/Dielectrics/MP-Dielectrics-filtered.extxyz", "mp-dielectric"), ("ferroelectric", f"{S}/Ferroelectrics/MP-Ferroelectrics.xyz", "mp-ferroelectric")):
    fr = read(path, ":"); size = len(fr); train_size = size - int(args.valid_fraction * size)
    idx = list(range(size)); rng = np.random.default_rng(args.seed); rng.shuffle(idx)
    tr, va = sorted(idx[:train_size]), sorted(idx[train_size:])
    for a in fr: a.info["head"] = head
    write(str(out / f"{tag}_train_seed{args.seed}.xyz"), [fr[i] for i in tr]); write(str(out / f"{tag}_valid_seed{args.seed}.xyz"), [fr[i] for i in va])
    (out / f"{tag}_valid_indices_{args.seed}.txt").write_text("\n".join(str(i) for i in idx[train_size:]) + "\n")
    key = "material_id" if tag == "dielectric" else "polar_mpid"
    vm = set(fr[i].info[key] for i in va); tm = set(fr[i].info[key] for i in tr)
    rec["sets"][tag] = {"file": path, "n": size, "n_train": len(tr), "n_valid": len(va), "valid_materials": len(vm), "valid_materials_also_in_train": len(vm & tm), "max_atoms": max(len(a) for a in fr)}
    print(tag, rec["sets"][tag])
(out / "split_record.json").write_text(json.dumps(rec, indent=1))
