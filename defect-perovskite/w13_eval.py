"""W1.3 stage 1: out-of-fold neutral floors by shell for the base_v2 cross-fit bases, and the
W0.6/W0.6a proxy thresholds (u_F, u_E) per size -- WRITTEN BEFORE the charged window is opened.

Every neutral frame is out-of-fold for exactly one fold (`null_oof.xyz` per fold: 79/159-atom
neutral vacancy frames; `valid.xyz`: 80-atom pristine). Forces are per component throughout
(a per-atom value is the RMS over the three Cartesian components). The foundation reference is
the FROZEN MH-1 `omat_pbe` head, never the fine-tuned model's replay head.

  python w13_eval.py --folds 0,1,2,3 --out ~/runs/w1_oof_stage1.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from ase.io import read
from mace.tools.scripts_utils import remove_pt_head
from mace.modules.dscc.prune import prune_elements
from mace.modules.dscc.train import vacancy_centre_full, near_field_categories
from mace.modules.dscc.admission import collective_coordinate
from mace.modules.dscc import data as dd
from mace.tools import torch_geometric as tg, AtomicNumberTable

SHELLS = [("2-4", 2.0, 4.0), ("4-8", 4.0, 8.0), ("8-10", 8.0, 10.0), ("10-12", 10.0, 12.0), ("12+", 12.0, 1e9)]
SPECIES = [17, 55, 82]


def load_base(path, head):
    m = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
    heads = list(getattr(m, "heads", []) or [])
    if len(heads) > 1:
        m = remove_pt_head(m, head)
    return prune_elements(m, SPECIES).double().eval()


def forces_energies(model, frames, ztable, r_max, device, batch_size=4):
    """Per-atom forces (eV/A) and per-frame energies (eV) of `model` on `frames`."""
    loader = tg.dataloader.DataLoader(dd.atomic_data(frames, ztable, r_max), batch_size=batch_size)
    F, E = [], []
    for b in loader:
        batch = b.to(device).to_dict()
        out = model(batch, training=False, compute_force=True)
        ptr = batch["ptr"].detach().cpu().numpy()
        f = out["forces"].detach().cpu().numpy()
        e = out["energy"].detach().cpu().numpy()
        for g in range(len(ptr) - 1):
            F.append(f[ptr[g]:ptr[g + 1]]); E.append(float(e[g]))
    return F, np.asarray(E)


def per_component(v):
    """|v| per component for a per-atom vector array: RMS over the three components."""
    return np.sqrt((np.asarray(v) ** 2).mean(axis=-1))


def shell_masks(atoms):
    """Radii and the W0.2 categories for one frame; None when there is no flanking pair."""
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
    cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
    numbers = atoms.get_atomic_numbers()
    d = collective_coordinate(pos, cell, numbers)
    found = vacancy_centre_full(pos, cell, numbers)
    if found is None:
        return None, None, d
    rad, flank = found
    cats = near_field_categories(pos, cell, numbers, flank)
    return rad.numpy(), {k: v.numpy() for k, v in cats.items()}, d


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt((x ** 2).mean())) if x.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3")
    ap.add_argument("--data", default=f"{W}/defect-perovskite/dataset_cf")
    ap.add_argument("--models", default="/home/alex/runs/base_v2_f{fold}/base_v2_f{fold}.model")
    ap.add_argument("--foundation", default="~/.cache/mace/macemh1model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--r_max", type=float, default=6.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many frames per file")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",")]
    torch.set_default_dtype(torch.float64)
    t0 = time.time()

    bases = {k: load_base(args.models.format(fold=k), "Default").to(args.device) for k in folds}
    found_model = load_base(args.foundation, "omat_pbe").to(args.device)
    ztable = AtomicNumberTable(SPECIES)
    print(f"models loaded [{time.time()-t0:.0f}s]", flush=True)

    rows = []          # one per out-of-fold neutral frame
    for k in folds:
        for name in ("null_oof.xyz", "valid.xyz"):
            frames = read(f"{args.data}/fold{k}/{name}", f":{args.limit}" if args.limit else ":")
            preds = {j: forces_energies(bases[j], frames, ztable, args.r_max, args.device, args.batch_size) for j in folds}
            fF, fE = forces_energies(found_model, frames, ztable, args.r_max, args.device, args.batch_size)
            for i, a in enumerate(frames):
                n = len(a)
                ref_f = a.arrays["REF_forces"] if "REF_forces" in a.arrays else a.get_forces()
                ref_e = float(a.info.get("REF_energy", a.info.get("energy", np.nan)))
                F_oof = preds[k][0][i]
                err = per_component(F_oof - ref_f)                       # per atom, per component
                stack = np.stack([preds[j][0][i] for j in folds])        # (n_folds, n_atoms, 3)
                sd = per_component(stack.std(axis=0, ddof=1))            # cross-fit spread per atom
                dis = per_component(F_oof - fF[i])                       # foundation disagreement
                rad, cats, d = shell_masks(a)
                row = {"fold": k, "file": name, "n": n, "id": a.info.get("id", ""),
                       "e_resid": float(preds[k][1][i] - ref_e), "e_ft": float(preds[k][1][i]),
                       "e_found": float(fE[i]), "e_sd_folds": float(np.std([preds[j][1][i] for j in folds], ddof=1)),
                       "counts": {int(z): int((a.get_atomic_numbers() == z).sum()) for z in SPECIES},
                       "d": d, "force_rmse": rms(err), "proxy_f": None, "shells": {}, "near": {}}
                if rad is not None:
                    for key, lo, hi in SHELLS:
                        m = (rad >= lo) & (rad < hi)
                        row["shells"][key] = {"rmse": rms(err[m]), "n": int(m.sum())}
                    win = (rad >= 2.0) & (rad < 4.0)
                    for key, mask in cats.items():
                        m = win & mask
                        row["near"][key] = {"rmse": rms(err[m]), "n": int(m.sum())}
                    band = win
                else:
                    band = np.ones(n, dtype=bool)          # pristine: no vacancy, whole frame
                row["proxy_f"] = (dis + sd)[band].tolist()
                row["proxy_f_dis"] = dis[band].tolist()
                row["proxy_f_sd"] = sd[band].tolist()
                row["proxy_f_terms"] = {"dis_p95": float(np.percentile(dis[band], 95)),
                                        "sd_p95": float(np.percentile(sd[band], 95))}
                rows.append(row)
            print(f"fold {k} {name}: {len(frames)} frames [{time.time()-t0:.0f}s]", flush=True)
    json.dump(rows, open(os.path.expanduser(args.out), "w"), indent=1)
    print("wrote", args.out, len(rows), "frames", flush=True)


if __name__ == "__main__":
    main()
