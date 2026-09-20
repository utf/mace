"""Two-model plan §2: reference and target evaluation on the published splits, with the paper
repo's own metric definitions (`scalar_metrics` / diag-offdiag split imported from
`plot_foundation.py`) plus the training-log conventions (all-component RMSE; polarisation
folded with the loss's `fold_polarization`). Works for the 1.0.2 model (heads pt_head /
mp-dielectric / mp-ferroelectric) and for Model A / B (heads omat_pbe / mp-dielectric /
mp-ferroelectric). Every dataset is evaluated with the model's replay head and with the
dedicated head. float64 unless --dtype float32.

  python defect-perovskite/mh1f2_ref_eval.py --model ~/src/macefield/1.0.2/MACEField-MH-0-omat-dielectric.model \
      --replay_head pt_head --out ~/runs/mh1f2/ref/ref_metrics_1.0.2.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase.io import read
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, "/home/alex/src/2025-04-mace-field/scripts/Foundation")
from mace import data as mace_data, tools          # noqa: E402
from mace.tools import torch_geometric as tg       # noqa: E402
from mace.modules.loss import fold_polarization    # noqa: E402
from plot_foundation import scalar_metrics         # noqa: E402  (the paper's metric code)

S = "/home/alex/src/2025-04-mace-field/scripts"; D = "/home/alex/runs/mh1f2/data"
# primary: the recipe's own seed-123 split of the full released files (mh1f2_split.py); extras: the from-scratch test files
DATASETS = {"dielectric_train": f"{D}/dielectric_train_seed123.xyz", "dielectric_valid": f"{D}/dielectric_valid_seed123.xyz",
            "ferroelectric_train": f"{D}/ferroelectric_train_seed123.xyz", "ferroelectric_valid": f"{D}/ferroelectric_valid_seed123.xyz",
            "dielectric_test_fromscratch": f"{S}/Dielectrics/MP-Dielectrics-filtered-test.xyz", "ferroelectric_test_fromscratch": f"{S}/Ferroelectrics/MP-Ferroelectrics-test.xyz",
            "dielectric_valid_fromscratch": f"{S}/Dielectrics/MP-Dielectrics-filtered-valid.xyz", "ferroelectric_valid_fromscratch": f"{S}/Ferroelectrics/MP-Ferroelectrics-valid.xyz"}
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--replay_head", default="pt_head"); ap.add_argument("--out", required=True)
ap.add_argument("--replay_holdout", default="/home/alex/runs/mh1f2/data/replay_holdout_dft.xyz")
ap.add_argument("--datasets", default="dielectric_train,dielectric_valid,ferroelectric_train,ferroelectric_valid,dielectric_test_fromscratch,ferroelectric_test_fromscratch"); ap.add_argument("--heads", default="dedicated,replay")
ap.add_argument("--device", default="cuda"); ap.add_argument("--dtype", default="float64"); ap.add_argument("--atoms_per_batch", type=int, default=64)
ap.add_argument("--max_atoms", type=int, default=100000)
args = ap.parse_args()
DT = torch.float64 if args.dtype == "float64" else torch.float32
torch.set_default_dtype(DT); dev = args.device; t0 = time.time()
m = torch.load(args.model, map_location=dev, weights_only=False).to(dev).to(DT).eval()
zt = tools.AtomicNumberTable([int(z) for z in m.atomic_numbers]); table = set(zt.zs)
ks = mace_data.KeySpecification.from_defaults()
ks.info_keys.update(energy="REF_energy", stress="REF_stress", polarization="REF_polarization", polarizability="REF_polarizability", electric_field="REF_electric_field")
ks.arrays_keys.update(forces="REF_forces", becs="REF_becs")


def batches(frames):
    order = np.argsort([len(a) for a in frames]); cur, n = [], 0
    for i in order:
        if cur and n + len(frames[i]) > args.atoms_per_batch:
            yield cur; cur, n = [], 0
        cur.append(int(i)); n += len(frames[i])
    if cur: yield cur


def run(frames, head, kind):
    """kind: dielectric (Z*, alpha) | ferro (P folded, E, F) | replay (E, F, sigma)."""
    kw = dict(training=kind == "ferro", compute_force=kind != "dielectric", compute_stress=kind == "replay", compute_polarization=kind == "ferro",
              compute_becs=kind == "dielectric", compute_polarizability=kind == "dielectric")
    Z_ref, Z_pred, A_ref, A_pred, P_d, E_d, F2, nF, S_d = [], [], [], [], [], [], 0.0, 0, []
    for ids in batches(frames):
        fs = [frames[i] for i in ids]
        ads = [mace_data.AtomicData.from_config(mace_data.config_from_atoms(a, key_specification=ks, head_name=head), z_table=zt, cutoff=float(m.r_max), heads=list(m.heads)) for a in fs]
        b = next(iter(tg.dataloader.DataLoader(ads, batch_size=len(ads)))).to(dev)
        o = m(b.to_dict(), **kw)
        if kind == "dielectric":
            Z_ref.append(b.becs.detach().cpu().numpy().reshape(-1, 3, 3)); Z_pred.append(o["becs"].detach().cpu().numpy().reshape(-1, 3, 3))
            A_ref.append(b.polarizability.detach().cpu().numpy().reshape(-1, 3, 3)); A_pred.append(o["polarizability"].detach().cpu().numpy().reshape(-1, 3, 3))
        if kind == "ferro":
            dP, _ = fold_polarization(o["polarization"].detach(), b.polarization.view(-1, 3), b.cell)
            P_d.append(dP.detach().cpu().numpy())
        if kind in ("ferro", "replay"):
            n_at = (b.ptr[1:] - b.ptr[:-1]).to(DT)
            E_d.append(((o["energy"].detach() - b.energy) / n_at).cpu().numpy()); F2 += float(((o["forces"].detach() - b.forces) ** 2).sum()); nF += int(b.forces.numel())
        if kind == "replay":
            S_d.append((o["stress"].detach() - b.stress).cpu().numpy().reshape(-1, 9))
        del o
    res = {"n_frames": len(frames)}
    if kind == "dielectric":
        Zr, Zp, Ar, Ap = (np.concatenate(x, 0) for x in (Z_ref, Z_pred, A_ref, A_pred))
        d = np.eye(3, dtype=bool)
        res.update(rmse_becs=float(np.sqrt(np.mean((Zp - Zr) ** 2))), becs_diag=scalar_metrics(Zr[:, d], Zp[:, d]), becs_offdiag=scalar_metrics(Zr[:, ~d], Zp[:, ~d]),
                   rmse_polarizability=float(np.sqrt(np.mean((Ap - Ar) ** 2))), polarizability_diag=scalar_metrics(Ar[:, d], Ap[:, d]), polarizability_offdiag=scalar_metrics(Ar[:, ~d], Ap[:, ~d]),
                   n_atoms=int(Zr.shape[0]))
    if kind == "ferro":
        P = np.concatenate(P_d, 0)
        res.update(rmse_polarization_folded_e_A2=float(np.sqrt(np.mean(P ** 2))), rmse_polarization_folded_uC_cm2=1602.176634 * float(np.sqrt(np.mean(P ** 2))))
    if kind in ("ferro", "replay"):
        res.update(rmse_e_meV_atom=1e3 * float(np.sqrt(np.mean(np.square(np.concatenate(E_d))))), rmse_f_meV_A=1e3 * float(np.sqrt(F2 / nF)))
    if kind == "replay":
        res.update(rmse_stress_meV_A3=1e3 * float(np.sqrt(np.mean(np.square(np.concatenate(S_d, 0))))))
    return res


out = {"model": args.model, "dtype": args.dtype, "replay_head": args.replay_head, "datasets": {}}
for name in args.datasets.split(","):
    frames = [a for a in read(DATASETS[name], ":") if len(a) <= args.max_atoms]
    n_all = len(frames); frames = [a for a in frames if set(a.numbers.tolist()) <= table]
    kind = "dielectric" if name.startswith("dielectric") else "ferro"
    dedicated = "mp-dielectric" if kind == "dielectric" else "mp-ferroelectric"
    out["datasets"][name] = {"n_in_table": len(frames), "n_total": n_all}
    for tag in args.heads.split(","):
        head = dedicated if tag == "dedicated" else args.replay_head
        if head not in list(m.heads): continue
        out["datasets"][name][head] = run(frames, head, kind)
        r = out["datasets"][name][head]
        print(f"{name:20s} {head:16s} n={len(frames):5d} " + (f"Z* {r['rmse_becs']:.4f} (diag {r['becs_diag']['rmse']:.4f} off {r['becs_offdiag']['rmse']:.4f})  alpha {r['rmse_polarizability']:.4f} (diag {r['polarizability_diag']['rmse']:.3f} off {r['polarizability_offdiag']['rmse']:.3f})" if kind == "dielectric"
              else f"P folded {r['rmse_polarization_folded_e_A2']:.5f} e/A2 = {r['rmse_polarization_folded_uC_cm2']:.2f} uC/cm2  E {r['rmse_e_meV_atom']:.1f} meV/atom  F {r['rmse_f_meV_A']:.1f} meV/A") + f"  ({time.time()-t0:.0f}s)", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
if args.replay_holdout and Path(args.replay_holdout).exists():
    frames = [a for a in read(args.replay_holdout, ":") if set(a.numbers.tolist()) <= table]
    out["replay_holdout"] = {args.replay_head: run(frames, args.replay_head, "replay"), "n": len(frames)}
    r = out["replay_holdout"][args.replay_head]
    print(f"replay hold-out ({args.replay_head}) n={len(frames)}: E {r['rmse_e_meV_atom']:.2f} meV/atom  F {r['rmse_f_meV_A']:.2f} meV/A  sigma {r['rmse_stress_meV_A3']:.3f} meV/A3", flush=True)
Path(args.out).write_text(json.dumps(out, indent=1)); print("saved", args.out, f"{time.time()-t0:.0f}s")
