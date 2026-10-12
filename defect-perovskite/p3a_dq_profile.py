"""Phase 3a: the shell profile of the vertical hole dq on the embedded +1 structure (per-atom dq vs distance from the
vacancy, by species; shell sums in 1 A bins; cumulative sum). Saves json and png."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace import tools                                                     # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.tools import torch_geometric as tg                               # noqa: E402
from dscc_embed_ladder import KEY_TOL, embed, relative, vacancy_site        # noqa: E402
from p3a_common import load_cell                                           # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--core", required=True); ap.add_argument("--cell", required=True)
ap.add_argument("--tiling", default="3,3,4"); ap.add_argument("--r_core", type=float, default=8.0); ap.add_argument("--device", default="cpu"); ap.add_argument("--out", required=True)
ap.add_argument("--sigma_s", type=float, default=-1.0, help="diagnostic override of the fill smearing (eV); <0 = the model's")
args = ap.parse_args(); torch.set_default_dtype(torch.float64); dev = args.device
unit = load_cell(args.cell); tt = tuple(int(x) for x in args.tiling.split(","))
src = json.load(open(Path(args.core).expanduser())); vac0, cell0 = vacancy_site(unit, tuple(src["tiling"]))
rel0 = np.array(src["rel_positions"]); u = np.array(src["displacements"])
core_disp = {tuple(np.round(rel0[i] / KEY_TOL).astype(np.int64)): u[i] for i in range(len(u))}
atoms, *_ = embed(unit, tt, core_disp, (vac0, cell0), args.r_core); vac, cell = vacancy_site(unit, tt)
r = np.linalg.norm(relative(np.array(atoms.get_positions()), vac, cell), axis=1); syms = np.array(atoms.get_chemical_symbols())
model = torch.load(Path(args.model).expanduser(), weights_only=False, map_location=dev).to(dev).eval(); model.base = model.base.double()
z3 = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
if args.sigma_s > 0:
    model.sigma_s = float(args.sigma_s)
print('sigma_s', model.sigma_s, flush=True)
a = atoms.copy(); a.info.update({"carrier_counts": np.array([0, 0, 1, 0], float), "cell_charge": 1, "config_type": "dq", "source_dir": "dq"})
b = next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, model.r_cut), batch_size=1))).to(dev).to_dict()
with torch.no_grad():
    dq = model(b, compute_force=False)["dq"].detach().cpu().numpy()
order = np.argsort(r); edges = np.arange(0, 25, 1.0)
shell = [{"r_lo": float(lo), "r_hi": float(lo + 1), "n": int(((r > lo) & (r <= lo + 1)).sum()), "sum_dq": float(dq[(r > lo) & (r <= lo + 1)].sum()),
          "max_abs_dq": float(np.abs(dq[(r > lo) & (r <= lo + 1)]).max()) if ((r > lo) & (r <= lo + 1)).any() else 0.0,
          "by_species": {s: float(dq[(r > lo) & (r <= lo + 1) & (syms == s)].sum()) for s in ("Pb", "Cs", "Cl")}} for lo in edges]
res = {"N": len(atoms), "tiling": list(tt), "sigma_s": float(model.sigma_s), "dq_sum": float(dq.sum()), "n_eff": float(1 / (dq ** 2).sum()), "shells": shell,
       "cum": [(float(R), float(dq[r <= R].sum())) for R in np.arange(1, 25, 1.0)], "n_negative": int((dq < 0).sum()), "sum_negative": float(dq[dq < 0].sum()),
       "atoms": [(float(r[i]), syms[i], float(dq[i])) for i in order]}
json.dump(res, open(Path(args.out).expanduser(), "w"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
col = {"Pb": "tab:red", "Cs": "tab:green", "Cl": "tab:blue"}
for s in ("Pb", "Cs", "Cl"):
    m = syms == s; ax[0].scatter(r[m], np.abs(dq[m]), s=9, c=col[s], label=s, alpha=0.7)
    ax[0].scatter(r[m & (dq < 0)], np.abs(dq[m & (dq < 0)]), s=30, facecolors="none", edgecolors="k", linewidths=0.5)
ax[0].set_yscale("log"); ax[0].set_ylim(1e-6, 1); ax[0].set_xlabel("distance from the vacancy (Å)"); ax[0].set_ylabel("|dq| per atom (e)"); ax[0].legend(); ax[0].axvline(12, ls=":", c="k")
ax[0].set_title("per atom (circled = negative dq)")
c = np.array([x["sum_dq"] for x in shell]); mid = edges + 0.5
bot = np.zeros_like(c)
for s in ("Pb", "Cs", "Cl"):
    v = np.array([x["by_species"][s] for x in shell]); ax[1].bar(mid, v, width=0.9, bottom=bot, color=col[s], label=s); bot += v
ax[1].set_yscale("symlog", linthresh=1e-3); ax[1].set_xlabel("shell (Å)"); ax[1].set_ylabel("Σdq in the 1 Å shell (e)"); ax[1].axvline(12, ls=":", c="k"); ax[1].legend(); ax[1].set_title("shell sums")
R = [x[0] for x in res["cum"]]; C = [x[1] for x in res["cum"]]
ax[2].plot(R, C, "o-"); ax[2].axhline(0.9, ls="--", c="gray"); ax[2].axvline(12, ls=":", c="k"); ax[2].set_ylim(0, 1.02); ax[2].set_xlabel("R (Å)"); ax[2].set_ylabel("Σdq(r ≤ R)"); ax[2].set_title(f"cumulative; N_eff {res['n_eff']:.2f}, {res['N']} atoms")
fig.suptitle(f"Vertical hole dq on the embedded +1 structure, {res['N']} atoms (bw5 s0, provisional), sigma_s {model.sigma_s:.4f} eV"); fig.tight_layout()
fig.savefig(Path(args.out).expanduser().with_suffix(".png"), dpi=130)
print("N_eff %.2f, sum %.3f, negative: %d atoms sum %.3f" % (res["n_eff"], res["dq_sum"], res["n_negative"], res["sum_negative"]))
for x in shell: print(f"{x['r_lo']:4.0f}-{x['r_hi']:<3.0f} n={x['n']:4d} sum {x['sum_dq']:+.4f} max|dq| {x['max_abs_dq']:.4f}  Pb {x['by_species']['Pb']:+.4f} Cs {x['by_species']['Cs']:+.4f} Cl {x['by_species']['Cl']:+.4f}")
