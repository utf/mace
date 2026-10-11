"""Plan P Phase 2c, Step 0: geometry diagnostics (no training).

0.1  Dense head eigenvalues (VBM, CBM, gap, absolute eV) on pristine tilings of the LABEL cell (the static
     pristine cell the Gamma-gap term acts on) and of the base-RELAXED cell (the Phase 0 ladder's cell):
     1x1x1 (the 80-atom anchor: the label cell must give the registered 2.400 eV), 2x2x2, 3x3x3 if memory allows.
0.2  The +1 core embedded at 639 atoms: (a) relaxed tiling with the existing core displacements (core.json,
     relaxed by the base-v2 head); (b) label tiling, core displacements mapped by atom index, re-relaxed by this
     checkpoint with the far field fixed (r > R_core); (a') relaxed tiling re-relaxed the same way (diagnostic
     row, registered 2026-10-11). Instruments = the gate's: reference-fill separation and N_eff on the
     neutral-labelled structure; +1-forward dq sums within 3/5/8/12/16 A and participation; Pb-Pb across the
     vacancy.
0.3  Separation and N_eff of every 79-atom +1 training frame against the Pb-Pb distance across the vacancy.
0.4  Sign-resolved dq by shell on the first 79-atom +1 frame, for the reference-fill hole and for the +1 forward.

    python defect-perovskite/p2c_step0.py --model RUN/model_avg.pt [--state RUN/model_avg.pt] --tag w01_bs_s2 \
        --out_dir ~/runs/dscc/p2c/w01_bs_s2 --device cuda --stages 0.1 0.2 0.3 0.4
"""
from __future__ import annotations
import argparse, json, sys, time, traceback
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from ase.optimize import BFGS
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace import tools                                                     # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.precondition import PreconditionConfig, bound_state_precondition   # noqa: E402
from mace.modules.dscc.scf import two_fillings                             # noqa: E402
from mace.modules.dscc.species import S_REF, State, neutral_count          # noqa: E402
from mace.modules.dscc.train import vacancy_centre_full                    # noqa: E402
from mace.tools import torch_geometric as tg                               # noqa: E402
from dscc_embed_ladder import KEY_TOL, embed, relative, vacancy_site        # noqa: E402
from dscc_relax import DSCCCalculator                                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--state", default=None, help="state dict to load into --model (v7 runs)")
ap.add_argument("--tag", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--label_cell", default=str(HERE / "static_pristine_cell.json"))
ap.add_argument("--relaxed_cell", default="/home/alex/runs/dscc/p0/pristine_relaxed_v2.json")
ap.add_argument("--core", default="/home/alex/runs/dscc/p0/ladder3_var_large/core.json")
ap.add_argument("--frames", default=str(HERE / "dataset_pbe" / "train.xyz"))
ap.add_argument("--stages", nargs="+", default=["0.1", "0.2", "0.3", "0.4"])
ap.add_argument("--tilings_01", nargs="+", default=["1,1,1", "2,2,2", "3,3,3"])
ap.add_argument("--r_core", type=float, default=8.0); ap.add_argument("--fmax", type=float, default=0.002); ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--device", default="cuda")
ap.add_argument("--charge_03", type=int, default=1, help="cell_charge of the 79-atom frames read in 0.3 (1 = the plan; 0 = the neutral-vacancy frames for comparison)")
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
OUT = Path(args.out_dir).expanduser(); OUT.mkdir(parents=True, exist_ok=True)
STATUS = OUT / "STATUS"; T0 = time.time()


def status(msg):
    line = f"{time.strftime('%F %T')} [{time.time()-T0:6.0f}s] {msg}"
    print(line, flush=True); open(STATUS, "a").write(line + "\n")


def load_model(device):
    m = torch.load(Path(args.model).expanduser(), weights_only=False, map_location=device)
    if args.state:
        sd = torch.load(Path(args.state).expanduser(), weights_only=False, map_location=device)
        if isinstance(sd, dict):
            m.load_state_dict(sd)
    m = m.to(device).eval(); m.base = m.base.double()
    return m


dev = args.device
model = load_model(dev)
z3 = tools.AtomicNumberTable([17, 55, 82]); cfg = PreconditionConfig(); sigma_s = float(model.sigma_s)
status(f"model {args.model} (state {args.state}) tag {args.tag} device {dev} sigma_s {sigma_s} precondition {cfg}")


def cell_from_json(path):
    doc = json.load(open(Path(path).expanduser()))
    return Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)


label_unit = cell_from_json(args.label_cell); relaxed_unit = cell_from_json(args.relaxed_cell)
UNITS = {"label": label_unit, "relaxed": relaxed_unit}


def batch_of(atoms, counts=(0, 0, 0, 0), charge=0, device=None):
    a = atoms.copy(); a.info.update({"carrier_counts": np.array(counts, float), "cell_charge": int(charge), "config_type": "p2c", "source_dir": "p2c"})
    return next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, model.r_cut), batch_size=1))).to(device or dev).to_dict()


def head_H(atoms):
    """The head's H0 on a neutral-labelled structure (the boundstate path), and the batch."""
    data = batch_of(atoms)
    H, numbers = model.filled_hamiltonians(data)[0]              # Phase 2c: H0, or H0 - diag(W) for arm B_w
    return H, [int(z) for z in numbers], data


def band_edges(atoms):
    H, numbers, _ = head_H(atoms)
    n_up, _ = S_REF.counts(neutral_count(numbers))
    eps = torch.linalg.eigvalsh(H)
    return {"n_atoms": len(atoms), "dim": int(H.shape[-1]), "n_up": int(n_up), "VBM": float(eps[n_up - 1]), "CBM": float(eps[n_up]),
            "gap": float(eps[n_up] - eps[n_up - 1]), "eps_min": float(eps[0]), "eps_max": float(eps[-1]), "eps_mean_occupied": float(eps[:n_up].mean())}


def pb_pb_across_vacancy(atoms):
    pos = torch.tensor(np.array(atoms.get_positions())); cell = torch.tensor(np.array(atoms.get_cell()))
    found = vacancy_centre_full(pos, cell, atoms.get_atomic_numbers())
    if found is None:
        return None, None
    rad, flank = found
    return float(2.0 * rad[flank[0]]), rad.numpy()


def plus1_stats(atoms, r):
    """The +1 forward: dq sums within shells of the vacancy, participation, top sites, E_near/E_far diagnostics."""
    b = batch_of(atoms, (0, 0, 1, 0), 1)
    with torch.no_grad():
        out = model(dict(b), compute_force=False)
    dq = out["dq"].detach().cpu().numpy(); dqn = dq / dq.sum(); part = 1.0 / float((dqn ** 2).sum())
    syms = atoms.get_chemical_symbols(); top = np.argsort(-np.abs(dq))[:4]
    diag = out.get("diagnostics", {})
    row = {"dq_sum": float(dq.sum()), "participation_plus1": part, "within": {str(R): float(dq[r <= R].sum()) for R in (3, 5, 8, 12, 16)},
           "top": [(syms[i], float(r[i]), float(dq[i])) for i in top], "head_energy_eV": float(out["head_energy"][0])}
    for k in ("e_host", "e_near", "e_far", "e_madelung", "q0_sum"):
        if k in diag:
            v = diag[k]; row[k] = float(v[0]) if hasattr(v, "__len__") else float(v)
    return row, dq


def precondition_row(atoms):
    data = batch_of(atoms)
    rec = bound_state_precondition(model, [data], cfg)["records"][0]
    return {"separation_eV": rec["separation"], "n_eff_hole": rec["n_eff"], "passed_C5": rec["passed"]}


def relax_core(atoms, vac, cell, label):
    """BFGS on the atoms within r_core of the ideal vacancy site, the rest fixed; CPU fallback on OOM."""
    global model
    r = np.linalg.norm(relative(np.array(atoms.get_positions()), vac, cell), axis=1)
    a = atoms.copy(); a.set_constraint(FixAtoms(indices=[int(i) for i in np.nonzero(r > args.r_core)[0]]))
    for device in ([dev, "cpu"] if dev != "cpu" else ["cpu"]):
        try:
            if device != dev:
                status(f"{label}: retrying the relaxation on CPU"); model = load_model("cpu")
            a.calc = DSCCCalculator(model, [0, 0, 1, 0], 1, device=device)
            e0 = a.get_potential_energy(); f0 = float(np.abs(a.get_forces()).max()); t = time.time()
            opt = BFGS(a, logfile=str(OUT / f"relax_{label}.log"), maxstep=0.1)
            converged = bool(opt.run(fmax=args.fmax, steps=args.steps))
            e1 = a.get_potential_energy(); f1 = float(np.abs(a.get_forces()).max())
            u = np.array(a.get_positions()) - np.array(atoms.get_positions())
            info = {"device": device, "converged": converged, "steps": int(opt.get_number_of_steps()), "seconds": time.time() - t, "E_start": e0, "E_relaxed": e1,
                    "dE_relax": e1 - e0, "fmax_start": f0, "fmax_final": f1, "n_free": int((r <= args.r_core).sum()), "max_disp_A": float(np.linalg.norm(u, axis=1).max()),
                    "rms_disp_core_A": float(np.sqrt((np.linalg.norm(u[r <= args.r_core], axis=1) ** 2).mean()))}
            if device != dev:
                model = load_model(dev)
            a.set_constraint(); a.calc = None
            return a, info
        except torch.OutOfMemoryError as ex:
            status(f"{label}: out of memory on {device}: {str(ex)[:120]}"); torch.cuda.empty_cache()
            if device == "cpu":
                raise
    raise RuntimeError("unreachable")


results = {"tag": args.tag, "model": args.model, "state": args.state, "sigma_s": sigma_s, "precondition": cfg.__dict__, "r_core": args.r_core}


def save():
    json.dump(results, open(OUT / "step0.json", "w"), indent=1, default=str)


# ---------------------------------------------------------------- 0.1 band edges on pristine tilings
if "0.1" in args.stages:
    status("stage 0.1 start")
    rows = []
    for name, unit in UNITS.items():
        for t in args.tilings_01:
            tt = tuple(int(x) for x in t.split(","))
            atoms = unit.repeat(tt)
            try:
                row = {"cell": name, "tiling": list(tt), **band_edges(atoms)}
            except torch.OutOfMemoryError as ex:
                torch.cuda.empty_cache(); row = {"cell": name, "tiling": list(tt), "n_atoms": len(atoms), "error": "out of memory"}
                status(f"0.1 {name} {t}: out of memory ({str(ex)[:80]})")
            rows.append(row)
            status(f"0.1 {name} {t} N={len(atoms)}: " + (f"VBM {row['VBM']:.4f} CBM {row['CBM']:.4f} gap {row['gap']:.4f} eV (dim {row['dim']}, eps_min {row['eps_min']:.3f})" if "gap" in row else row["error"]))
            results["0.1"] = rows; save()
    status("stage 0.1 done")

# ---------------------------------------------------------------- 0.2 embedded +1 core at 639 atoms
if "0.2" in args.stages:
    status("stage 0.2 start")
    src = json.load(open(Path(args.core).expanduser())); src_t = tuple(src["tiling"])
    vac_r, cell_r = vacancy_site(relaxed_unit, src_t); vac_l, cell_l = vacancy_site(label_unit, src_t)
    rel0 = np.array(src["rel_positions"]); u = np.array(src["displacements"])
    core_disp = {tuple(np.round(rel0[i] / KEY_TOL).astype(np.int64)): u[i] for i in range(len(u))}
    emb_a, moved, missing, inside = embed(relaxed_unit, src_t, core_disp, (vac_r, cell_r), args.r_core)
    status(f"0.2 (a) relaxed tiling {src_t}: {len(emb_a)} atoms, core {moved} moved / {missing} missing of {inside} within {args.r_core} A")
    # the ideal tilings (vacancy removed) in both lattices share the atom ordering: map the core displacements by index
    from mace.modules.dscc.ladder import tiling_ladder
    ideal_r = tiling_ladder(relaxed_unit, [src_t], vacancy_species=17, vacancy_index=0)[0]
    ideal_l = tiling_ladder(label_unit, [src_t], vacancy_species=17, vacancy_index=0)[0]
    disp_by_index = np.array(emb_a.get_positions()) - np.array(ideal_r.get_positions())
    disp_by_index -= np.round(disp_by_index @ np.linalg.inv(cell_r)) @ cell_r
    start_b = ideal_l.copy(); start_b.set_positions(np.array(ideal_l.get_positions()) + disp_by_index)
    structures = [("a_relaxed_tiling_existing_core", emb_a, vac_r, cell_r, False), ("b_label_tiling_core_rerelaxed", start_b, vac_l, cell_l, True),
                  ("a_prime_relaxed_tiling_core_rerelaxed", emb_a, vac_r, cell_r, True)]
    rows = []
    for label, atoms, vac, cell, do_relax in structures:
        row = {"label": label, "n_atoms": len(atoms)}
        try:
            if do_relax:
                atoms, info = relax_core(atoms, vac, cell, label); row["relaxation"] = info
                status(f"0.2 {label}: relaxed {info['steps']} steps on {info['device']}, converged {info['converged']}, fmax {info['fmax_final']:.4f}, dE {info['dE_relax']:+.4f} eV, max disp {info['max_disp_A']:.3f} A ({info['seconds']:.0f} s)")
            r = np.linalg.norm(relative(np.array(atoms.get_positions()), vac, cell), axis=1)
            pbpb, _ = pb_pb_across_vacancy(atoms); row["pb_pb_across_vacancy_A"] = pbpb
            row.update(precondition_row(atoms))
            st, _ = plus1_stats(atoms, r); row.update(st)
            row["positions"] = np.array(atoms.get_positions()).tolist(); row["cell"] = np.array(atoms.get_cell()).tolist(); row["numbers"] = atoms.get_atomic_numbers().tolist()
            status(f"0.2 {label}: separation {row['separation_eV']:.3f} eV, N_eff(hole) {row['n_eff_hole']:.2f}, +1 participation {row['participation_plus1']:.1f}, "
                   f"dq within 3/5/8/12/16 A " + " ".join(f"{row['within'][k]:.3f}" for k in ("3", "5", "8", "12", "16")) + f", Pb-Pb {pbpb}")
        except Exception as ex:
            row["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"; status(f"0.2 {label}: FAILED {row['error']}"); traceback.print_exc()
        rows.append(row); results["0.2"] = rows; save()
    status("stage 0.2 done")

# ---------------------------------------------------------------- 0.3 separation / N_eff vs Pb-Pb on the +1 79-atom frames
if "0.3" in args.stages:
    status("stage 0.3 start")
    frames = [a for a in read(args.frames, index=":") if int(a.info.get("cell_charge", 0)) == args.charge_03 and len(a) == 79]
    rows = []; n_none = 0
    for i, a in enumerate(frames):
        pbpb, _ = pb_pb_across_vacancy(a)
        if pbpb is None:
            n_none += 1
        rec = precondition_row(a)
        rows.append({"index": i, "pb_pb_A": pbpb, **rec})
        if (i + 1) % 100 == 0:
            status(f"0.3 {i+1}/{len(frames)} frames")
    d = np.array([r["pb_pb_A"] for r in rows if r["pb_pb_A"] is not None]); s = np.array([r["separation_eV"] for r in rows if r["pb_pb_A"] is not None]); n = np.array([r["n_eff_hole"] for r in rows if r["pb_pb_A"] is not None])
    bins = np.arange(np.floor(d.min() * 4) / 4, d.max() + 0.25, 0.25); binned = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d >= lo) & (d < hi)
        if m.any():
            binned.append({"pb_pb_lo": float(lo), "pb_pb_hi": float(hi), "n": int(m.sum()), "separation_p50": float(np.median(s[m])), "n_eff_p50": float(np.median(n[m])), "pass_fraction": float(np.mean((s[m] >= cfg.delta_c) & (n[m] <= cfg.n_loc)))})
    core_pbpb = {}
    try:
        for row in results.get("0.2", []):
            core_pbpb[row["label"]] = row.get("pb_pb_across_vacancy_A")
    except Exception:
        pass
    results["0.3"] = {"cell_charge": args.charge_03, "n_frames": len(frames), "n_without_flank_pair": n_none, "pearson_sep_vs_pbpb": float(np.corrcoef(d, s)[0, 1]), "pearson_neff_vs_pbpb": float(np.corrcoef(d, n)[0, 1]),
                      "pb_pb_range": [float(d.min()), float(d.max())], "binned": binned, "core_pb_pb": core_pbpb, "frames": rows}
    save(); status(f"0.3 done: {len(frames)} frames ({n_none} without a flanking pair); Pb-Pb {d.min():.2f}-{d.max():.2f} A; corr(sep, Pb-Pb) {results['0.3']['pearson_sep_vs_pbpb']:+.3f}, corr(N_eff, Pb-Pb) {results['0.3']['pearson_neff_vs_pbpb']:+.3f}")
    for b in binned:
        status(f"0.3 bin {b['pb_pb_lo']:.2f}-{b['pb_pb_hi']:.2f} A: n {b['n']}, separation p50 {b['separation_p50']:.3f} eV, N_eff p50 {b['n_eff_p50']:.2f}, pass {b['pass_fraction']:.2f}")

# ---------------------------------------------------------------- 0.4 sign-resolved dq by shell on one +1 79-atom frame
if "0.4" in args.stages:
    status("stage 0.4 start")
    fr = next(a for a in read(args.frames, index=":") if int(a.info.get("cell_charge", 0)) == 1 and len(a) == 79)
    pbpb, rad = pb_pb_across_vacancy(fr)
    H, numbers, _ = head_H(fr)
    n_ref = neutral_count(numbers); n_up, n_dn = S_REF.counts(n_ref)
    with torch.no_grad():
        hole_dq = two_fillings(H, State(1, -1, 0).counts(n_ref), (n_up, n_dn), sigma_s).dq.detach().cpu().numpy()
    st, plus_dq = plus1_stats(fr, rad)
    shells = [(0, 3), (3, 5), (5, 8), (8, 12), (12, 99)]
    def resolve(dq):
        out = {}
        for lo, hi in shells:
            m = (rad > lo) & (rad <= hi)
            out[f"{lo}-{hi}"] = {"n_atoms": int(m.sum()), "positive": float(dq[m][dq[m] > 0].sum()), "negative": float(dq[m][dq[m] < 0].sum()), "abs": float(np.abs(dq[m]).sum())}
        dqn = dq / dq.sum()
        return {"shells": out, "sum": float(dq.sum()), "sum_abs": float(np.abs(dq).sum()), "n_eff_raw": float(1.0 / (dq ** 2).sum()), "participation_normalised": float(1.0 / (dqn ** 2).sum()),
                "largest": sorted([(fr.get_chemical_symbols()[i], float(rad[i]), float(dq[i])) for i in np.argsort(-np.abs(dq))[:6]], key=lambda x: -abs(x[2]))}
    results["0.4"] = {"frame": "first 79-atom +1 training frame", "pb_pb_A": pbpb, "reference_fill_hole": resolve(hole_dq), "plus1_forward": resolve(plus_dq), "plus1_stats": st}
    save()
    for key in ("reference_fill_hole", "plus1_forward"):
        r_ = results["0.4"][key]
        status(f"0.4 {key}: sum {r_['sum']:+.3f}, sum|dq| {r_['sum_abs']:.3f}, N_eff(raw) {r_['n_eff_raw']:.2f}, participation(normalised) {r_['participation_normalised']:.2f}; by shell (+/-): "
               + "; ".join(f"{k} {v['positive']:+.3f}/{v['negative']:+.3f}" for k, v in r_["shells"].items()))
    status("stage 0.4 done")
status("ALL DONE")
