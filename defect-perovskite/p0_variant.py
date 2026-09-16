"""Plan P Phase 0.3: the polarisation variant by surgery on a shipped W6 checkpoint, gated
once against W6 on a charged training frame, then timed at the ladder sizes.

  * host_coupling -> polarisation, provider attached, r_split set (both new kernels)
  * gate: analytic route refused; dq, band energy, e_madelung identical to W6 on the same frame
    (only e_host differs); one central-difference force check with the real base and provider
  * timing / peak memory of one energy+force call at the tiling sizes (charged +1)

    python defect-perovskite/p0_variant.py --w6 ~/runs/dscc/dscc_w6_s1/model.pt --r_split 3.6 \
        --cell ~/runs/dscc/p0/pristine_relaxed_v2.json --tilings 2,2,2 2,2,3 --device cuda \
        --out ~/runs/dscc/p0/p0_variant_s1.json --save ~/runs/dscc/p0/w6_s1_polarisation.pt
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.io import read
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mace import tools                                            # noqa: E402
from mace.modules.dscc import data as dd                          # noqa: E402
from mace.modules.dscc.analytic import analytic_refusal, capture_refusal   # noqa: E402
from mace.modules.dscc.field import load_field_provider           # noqa: E402
from mace.modules.dscc.ladder import tiling_ladder                # noqa: E402
from mace.tools import torch_geometric                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--w6", required=True)
ap.add_argument("--ckpt", default="/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model")
ap.add_argument("--head", default="mp-dielectric")
ap.add_argument("--r_split", type=float, required=True)
ap.add_argument("--eps_inf", type=float, default=4.0)
ap.add_argument("--cell", required=True)
ap.add_argument("--frames", default=str(HERE / "dataset_pbe" / "train.xyz"))
ap.add_argument("--tilings", nargs="*", default=["2,2,2"])
ap.add_argument("--device", default="cuda")
ap.add_argument("--fd", type=int, default=2, help="number of FD force components checked")
ap.add_argument("--out", required=True)
ap.add_argument("--save", default="")
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
dev = args.device


def load_w6():
    m = torch.load(args.w6, map_location="cpu", weights_only=False)
    m.base = m.base.double()      # the W6 checkpoints carry a float32 base; Phase 0 evaluates in float64
    return m


def batch_of(atoms, counts, charge, r_cut):
    a = atoms.copy(); a.info.update({"carrier_counts": np.array(counts, dtype=float), "cell_charge": charge, "config_type": "p0", "source_dir": "p0"})
    z = tools.AtomicNumberTable([17, 55, 82])
    return next(iter(torch_geometric.dataloader.DataLoader(dd.atomic_data([a], z, r_cut), batch_size=1))).to(dev).to_dict()


def surgery(m):
    prov = load_field_provider(args.ckpt, head=args.head, device="cpu")
    m.host_coupling = "polarisation"
    m.field_provider = prov
    m.r_split = float(args.r_split)
    from dataclasses import replace
    m.kernel = replace(m.kernel, eps_inf=float(args.eps_inf))
    return m


w6 = load_w6().to(dev).eval()
var = surgery(load_w6()).to(dev).eval()
res = {"w6": args.w6, "r_split": args.r_split, "eps_inf": args.eps_inf, "w6_r_split": float(w6.r_split), "w6_eps_inf": float(w6.kernel.eps_inf),
       "refusals": {"analytic": analytic_refusal(var), "capture": capture_refusal(var)}}
assert res["refusals"]["analytic"] and res["refusals"]["capture"]
res["refusals"]["w6_analytic"] = analytic_refusal(w6)   # W6 as shipped reads block 0: cotangent route too

# ---- gate on one charged training frame (the first +1 frame at 79 atoms)
frames = read(args.frames, index=":")
fr = next(a for a in frames if int(a.info.get("cell_charge", 0)) == 1 and len(a) == 79)
counts = np.asarray(fr.info["carrier_counts"], dtype=float)
b = batch_of(fr, counts, 1, w6.r_cut)
o6 = w6(dict(b), compute_force=True)
ov = var(dict(b), compute_force=True)
d6, dv = o6["diagnostics"], ov["diagnostics"]
gate = {"dq_max_abs_diff": float((o6["dq"] - ov["dq"]).abs().max()), "dq_sum_w6": float(o6["dq"].sum()), "dq_sum_var": float(ov["dq"].sum()),
        "e_madelung_w6": d6["e_madelung"][0], "e_madelung_var": dv["e_madelung"][0],
        "e_host_w6": d6["e_host"][0], "e_cP_var": dv["e_host"][0], "e_pp_var": float(ov["e_pp"][0]),
        "head_energy_w6": float(o6["head_energy"][0]), "head_energy_var": float(ov["head_energy"][0]),
        "band_energy_w6": float(o6["head_energy"][0]) - d6["e_host"][0] - d6["e_madelung"][0],
        "band_energy_var": float(ov["head_energy"][0]) - dv["e_host"][0] - dv["e_madelung"][0],
        "force_rms_diff_meV_A": 1e3 * float(((o6["forces"] - ov["forces"]) ** 2).mean().sqrt()),
        "force_rmse_vs_label_w6": 1e3 * float(((o6["forces"].cpu() - torch.tensor(fr.arrays["REF_forces"])) ** 2).mean().sqrt()),
        "force_rmse_vs_label_var": 1e3 * float(((ov["forces"].cpu() - torch.tensor(fr.arrays["REF_forces"])) ** 2).mean().sqrt())}
gate["band_energy_diff"] = gate["band_energy_var"] - gate["band_energy_w6"]
# FD force on the variant
fd = []
pos0 = fr.get_positions(); h = 1e-3
rng = np.random.default_rng(0)
for k in range(args.fd):
    a_, c_ = int(rng.integers(len(fr))), int(rng.integers(3))
    def e(shift):
        f2 = fr.copy(); p = pos0.copy(); p[a_, c_] += shift; f2.set_positions(p)
        return float(var(batch_of(f2, counts, 1, w6.r_cut), compute_force=False)["energy"][0])
    fdv = -(e(h) - e(-h)) / (2 * h)
    fd.append({"atom": a_, "comp": c_, "fd": fdv, "analytic": float(ov["forces"][a_, c_]), "diff_meV_A": 1e3 * abs(fdv - float(ov["forces"][a_, c_]))})
gate["fd"] = fd
res["gate"] = gate
print("gate:", json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in gate.items() if k != "fd"}), flush=True)
print("fd:", fd, flush=True)
if args.save:
    torch.save(var.cpu(), args.save); var.to(dev)
    res["saved"] = args.save

# ---- timing at the ladder sizes
doc = json.load(open(Path(args.cell).expanduser()))
unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
timing = []
for t in args.tilings:
    tt = tuple(int(x) for x in t.split(","))
    cell = tiling_ladder(unit, [tt], vacancy_species=17, vacancy_index=0)[0]
    for name, m in (("w6", w6), ("polarisation", var)):
        row = {"tiling": t, "n_atoms": len(cell), "model": name, "device": dev}
        try:
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t0 = time.time()
            out = m(batch_of(cell, [0, 0, 1, 0], 1, m.r_cut), compute_force=True)
            if dev == "cuda":
                torch.cuda.synchronize()
            row["seconds"] = time.time() - t0
            row["energy"] = float(out["energy"][0]); row["fmax"] = float(out["forces"].abs().max())
            if dev == "cuda":
                row["peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
            del out
        except torch.OutOfMemoryError as ex:
            row["error"] = "OOM: " + str(ex)[:200]
        if dev == "cuda":
            torch.cuda.empty_cache()
        timing.append(row)
        print("timing:", row, flush=True)
res["timing"] = timing
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1)
print("saved", args.out)
