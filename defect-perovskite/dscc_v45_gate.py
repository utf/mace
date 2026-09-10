"""v4.5 gate before the restart (amendment, last bullet): on real charged frames of both
sizes the v4.5 solver (Levenberg-Marquardt damping, tangent predictor, warm starts) must
agree with the previous implementation (backtracking Newton, no predictor) on the fixed
points, the band identity, the forces (finite differences) and the implicit parameter
gradients, to tolerance; iteration counts and step times are recorded.
    python defect-perovskite/dscc_v45_gate.py --device cuda --out ~/runs/dscc/v45_gate.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mace.modules.dscc.model import MACEDSCC
from mace.modules.dscc.kernels import KernelConfig
from mace.modules.dscc.scf import ScfOptions
from mace.modules.dscc.train import load_frames, to_device
from mace.modules.dscc import data as dd
from mace.tools import torch_geometric as tg, AtomicNumberTable
torch.set_default_dtype(torch.float64)

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda")
ap.add_argument("--base", default="/home/alex/runs/aprime_prod/aprime_prod_base.pt")
ap.add_argument("--h0", default="/home/alex/runs/dscc/dscc_arm1_full_s0/h0_state.pt")
ap.add_argument("--out", default="/home/alex/runs/dscc/v45_gate.json")
ap.add_argument("--n79", type=int, default=8)
ap.add_argument("--n159", type=int, default=4)
ap.add_argument("--lambdas", default="0.05,0.5")
args = ap.parse_args()
DEV = args.device
OLD = dict(damping="backtrack", predictor=False)     # the pre-v4.5 solver
NEW = dict()                                          # the v4.5 defaults (deferred damped Newton, predictor off)
VARIANTS = (("new_pred", dict(damping="newton", predictor=True)), ("new_lm", dict(damping="lm", predictor=False)))

base = torch.load(args.base, weights_only=False, map_location="cpu").double()
model = MACEDSCC(base, r_cut=10.0, directional=True, coupling=True, route_b=False, kernel=KernelConfig(regime="B"),
                 scf=ScfOptions(n_max=100)).to(DEV)
model.load_h0_from(args.h0); model.set_coupling_mode("full")
frames = load_frames(f"{HERE}/dataset_pbe/train.xyz", f"{HERE}/dataset_pbe/valid.xyz")
metas = [dd.frame_meta(i, a, "CsPbCl3", pristine_atoms=80) for i, a in enumerate(frames)]
pristine = [m.index for m in metas if m.n_atoms == 80 and m.state.Q == 0][:16]
z = AtomicNumberTable(model.atomic_numbers)
model.set_pristine_reference([to_device(b, DEV) for b in tg.dataloader.DataLoader(dd.atomic_data([frames[i] for i in pristine], z, 10.0), batch_size=16)])
sets = {79: [m.index for m in metas if m.state.Q != 0 and m.n_atoms == 79][:args.n79],
        159: [m.index for m in metas if m.state.Q != 0 and m.n_atoms == 159][:args.n159]}
params = [p for p in model.parameters() if p.requires_grad]


def batches_of(idx, bs=4):
    ds = dd.atomic_data([frames[i] for i in idx], z, 10.0)
    return [to_device(b, DEV) for b in tg.dataloader.DataLoader(ds, batch_size=bs)]


def set_options(**kw):
    model.scf_options = ScfOptions(n_max=100, **kw)


def run(batch, training=False, warm=None):
    model.train(training); torch.cuda.synchronize() if DEV.startswith("cuda") else None
    t0 = time.time()
    if training:
        out = model(dict(batch), training=True, compute_force=True, warm_start=warm)
    else:
        with torch.no_grad():
            out = model(dict(batch), compute_force=True, warm_start=warm)
    torch.cuda.synchronize() if DEV.startswith("cuda") else None
    out["_time"] = time.time() - t0
    return out


def split(t, ptr):
    return [t[int(ptr[g]):int(ptr[g + 1])] for g in range(int(ptr.numel() - 1))]


def set_lambda(lam):
    with torch.no_grad():
        model.lambda_raw.fill_(torch.logit(torch.tensor(lam / model.lambda_max)))


report = {"lambdas": {}, "tolerances": {"dq": 1e-7, "energy_eV": 1e-8, "force_eV_A": 1e-7, "grad_rel": 1e-6,
                                        "band_identity_eV": 1e-9, "fd_force_eV_A": 1e-5, "tol_root": ScfOptions().tol_root}}
ok_all = True
for lam in [float(x) for x in args.lambdas.split(",")]:
    set_lambda(lam)
    rep = {}
    for nat, idx in sets.items():
        r = {"frames": idx, "old": {}, "new": {}}
        stored = {}
        # (a) inference fixed points, energies, forces, band identity, iterations, time
        for name, _ in VARIANTS:
            r[name] = {}
        for name, kw in (("old", OLD), ("new", NEW)) + VARIANTS:
            set_options(**kw)
            outs = [run(b) for b in batches_of(idx)]
            r[name]["iterations"] = sum((o["diagnostics"]["iterations"] for o in outs), [])
            r[name]["fills"] = sum((o["diagnostics"].get("fills", []) for o in outs), [])
            r[name]["converged"] = sum((o["diagnostics"]["converged"] for o in outs), [])
            r[name]["commutator_max"] = max(sum((o["diagnostics"]["commutator"] for o in outs), []))
            r[name]["band_minus_primary_max"] = max(abs(x) for x in sum((o["diagnostics"]["band_minus_primary"] for o in outs), []))
            r[name]["time_per_batch"] = float(np.mean([o["_time"] for o in outs]))
            r[name]["energy"] = torch.cat([o["energy"].detach().cpu() for o in outs])
            r[name]["forces"] = torch.cat([o["forces"].detach().cpu() for o in outs])
            r[name]["dq"] = torch.cat([o["dq"].detach().cpu() for o in outs])
            if name == "new":
                for b, o in zip(batches_of(idx), outs):
                    pass
        r["dq_diff"] = float((r["new"]["dq"] - r["old"]["dq"]).abs().max())
        for name, _ in VARIANTS:
            r[name]["dq_diff_vs_old"] = float((r[name]["dq"] - r["old"]["dq"]).abs().max())
        r["energy_diff"] = float((r["new"]["energy"] - r["old"]["energy"]).abs().max())
        r["force_diff"] = float((r["new"]["forces"] - r["old"]["forces"]).abs().max())
        # (b) implicit parameter gradients of a force loss, old vs new
        grads = {}
        for name, kw in (("old", OLD), ("new", NEW)):
            set_options(**kw)
            g_sum = None
            for b in batches_of(idx):
                o = run(b, training=True)
                loss = ((o["forces"] - b["forces"]) ** 2).sum()
                g = torch.autograd.grad(loss, params, allow_unused=True)
                g = [torch.zeros_like(p) if gi is None else gi.detach() for p, gi in zip(params, g)]
                g_sum = g if g_sum is None else [a + c for a, c in zip(g_sum, g)]
            grads[name] = g_sum
        r["grad_rel_diff"] = max(float((a - c).abs().max() / max(float(c.abs().max()), 1e-12)) for a, c in zip(grads["new"], grads["old"]))
        r["grad_norm"] = float(sum(float(c.norm() ** 2) for c in grads["old"]) ** 0.5)
        # (c) warm start after a parameter move (v4.5 later visits): the stored dq of the NEW
        # continuation at these parameters, then lambda and U moved by 2 %, H0 by 1e-3 relative
        set_options(**NEW)
        outs = [run(b) for b in batches_of(idx)]
        stored = sum((split(o["dq"].detach(), b["ptr"]) for b, o in zip(batches_of(idx), outs)), [])
        saved = [p.detach().clone() for p in model.parameters()]
        with torch.no_grad():
            for p in model.parameters():
                if p is model.lambda_raw or p is model.u_raw:
                    p.mul_(1.02)
                elif p.requires_grad:
                    p.add_(1e-3 * p.abs().mean() * torch.randn_like(p))
        outs_c = [run(b) for b in batches_of(idx)]
        outs_w = [run(b, warm=[w for w in stored[k * 4:(k + 1) * 4]]) for k, b in enumerate(batches_of(idx))]
        set_options(**OLD)
        outs_wo = [run(b, warm=[w for w in stored[k * 4:(k + 1) * 4]]) for k, b in enumerate(batches_of(idx))]
        outs_co = [run(b) for b in batches_of(idx)]
        set_options(**NEW)
        dq_c = torch.cat([o["dq"].cpu() for o in outs_c]); dq_w = torch.cat([o["dq"].cpu() for o in outs_w])
        dq_wo = torch.cat([o["dq"].cpu() for o in outs_wo]); dq_co = torch.cat([o["dq"].cpu() for o in outs_co])
        r["warm"] = {"dq_diff_vs_continuation": float((dq_w - dq_c).abs().max()),
                     "old_solver_warm_vs_new_warm": float((dq_wo - dq_w).abs().max()),
                     "old_solver_cont_vs_new_cont": float((dq_co - dq_c).abs().max()),
                     "old_solver_warm_vs_old_cont": float((dq_wo - dq_co).abs().max()),
                     "old_iterations": sum((o["diagnostics"]["iterations"] for o in outs_wo), []),
                     "old_continuation_iterations": sum((o["diagnostics"]["iterations"] for o in outs_co), []),
                     "force_diff": float((torch.cat([o["forces"].cpu() for o in outs_w]) - torch.cat([o["forces"].cpu() for o in outs_c])).abs().max()),
                     "iterations": sum((o["diagnostics"]["iterations"] for o in outs_w), []),
                     "continuation_iterations": sum((o["diagnostics"]["iterations"] for o in outs_c), []),
                     "converged": sum((o["diagnostics"]["converged"] for o in outs_w), []),
                     "warm_started": [o["diagnostics"].get("warm_started") for o in outs_w],
                     "time_per_batch": float(np.mean([o["_time"] for o in outs_w])),
                     "continuation_time_per_batch": float(np.mean([o["_time"] for o in outs_c]))}
        # warm-started TRAINING step (implicit gradient at the warm fixed point) vs continuation
        g_c = g_w = None
        for k, b in enumerate(batches_of(idx)):
            for which, warm in (("c", None), ("w", stored[k * 4:(k + 1) * 4])):
                o = run(b, training=True, warm=warm)
                g = torch.autograd.grad(((o["forces"] - b["forces"]) ** 2).sum(), params, allow_unused=True)
                g = [torch.zeros_like(p) if gi is None else gi.detach() for p, gi in zip(params, g)]
                if which == "c":
                    g_c = g if g_c is None else [a + c for a, c in zip(g_c, g)]
                else:
                    g_w = g if g_w is None else [a + c for a, c in zip(g_w, g)]
        r["warm"]["grad_rel_diff"] = max(float((a - c).abs().max() / max(float(c.abs().max()), 1e-12)) for a, c in zip(g_w, g_c))
        with torch.no_grad():
            for p, q in zip(model.parameters(), saved):
                p.copy_(q)
        # (d) per-graph path (a mixed-size batch) vs the batched path, NEW
        if nat == 79 and sets[159]:
            mixed = batches_of([idx[0], sets[159][0]], bs=2)[0]
            o = run(mixed)
            assert o["diagnostics"].get("batched", False) is False
            d79 = o["dq"][:79].cpu(); r["per_graph_vs_batched_dq"] = float((d79 - r["new"]["dq"][:79]).abs().max())
        for key in ("energy", "forces", "dq"):
            for name in ("old", "new") + tuple(n for n, _ in VARIANTS):
                r[name].pop(key)
        rep[nat] = r
        ok = (r["dq_diff"] < 1e-7 and r["energy_diff"] < 1e-8 and r["force_diff"] < 1e-7 and r["grad_rel_diff"] < 1e-6
              and r["new"]["band_minus_primary_max"] < 1e-9 and all(r["new"]["converged"])
              and r["warm"]["dq_diff_vs_continuation"] < ScfOptions().tol_root and all(r["warm"]["converged"])
              and r["warm"]["grad_rel_diff"] < 1e-6 and r["new"]["commutator_max"] < ScfOptions().tol_c)
        r["pass"] = bool(ok); ok_all = ok_all and ok
        print(f"lambda {lam} nat {nat}: dq {r['dq_diff']:.1e} E {r['energy_diff']:.1e} F {r['force_diff']:.1e} grad {r['grad_rel_diff']:.1e} "
              f"band {r['new']['band_minus_primary_max']:.1e} comm {r['new']['commutator_max']:.1e} | iters old {r['old']['iterations']} new {r['new']['iterations']} (fills {r['new']['fills']}) " + " ".join(f"{n} {r[n]['iterations']} (fills {r[n]['fills']}, conv {all(r[n]['converged'])}, dq {r[n]['dq_diff_vs_old']:.1e})" for n, _ in VARIANTS) + " "
              f"warm {r['warm']['iterations']} (cont {r['warm']['continuation_iterations']}) | time/batch old {r['old']['time_per_batch']:.1f} new {r['new']['time_per_batch']:.1f} warm {r['warm']['time_per_batch']:.1f} s "
              f"| warm dq {r['warm']['dq_diff_vs_continuation']:.1e} grad {r['warm']['grad_rel_diff']:.1e} "
              f"(old solver: warm-vs-new-warm {r['warm']['old_solver_warm_vs_new_warm']:.1e}, cont-vs-new-cont {r['warm']['old_solver_cont_vs_new_cont']:.1e}, "
              f"warm-vs-cont {r['warm']['old_solver_warm_vs_old_cont']:.1e}, iters {r['warm']['old_iterations']} cont {r['warm']['old_continuation_iterations']}) | {'PASS' if ok else 'FAIL'}", flush=True)
    # (e) finite-difference force spot check with the NEW solver, two 79-atom frames, one atom each
    set_options(**NEW)
    fd = []
    for i in sets[79][:2]:
        b = batches_of([i], bs=1)[0]
        o = run(b)
        j = 12; h = 2e-3
        for c in range(2):
            e = []
            for sgn in (+1, -1):
                bb = dict(b); pos = b["positions"].clone(); pos[j, c] += sgn * h; bb["positions"] = pos
                with torch.no_grad():
                    e.append(float(model(bb, compute_force=False)["energy"][0]))
            fd.append({"frame": int(i), "atom": j, "dir": c, "fd": -(e[0] - e[1]) / (2 * h), "analytic": float(o["forces"][j, c])})
    fd_err = max(abs(x["fd"] - x["analytic"]) for x in fd)
    rep["fd_forces"] = {"cases": fd, "max_abs_err": fd_err}
    ok_all = ok_all and fd_err < 1e-5
    print(f"lambda {lam} FD forces: max |FD - analytic| {fd_err:.2e} eV/A", flush=True)
    report["lambdas"][str(lam)] = rep
report["pass"] = bool(ok_all)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(report, open(args.out, "w"), indent=1, default=str)
print("GATE", "PASS" if ok_all else "FAIL", "->", args.out)
