"""Plan section 8: the 2x benchmark. Wall time per MD-style step (energy + forces) of the
D-SCC model against the base alone, on the named hardware, fixed batch size, fixed
`tol_q`/`tol_E`, warm-started along a registered trajectory (consecutive frames of one MD
run, `dq` of the previous frame as the start), the base's early feature block recomputed
for derivatives. Reports median and 95th percentile of the ratio; both must be <= 2.
    python defect-perovskite/dscc_benchmark.py --model /home/alex/runs/dscc/<run>/model_dscc.pt --device cuda --frames 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mace import tools                                            # noqa: E402
from mace.modules.dscc import data as dd                          # noqa: E402
from mace.modules.models import ScaleShiftMACE                    # noqa: E402
from mace.tools import torch_geometric                            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset", default=str(HERE / "dataset_pbe"))
    ap.add_argument("--frames", type=int, default=40, help="consecutive charged frames of one source (the trajectory)")
    ap.add_argument("--size", type=int, default=79)
    ap.add_argument("--out", default="")
    ap.add_argument("--predictor", type=int, default=0, help="v5 W2: warm start from the secant of the last two fixed points (2 dq_n - dq_{n-1})")
    ap.add_argument("--no_analytic", action="store_true", help="O3: fall back to the autograd cotangent route for the head force (the analytic kernel's reference)")
    ap.add_argument("--fixed_cell", action="store_true", help="build the trajectory from the FIRST frame by small thermal displacements at its own cell -- what a relaxation or NVT/NVE step looks like, and the regime CUDA-graph capture applies to (the registered 79-atom frames are NPT snapshots whose cells differ frame to frame)")
    ap.add_argument("--graphed", action="store_true", help="O3: run the model leg through CUDA-graph capture (and time a captured base leg alongside the eager one)")
    ap.add_argument("--both", action="store_true", help="also time the other force route, interleaved frame by frame")
    ap.add_argument("--check", action="store_true", help="compare the two force routes on the first frame before timing")
    ap.add_argument("--tol_q", type=float, default=None, help="override the solver's inference tolerance (the registered value is used when unset)")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    import ase.io
    frames = [a for a in ase.io.read(f"{args.dataset}/train.xyz", ":") if int(a.info["cell_charge"]) != 0 and len(a) == args.size]
    frames = frames[: args.frames]
    if args.fixed_cell:
        seed0 = frames[0]
        frames = []
        for i in range(args.frames):
            a = seed0.copy()
            a.info.update(seed0.info)
            if i:
                a.rattle(stdev=0.02, seed=1000 + i)
            frames.append(a)
    model = torch.load(args.model, weights_only=False, map_location=args.device).to(args.device).eval()
    z_table = tools.AtomicNumberTable(model.atomic_numbers)
    base = model.base

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    def batch(atoms, cutoff):
        d = dd.atomic_data([atoms], z_table, cutoff)
        return next(iter(torch_geometric.dataloader.DataLoader(d, batch_size=1))).to(args.device).to_dict()

    # W1 (v5) put the frozen base in float32 with the head in float64. The base-alone leg is
    # timed IN THE BASE'S OWN PRECISION, cast outside the timed region: that is what a
    # base-only MD step costs, and putting the cast inside would charge the denominator for
    # work the comparison does not do. The model leg pays its own casts, which is correct --
    # they are part of what the head costs.
    base_dtype = model.base_dtype()

    def base_batch(atoms):
        return model._cast_floats(batch(atoms, float(base.r_max)), base_dtype)

    if args.check:
        b = batch(frames[0], model.r_cut)
        model.analytic_head_forces = True
        fast = model(dict(b), compute_force=True)["forces"].detach()
        fast2 = model(dict(b), compute_force=True)["forces"].detach()
        model.analytic_head_forces = False
        ref = model(dict(b), compute_force=True)["forces"].detach()
        # The floor is the BASE: its scatters are non-deterministic on CUDA, so two identical
        # calls differ by ~1e-6 eV/A. Quoting the route difference without it would be
        # attributing the base's noise to the kernel.
        print(f"# force routes: max|analytic - autograd| = {float((fast - ref).abs().max()):.3e} "
              f"eV/A, same-route repeat (the floor) {float((fast - fast2).abs().max()):.3e}, "
              f"on |F|max = {float(ref.abs().max()):.3f}", file=sys.stderr)
    model.analytic_head_forces = not args.no_analytic
    # warm-up. THREE frames, not one: the O3 head compiles its assembly and its analytic
    # force kernel on first use, and consecutive frames have different edge counts -- with one
    # warm-up frame the second frame would pay a recompile inside the timed region.
    for w in frames[:3]:
        model(batch(w, model.r_cut), compute_force=True)
        if args.both:
            model.analytic_head_forces = not model.analytic_head_forces
            model(batch(w, model.r_cut), compute_force=True)
            model.analytic_head_forces = not model.analytic_head_forces
    ScaleShiftMACE.forward(base, base_batch(frames[0]), compute_force=True); sync()
    if args.tol_q is not None:
        import dataclasses
        model.scf_options = dataclasses.replace(model.scf_options, tol_q=args.tol_q, tol_q_inference=args.tol_q)
    graphed = None
    t_graph_base = []              # --graphed: the base leg captured, the second denominator
    if args.graphed:
        from mace.modules.dscc.capture import GraphedStep

        def capture(atoms):
            return GraphedStep(model, batch(atoms, model.r_cut))

        from mace.modules.dscc.capture import edge_bucket, pad_batch

        class _CapturedBase:
            """The base alone, captured, so the ratio can be quoted against BOTH denominators.
            The registered criterion was written against an eager base, which is launch-bound;
            graphing the model makes it faster and the ratio harder, so the absolute
            milliseconds are the number that carries across versions.

            Its edge list is bucketed the same way the model's is. Without that, a frame whose
            base graph has a different edge count would silently replay the CAPTURED frame's
            neighbours -- a timing leg that measures the wrong thing and says nothing."""

            def __init__(self, bb):
                self.capacity = edge_bucket(int(bb["edge_index"].shape[1]))
                self.data = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                             for k, v in pad_batch(dict(bb), self.capacity, float(base.r_max)).items()}
                self.data["positions"].requires_grad_(True)

                st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    for _ in range(3):
                        o = ScaleShiftMACE.forward(base, dict(self.data), compute_force=True)
                        del o
                torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
                self.g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.g):
                    self.out = ScaleShiftMACE.forward(base, dict(self.data), compute_force=True)
                torch.cuda.synchronize()

            def fits(self, bb):
                return int(bb["edge_index"].shape[1]) <= self.capacity

            def __call__(self, bb):
                padded = pad_batch(dict(bb), self.capacity, float(base.r_max))
                with torch.no_grad():
                    for k in ("positions", "cell", "shifts", "unit_shifts", "edge_index"):
                        self.data[k].copy_(padded[k])
                self.g.replay()
                return self.out

        captured_base = None
    t_alt = []                     # --both: the other force route, interleaved frame by frame
    t_head, t_base, iters = [], [], []
    warm = None
    history = []
    for atoms in frames:
        bb = base_batch(atoms)
        sync(); t0 = time.perf_counter()
        ScaleShiftMACE.forward(base, bb, compute_force=True); sync()
        t_base.append(time.perf_counter() - t0)
        bh = batch(atoms, model.r_cut)
        if args.graphed:
            if graphed is None or not graphed.fits(bh):
                graphed = capture(atoms)                              # a new bucket: one re-capture
            sync(); t0 = time.perf_counter()
            out = graphed(bh)
            sync(); t_head.append(time.perf_counter() - t0)
            if captured_base is None or not captured_base.fits(bb):
                captured_base = _CapturedBase(bb)
            sync(); t0 = time.perf_counter()
            captured_base(bb)
            sync(); t_graph_base.append(time.perf_counter() - t0)
        else:
            sync(); t0 = time.perf_counter()
            out = model(bh, compute_force=True, warm_start=warm)      # initialisation iii along the trajectory
            sync(); t_head.append(time.perf_counter() - t0)
        if args.both:
            # Interleaved in the SAME loop: the GPU's clocks drift between runs by ~15 %, so
            # two separate invocations cannot be compared frame for frame.
            model.analytic_head_forces = not model.analytic_head_forces
            bh2 = batch(atoms, model.r_cut)
            sync(); t0 = time.perf_counter()
            model(bh2, compute_force=True, warm_start=warm)
            sync(); t_alt.append(time.perf_counter() - t0)
            model.analytic_head_forces = not model.analytic_head_forces
        history.append(out["dq"].detach()); history = history[-2:]
        # v5 W2 closing item: the tangent along the trajectory as the secant of the last two
        # fixed points (the geometry response is not formed); plain warm start otherwise.
        warm = [2 * history[-1] - history[-2]] if (args.predictor and len(history) == 2 and history[-1].shape == history[-2].shape) else [history[-1]]
        iters.append(out.get("diagnostics", {}).get("iterations", [0])[0] if "diagnostics" in out else 0)
    ratio = np.array(t_head) / np.array(t_base)
    report = {"model": args.model, "device": args.device, "hardware": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
              "n_frames": len(frames), "size": args.size, "coupling": bool(model.coupling), "route_b": bool(model.route_b),
              "base_dtype": str(base_dtype), "r_max_base": float(base.r_max), "r_cut_head": float(model.r_cut),
              "base_ms_median": 1000 * float(np.median(t_base)), "model_ms_median": 1000 * float(np.median(t_head)),
              "ratio_median": float(np.median(ratio)), "ratio_p95": float(np.percentile(ratio, 95)),
              "scf_iterations_median": float(np.median(iters)) if iters else None, "predictor": bool(args.predictor), "tol_q": float(model.scf_options.tol_q_inference if args.tol_q is None else args.tol_q),
              "graphed": bool(args.graphed), "fixed_cell": bool(args.fixed_cell),
              "graphed_base_ms_median": (1000 * float(np.median(t_graph_base))) if t_graph_base else None,
              "ratio_median_vs_graphed_base": (float(np.median(t_head)) / float(np.median(t_graph_base))) if t_graph_base else None,
              "alt_route_ms_median": (1000 * float(np.median(t_alt))) if t_alt else None,
              "analytic_head_forces": bool(getattr(model, "analytic_head_forces", False)),
              "passes_2x": bool(np.median(ratio) <= 2.0 and np.percentile(ratio, 95) <= 2.0),
              "note_fixed_cell": ("NOT the registered P4.3 trajectory: the frames are the first one "
                                  "displaced by 0.02 A, at its own cell, which is what a relaxation or an "
                                  "NVT/NVE step is and the regime CUDA-graph capture applies to. Quote this "
                                  "number with that qualifier.") if args.fixed_cell else None,
              "note": "the model's time includes the base's full forward (the head needs block 0 attached; E_base is not cached at inference)"}
    print(json.dumps(report, indent=1))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
