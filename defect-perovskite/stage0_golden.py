#!/usr/bin/env python3
"""Stage 0 of transition plan v8: the golden reference of v6 behaviour, and its comparison.

WHAT "BIT-IDENTICAL" MEANS HERE, operationally. Section 3 of the plan accepts Stage 0 only
if the `count_fill` dispatch reproduces the v6 forward "in chemical potentials, occupations,
P, band free energy, total energy, forces and stress on fixed pristine, V_Cl0 and V_Cl+
batches". A tolerance would make that a metric; this script makes it a unit test. `capture`
runs the v6 code (the branch point, f8c61eb) on a trained model over fixed batches, checks
that the SAME forward run twice is `torch.equal` on every field -- so the platform floor is
literally zero and a later mismatch cannot be blamed on nondeterminism -- and writes every
tensor out. `compare` runs the current code on the same batches and asserts `torch.equal`
field by field.

WHAT IS CAPTURED, and why the output dict is not enough. The model returns energies, forces
and stress, but not the fills: `mu`, the occupations, and `P` live inside `head_energy_hf`
and `batched_head_energy_hf` and are consumed there. They are recorded by wrapping those two
functions and `find_mu` in the head's module namespace -- a capture-time hook, not a model
change -- so what is written is exactly what the forward used, including the batched
bisection's own bracket (a single-spectrum recomputation would reach a different `mu` in the
last bits). Once the dispatch of section 2.4 exists it returns these itself and `compare`
reads them from the dispatch; the hook is the v6 side of that comparison.

BOTH SOLVER PATHS AND BOTH MODES. A size-uniform batch takes `_batched_solve`, a mixed-size
or single-graph batch takes `_loop_solve`; `training=True` builds the density-response graph
and `training=False` does not. The value must be the same on every path, and the golden
covers all four combinations so a dispatch that is exact on one path and approximate on
another fails here rather than in a training run.

CPU, ONE THREAD, AND THE TORCHSCRIPT OPTIMISER OFF. The first capture found the same forward
run twice differing in the trunk's FORCES -- 7e-9 under the mixed policy, one ulp in float64
-- with every energy and every head internal identical. It was not BLAS threading and not
MKL's kernel choice (both were pinned and it persisted); it was TorchScript's profiling
executor, which re-optimises e3nn's scripted tensor products after the first run and fuses
the backward into a different summation order. Runs 2, 3, 4 agreed with each other and not
with run 1. The graph executor is therefore held at its unoptimised graph for both `capture`
and `compare`, which makes the first run the same as every later one, and `compare` refuses
a different thread count.

    python defect-perovskite/stage0_golden.py capture --model ~/runs/arma_models/arma_s1.model
    python defect-perovskite/stage0_golden.py compare --model ~/runs/arma_models/arma_s1.model

The tensors go to `~/runs/golden/<name>.pt` (tens of MB; not in git) and a manifest of
SHA-256 digests per field to `defect-perovskite/golden/<name>.json` (in git), so the
identity claim is checkable from the repository alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from mace import data as mace_data
from mace import tools
from mace.data.defects import prepare_defect_configurations
from mace.modules import defect_counting, defect_madelung
from mace.modules.defect_context import ForwardContext
from mace.tools import torch_geometric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import KEYSPEC, _assert_repo  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset_cf"
Z_TABLE = tools.AtomicNumberTable([17, 55, 82])
THREADS = 1
GOLDEN_FORMAT = 1

# The output-dict fields that are tensors and carry a number the plan names, plus the
# per-term pieces. Fields that are None on this model (image_compensation, bound_switch,
# hessian, ...) are recorded as absent and must stay absent.
OUTPUT_KEYS = (
    "energy", "base_energy", "delta_energy", "correction_energy", "delta_sr_energy",
    "node_energy", "forces", "base_forces", "delta_forces",
    "virials", "stress", "carrier_alpha", "carrier_eps_mean", "logit_gap", "delta_u",
    "carrier_readouts", "carrier_logits", "defect_features", "trunk_block0",
    "image_compensation", "bound_switch",
    # Stage 1.2: the frontier term and its per-graph diagnostics.
    "base_trunk_energy", "frontier_energy", "frontier_ref_energy", "frontier_w",
    "frontier_p", "frontier_q_F", "frontier_w_ref", "frontier_min_weight",
)
# Stage 1.2 retired the carrier branch: these v6 fields no longer exist on the forward.
RETIRED_KEYS = ("out.delta_lr_ref_energy", "out.latent_charges", "out.latent_charges_host",
                "out.latent_charges_carrier", "out.screening_amplitude")
STAGE12_KEYS = tuple(f"out.{k}" for k in OUTPUT_KEYS[-8:])
# Fields the frontier term changes BY DESIGN on a record with a charged graph: the totals
# and their derivatives. Everything the head computes -- the spectrum, the fills, the
# Madelung shift, the band term, the trunk -- must still be bit-identical.
STAGE12_VALUE_FIELDS = ("out.energy", "out.correction_energy", "out.delta_energy")


# ------------------------------------------------------------------------------ frames

def frame_selection():
    """Fixed frames by (file, index): the first of each class in file order.

    Chosen by composition and counter, never by a label: 80 atoms with zero counter is the
    pristine cell, 79 with zero counter the neutral vacancy (a doublet, `m_s_ref_doubled`
    = 1), 79 and 159 with `(0, 0, 1, 0)` the charged vacancy. Indices are recorded in the
    manifest so the selection is reproducible without this function.
    """
    from ase.io import read

    def pick(path, natoms, charged, n):
        out = []
        for i, a in enumerate(read(str(path), index=":")):
            c = a.info.get("carrier_counts")
            is_charged = c is not None and int(np.asarray(c).sum()) != 0
            if len(a) == natoms and is_charged == charged:
                out.append((str(path.relative_to(HERE)), i, a))
                if len(out) == n:
                    break
        if len(out) != n:
            raise RuntimeError(f"only {len(out)} of {n} frames with {natoms} atoms, "
                               f"charged={charged} in {path}")
        return out

    return dict(
        pristine=pick(DATA / "fold0" / "valid.xyz", 80, False, 2),
        vcl0_79=pick(DATA / "fold0" / "train.xyz", 79, False, 2),
        vcl0_159=pick(DATA / "fold0" / "train.xyz", 159, False, 1),
        qp1_79=pick(DATA / "eval_qp1.xyz", 79, True, 2),
        qp1_159=pick(DATA / "eval_qp1.xyz", 159, True, 1),
    )


def batch_plan(frames):
    """Five batches covering both solver paths, neutral and charged, both sizes."""
    f = {k: [a for _, _, a in v] for k, v in frames.items()}
    return {
        # size-uniform, charged -> _batched_solve with the response built
        "qp1_79_pair": [f["qp1_79"][0], f["qp1_79"][1]],
        # single graph -> _loop_solve, the 636-state matrix
        "qp1_159_single": [f["qp1_159"][0]],
        # mixed sizes -> _loop_solve; pristine, neutral doublet and charged in one batch
        "mixed": [f["pristine"][0], f["vcl0_79"][0], f["qp1_79"][0]],
        # size-uniform, neutral -> _batched_solve with no response (the skip)
        "pristine_pair": [f["pristine"][0], f["pristine"][1]],
        # size-uniform 159, one neutral and one charged graph
        "big_pair": [f["vcl0_159"][0], f["qp1_159"][0]],
    }


def atomic_data(atoms_list, cutoff):
    configs = [mace_data.config_from_atoms(a, key_specification=KEYSPEC) for a in atoms_list]
    prepare_defect_configurations(configs)
    return [mace_data.AtomicData.from_config(c, z_table=Z_TABLE, cutoff=cutoff)
            for c in configs]


def make_batch(atoms_list, cutoff):
    atomic = atomic_data(atoms_list, cutoff)
    loader = torch_geometric.dataloader.DataLoader(atomic, batch_size=len(atomic),
                                                   shuffle=False)
    return next(iter(loader))


def class_table_frames(frames, cutoff):
    """The golden frames as keyed `AtomicData`, PRISTINE FIRST, then the neutral vacancies:
    the order `ensure_class_table` reads the reference geometry of each class from. Every
    driver on arma_s1 builds its table from this list, so their tables agree."""
    from mace.modules.defect_cache import attach_frame_keys

    order = [frames["pristine"][0][2], frames["vcl0_79"][0][2], frames["vcl0_159"][0][2]]
    atomic = atomic_data(order, cutoff)
    attach_frame_keys(atomic, z_table=Z_TABLE)
    return atomic


def ensure_table(model, ctx, frames, log=True):
    from mace.modules.defect_composition import ensure_class_table

    return ensure_class_table(model, class_table_frames(frames, ctx.cutoff), log=log)


# ------------------------------------------------------------------------------ hooks

class HeadRecorder:
    """Records what the v6 head computed inside the fills, by wrapping module globals.

    `find_mu` is patched in `defect_counting`'s namespace, which is where every fill reads
    it from, so each call's output lands here in call order. The two energy routines are
    wrapped to keep their returns: `(energy, lam, psi, P_now, P_ref)` per graph on the loop
    path and `(energy, lam, D, refs)` on the batched one.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.mu = []
        self.loop = []
        self.batched = []
        self.madelung = []

    def __enter__(self):
        self._orig = (defect_counting.find_mu, defect_counting.head_energy_hf,
                      defect_counting.batched_head_energy_hf,
                      defect_madelung.MadelungOnSite.on_site_shift)
        rec = self

        def find_mu(*a, **k):
            out = rec._orig[0](*a, **k)
            rec.mu.append(out.detach().clone())
            return out

        def head_energy_hf(*a, **k):
            out = rec._orig[1](*a, **k)
            rec.loop.append(tuple(t.detach().clone() for t in out))
            return out

        def batched_head_energy_hf(*a, **k):
            out = rec._orig[2](*a, **k)
            e, lam, d, refs = out
            rec.batched.append((e.detach().clone(), lam.detach().clone(),
                                d.detach().clone(), tuple(r.detach().clone() for r in refs)))
            return out

        def on_site_shift(self_, *a, **k):
            out = rec._orig[3](self_, *a, **k)
            rec.madelung.append(out.detach().clone())
            return out

        defect_counting.find_mu = find_mu
        defect_counting.head_energy_hf = head_energy_hf
        defect_counting.batched_head_energy_hf = batched_head_energy_hf
        defect_madelung.MadelungOnSite.on_site_shift = on_site_shift
        return self

    def __exit__(self, *exc):
        (defect_counting.find_mu, defect_counting.head_energy_hf,
         defect_counting.batched_head_energy_hf,
         defect_madelung.MadelungOnSite.on_site_shift) = self._orig

    def flatten(self):
        out = {}
        for i, m in enumerate(self.mu):
            out[f"mu.{i}"] = m
        for i, (e, lam, psi, p_now, p_ref) in enumerate(self.loop):
            out[f"loop.{i}.energy"] = e
            out[f"loop.{i}.lam"] = lam
            out[f"loop.{i}.P"] = p_now
            out[f"loop.{i}.P_ref"] = p_ref
        for i, (e, lam, d, refs) in enumerate(self.batched):
            out[f"batched.{i}.energy"] = e
            out[f"batched.{i}.lam"] = lam
            out[f"batched.{i}.D"] = d
            out[f"batched.{i}.n_maj_ref"] = refs[0]
            out[f"batched.{i}.n_min_ref"] = refs[1]
        for i, m in enumerate(self.madelung):
            out[f"madelung.{i}"] = m
        return out


# ------------------------------------------------------------------------------ forward

MODES = {
    "eval_force_stress": dict(training=False, compute_force=True, compute_stress=True),
    "train_force": dict(training=True, compute_force=True, compute_stress=False),
}


def run_once(model, ctx, batch, mode):
    """One forward under `mode`; returns a flat `{field: tensor | None}` record."""
    rec = HeadRecorder()
    d = ctx.forward_dict(batch, requires_grad=True)
    with rec, torch.enable_grad():
        out = model(d, **MODES[mode])
    flat = {}
    for k in OUTPUT_KEYS:
        v = out.get(k)
        flat[f"out.{k}"] = None if v is None else v.detach().clone()
    flat.update(rec.flatten())
    return flat


def records_equal(a, b):
    """Field-wise `torch.equal`; returns the list of (field, max |diff| or reason)."""
    bad = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if x is None and y is None:
            continue
        if x is None or y is None:
            bad.append((k, "present on one side only"))
            continue
        if tuple(x.shape) != tuple(y.shape) or x.dtype != y.dtype:
            bad.append((k, f"shape/dtype {tuple(x.shape)}/{x.dtype} vs "
                           f"{tuple(y.shape)}/{y.dtype}"))
            continue
        if not torch.equal(x, y):
            diff = (x.double() - y.double()).abs().max()
            bad.append((k, f"max|diff| = {float(diff):.3e}"))
    return bad


def digest(t):
    if t is None:
        return None
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


def git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE.parent,
                                   text=True).strip()


def model_digest(model):
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def setup(args):
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    # The profiling executor's re-optimisation is the run-to-run difference (see the module
    # docstring); the unoptimised graph is the one being pinned.
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._set_graph_executor_optimize(False)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    model = torch.load(args.model, map_location="cpu", weights_only=False).to("cpu").eval()
    ctx = ForwardContext.production(model, device="cpu")
    frames = frame_selection()
    # Stage 1.2: the frontier term needs the class table on every charged graph; a model
    # pickled before the table existed gets it here, from the golden frames.
    if getattr(args, "class_table", True):
        ensure_table(model, ctx, frames, log=False)
    batches = {name: make_batch(atoms, ctx.cutoff) for name, atoms in batch_plan(frames).items()}
    return model, ctx, frames, batches


def capture(args):
    model, ctx, frames, batches = setup(args)
    records = {}
    for bname, batch in batches.items():
        for mode in MODES:
            first = run_once(model, ctx, batch, mode)
            second = run_once(model, ctx, batch, mode)
            bad = records_equal(first, second)
            if bad:
                raise RuntimeError(
                    f"{bname}/{mode}: the same forward run twice is not bit-identical -- "
                    f"{bad[:5]}; fix the capture environment, not the tolerance")
            records[f"{bname}/{mode}"] = first
            print(f"  {bname:16s} {mode:18s} E = "
                  f"{[round(float(x), 6) for x in first['out.energy']]}  "
                  f"({len(first)} fields, repeat identical)")
    meta = dict(
        golden_format=GOLDEN_FORMAT, git_sha=git_sha(), model=str(args.model),
        model_sha256=model_digest(model), threads=THREADS, torch=torch.__version__,
        platform=platform.platform(), cutoff=ctx.cutoff, eps_inf=ctx.eps_inf,
        default_dtype=str(torch.get_default_dtype()),
        precision_policy=getattr(model, "precision_policy", "uniform"),
        frames={k: [(p, i) for p, i, _ in v] for k, v in frames.items()},
        batches={k: [len(a) for a in v] for k, v in batch_plan(frames).items()},
        modes=MODES,
    )
    out_pt = Path(args.out).expanduser()
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(meta=meta, records=records), out_pt)
    manifest = dict(meta=meta, digests={
        name: {k: digest(v) for k, v in rec.items()} for name, rec in records.items()})
    out_json = Path(args.manifest)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    n_fields = sum(len(r) for r in records.values())
    print(f"golden: {len(records)} records, {n_fields} fields -> {out_pt} ({out_json})")


# Fields a derivative fix may change: everything that is a force, a virial, a stress or a
# density response. Under `--energies-only` these are reported but not counted as failures;
# every energy, chemical potential, occupation, spectrum and density matrix must still be
# bit-identical (Stage 1's first two steps change derivatives, not values).
DERIVATIVE_FIELDS = ("out.forces", "out.base_forces", "out.delta_forces", "out.virials",
                     "out.stress")


def _allowed_stage12(name, bad):
    """Stage 1.2's allow-list: retired and new fields may be present on one side only on
    every record; the totals and the derivatives may differ on a record with a charged
    graph; NOTHING may differ on a neutral record."""
    presence = [b for b in bad if b[0] in RETIRED_KEYS or b[0] in STAGE12_KEYS]
    if name.startswith("pristine_pair"):
        return presence
    values = [b for b in bad if b[0] in STAGE12_VALUE_FIELDS or b[0] in DERIVATIVE_FIELDS]
    return presence + values


def compare(args):
    ref = torch.load(Path(args.ref).expanduser(), map_location="cpu", weights_only=False)
    energies_only = bool(getattr(args, "energies_only", False))
    stage12 = bool(getattr(args, "stage12", False))
    meta = ref["meta"]
    model, ctx, frames, batches = setup(args)
    if model_digest(model) != meta["model_sha256"]:
        raise RuntimeError("the model differs from the one the golden was captured with")
    if torch.get_num_threads() != meta["threads"]:
        raise RuntimeError(f"thread count {torch.get_num_threads()} != {meta['threads']}")
    if {k: [(p, i) for p, i, _ in v] for k, v in frames.items()} != {
            k: [tuple(x) for x in v] for k, v in meta["frames"].items()}:
        raise RuntimeError("the frame selection differs from the golden's")
    failures = {}
    for name, golden in ref["records"].items():
        bname, mode = name.split("/")
        now = run_once(model, ctx, batches[bname], mode)
        bad = records_equal(golden, now)
        allowed = [b for b in bad if b[0] in DERIVATIVE_FIELDS] if energies_only else []
        if stage12:
            allowed = _allowed_stage12(name, bad)
        bad = [b for b in bad if b not in allowed]
        status = "identical" if not bad else f"{len(bad)} field(s) differ"
        if allowed:
            status += f" ({len(allowed)} field(s) differ, allowed)"
        print(f"  {name:36s} {status}")
        for k, why in bad + allowed:
            print(f"      {k}: {why}")
        if bad:
            failures[name] = bad
    print(f"golden {meta['git_sha'][:10]} vs HEAD {git_sha()[:10]}: "
          f"{len(ref['records']) - len(failures)}/{len(ref['records'])} records identical"
          + (" (energies only)" if energies_only else "")
          + (" (Stage 1.2 allow-list: head fields and neutral records exact)" if stage12
             else ""))
    return 0 if not failures else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("capture", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--out" if name == "capture" else "--ref",
                       default="~/runs/golden/v6_arma_s1.pt")
        if name == "capture":
            s.add_argument("--manifest", default=str(HERE / "golden" / "v6_arma_s1.json"))
        else:
            s.add_argument("--energies-only", action="store_true",
                           help="derivative fields may differ (Stage 1 derivative fixes)")
            s.add_argument("--stage12", action="store_true",
                           help="Stage 1.2 allow-list: the frontier term changes the totals "
                                "of charged records; head fields and neutral records exact")
    args = p.parse_args(argv)
    _assert_repo()
    return capture(args) if args.cmd == "capture" else compare(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
