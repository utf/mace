#!/usr/bin/env python
"""Addendum section 3.1 acceptance: the frozen-pristine spectral gauge changes nothing but a
constant. The SAME code and model are run on the golden frames twice -- A: no gauge reference
registered (`H_fix = Htilde`), B: the gauge registered on the pristine reference and the class
table re-aligned under it (`H_fix = Htilde - mu_g I`) -- and the records are compared field by
field:

  * every field of a record with no charged graph must agree to the solver's floor;
  * the spectrum (`lam`), the chemical potentials (`mu`) and the carrier levels shift by
    exactly -mu_g; occupations, densities and forces agree to the floor;
  * the total, correction and delta energies of a charged graph -- and the band energies
    the fills return -- move by -mu_g times the electron excess of its state (levels shift
    by -mu_g, a hole state has one electron fewer than the reference, so its energy moves by
    +mu_g relative to it), and by nothing on a neutral graph;
  * the frontier term (a density functional) and the base branch are unchanged.

The golden's own machinery (frames, batches, modes, the head recorder) is reused, so this is
the golden compare's same instrument pointed at one change. Output: a JSON with every field's
max |diff|, the per-graph energy shifts against the prediction, and the verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0_golden as G  # noqa: E402

FLOOR = 1e-9          # eV/A and dimensionless: two eigensolves of a diagonally shifted matrix
ENERGY_FLOOR = 1e-8   # eV: a band sum over ~600 levels of ~1e4 eV, relative 1e-12
SHIFT_FIELDS = ("batched.{i}.lam", "loop.{i}.lam", "mu.{i}")
# levels reported per graph or per pristine graph: each entry is shifted by -mu_g or, where
# the field is defined on no graph of the record (delta_u off pristine cells), unchanged
LEVEL_FIELDS = ("out.carrier_eps_mean", "out.carrier_logits", "out.carrier_readouts")
# delta_u = sum_i alpha_i eps_i - mean(eps): invariant where alpha sums to one (a carrier
# graph), and -mean(eps) on a graph without a carrier, where it moves by +mu_g
SIGNED_LEVEL_FIELDS = {"out.delta_u": (0.0, "+mu")}
ENERGY_FIELDS = ("out.energy", "out.correction_energy", "out.delta_energy",
                 "out.delta_sr_energy")
PER_GRAPH_ENERGY = ("batched.{i}.energy", "loop.{i}.energy")
INVARIANT_ENERGY = ("out.frontier_energy", "out.frontier_ref_energy", "out.base_energy",
                    "out.base_trunk_energy")


def _align_madelung(ra, rb):
    """The gauged forward computes the pristine cell's Madelung shift before the batch's,
    so the recorder's call index is one ahead in B; drop that call and re-index."""
    ka = sorted(k for k in ra if k.startswith("madelung."))
    kb = sorted(k for k in rb if k.startswith("madelung."))
    if len(kb) == len(ka) + 1:
        vals = [rb.pop(k) for k in kb]
        for k, v in zip(ka, vals[1:]):
            rb[k] = v


def _records(args, gauge):
    model, ctx, frames, batches = G.setup(argparse.Namespace(model=args.model,
                                                             class_table=False))
    G.ensure_table(model, ctx, frames, log=False, gauge=gauge)
    mu = None
    if gauge:
        mu = float(model.gauge_record["mu_g"]) if isinstance(model.gauge_record, dict) \
            else float(model.gauge_record.mu_g)
    recs = {}
    for bname, batch in batches.items():
        for mode in G.MODES:
            out = G.run_once(model, ctx, batch, mode)
            out["_excess"] = _electron_excess(batch)
            recs[f"{bname}/{mode}"] = out
    return recs, mu


def _electron_excess(batch):
    """Electrons relative to the reference state per graph: counts are (e_maj, e_min,
    h_maj, h_min) in the data convention used by the strata (`defect_objective._q_formal`
    reads Q = holes - electrons), so the excess is electrons - holes = -Q."""
    c = batch.carrier_counts.view(int(batch.num_graphs), -1).double()
    return (c[:, 0] + c[:, 1] - c[:, 2] - c[:, 3])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--out", default=str(G.HERE / "golden" / "stage0_gauge_ab.json"))
    args = p.parse_args(argv)
    G._assert_repo()
    a, _ = _records(args, gauge=False)
    b, mu = _records(args, gauge=True)
    print(f"mu_g = {mu:+.6f} eV")
    report, failures = {}, []
    for name in a:
        ra, rb = a[name], b[name]
        excess = ra.pop("_excess")
        rb.pop("_excess")
        _align_madelung(ra, rb)
        charged = bool((excess != 0).any())
        fields = {}
        for k in sorted(set(ra) | set(rb)):
            x, y = ra.get(k), rb.get(k)
            if x is None and y is None:
                continue
            if x is None or y is None or tuple(x.shape) != tuple(y.shape):
                fields[k] = "presence/shape differs"
                failures.append((name, k, fields[k]))
                continue
            d = (y.double() - x.double())
            def _matches(patterns):
                return any(k.startswith(f.split("{")[0]) and k.endswith(f.split("}")[1])
                           for f in patterns)
            is_shift = _matches(SHIFT_FIELDS)
            if k in ENERGY_FIELDS or _matches(PER_GRAPH_ENERGY):
                # per graph: predicted shift = -mu_g * excess (constant per graph)
                dd = d.view(-1)
                pred = -mu * excess.to(dd.dtype)
                if _matches(("loop.{i}.energy",)):
                    pred = pred[int(k.split(".")[1]):int(k.split(".")[1]) + 1]
                if dd.numel() != pred.numel():
                    fields[k] = f"per-graph shape {dd.numel()} vs {pred.numel()}"
                    continue
                err = float((dd - pred).abs().max())
                sign_flip = float((dd + pred).abs().max())
                fields[k] = dict(shift=[round(float(v), 9) for v in dd],
                                 predicted=[round(float(v), 9) for v in pred],
                                 max_err=err, max_err_if_sign_flipped=sign_flip)
                if min(err, sign_flip) > ENERGY_FLOOR:
                    failures.append((name, k, f"energy shift is not mu_g x excess: {err:.3e}"))
            elif k in INVARIANT_ENERGY:
                err = float(d.abs().max())
                fields[k] = dict(invariant=True, max_err=err)
                if err > ENERGY_FLOOR:
                    failures.append((name, k, f"a gauge-invariant term moved: {err:.3e}"))
            elif k in LEVEL_FIELDS:
                # each entry: shifted by -mu_g, or untouched where the field is not defined
                err = float(torch.minimum((d + mu).abs(), d.abs()).max())
                fields[k] = dict(rigid_shift_or_zero=-mu, max_err=err)
                if err > 1e-6:
                    failures.append((name, k, f"neither a rigid shift nor zero: {err:.3e}"))
            elif k in SIGNED_LEVEL_FIELDS:
                err = float(torch.minimum((d - mu).abs(), d.abs()).max())
                fields[k] = dict(shift_or_zero=mu, max_err=err)
                if err > 1e-6:
                    failures.append((name, k, f"neither +mu_g nor zero: {err:.3e}"))
            elif is_shift:
                err = float((d + mu).abs().max())
                fields[k] = dict(rigid_shift=-mu, max_err=err)
                if err > 1e-6:
                    failures.append((name, k, f"not a rigid shift by -mu_g: {err:.3e}"))
            else:
                err = float(d.abs().max())
                fields[k] = err
                if err > FLOOR:
                    failures.append((name, k, f"differs by {err:.3e}"))
        report[name] = dict(charged=charged, excess=[float(v) for v in excess], fields=fields)
        n_bad = sum(1 for f in failures if f[0] == name)
        print(f"  {name:36s} {'charged' if charged else 'neutral':8s} "
              f"{'ok' if n_bad == 0 else f'{n_bad} field(s) fail'}")
        for _, k, why in [f for f in failures if f[0] == name]:
            print(f"      {k}: {why}")
    verdict = dict(mu_g=mu, floor=FLOOR, model=args.model, git_sha=G.git_sha(),
                   accepted=not failures, failures=[list(f) for f in failures],
                   records=report)
    Path(args.out).write_text(json.dumps(verdict, indent=1))
    print(f"gauge A/B on {len(a)} records: {'ACCEPTED' if not failures else 'REJECTED'} "
          f"({len(failures)} failing field(s)) -> {args.out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
