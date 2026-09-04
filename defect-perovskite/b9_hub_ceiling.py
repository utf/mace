#!/usr/bin/env python3
"""F11 and F12: is the hopping modulation's stop the lever b7's null could not test?

WHY b7 DID NOT SETTLE THIS. b7 varied the GLOBAL radial envelope at initialisation, where the
learned environment modulation is identically 1 by construction (the site MLP starts near
zero). So it measured what a uniform, host-wide change does and found every candidate worse.
It never exercised the DEFECT-LOCAL lever, which is the only one that can raise the hub
coupling without inflating the bandwidth everywhere -- and b3 found that lever already at its
stop on a third of hub bonds. b7's null is scoped to the global knob; this is the local one.

1a -- THE STOP IDENTITY (F11). `t = v0 * radial(r) * (1 + hop_range * tanh(pre))`, so the
modulation is recovered exactly as `pw = t / (v0 * radial)` and `tanh(pre) = (pw - 1) /
hop_range`. A bond is AT ITS STOP when |tanh(pre)| > 1 - eps. Reported per bond type, because
b3 found pp-sigma pinned at the UPPER stop and ss-sigma at the LOWER one on the same bond --
"33% saturated" pooled over channels hides that the head is pushing two of them in opposite
directions.

1b -- THE WHAT-IF (F12). The hub edge's four integrals are scaled by 1.25 and 1.5 in the
trained models, and two things are read: the 79-atom force loss (does the fit improve where
98% of the data lives?) and the 159-atom F4 slope (does the trend move toward -0.134?).

SCORING CAVEAT, REGISTERED BEFORE THE RUN. This is a fixed-density intervention on a
variational quantity: the occupations are re-solved but the rest of the Hamiltonian is frozen
at parameters fitted under the old coupling, so the MAGNITUDE of either response is not a
prediction of what a retrained model would give. F12 is scored on SIGN AND DIRECTION of both
responses only. The rerun is the measurement; the what-if buys the decision about whether to
spend it.

Evaluation-only: the vacancy assignment picks the bond and never reaches the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from mace import tools
from mace.modules.defect_context import ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_size_slopes import stratified  # noqa: E402
from b3_hopping_channel import BOND_NAMES, HubProbe, hub_of  # noqa: E402
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import adopt_model_dtype, make_batches  # noqa: E402
from s3_dehead_trend import CLEAN_NATOMS, fit_with_ci, hub_separation  # noqa: E402
from ta_band_edge import capture, load_frames, select  # noqa: E402

SMALL_NATOMS = 79
AT_BOUND = 0.98           # |tanh(pre)| above this counts as sitting at the stop
D_BINS = ((4.5, 5.5), (5.5, 6.5), (6.5, 8.0))
SCALES = (1.0, 1.25, 1.5)
F12_TARGET = -0.08        # F4 must move past this for F12 to hold


class ScaledHub(HubProbe):
    """`HubProbe` with a multiplicative factor on the hub edge instead of a deletion.

    Subclassed rather than copied: the wrapping is the delicate part (record the edge list in
    `forward`, act on the same edge order in `integrals`, restore both on exit) and a second
    implementation of it would be a second forward pass to keep in step.
    """

    def __init__(self, head, scale=1.0):
        super().__init__(head, drop=False)
        self.scale = float(scale)

    def __enter__(self):
        out = super().__enter__()
        inner = self.h.integrals

        def integrals(fi, fj, r, si, sj):
            v = inner(fi, fj, r, si, sj)          # captures the hub edge as a side effect
            ei = self.captured.get("edge_index")
            if ei is None or self.pair is None or self.scale == 1.0:
                return v
            from b3_hopping_channel import _hub_edge_mask

            mask = _hub_edge_mask(ei, *self.pair)
            factor = torch.where(mask, self.scale, 1.0).to(v.dtype).unsqueeze(-1)
            return v * factor

        self.h.integrals = integrals
        return out


def hub_pair_weights(model, atoms, z_table, cutoff, device, ctx, pair):
    """`tanh(pre)` per bond type on the hub edge, recovered from the identity."""
    batch, frs = make_batches([atoms], z_table, cutoff, 1, device)[0]
    with ScaledHub(model.spectral, scale=1.0) as probe:
        probe.pair = pair
        capture(model, batch, ctx=ctx, frames=frs)
        cap = dict(probe.captured)
    r = cap.get("hub_r", float("nan"))
    if not np.isfinite(r) or cap.get("hub_n_edges", 0) == 0:
        return None, r
    h = model.spectral.h
    v0 = h.v0(torch.tensor([2], device=device),
              torch.tensor([2], device=device))[0].detach().cpu().numpy()
    rad = float(h.radial(torch.tensor([r], dtype=torch.float64)))
    with np.errstate(invalid="ignore", divide="ignore"):
        pw = cap["hub_v"] / (v0 * rad)
    return (pw - 1.0) / h.hop_range, r


def force_loss(model, frames, pairs, z_table, cutoff, device, ctx, scale):
    """Mean squared force error over frames, with the hub bond scaled."""
    total, n = 0.0, 0
    for atoms, pair in zip(frames, pairs):
        batch, frs = make_batches([atoms], z_table, cutoff, 1, device)[0]
        with ScaledHub(model.spectral, scale=scale) as probe:
            probe.pair = pair
            d = ctx.forward_dict(batch, frs, requires_grad=True)
            out = model(d, training=False, compute_force=True)
        err = (out["forces"] - batch.forces).detach()
        total += float((err ** 2).mean())
        n += 1
    return total / max(n, 1)


def delta_sr(model, frames, pairs, z_table, cutoff, device, ctx, scale):
    vals = []
    for atoms, pair in zip(frames, pairs):
        batch, frs = make_batches([atoms], z_table, cutoff, 1, device)[0]
        with ScaledHub(model.spectral, scale=scale) as probe:
            probe.pair = pair
            _, out = capture(model, batch, ctx=ctx, frames=frs)
        vals.append(float(out["delta_sr_energy"].reshape(-1)[0]))
    return np.array(vals)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--n-small", type=int, default=48)
    ap.add_argument("--eps-inf", type=float, default=None,
                    help="per-host input; read off the model when omitted")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    rng = np.random.default_rng(args.seed)
    charged = select(load_frames(args.data), charged=True)

    big = [a for a in charged if len(a) == CLEAN_NATOMS]
    big_pairs = [hub_of(a) for a in big]
    big = [a for a, p in zip(big, big_pairs) if p is not None]
    big_pairs = [p for p in big_pairs if p is not None]
    d_big = np.array([hub_separation(a) for a in big])

    small_all = [a for a in charged if len(a) == SMALL_NATOMS]
    d_all = np.array([hub_separation(a) for a in small_all])
    ok = np.isfinite(d_all)
    small_all = [a for a, o in zip(small_all, ok) if o]
    idx = stratified(small_all, d_all[ok], args.n_small, rng)
    small = [small_all[i] for i in idx]
    small_pairs = [hub_of(a) for a in small]
    small = [a for a, p in zip(small, small_pairs) if p is not None]
    small_pairs = [p for p in small_pairs if p is not None]

    print(f"{len(big)} charged {CLEAN_NATOMS}-atom frames (d {d_big.min():.2f}-"
          f"{d_big.max():.2f} A), {len(small)} of {SMALL_NATOMS} atoms", flush=True)
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))

    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        # The batches built for this model must carry ITS dtype: AtomicData uses the
        # process default, which is float32, while the joint run trains at float64.
        adopt_model_dtype(model)
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        name = Path(mp).name
        row = {"model": name, "hop_range": float(model.spectral.h.hop_range)}

        # ------------------------------------------------------------- 1a, F11
        tanh_all, d_all_hub = [], []
        for atoms, pair in zip(big + small, big_pairs + small_pairs):
            t, r = hub_pair_weights(model, atoms, z, cutoff, args.device, ctx, pair)
            if t is not None:
                tanh_all.append(t)
                d_all_hub.append(r)
        tanh_all = np.array(tanh_all, dtype=float)          # [n_frames, 4]
        d_all_hub = np.array(d_all_hub, dtype=float)
        row["n_bonds"] = int(tanh_all.shape[0])
        row["stops"] = {}
        for b, bond in enumerate(BOND_NAMES):
            v = tanh_all[:, b]
            at = np.abs(v) > AT_BOUND
            row["stops"][bond] = dict(
                mean_tanh=float(np.nanmean(v)),
                at_bound=float(np.nanmean(at)),
                upper=float(np.nanmean(at & (v > 0))),
                lower=float(np.nanmean(at & (v < 0))),
                by_d={f"{lo}-{hi}": float(np.nanmean(at[(d_all_hub >= lo)
                                                        & (d_all_hub < hi)]))
                      if ((d_all_hub >= lo) & (d_all_hub < hi)).any() else float("nan")
                      for lo, hi in D_BINS})

        # ------------------------------------------------------------- 1b, F12
        row["whatif"] = {}
        for scale in SCALES:
            fl = force_loss(model, small, small_pairs, z, cutoff, args.device, ctx, scale)
            sr = delta_sr(model, big, big_pairs, z, cutoff, args.device, ctx, scale)
            good = np.isfinite(sr) & np.isfinite(d_big)
            slope, lo, hi, corr = fit_with_ci(d_big[good], sr[good])
            row["whatif"][f"{scale}"] = dict(force_loss_79=fl, f4_slope_159=slope,
                                             f4_ci=[lo, hi], f4_corr=corr)
        rows.append(row)

        base = row["whatif"]["1.0"]
        print(f"\n  {name}  (hop_range {row['hop_range']})")
        for bond in BOND_NAMES:
            d = row["stops"][bond]
            print(f"    {bond:9s} mean tanh {d['mean_tanh']:+.3f}  at bound "
                  f"{100 * d['at_bound']:5.1f}% ({100 * d['upper']:.0f}% upper / "
                  f"{100 * d['lower']:.0f}% lower)   by d: " +
                  "  ".join(f"{k} {100 * v:4.0f}%" for k, v in d["by_d"].items()))
        for scale in SCALES:
            w = row["whatif"][f"{scale}"]
            print(f"    x{scale:<5}  force loss(79) {w['force_loss_79']:.5f} "
                  f"({100 * (w['force_loss_79'] / base['force_loss_79'] - 1):+.1f}%)   "
                  f"F4(159) {w['f4_slope_159']:+.4f} "
                  f"[{w['f4_ci'][0]:+.4f}, {w['f4_ci'][1]:+.4f}]", flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    if not rows:
        raise SystemExit("no models scored")

    print("\n=== F11: which stop, pooled over seeds ===")
    for bond in BOND_NAMES:
        m = np.array([r["stops"][bond]["mean_tanh"] for r in rows])
        a = np.array([r["stops"][bond]["at_bound"] for r in rows])
        u = np.array([r["stops"][bond]["upper"] for r in rows])
        by = {k: np.nanmean([r["stops"][bond]["by_d"][k] for r in rows])
              for k in rows[0]["stops"][bond]["by_d"]}
        print(f"  {bond:9s} mean tanh {m.mean():+.3f} +- {m.std():.3f}   at bound "
              f"{100 * a.mean():5.1f}% +- {100 * a.std():.1f}   upper share "
              f"{100 * u.mean() / max(a.mean(), 1e-9):.0f}%   by d: " +
              "  ".join(f"{k} {100 * v:4.0f}%" for k, v in by.items()))

    print("\n=== F12: the what-if, pooled over seeds ===")
    base_fl = np.array([r["whatif"]["1.0"]["force_loss_79"] for r in rows])
    base_f4 = np.array([r["whatif"]["1.0"]["f4_slope_159"] for r in rows])
    print(f"  x1.00   force loss(79) {base_fl.mean():.5f} +- {base_fl.std():.5f}   "
          f"F4(159) {base_f4.mean():+.4f} +- {base_f4.std():.4f}")
    verdict = None
    for scale in SCALES[1:]:
        fl = np.array([r["whatif"][f"{scale}"]["force_loss_79"] for r in rows])
        f4 = np.array([r["whatif"][f"{scale}"]["f4_slope_159"] for r in rows])
        d_fl = float(np.mean(fl - base_fl))
        n_down = int((fl < base_fl).sum())
        print(f"  x{scale:<5}  force loss(79) {fl.mean():.5f} "
              f"({100 * (fl.mean() / base_fl.mean() - 1):+.1f}%, fell in {n_down}/{len(fl)} "
              f"seeds)   F4(159) {f4.mean():+.4f} +- {f4.std():.4f}")
        if scale == 1.25:
            verdict = dict(force_loss_falls=bool(d_fl < 0),
                           force_loss_seeds_down=n_down,
                           f4_past_target=bool(f4.mean() < F12_TARGET),
                           f4_mean=float(f4.mean()),
                           delta_force_loss=d_fl)

    verdict["holds"] = bool(verdict["force_loss_falls"] and verdict["f4_past_target"])
    payload = {"rows": rows, "f12": verdict, "at_bound_threshold": AT_BOUND,
               "f12_target": F12_TARGET}
    args.out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n  F12 at x1.25: force loss {'falls' if verdict['force_loss_falls'] else 'RISES'}"
          f" ({verdict['delta_force_loss']:+.6f}), F4 {verdict['f4_mean']:+.4f} vs target "
          f"{F12_TARGET}")
    print(f"  -> F12 {'HOLDS -- widen the modulation bound' if verdict['holds'] else 'FAILS -- the stop is not the lever'}")
    print("\n  Scored on SIGN AND DIRECTION only, as registered: this is a fixed-parameter\n"
          "  intervention on a variational quantity, so the magnitudes are not predictions.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
