#!/usr/bin/env python3
"""F7: is the Pb-Pb coupling channel that carries F4's slope suppressed by the envelope?

THE QUESTION. `dlambda/dd` in a two-centre picture is `-d|t|/dd`, so a head whose hub-bond
hopping is too small, or falls off too fast, cannot reproduce a label slope no matter how it
is trained. The head's radial envelope is

    radial(r) = exp(-(r - d_ref) / decay_length) * (1 - (r / r_cut)^6)^2

with `d_ref = 2.8 A`, `decay_length = 1.0 A`, `r_cut = 10 A`, while Harrison -- the rule the
`v0` table is INITIALISED from -- is a power law, `V = eta * hbar^2 / (m d^2)`. The two agree
at `d_ref` by construction and diverge fast: the hub Pb-Pb bond sits at 5-7 A, where an
exponential with a 1 A constant has fallen by `exp(-3)` to `exp(-4)` and a `d^-2` power law
has fallen by a factor of four. This measures the gap instead of asserting it.

FOUR THINGS, in increasing order of how much they decide.

  1. THE STATIC TABLE. `radial`, its two factors and its log-derivative against Harrison's,
     over 4.5-8 A. Pure function of the head's constants; no model needed. Establishes
     whether the taper is even active in the hub window (it is not: `(r/10)^6` is 0.05 at
     6 A) so that "inside a cutoff taper" can be answered rather than assumed.
  2. THE HUB BOND ITSELF. `t_i = v0_i * radial(d) * (1 + hop_range * tanh(pre))` decomposed
     factor by factor on the real hub edge of every frame, with the LEARNED `v0` -- because
     the optimiser may have compensated by inflating `v0`, and a ratio taken against the
     initialisation would miss that. F7's "beyond the pair-weight factor" is exactly the
     `(1 + 0.5 tanh)` term, which is bounded in [0.5, 1.5]: if the envelope gap exceeds that
     headroom, no amount of training closes it.
  3. THE CLEAN DERIVATIVE, and this is what the regression cannot give. F4's slope is fitted
     ACROSS frames, so it mixes the hub separation with everything else that differs between
     snapshots. Here the two hub Pb are displaced along their own axis by +-0.05 A on a fixed
     frame, and `d(delta_sr)/dd` and `dlambda/dd` are read as genuine derivatives of the same
     configuration.
  4. THE HUB-EDGE ABLATION, which names the channel. The same derivative is taken with the
     direct hub edge deleted from the Hamiltonian. If it survives, the d-dependence is
     superexchange through the ligand shell and the envelope is not the lever; if it
     collapses, the direct bond carries it and the envelope is.

Plus a forward-only envelope swap (`exp` -> `(d_ref/r)^2`) on the SAME derivative, with the
trained weights frozen. THAT IS OFF-DISTRIBUTION AND IS REPORTED AS A SENSITIVITY BOUND, not
as a prediction: these weights were fitted under the exponential, so the swap says how much
the observable can move for an envelope change, not where a retrained model would land.

Evaluation-only: the vacancy assignment picks the axis and never reaches the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from ase.geometry import get_distances

from mace import tools
from mace.modules.defect_context import EPS_INF_DEFAULT, ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from s3_dehead_trend import CLEAN_NATOMS, hub_separation  # noqa: E402
from ta_band_edge import capture, load_frames, select  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402

BOND_NAMES = ("ss_sigma", "sp_sigma", "pp_sigma", "pp_pi")
ETA = np.array([-1.40, 1.84, 3.24, -0.81])       # Harrison's universal coefficients
HBAR2_OVER_M = 7.62                              # eV A^2
PP_SIGMA = 2                                     # the sigma-like channel along the hub axis


# ------------------------------------------------------------------ 1. the static table


def envelope_table(head, lo=4.5, hi=8.0, step=0.25):
    """`radial` and Harrison side by side, as pure functions of the head's constants."""
    r = np.arange(lo, hi + 1e-9, step)
    d_ref, L, rc = head.d_ref, head.decay_length, head.r_cut
    expo = np.exp(-(r - d_ref) / L)
    taper = (1.0 - np.clip(r / rc, None, 1.0) ** 6) ** 2
    model = expo * taper
    harrison = (d_ref / r) ** 2
    # d ln f / dr for each. The taper's term is written out rather than differenced so the
    # number is exact where it matters (it is tiny below 7 A, which is the point).
    x6 = np.clip(r / rc, None, 1.0) ** 6
    dln_taper = np.where(x6 < 1.0, -12.0 * x6 / (r * (1.0 - x6)), -np.inf)
    return dict(
        r=r.tolist(), exponential=expo.tolist(), taper=taper.tolist(),
        model=model.tolist(), harrison=harrison.tolist(),
        ratio=(model / harrison).tolist(),
        dln_model=(-1.0 / L + dln_taper).tolist(),
        dln_harrison=(-2.0 / r).tolist(),
        d_ref=d_ref, decay_length=L, r_cut=rc)


# --------------------------------------------------------------------- edge bookkeeping


def hub_of(atoms):
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return None
    return int(site.shell[0]), int(site.shell[1])


def _hub_edge_mask(edge_index, a, b):
    src, dst = edge_index[0], edge_index[1]
    return ((src == a) & (dst == b)) | ((src == b) & (dst == a))


class HubProbe:
    """Wraps the head's `forward`/`integrals` to read, and optionally delete, the hub edge.

    WHY WRAP AND NOT REIMPLEMENT. Writing a second copy of the assembly here to insert a mask
    is how a diagnostic and the model come to disagree about the forward pass -- this project
    has paid for that four times. `forward` is wrapped only to record the edge list it was
    handed; `integrals` is wrapped to act on the same edge order it returns. Neither body is
    duplicated.
    """

    def __init__(self, head, drop=False, radial_override=None):
        self.h = head.h
        self.drop = bool(drop)
        self.radial_override = radial_override
        self.pair = None
        self.captured = {}
        self._fwd = self._int = self._rad = None

    def __enter__(self):
        h, self = self.h, self
        self._fwd, self._int = h.forward, h.integrals
        slot = self.captured

        def forward(*a, **k):
            slot["edge_index"] = k.get("edge_index", a[2] if len(a) > 2 else None)
            return self._fwd(*a, **k)

        def integrals(fi, fj, r, si, sj):
            v = self._int(fi, fj, r, si, sj)
            ei = slot.get("edge_index")
            if ei is None or self.pair is None:
                return v
            mask = _hub_edge_mask(ei, *self.pair)
            if mask.any():
                # The MINIMUM-image edge, not the first match. A 10 A graph cutoff on a
                # 159-atom cell puts several periodic images of the partner inside range, so
                # the pair appears more than once with different shifts; taking `nonzero()[0]`
                # would silently read whichever image the neighbour list happened to emit
                # first and report an image separation as the bond length.
                where = torch.nonzero(mask).reshape(-1)
                idx = int(where[int(torch.argmin(r[where]))])
                slot["hub_r"] = float(r[idx])
                slot["hub_v"] = v[idx].detach().float().cpu().numpy().copy()
                slot["hub_n_edges"] = int(mask.sum())
            else:
                slot["hub_r"] = float("nan")
                slot["hub_v"] = np.full(4, np.nan)
                slot["hub_n_edges"] = 0
            if self.drop:
                v = v * (~mask).unsqueeze(-1).to(v.dtype)
            return v

        h.forward, h.integrals = forward, integrals
        if self.radial_override is not None:
            self._rad = h.radial
            h.radial = self.radial_override
        return self

    def __exit__(self, *exc):
        self.h.forward, self.h.integrals = self._fwd, self._int
        if self._rad is not None:
            self.h.radial = self._rad
        return False


def power_law_radial(head):
    """Harrison's own `d^-2`, keeping the SAME cutoff taper so the graph stays finite."""
    d_ref, rc = head.h.d_ref, head.h.r_cut

    def radial(r):
        x = (r / rc).clamp(max=1.0)
        return (d_ref / r.clamp_min(1e-9)) ** 2 * (1.0 - x ** 6) ** 2

    return radial


# ------------------------------------------------------------------ 3/4. the derivative


def displaced(atoms, pair, delta):
    """A copy with the two hub atoms moved apart along their own axis by `delta` in total."""
    b = atoms.copy()
    b.info = dict(atoms.info)
    pos = b.get_positions()
    i, j = pair
    vec, dist = get_distances(pos[i][None], pos[j][None], cell=b.get_cell(), pbc=b.pbc)
    axis = vec[0, 0] / max(float(dist[0, 0]), 1e-12)
    pos[i] -= 0.5 * delta * axis
    pos[j] += 0.5 * delta * axis
    b.set_positions(pos)
    return b


def observables(model, atoms, z_table, cutoff, device, ctx, pair, drop=False,
                radial_override=None):
    """`(delta_sr, lambda_frontier)` on one frame, optionally without the direct hub edge.

    ONE forward, not two. `capture` already returns the model output beside the head
    internals, so reading `delta_sr_energy` from it costs nothing; calling the model and then
    `frontier_level` separately would run the 636x636 `eigh` twice per point, and there are
    seven points per frame per seed.
    """
    from mace.modules.defect_counting import VALENCE, changed_level_index

    batch, frs = make_batches([atoms], z_table, cutoff, 1, device)[0]
    with HubProbe(model.spectral, drop=drop, radial_override=radial_override) as probe:
        probe.pair = pair
        internals, out = capture(model, batch, ctx=ctx, frames=frs)
        cap = dict(probe.captured)
    sr = float(out["delta_sr_energy"].reshape(-1)[0])
    lam = internals["lam"][0, 0]
    lam = lam[lam < 500.0]
    n_total = sum(VALENCE[int(z)] for z in frs[0].get_atomic_numbers())
    k = changed_level_index(n_total, batch.carrier_counts.reshape(-1).tolist())
    lam_f = float(lam[k]) if 0 <= k < lam.numel() else float("nan")
    return sr, lam_f, cap


def derivative(model, atoms, z_table, cutoff, device, ctx, pair, step, **kw):
    plus = observables(model, displaced(atoms, pair, +step), z_table, cutoff, device, ctx,
                       pair, **kw)
    minus = observables(model, displaced(atoms, pair, -step), z_table, cutoff, device, ctx,
                        pair, **kw)
    return ((plus[0] - minus[0]) / (2 * step), (plus[1] - minus[1]) / (2 * step))


def mean_ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0, \
        int(v.size)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--step", type=float, default=0.05,
                    help="half the total hub displacement, A")
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    frames = [a for a in select(load_frames(args.data), charged=True)
              if len(a) == CLEAN_NATOMS][:args.frames]
    pairs = [hub_of(a) for a in frames]
    keep = [(a, p) for a, p in zip(frames, pairs) if p is not None]
    d_all = np.array([hub_separation(a) for a, _ in keep])
    print(f"{len(keep)} charged {CLEAN_NATOMS}-atom frames with a located hub, "
          f"d {d_all.min():.2f}-{d_all.max():.2f} A", flush=True)
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))

    payload = {"frames": len(keep), "step": args.step, "d": d_all.tolist(), "models": []}
    for mp in args.models:
        if not Path(mp).exists():
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        head = model.spectral
        cutoff = max(float(model.r_max), float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        name = Path(mp).name

        row = {"model": name, "envelope": envelope_table(head.h),
               "hop_range": float(head.h.hop_range),
               "on_site_range": float(head.h.on_site_range)}
        if "table" not in payload:
            payload["table"] = row["envelope"]      # identical across seeds; kept once

        v0 = head.h.v0(torch.tensor([2], device=args.device),
                       torch.tensor([2], device=args.device))[0].detach().cpu().numpy()
        row["v0_PbPb_learned"] = v0.tolist()
        row["v0_PbPb_harrison_at_dref"] = (ETA * HBAR2_OVER_M / head.h.d_ref ** 2).tolist()

        hub_v, hub_corr, hub_d = [], [], []
        sr_full, lam_full, sr_drop, lam_drop, sr_pow, lam_pow = [], [], [], [], [], []
        for atoms, pair in keep:
            _, _, cap = observables(model, atoms, z_table, cutoff, args.device, ctx, pair)
            if cap.get("hub_n_edges", 0) == 0:
                print(f"  {name}: hub edge ABSENT from the graph on one frame -- the "
                      "cutoff does not reach it; that alone answers F7", flush=True)
            hub_v.append(cap.get("hub_v", np.full(4, np.nan)))
            hub_d.append(cap.get("hub_r", float("nan")))
            # The learned pair modulation on this bond, recovered from the identity
            # t = v0 * radial(d) * corr: the factor F7 calls the pair weight.
            rad = float(head.h.radial(torch.tensor([cap.get("hub_r", np.nan)],
                                                   dtype=torch.float64))[0])
            with np.errstate(invalid="ignore", divide="ignore"):
                hub_corr.append(hub_v[-1] / (v0 * rad))

            a, b = derivative(model, atoms, z_table, cutoff, args.device, ctx, pair,
                              args.step)
            sr_full.append(a); lam_full.append(b)
            a, b = derivative(model, atoms, z_table, cutoff, args.device, ctx, pair,
                              args.step, drop=True)
            sr_drop.append(a); lam_drop.append(b)
            a, b = derivative(model, atoms, z_table, cutoff, args.device, ctx, pair,
                              args.step, radial_override=power_law_radial(head))
            sr_pow.append(a); lam_pow.append(b)

        hub_v = np.array(hub_v, dtype=float)
        hub_corr = np.array(hub_corr, dtype=float)
        hub_d = np.array(hub_d, dtype=float)
        harrison = ETA[None, :] * HBAR2_OVER_M / hub_d[:, None] ** 2

        row.update(
            hub_d=hub_d.tolist(),
            hub_t=hub_v.tolist(),
            hub_pair_weight=hub_corr.tolist(),
            hub_harrison=harrison.tolist(),
            hub_ratio_to_harrison=(hub_v / harrison).tolist(),
            d_delta_sr_dd=dict(zip(("full", "hub_dropped", "power_law"),
                                   (sr_full, sr_drop, sr_pow))),
            d_lambda_dd=dict(zip(("full", "hub_dropped", "power_law"),
                                 (lam_full, lam_drop, lam_pow))),
        )
        payload["models"].append(row)

        rat = (hub_v / harrison)[:, PP_SIGMA]
        pw = hub_corr[:, PP_SIGMA]
        m_sr, e_sr, _ = mean_ci(sr_full)
        m_dr, e_dr, _ = mean_ci(sr_drop)
        m_pw_, e_pw_, _ = mean_ci(sr_pow)
        m_lam, e_lam, _ = mean_ci(lam_full)
        print(f"  {name:20s} pp_sigma at the hub: t/Harrison {np.nanmean(rat):.3f} "
              f"+- {np.nanstd(rat):.3f}, pair weight {np.nanmean(pw):.3f} "
              f"(bounded [{1 - head.h.hop_range:.1f}, {1 + head.h.hop_range:.1f}])",
              flush=True)
        print(f"  {'':20s} d(delta_sr)/dd  full {m_sr:+.4f}+-{e_sr:.4f}   "
              f"hub dropped {m_dr:+.4f}+-{e_dr:.4f}   power law {m_pw_:+.4f}+-{e_pw_:.4f} "
              f"eV/A", flush=True)
        print(f"  {'':20s} dlambda/dd      full {m_lam:+.4f}+-{e_lam:.4f} eV/A", flush=True)
        args.out.write_text(json.dumps(payload, indent=2, default=float))
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.out.write_text(json.dumps(payload, indent=2, default=float))

    # ------------------------------------------------------------------- the verdict
    if payload["models"]:
        t = payload["table"]
        print("\n=== the envelope against Harrison (pure function of the constants) ===")
        print(f"  {'r (A)':>7} {'exp':>9} {'taper':>8} {'model':>9} {'harrison':>9} "
              f"{'ratio':>7} {'dln model':>10} {'dln harr':>9}")
        for k, r in enumerate(t["r"]):
            if r < 4.75 or r > 7.75:
                continue
            print(f"  {r:7.2f} {t['exponential'][k]:9.4f} {t['taper'][k]:8.4f} "
                  f"{t['model'][k]:9.4f} {t['harrison'][k]:9.4f} {t['ratio'][k]:7.3f} "
                  f"{t['dln_model'][k]:10.3f} {t['dln_harrison'][k]:9.3f}")

        def pooled(field, arm):
            return np.concatenate([np.asarray(m[field][arm], dtype=float)
                                   for m in payload["models"]])

        full, drop, pw = (pooled("d_delta_sr_dd", a)
                          for a in ("full", "hub_dropped", "power_law"))
        lf, ld = (pooled("d_lambda_dd", a) for a in ("full", "hub_dropped"))
        print("\n=== the clean derivative, pooled over seeds and frames ===")
        print(f"  d(delta_sr)/dd   full        {np.nanmean(full):+.4f} "
              f"+- {np.nanstd(full):.4f} eV/A")
        print(f"                   hub dropped {np.nanmean(drop):+.4f} "
              f"+- {np.nanstd(drop):.4f} eV/A   "
              f"(retained {100 * np.nanmean(drop) / np.nanmean(full):.0f}%)")
        print(f"                   power law   {np.nanmean(pw):+.4f} "
              f"+- {np.nanstd(pw):.4f} eV/A   "
              f"(x{np.nanmean(pw) / np.nanmean(full):.2f} vs full)")
        print(f"  dlambda/dd       full        {np.nanmean(lf):+.4f} "
              f"+- {np.nanstd(lf):.4f} eV/A")
        print(f"                   hub dropped {np.nanmean(ld):+.4f} "
              f"+- {np.nanstd(ld):.4f} eV/A")
        rat = np.concatenate([np.asarray(m["hub_ratio_to_harrison"],
                                         dtype=float)[:, PP_SIGMA]
                              for m in payload["models"]])
        headroom = 1.0 + float(payload["models"][0].get("hop_range", 0.5))
        print(f"\n  F7 asks whether t_PbPb sits below Harrison by more than the pair-weight "
              f"factor.\n  pp_sigma t/Harrison at the hub: {np.nanmean(rat):.3f} "
              f"+- {np.nanstd(rat):.3f}; the bounded modulation can supply at most "
              f"x{headroom:.1f}.")
        print(f"  -> F7 {'FIRES' if np.nanmean(rat) < 1.0 / headroom else 'does not fire'}")

        # The pair weight is the head's only lever on this bond, so whether it is PINNED
        # says whether the head is straining against the envelope or content with it.
        pw_all = np.concatenate([np.asarray(m["hub_pair_weight"], dtype=float)[:, PP_SIGMA]
                                 for m in payload["models"]])
        hop_range = float(payload["models"][0]["hop_range"])
        at_bound = float(np.mean(np.abs(np.abs(pw_all - 1.0) - hop_range) < 0.02 * hop_range))
        print(f"  pp_sigma pair weight {np.nanmean(pw_all):.4f}; "
              f"{100 * at_bound:.0f}% of hub bonds sit within 2% of the bound "
              f"[{1 - hop_range:.2f}, {1 + hop_range:.2f}]")

        # The "one number" a length-constant fix would have to change, reported so that §2's
        # choice is a measurement and not a preference: the decay length whose log-slope
        # matches Harrison's at the median hub separation.
        d_med = float(np.nanmedian(np.concatenate(
            [np.asarray(m["hub_d"], dtype=float) for m in payload["models"]])))
        print(f"\n  At the median hub separation {d_med:.2f} A, Harrison's log-slope is "
              f"{-2 / d_med:+.3f} /A and the exponential's is "
              f"{-1 / t['decay_length']:+.3f} /A.")
        print(f"  Matching them there needs decay_length = {d_med / 2:.2f} A "
              f"(currently {t['decay_length']:.2f}); matching at d_ref = {t['d_ref']} A "
              f"needs {t['d_ref'] / 2:.2f} A.")
        print("  Those two disagree because an exponential cannot have a power law's "
              "log-slope at\n  two separations at once -- which is itself the argument for "
              "the power law over a retune.")

        print("\n=== the derivative against the hub separation, pooled ===")
        d_pool = np.concatenate([np.asarray(m["hub_d"], dtype=float)
                                 for m in payload["models"]])
        for lo, hi in ((4.5, 5.5), (5.5, 6.5), (6.5, 8.0)):
            m = (d_pool >= lo) & (d_pool < hi)
            if m.sum() < 2:
                continue
            print(f"  d {lo:.1f}-{hi:.1f} A  n {int(m.sum()):3d}   full "
                  f"{np.nanmean(full[m]):+.4f}   hub dropped {np.nanmean(drop[m]):+.4f}   "
                  f"power law {np.nanmean(pw[m]):+.4f}   t/Harrison "
                  f"{np.nanmean(rat[m]):.3f}")
        print("\n  The power-law column is a SENSITIVITY BOUND, not a prediction: these "
              "weights were\n  fitted under the exponential envelope, so it says how far the "
              "observable can move,\n  not where a retrained model lands.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
