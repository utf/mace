#!/usr/bin/env python3
"""Size-extensivity comparison across models: is ``alpha`` on the defect flat in ``N``?

This is the acceptance test of the size plan, section 7, drawn rather than tabulated. Each
model contributes one curve per live carrier channel: the share of the attention landing on
the defect shell against cell size. A size-transferable model gives a horizontal line. The
failure this whole exercise is about is a curve that falls, and it falls as ``1/N`` once
``N`` passes ``k e^{gap}``.

Two things are deliberately *not* plotted. ``gap_l`` (max logit minus mean) is not the gap
in ``N* = k e^{gap}``: it measures one atom against the average and runs 2-4x larger than
the collective shell-versus-bulk value that actually governs dilution. And ``size_viol``,
the hinge's own diagnostic, says whether the term is satisfied, not whether the model is
size-stable -- those came apart in testing, where driving the violation to zero produced a
carrier sitting on non-defect atoms.

A channel holding ~0 weight on the shell at every size is flat, but flat at zero: that is
an absent carrier, not a stable one, and it is drawn dashed and excluded from the verdict.

    python plot_size_comparison.py --runs bfull_128ch_L1_s1 efull_128ch_L1_s2 esize_128ch_L1_s2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

CHANNELS = ("e_maj", "e_min", "h_maj", "h_min")
# A channel counts as holding the defect only if its shell weight beats the *uniform*
# expectation by this factor. An absolute cut cannot work: uniform attention already puts
# k/N on the shell, which is 0.086 at N = 70 -- so a dead channel reads 0.087 and would
# pass any fixed threshold below that, then appear in the chart as a real carrier.
PRESENT_FACTOR = 3.0
SHELL = 6


def holds_defect(sizes, alpha) -> bool:
    """Does this channel beat uniform attention on the shell at the smallest cell?"""
    if not len(sizes):
        return False
    uniform = SHELL / float(sizes[0])
    return float(alpha[0]) > PRESENT_FACTOR * uniform


def parse(path: Path) -> dict:
    """Per-channel (sizes, alpha-on-shell, implied gap) from an alpha_dilution.py log."""
    rows: dict = {c: {"n": [], "alpha": []} for c in CHANNELS}
    gaps: dict = {}
    for line in path.read_text().splitlines():
        table = re.match(
            r"\s*(\d+)\s+(e_maj|e_min|h_maj|h_min)\s+([\d.]+)\s+([\d.-]+)\s+([\d.-]+)",
            line,
        )
        if table:
            rows[table.group(2)]["n"].append(int(table.group(1)))
            rows[table.group(2)]["alpha"].append(float(table.group(4)))
        summary = re.match(r"\s+implied gap\s+([-\d.]+)", line)
        if summary and gaps.get("_pending"):
            gaps[gaps.pop("_pending")] = float(summary.group(1))
        named = re.match(r"\s+(e_maj|e_min|h_maj|h_min)\s+alpha on shell", line)
        if named:
            gaps["_pending"] = named.group(1)
    gaps.pop("_pending", None)
    return {"rows": rows, "gaps": gaps}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    here = Path(__file__).resolve().parent
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run names; reads ~/runs/dilution_<name>.log")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--log-dir", type=Path, default=Path.home() / "runs")
    parser.add_argument("--out", type=Path, default=here / "size_comparison.png")
    args = parser.parse_args()

    labels = args.labels or args.runs
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must match --runs in length")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(args.runs)))
    markers = ["o", "s", "^", "D", "v", "P"]

    parsed = {}
    for name in args.runs:
        path = args.log_dir / f"dilution_{name}.log"
        if not path.is_file():
            print(f"  missing {path}, skipping")
            continue
        parsed[name] = parse(path)
    if not parsed:
        raise SystemExit("no dilution logs found")

    live = [
        c for c in CHANNELS
        if any(
            holds_defect(data["rows"][c]["n"], data["rows"][c]["alpha"])
            for data in parsed.values()
        )
    ]
    figure, axes = plt.subplots(
        1, len(live), figsize=(3.3 * len(live), 3.4), squeeze=False
    )
    axes = axes[0]

    for index, (name, data) in enumerate(parsed.items()):
        label = labels[args.runs.index(name)]
        for axis, channel in zip(axes, live):
            sizes = np.array(data["rows"][channel]["n"], dtype=float)
            alpha = np.array(data["rows"][channel]["alpha"], dtype=float)
            if not len(sizes):
                continue
            present = holds_defect(sizes, alpha)
            axis.plot(
                sizes, alpha, marker=markers[index % len(markers)], ms=4.5, lw=1.4,
                color=colours[index], alpha=0.9 if present else 0.35,
                linestyle="-" if present else "--",
                label=f"{label}  (gap {data['gaps'].get(channel, float('nan')):.1f})",
            )

    for axis, channel in zip(axes, live):
        axis.set_xscale("log")
        axis.set_xlabel("atoms in cell $N$")
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(channel, fontsize=10)
        axis.axhline(1.0, lw=0.6, color="0.8", zorder=0)
        axis.legend(frameon=False, fontsize=7.2, loc="lower left")
    axes[0].set_ylabel(r"$\alpha$ on the defect shell")
    figure.suptitle(
        "Flat = size-transferable.  Falling = the correction dilutes into the bulk.",
        fontsize=9.5, y=1.02,
    )
    figure.tight_layout()
    figure.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}\n")

    width = max(len(l) for l in labels) + 2
    print(f"{'model':{width}s} " + " ".join(f"{c:>18s}" for c in live))
    print(f"{'':{width}s} " + " ".join(f"{'gap   N*   drop':>18s}" for _ in live))
    for name, data in parsed.items():
        label = labels[args.runs.index(name)]
        cells = []
        for channel in live:
            alpha = data["rows"][channel]["alpha"]
            gap = data["gaps"].get(channel, float("nan"))
            if not holds_defect(data["rows"][channel]["n"], alpha):
                cells.append(f"{'absent':>18s}")
                continue
            star = 6 * np.exp(gap)
            cells.append(f"{gap:6.2f} {star:8.0f} {alpha[0] - alpha[-1]:+5.2f}")
        print(f"{label:{width}s} " + " ".join(cells))
    print(
        "\ndrop = alpha lost between the smallest and largest cell; ~0 is the target.\n"
        "'absent' means the channel never held the defect, so its flatness is vacuous."
    )


if __name__ == "__main__":
    main()
