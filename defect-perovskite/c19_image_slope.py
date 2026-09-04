#!/usr/bin/env python3
"""Why does the image term triple F4? Knock it out and re-measure.

THE OBSERVATION. Arm B's F4 -- `d(delta_SR)/dd` on 159-atom charged frames, full range -- is
−0.3200 ± 0.0096 against a gate band of [−0.142, −0.063], with 0 of 4 seeds inside. Arm A,
identical but for the image compensation, is −0.0874 ± 0.0142 with 5 of 6 inside. The term
is the only difference between the arms, so it is the only candidate; this measures its
contribution instead of inferring it.

THE METHOD. `image_potential` is patched to return zeros on the same trained arm B models,
which makes `comp = -s * phi_img / eps_inf` identically zero and removes the term from the
Hamiltonian without touching a weight. F4 is then re-measured by the same scorer.

WHAT THE ANSWER MEANS. If F4 returns to arm A's ≈ −0.09 with the term off, the term
contributes the whole −0.23 eV/Å excess, and the question becomes why the image potential
rides on the hub separation `d` so much harder than the labels do. If F4 stays steep, the
term is not the cause and the arms differ for some other reason -- which would be a surprise
worth having.

A knockout at fixed weights answers "how much of the present slope does this term
contribute", not "what would training without it give". Arm A is the second answer, and the
two together bracket it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])


def _child(mode: str, models: list[str], out: str, device: str, reference: str) -> None:
    sys.path.insert(0, REPO)
    sys.path.insert(0, REPO + "/defect-perovskite")
    import mace  # noqa: F401
    import torch

    if mode == "off":
        import mace.modules.defect_models as dm
        import mace.modules.defect_image as di

        def _zero(ewald, q_c, positions, cell, batch, num_graphs, *a, **k):
            return torch.zeros_like(q_c)

        di.image_potential = _zero
        dm.image_potential = _zero          # if it was imported by name anywhere
        print("image_potential PATCHED TO ZERO", flush=True)
    else:
        print("image term as trained", flush=True)

    sys.argv = (["b10_adoption.py", "--models"] + models
                + ["--fold-prefix", "aprime_f", "--cf-runs", str(Path.home() / "runs"),
                   "--reference-json", reference, "--device", device, "--out", out])
    import b10_adoption

    b10_adoption.main()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--reference-json", default=str(Path.home() / "runs"
                                                    / "aprime_reference.json"))
    ap.add_argument("--out-prefix", default=str(Path.home() / "runs" / "image_slope"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        _child(args.child, args.models, f"{args.out_prefix}_{args.child}.json",
               args.device, args.reference_json)
        return

    for mode in ("on", "off"):
        print(f"\n{'=' * 70}\n=== image term {mode}\n{'=' * 70}", flush=True)
        subprocess.run([sys.executable, __file__, "--child", mode,
                        "--models"] + args.models
                       + ["--reference-json", args.reference_json,
                          "--out-prefix", args.out_prefix, "--device", args.device],
                       check=False)
    print("\nCompare the two `F4 delta_sr slope` pooled lines. The difference is the image "
          "term's contribution to the hub-separation slope at fixed weights.")


if __name__ == "__main__":
    main()
