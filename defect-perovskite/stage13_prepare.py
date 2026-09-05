#!/usr/bin/env python3
"""Plan v8 Stage 1.3, the forward-only half: the Stage B cohort under the three Madelung
arms, as model files the existing gate scorers can read.

For each Stage B seed the model is loaded, given the class table (built the way the trainer
builds it: over the training and validation sets in file order, first frame of each class
as its reference geometry -- the Stage B checkpoints predate the table), set to one of
`full` (a: as is), `long_range` (b: the short-range part removed at r_split, the
first-block cutoff) or `off` (c: removed entirely), and saved under
`~/runs/s13_<arm>_models/s13_<arm>_s<k>.model` (plus the `<tag>_s<k>/<tag>_s<k>.model`
layout the gate queue expects). The learned static charges Z of each seed are printed --
under (a) and (b) they are the same numbers, since no retrain happens here.

    python defect-perovskite/stage13_prepare.py --seeds 1 2 3 4 5 6 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

import mace  # noqa: F401
from ase.io import read

from mace import data as mace_data
from mace import tools
from mace.data.defects import prepare_defect_configurations
from mace.modules.defect_cache import attach_frame_keys
from mace.modules.defect_composition import ensure_class_table
from mace.modules.defect_madelung import MADELUNG_RANGES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_gates import KEYSPEC  # noqa: E402

HERE = Path(__file__).resolve().parent
ARMS = {"full": "a", "long_range": "b", "off": "c"}


def frames_for_table(data_dir: Path, z_table, cutoff: float):
    atoms = read(str(data_dir / "train.xyz"), index=":") + read(str(data_dir / "valid.xyz"),
                                                              index=":")
    configs = [mace_data.config_from_atoms(a, key_specification=KEYSPEC) for a in atoms]
    prepare_defect_configurations(configs)
    out = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=cutoff)
           for c in configs]
    attach_frame_keys(out, z_table=z_table)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path.home() / "runs")
    p.add_argument("--source-tag", default="stageb")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--data", type=Path, default=HERE / "dataset_pbe")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    torch.set_default_dtype(torch.float64)
    frames = None
    for seed in args.seeds:
        src = args.runs / f"{args.source_tag}_s{seed}" / f"{args.source_tag}_s{seed}.model"
        model = torch.load(src, map_location="cpu", weights_only=False).to(args.device).eval()
        z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        if frames is None:
            t0 = time.time()
            frames = frames_for_table(args.data, z_table, float(model.r_max))
            print(f"{len(frames)} frames for the class table ({time.time() - t0:.0f} s)")
        model.composition_classes = None
        t0 = time.time()
        table = ensure_class_table(model, frames, log=False)
        counted = {k: (r["n_e"], r["n_h"], r["q_core"], r["tier"])
                   for k, r in table["classes"].items()}
        z = [round(float(v), 4) for v in model.madelung.z.detach().cpu()]
        print(f"seed {seed}: table in {time.time() - t0:.0f} s, classes {counted}; Z {z}")
        for mode, arm in ARMS.items():
            model.madelung_range = mode
            tag = f"s13{arm}"
            d = args.runs / f"{tag}_models"
            d.mkdir(exist_ok=True)
            out = d / f"{tag}_s{seed}.model"
            torch.save(model.cpu(), out)
            model.to(args.device)
            layout = args.runs / f"{tag}_s{seed}"
            layout.mkdir(exist_ok=True)
            link = layout / f"{tag}_s{seed}.model"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(out)
            print(f"  wrote {out} (madelung_range={mode})")
        model.madelung_range = "full"


if __name__ == "__main__":
    main()
