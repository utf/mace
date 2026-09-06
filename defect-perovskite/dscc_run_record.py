"""Regenerate `run_record.json` for runs trained before the record dump was restored: the
config from the launcher's `config {...}` log line, the folds recomputed (deterministic:
the cross-fit files and the seeded group split), strata from the metas.
    python defect-perovskite/dscc_run_record.py --runs /home/alex/runs/dscc/dscc_arm1_*
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mace.modules.dscc import data as dd                          # noqa: E402
from mace.modules.dscc.train import TrainConfig, load_frames, outer_folds, stratum_id   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", default=str(HERE / "dataset_pbe"))
    ap.add_argument("--cf_dir", default=str(HERE / "dataset_cf"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
    metas = [dd.frame_meta(i, a, "CsPbCl3", 80) for i, a in enumerate(frames)]
    for run in args.runs:
        run = Path(run)
        if (run / "run_record.json").exists() and not args.force:
            continue
        log = (run / "train.log").read_text()
        m = re.search(r" config (\{.*\})$", log, re.M)
        cfg = json.loads(m.group(1))
        cfg = {k: v for k, v in cfg.items() if k in TrainConfig.__dataclass_fields__}
        tc = TrainConfig(**cfg)
        fold_of = outer_folds(frames, metas, args.cf_dir, tc.n_folds, tc.seed)
        charged = [x for x in metas if x.state.Q != 0]
        train_idx = [x.index for x in charged if fold_of.get(x.index) != tc.fold]
        held_idx = [x.index for x in charged if fold_of.get(x.index) == tc.fold]
        strata = {}
        for i in train_idx:
            strata.setdefault(stratum_id(metas[i]), []).append(i)
        record = {"config": cfg, "strata": {k: {"n": len(v), "total_weight": 1.0} for k, v in strata.items()},
                  "n_train": len(train_idx), "n_held": len(held_idx), "fold_of": {str(k): v for k, v in fold_of.items()},
                  "regenerated_from": "train.log"}
        json.dump(record, open(run / "run_record.json", "w"), indent=1, default=str)
        print(run.name, "record regenerated: train", len(train_idx), "held", len(held_idx))


if __name__ == "__main__":
    main()
