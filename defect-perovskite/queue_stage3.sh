#!/usr/bin/env bash
# Stage 3: Edit 4, the electron-counting head over a uniform s+p basis.
#
# LEARNING RATE 0.05, AND A STAGE-2 CONTROL AT THE SAME RATE.
#
# The first Stage-3 launch used lr = 0.01, the rate Stages 1-2 ran at, and it did not train:
# force 5.34 -> 4.81 over 40 epochs, an optimiser barely moving. At lr = 0.05 the same
# configuration falls from 3.71 to 0.0092 in five epochs. At 0.20 it diverges (Z reaching
# -5.8, +7.1). That run is void and is not reported.
#
# The reason Stage 3 needs a different rate is structural, not a tuning accident: it starts
# at force ~5 where Stages 1-2 started at ~0.004. The counting head's correction is eV-scale
# at initialisation and has to travel three orders of magnitude; the earlier heads only had
# to refine.
#
# Changing the rate between stages weakens "everything held fixed but the edit", so the
# STAGE-2 ARM IS RE-RUN AT lr 0.05 ALONGSIDE. That removes the confound rather than arguing
# about it, and it re-establishes the baseline Stage 3 has to beat under matched settings.
#
# WATCH ITEM: any seed whose pristine split fraction exceeds 0.30 is reseeded, not
# interpreted. stage_run marks them; the summary lists them.
#
# b3 GPUs 4-7 only.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
ARCH=$R/r2_h3_anneal_s5/r2_h3_anneal_s5.model
BASE=$R/e0_base_s1/e0_base_s1.model
cd "$W" || exit 1
for f in "$ARCH" "$BASE"; do [ -f "$f" ] || { echo "ABORT: missing $f"; exit 1; }; done
grep -q 'superatom_reseed' defect-perovskite/stage_run.py || {
    echo "ABORT: stale stage_run.py on this machine (no superatom watch)"; exit 1; }

cell () {  # gpu stage seedstart tag
    CUDA_VISIBLE_DEVICES=$1 python defect-perovskite/stage_run.py \
        --arch "$ARCH" --base "$BASE" --stage "$2" --madelung on \
        --frames 48 --epochs 60 --lr 0.05 --seeds 3 --seed-start "$3" --e-gap 2.4 \
        --save-dir "$R/stage3_models" --device cuda \
        --out "$R/$4.json" > "$R/$4.log" 2>&1
    echo "  done $4 (exit $?)"
}

echo "=== Stage 3 + matched Stage-2 control starting $(date +%F' '%H:%M:%S) ==="
cell 4 3 1 s3_a   &
cell 5 3 4 s3_b   &
cell 6 2 1 s2lr_a &
cell 7 2 4 s2lr_b &
wait
echo "=== training complete $(date +%F' '%H:%M:%S) ==="

python - <<'PY' | tee "$R/stage3_summary.txt"
import json, glob
import numpy as np
for pat, label in (("s3_[ab]", "STAGE 3 (counting, s+p)"),
                   ("s2lr_[ab]", "STAGE 2 control (bounded, s-only)")):
    rows = []
    for f in sorted(glob.glob(f'/home/alex/runs/{pat}.json')):
        try:
            rows += [r for r in json.load(open(f)) if 'error' not in r]
        except Exception:
            pass
    if not rows:
        print(f"{label}: no rows")
        continue
    flagged = [r['seed'] for r in rows if r.get('superatom_reseed')]
    keep = [r for r in rows if not r.get('superatom_reseed')]
    m = lambda k, g: np.mean([r[k] for r in g if k in r])
    sd = lambda k, g: np.std([r[k] for r in g if k in r])
    print(f"\n{label}  n={len(rows)}  (reseed-flagged: {flagged if flagged else 'none'})")
    if keep:
        print(f"  KEPT n={len(keep)}  axial_red {m('axial_red', keep):+.3f}"
              f"+-{sd('axial_red', keep):.3f}  rmse_all {m('rmse_all', keep):.1f}"
              f"+-{sd('rmse_all', keep):.1f}  N_eff {m('neff', keep):6.2f}"
              f"  null_ratio {m('null_ratio', keep):.3f}"
              f"  split {m('split_fraction', keep):.4f}")
    for r in sorted(rows, key=lambda x: x['seed']):
        print(f"    seed {r['seed']}  axial_red {r['axial_red']:+.3f}  "
              f"rmse {r['rmse_all']:.1f}  force {r.get('force_final', float('nan')):.5f}  "
              f"split {r.get('split_fraction', float('nan')):.4f}"
              + ("  RESEED" if r.get('superatom_reseed') else ""))
PY

echo "=== D-2 on Stage-3 models $(date +%F' '%H:%M:%S) ==="
CUDA_VISIBLE_DEVICES=4 python defect-perovskite/d2_bins.py \
    --models "$R"/stage3_models/s3_on_s*.model --limit 24 \
    --device cuda --out "$R/d2_stage3.json" > "$R/d2_stage3.log" 2>&1
echo "=== Stage 3 complete $(date +%F' '%H:%M:%S) ==="
