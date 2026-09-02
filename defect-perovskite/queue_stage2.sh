#!/usr/bin/env bash
# Stage 2: Edit 3 on top of Edit 1 -- bounded elements, t_min and the learned decay removed.
#
# TWO Z ARMS, same six seeds as Stage 1. The plan specifies learnable Z, so learnable Z is
# the arm of record; the nominal-Z arm is the diagnostic control Stage 1 said was needed
# (learning Z lost 0.11 in axial_red and 2 meV/A, and diverged on two seeds). Running both
# means the Z question is answered whichever way it is decided, without amending the plan
# unilaterally.
#
# Stage 2's gates:
#   * force parity within the SEED SPREAD of Stage 1 -- learned +-7.2, nominal +-6.6
#   * pristine spectrum: bands, not superatom. Stage 1's V3 head reads split fraction ~0.76
#     against an even-spacing reference of 0.20, so there is a number to improve on.
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
# The tag must distinguish the arms, or one overwrites the other. It did not, once.
grep -q '_frz' defect-perovskite/stage_run.py || { echo "ABORT: stale stage_run.py"; exit 1; }

cell () {  # gpu extra-flags seedstart tag
    CUDA_VISIBLE_DEVICES=$1 python defect-perovskite/stage_run.py \
        --arch "$ARCH" --base "$BASE" --stage 2 --madelung on $2 \
        --frames 48 --epochs 40 --seeds 3 --seed-start "$3" \
        --save-dir "$R/stage2_models" --device cuda \
        --out "$R/$4.json" > "$R/$4.log" 2>&1
    echo "  done $4 (exit $?)"
}

echo "=== Stage 2 starting $(date +%F' '%H:%M:%S) ==="
cell 4 ""           1 s2_on_a  &
cell 5 ""           4 s2_on_b  &
cell 6 "--freeze-z" 1 s2_frz_a &
cell 7 "--freeze-z" 4 s2_frz_b &
wait
echo "=== Stage 2 training complete $(date +%F' '%H:%M:%S) ==="

python - <<'PY' | tee "$R/stage2_summary.txt"
import json, glob
import numpy as np
for tag, label in (("s2_on", "LEARNED Z"), ("s2_frz", "NOMINAL Z")):
    rows = []
    for f in sorted(glob.glob(f'/home/alex/runs/{tag}_*.json')):
        try:
            rows += [r for r in json.load(open(f)) if 'error' not in r]
        except Exception:
            pass
    if not rows:
        continue
    m = lambda k: np.mean([r[k] for r in rows if k in r])
    sd = lambda k: np.std([r[k] for r in rows if k in r])
    print(f"{label:10s} n={len(rows)}  axial_red {m('axial_red'):+.3f}+-{sd('axial_red'):.3f}"
          f"  rmse_all {m('rmse_all'):.1f}+-{sd('rmse_all'):.1f}"
          f"  N_eff {m('neff'):6.2f}  null_ratio {m('null_ratio'):.3f}")
    sf = [r['split_fraction'] for r in rows if 'split_fraction' in r]
    if sf:
        print(f"           pristine split fraction {np.mean(sf):.4f} "
              f"(Stage 1 V3 head: 0.76; even spacing: 0.20)")
PY

echo "=== D-2 on Stage-2 models $(date +%F' '%H:%M:%S) ==="
CUDA_VISIBLE_DEVICES=4 python defect-perovskite/d2_bins.py \
    --models "$R"/stage2_models/s2_on_s*.model --limit 24 \
    --device cuda --out "$R/d2_stage2_on.json" > "$R/d2_stage2_on.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 python defect-perovskite/d2_bins.py \
    --models "$R"/stage2_models/s2_on_frz_s*.model --limit 24 \
    --device cuda --out "$R/d2_stage2_frz.json" > "$R/d2_stage2_frz.log" 2>&1 &
# The standing sensitivity diagnostic, per Edit 3's spec.
CUDA_VISIBLE_DEVICES=6 python defect-perovskite/d1_sensitivity.py \
    --models "$R"/stage2_models/s2_on_frz_s*.model --limit 16 --probes 8 \
    --device cuda --out "$R/d1_stage2.json" > "$R/d1_stage2.log" 2>&1 &
wait
echo "=== Stage 2 complete $(date +%F' '%H:%M:%S) ==="
