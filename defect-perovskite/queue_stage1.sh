#!/usr/bin/env bash
# Stage 1: Edit 1 (Madelung on-site) + Edit 2 (response channel deleted), head-only retrain.
#
# SIX seeds per arm, not three. Stage 2's gate is "force parity within the seed spread of
# Stage 1", and r1_matrix's own note says a spread estimated from three points is the weakest
# link in a verdict. Six per arm also matches the A/B convention and fits GPUs 4-7 exactly:
# one arm-half per GPU.
#
# THE CONTROL IS THE OFF ARM, NOT THE ARCHIVED COHORT. The archived models trained under
# T-B's edge and gap losses; this harness is force-only. Comparing ON against those numbers
# would conflate the edit with the loss change, so both arms are run here and every gate is
# read ON vs OFF. The archived numbers are background only.
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

cell () {  # gpu madelung seeds seedstart tag
    CUDA_VISIBLE_DEVICES=$1 python defect-perovskite/stage_run.py \
        --arch "$ARCH" --base "$BASE" --stage 1 --madelung "$2" \
        --frames 48 --epochs 40 --seeds "$3" --seed-start "$4" \
        --save-dir "$R/stage1_models" --device cuda \
        --out "$R/$5.json" > "$R/$5.log" 2>&1
    echo "  done $5 (exit $?)"
}

echo "=== Stage 1 starting $(date +%F' '%H:%M:%S) ==="
cell 4 on  3 1 s1_on_a  &
cell 5 on  3 4 s1_on_b  &
cell 6 off 3 1 s1_off_a &
cell 7 off 3 4 s1_off_b &
wait
echo "=== Stage 1 training complete $(date +%F' '%H:%M:%S) ==="

python - <<'PY' | tee "$R/stage1_summary.txt"
import json, glob
import numpy as np
rows = []
for f in sorted(glob.glob('/home/alex/runs/s1_*.json')):
    try:
        rows += [r for r in json.load(open(f)) if 'error' not in r]
    except Exception:
        pass
for arm, flag in (('ON', True), ('OFF', False)):
    g = [r for r in rows if r.get('madelung') is flag]
    if not g:
        continue
    def m(k):
        return np.mean([r[k] for r in g])
    def sd(k):
        return np.std([r[k] for r in g])
    print(f"{arm:4s} n={len(g)}  axial_red {m('axial_red'):+.3f}+-{sd('axial_red'):.3f}"
          f"  rmse_all {m('rmse_all'):.1f}+-{sd('rmse_all'):.1f}"
          f"  rmse_nbhd {m('rmse_nbhd'):.1f}+-{sd('rmse_nbhd'):.1f}"
          f"  N_eff {m('neff'):6.2f}  null_ratio {m('null_ratio'):.3f}")
    if flag:
        z = np.array([r['z'] for r in g if 'z' in r])
        if z.size:
            print(f"     Z (Cl, Cs, Pb) mean {np.round(z.mean(0), 3).tolist()}  "
                  f"sd {np.round(z.std(0), 3).tolist()}")
PY

# The D-2 gate, read ON vs OFF on models this stage produced.
echo "=== D-2 on Stage-1 models $(date +%F' '%H:%M:%S) ==="
CUDA_VISIBLE_DEVICES=4 python defect-perovskite/d2_bins.py \
    --models "$R"/stage1_models/s1_on_s*.model --limit 24 \
    --device cuda --out "$R/d2_stage1_on.json" > "$R/d2_stage1_on.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 python defect-perovskite/d2_bins.py \
    --models "$R"/stage1_models/s1_off_s*.model --limit 24 \
    --device cuda --out "$R/d2_stage1_off.json" > "$R/d2_stage1_off.log" 2>&1 &
wait
echo "=== Stage 1 complete $(date +%F' '%H:%M:%S) ==="
grep -E "dL=" "$R/d2_stage1_on.log" | tail -6
grep -E "dL=" "$R/d2_stage1_off.log" | tail -6
