#!/usr/bin/env bash
# D-series: diagnostics on existing checkpoints, no retraining. Section 1 of the
# Madelung-in-H plan, run BEFORE any Stage-1 training.
#
# Cohort: the 12 A/B cells (delocalised-but-fitting) plus F2's six localised cells, so the
# defect-near/bulk-like contrast is measured across both behaviours rather than within one.
#
# b3 GPUs 4-7 only.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$W" || exit 1

AB=("$R"/ab_models/v3_o*_fm0.2_gap2.4_s*.model)
# F2's six localised cells, named explicitly -- they are a measured subset, not a glob.
LOC=("$R/tbv3_models/v3_fm0.2_gap2.2_s4.model" "$R/tbv3_models/v3_fm0.2_gap2.2_s5.model" \
     "$R/tbv3_models/v3_fm0.2_gap2.4_s5.model" "$R/tbv3_models/v3_fm0.2_gap2.4_s6.model" \
     "$R/tbv3_models/v3_fm0.3_gap2.4_s1.model" "$R/tbv3_models/v3_fm0.3_gap2.4_s4.model")
ALL=("${AB[@]}" "${LOC[@]}")
echo "cohort: ${#ALL[@]} models"

echo "=== D-series starting $(date +%F' '%H:%M:%S) ==="
CUDA_VISIBLE_DEVICES=4 python defect-perovskite/d1_sensitivity.py \
    --models "${ALL[@]}" --limit 16 --n-pristine 12 --probes 8 \
    --device cuda --out "$R/d1_sensitivity.json" > "$R/d1_sensitivity.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=5 python defect-perovskite/d2_bins.py \
    --models "${ALL[@]}" --limit 24 --n-pristine 8 \
    --device cuda --out "$R/d2_bins.json" > "$R/d2_bins.log" 2>&1 &
P2=$!
wait $P1; echo "  d1 exit $?"
wait $P2; echo "  d2 exit $?"
echo "=== D-series complete $(date +%F' '%H:%M:%S) ==="
tail -6 "$R/d1_sensitivity.log"
tail -6 "$R/d2_bins.log"
