#!/usr/bin/env bash
# Section 4: the forward-only steps, on the SIX EXISTING models, under the corrected kernel.
#
# READ-ONLY. Nothing here retrains and nothing here edits the model. Every output is an
# interpretive number or a config parameter for section 5 -- the plan is explicit that these
# feed the rerun as configuration, never as edits.
#
# The models were TRAINED under the retired self-image subtraction and are evaluated here
# under the full sum, so these numbers are "what the old weights predict through the corrected
# kernel". That is deliberately not the same thing as a retrained result: the per-species
# on-site potential has moved by order A_ii*Z/eps_inf, so eps0, Z and c are no longer at
# their fitted optimum. Section 5 is what settles the physics; this is what says how far the
# correction moves things before anything re-settles.
#
# Outputs carry an _fs tag so the pre-deletion numbers stay on disk for comparison.
# b3 GPUs 4-7 only.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$W" || exit 1

MODELS=("$R"/s3wired_models/s3_on_s*.model)
[ -e "${MODELS[0]}" ] || { echo "ABORT: no wired models in $R/s3wired_models"; exit 1; }

# The kernel must actually be the corrected one. A stale sync would re-measure the old
# convention under a new filename, which is worse than not running.
grep -q 'no longer accepts a self-potential' mace/modules/defect_madelung.py || {
    echo "ABORT: stale defect_madelung.py -- the self-image subtraction is still live"; exit 1; }
grep -q 'self_potential_of' mace/modules/defect_models.py && {
    echo "ABORT: defect_models.py still calls self_potential_of"; exit 1; }

echo "=== section 4 forward-only, ${#MODELS[@]} models, $(date +%F' '%H:%M:%S) ==="

CUDA_VISIBLE_DEVICES=4 python -u defect-perovskite/s3_dehead_trend.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s4_f4_fs.json" \
    > "$R/s4_f4_fs.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u defect-perovskite/s3_lambda_d.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s4_f5_fs.json" \
    > "$R/s4_f5_fs.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 python -u defect-perovskite/s3_dilution.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s4_dilution_fs.json" \
    > "$R/s4_dilution_fs.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 python -u defect-perovskite/s4_onsite_delta.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s4_onsite_delta.json" \
    > "$R/s4_onsite_delta.log" 2>&1 &
wait
echo "=== gates and deltas done $(date +%F' '%H:%M:%S); starting the two scans ==="

CUDA_VISIBLE_DEVICES=4 python -u defect-perovskite/s4_tel_scan.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s4_tel_scan.json" \
    > "$R/s4_tel_scan.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u defect-perovskite/s4_saturation.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s4_saturation.json" \
    > "$R/s4_saturation.log" 2>&1 &
wait
echo "=== section 4 complete $(date +%F' '%H:%M:%S) ==="
for f in s4_f4_fs s4_f5_fs s4_dilution_fs s4_onsite_delta s4_tel_scan s4_saturation; do
    echo; echo "--- $f ---"
    grep -vE "Warning|warn|^ *$|shape = |torch.load|openequivariance|alternative|visitor" \
        "$R/$f.log" | tail -14
done
