#!/usr/bin/env bash
# Section 6: the standing gate table, on the SECTION-5 models (full-sum kernel).
# F4 + F5 on one card, dilution on another. b3 GPUs 4-7 only.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$W" || exit 1
MODELS=("$R"/s5_models/s3_on_s*.model)
[ -e "${MODELS[0]}" ] || { echo "ABORT: no section-5 models in $R/s5_models"; exit 1; }
echo "=== section 6 gates on ${#MODELS[@]} models, $(date +%F' '%H:%M:%S) ==="

CUDA_VISIBLE_DEVICES=4 python -u defect-perovskite/s3_dehead_trend.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s6_f4.json" > "$R/s6_f4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u defect-perovskite/s3_lambda_d.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s6_f5.json" > "$R/s6_f5.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 python -u defect-perovskite/s3_dilution.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s6_dilution.json" \
    > "$R/s6_dilution.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 python -u defect-perovskite/s3_f4_discriminate.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s6_f4disc.json" \
    > "$R/s6_f4disc.log" 2>&1 &
wait
echo "=== section 6 complete $(date +%F' '%H:%M:%S) ==="
for f in s6_f4 s6_f5 s6_dilution s6_f4disc; do
    echo; echo "--- $f ---"
    grep -vE "Warning|warn|^ *$|shape = |torch.load|openequivariance|alternative|visitor|WARNING" \
        "$R/$f.log" | tail -12
done
