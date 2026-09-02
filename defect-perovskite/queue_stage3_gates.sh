#!/usr/bin/env bash
# Section 3, gates 1 and 3, on the SIX WIRED models -- not on the frozen-P ones.
#
#   F4  d(dE_head)/d(d_hub) on the 159-atom charged subset, per seed with its own CI.
#       Pass: right sign and within 3x of -0.134 eV/A.
#   F5  corr(lambda_frontier, d_hub) on the same subset. Pass: positive.
#
# Both refuse to pool cell sizes. The 79-atom energy targets carry M1b's base-extrapolation
# slope of +0.37 eV/A, which the frozen base cannot absorb; the 159-atom charged frames are
# the cut that speaks for the labels.
#
# The dilution gate is NOT here -- it needs the two-size matched frames and is scored
# separately.
#
# b3 GPUs 4-7 only. One card is enough: this is inference on 17 frames.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$W" || exit 1

MODELS=("$R"/s3wired_models/s3_on_s*.model)
if [ ! -e "${MODELS[0]}" ]; then
    echo "ABORT: no wired models in $R/s3wired_models -- section 2 has not finished"
    exit 1
fi
echo "=== scoring ${#MODELS[@]} wired models ==="
printf '  %s\n' "${MODELS[@]##*/}"

echo
echo "=== F4: dE_head against d, 159-atom subset ==="
CUDA_VISIBLE_DEVICES=4 python defect-perovskite/s3_dehead_trend.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s3wired_f4.json" \
    2>&1 | grep -vE "Warning|warn|^ *$"

echo
echo "=== F5: lambda_frontier against d, same subset ==="
CUDA_VISIBLE_DEVICES=4 python defect-perovskite/s3_lambda_d.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s3wired_f5.json" \
    2>&1 | grep -vE "Warning|warn|^ *$"
