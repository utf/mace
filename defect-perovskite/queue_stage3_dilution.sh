#!/usr/bin/env bash
# Section 3, gate 2: R_model dilution on the two-size frames, bound-conditioned.
# b3 GPUs 4-7 only. One card; this is inference.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$W" || exit 1

MODELS=("$R"/s3wired_models/s3_on_s*.model)
if [ ! -e "${MODELS[0]}" ]; then
    echo "ABORT: no wired models in $R/s3wired_models"; exit 1
fi
echo "=== dilution gate on ${#MODELS[@]} wired models ==="
CUDA_VISIBLE_DEVICES=4 python defect-perovskite/s3_dilution.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s3wired_dilution.json" \
    2>&1 | grep -vE "Warning|warn|^ *$|shape = |torch.load|openequivariance|alternative"
