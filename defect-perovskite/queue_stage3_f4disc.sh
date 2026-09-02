#!/usr/bin/env bash
# F4's follow-up: head or base? b3 GPUs 4-7 only.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$W" || exit 1
MODELS=("$R"/s3wired_models/s3_on_s*.model)
[ -e "${MODELS[0]}" ] || { echo "ABORT: no wired models"; exit 1; }
CUDA_VISIBLE_DEVICES=5 python -u defect-perovskite/s3_f4_discriminate.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s3wired_f4disc.json" \
    2>&1 | grep --line-buffered -vE "Warning|warn|^ *$|shape = |torch.load|openequivariance|alternative"
