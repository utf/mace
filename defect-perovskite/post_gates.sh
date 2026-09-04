#!/usr/bin/env bash
# What the running chain cannot do for itself, for one arm:  post_gates.sh <tag>
#
# The A/B chain parsed its gates() body at 10:31 on 4 Sep, before gate 10 and before c11
# and c12 existed. Rather than edit a script a running bash is reading by byte offset --
# which this cycle would have aborted the chain after wave 1 -- the missing steps live here
# and are launched by hand after each `=== gates for <tag> ===` stage finishes.
#
#   §2.2   c11_site_deviation   the per-site charge deviation per seed
#   gate 10 c3_tiling_drift     ideal AND thermal tiles with the bound switch (arm B only)
#   table  c12_gate_table       the ten-gate table, assembled from the JSON
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
R=$HOME/runs
export PYTHONPATH=$REPO PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$REPO" || exit 1

TAG="${1:-${TAG:-arma}}"
models=("$R/${TAG}_models/${TAG}_s"*.model)
[ -e "${models[0]}" ] || { echo "ABORT: no models under $R/${TAG}_models"; exit 1; }
echo "=== post-gates for $TAG: ${#models[@]} models  $(date +%F' '%H:%M:%S) ==="

echo "=== §2.2 per-site charge deviation, $TAG ==="
CUDA_VISIBLE_DEVICES=4 python -u "$HERE/c11_site_deviation.py" \
    --models "${models[@]}" --device cuda --out "$R/${TAG}_site_dev.json" 2>&1 | tail -25 || true

if [ "$TAG" = "armb" ]; then
    echo "=== gate 10: tiling drift, ideal and thermal, with the bound switch ==="
    CUDA_VISIBLE_DEVICES=5 python -u "$HERE/c3_tiling_drift.py" \
        --models "${models[@]}" --thermal 3 \
        --device cuda --out "$R/${TAG}_tiling.json" 2>&1 | tail -60 || true
fi

echo "=== the ten-gate table, $TAG ==="
python -u "$HERE/c12_gate_table.py" --tag "$TAG" --out "$R/${TAG}_gates.md" 2>&1 | tail -40 || true
echo "=== post-gates complete $(date +%F' '%H:%M:%S) ==="
