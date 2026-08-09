#!/usr/bin/env bash
# The four-model size-extensivity comparison, run once the size model finishes.
#
#   b_128ch_L1_s1        small data,  L1
#   bfull_128ch_L1_s2    larger data, L1                  (best L1 by dE)
#   efull_128ch_L1_s2    larger data, L1 + long range
#   esize_128ch_L1_s2    larger data, L1 + long range + size hinge (1e-4)
#
# Consecutive pairs differ by one thing each, so every step in the chart is attributable.
# alpha_dilution is single points only -- no relaxation -- so all four together cost minutes.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=""

RUNS=(b_128ch_L1_s1 bfull_128ch_L1_s2 efull_128ch_L1_s2 esize_128ch_L1_s2)

while pgrep -f "cli\.run_train.*esize_128ch" >/dev/null 2>&1; do sleep 120; done
sleep 30  # let the final model land on disk

for name in "${RUNS[@]}"; do
    model="$HOME/runs/${name}/${name}.model"
    [ -f "$model" ] || { echo "missing $model"; continue; }
    OMP_NUM_THREADS=8 python -u "${HERE}/alpha_dilution.py" \
        --model "$model" --data-dir "${HERE}/dataset_full" --device cpu \
        --repeats 3,3,1 4,4,2 6,6,2 8,8,2 10,10,3 12,12,3 \
        > "$HOME/runs/dilution_${name}.log" 2>&1 &
done
wait

python -u "${HERE}/plot_size_comparison.py" \
    --runs "${RUNS[@]}" \
    --labels "small data L1" "larger data L1" "+ long range" "+ size hinge" \
    --out "${HERE}/size_comparison.png" \
    > "$HOME/runs/size_comparison.txt" 2>&1
echo "comparison written to ${HERE}/size_comparison.png"
cat "$HOME/runs/size_comparison.txt"
