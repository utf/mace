#!/usr/bin/env bash
# The acceptance test of section 7: is alpha on the defect shell FLAT in N?
#
# Everything else measured so far -- gap_l, size_viol, participation at one cell size --
# is upstream of the thing that actually matters. Only this says whether the correction
# survives a larger cell.
#
# Four models, chosen so the term's effect is attributable:
#   cal_off         gamma 0,   size off             control
#   cal_w1          gamma 0,   size 1.0             the ONLY run where the constraint closed
#   beta_seed_off   gamma 1.5, size off             seeded control
#   beta_seed_size  gamma 1.5, size 2.5e-5          seeded, constraint still violated
#
# cal_w1 is the real test. If closing the constraint does not flatten alpha, the term is
# not solving the problem it was designed for, whatever its own diagnostics say.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=""

for name in cal_off cal_w1 beta_seed_off beta_seed_size; do
    model="$HOME/runs/${name}/${name}.model"
    [ -f "$model" ] || { echo "missing $model"; continue; }
    OMP_NUM_THREADS=8 python -u "${HERE}/alpha_dilution.py" \
        --model "$model" --data-dir "${HERE}/dataset_full" --device cpu \
        --repeats 3,3,1 4,4,2 6,6,2 8,8,2 10,10,3 \
        > "$HOME/runs/dilution_${name}.log" 2>&1 &
done
wait
echo "dilution checks complete"
