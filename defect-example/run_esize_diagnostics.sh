#!/usr/bin/env bash
# Parity and the finite-size convergence ladder for the L1 + long-range + size-hinge model.
#
# Parity goes on the GPU (float32, now free); the ladder stays on CPU in float64. That is
# not a fallback -- this card runs float64 at 1/32 rate, and the ladder is measuring drifts
# of a few meV on total energies of ~-37,000 eV, where float32 noise is ~2 meV.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
MODEL="$HOME/runs/esize_128ch_L1_s2/esize_128ch_L1_s2.model"

(
    export CUDA_VISIBLE_DEVICES=0
    python -u "${HERE}/plot_parity.py" --model "$MODEL" --data-dir "${HERE}/dataset_full" \
        --split valid --out "${HERE}/parity_esize_valid.png" --drop-tiny-cells 100
    python -u "${HERE}/plot_parity.py" --model "$MODEL" --data-dir "${HERE}/dataset_full" \
        --split train --out "${HERE}/parity_esize_train.png" --drop-tiny-cells 100
) > "$HOME/runs/parity_esize.log" 2>&1 &

(
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=40
    python -u "${HERE}/defect_size_extensivity.py" --model "$MODEL" \
        --data-dir "${HERE}/dataset_full" \
        --repeats 4,4,2 5,5,2 6,6,2 8,8,2 10,10,3 12,12,3 14,14,3 \
        --device cpu --fmax 0.03 --steps 400 \
        --out-json "${HERE}/size_convergence_esize.json" \
        --out-plot "${HERE}/size_convergence_esize.png"
) > "$HOME/runs/ladder_esize.log" 2>&1 &

wait
echo "esize diagnostics complete"
