#!/usr/bin/env bash
# Finite-size convergence ladder, 256 -> 2400 atoms, reproducing the NEP paper's figure.
#
# Runs on CPU by design: the GPU is occupied by training, and this is a long job made of
# many small force calls rather than one big one, so it parallelises across cores well.
# Move it to --device cuda once the GPU is free -- it will be far faster there.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${THREADS:-32}"
MODEL="${MODEL:-$HOME/runs/b_128ch_L1_s1/b_128ch_L1_s1.model}"
python -u "${HERE}/defect_size_extensivity.py" \
    --model "$MODEL" \
    --repeats ${REPEATS:-4,4,2 5,5,2 6,6,2 8,8,2 10,10,3} \
    --device cpu --fmax "${FMAX:-0.02}" --steps "${STEPS:-500}" \
    --out-json "${HERE}/size_convergence.json" \
    --out-plot "${HERE}/size_convergence.png"
