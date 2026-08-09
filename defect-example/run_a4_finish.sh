#!/usr/bin/env bash
# Bring the Stage A4 baseline to a uniform 40 epochs on all three seeds.
# Seeds 1-2 are already past 40's worth of work and are stopped once they reach it;
# seed 3 died of CUDA OOM at epoch 14 (concurrent benchmarks on the same GPU) and is
# rerun from scratch, alone, so it cannot be starved again.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

until [ "$(grep -cE 'Epoch 39: head' "$HOME/runs/a4_v4_s1.log" 2>/dev/null || echo 0)" -ge 1 ] \
   && [ "$(grep -cE 'Epoch 39: head' "$HOME/runs/a4_v4_s2.log" 2>/dev/null || echo 0)" -ge 1 ]; do
    sleep 20
done
for p in $(pgrep -f "name=a4_v4_s1" ; pgrep -f "name=a4_v4_s2"); do kill "$p" 2>/dev/null || true; done
sleep 10
echo "seeds 1-2 stopped at 40 epochs"

rm -rf "$HOME/runs/a4_v4_s3" "$HOME/runs/a4_v4_s3.log"
NAME=a4_v4_s3 WORK_DIR="$HOME/runs/a4_v4_s3" DATA_DIR="${HERE}/dataset" \
MAX_NUM_EPOCHS=40 NUM_CHANNELS=8 MAX_L=0 NUM_RADIAL_BASIS=4 R_MAX=4.0 \
BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float32 \
USE_EMA=False PATIENCE=250 SEED=3 \
"${HERE}/train_defect_model.sh" > "$HOME/runs/a4_v4_s3.log" 2>&1
echo "seed 3 complete"

exec env EPOCHS=40 "${HERE}/run_stage_d.sh"
