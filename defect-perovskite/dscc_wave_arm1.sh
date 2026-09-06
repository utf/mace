#!/usr/bin/env bash
# Arm 1 (plan section 7, v4.2): full H0 vs scalar-only control, Phi = 0, six seeds each,
# outer fold = seed mod 4. Launches the runs listed in RUNS on the GPUs listed in GPUS,
# two per GPU, through b3_run.sh (one log per run under ~/runs). Usage on b3:
#   bash defect-perovskite/dscc_wave_arm1.sh "4 5 7" "full:0 ctrl:0 full:1 ctrl:1 full:2 ctrl:2"
# and locally: bash defect-perovskite/dscc_wave_arm1.sh "0" "full:3"
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
GPUS=($1); RUNS=($2)
EPOCHS=${EPOCHS:-60}
slot=0
for run in "${RUNS[@]}"; do
  kind=${run%%:*}; seed=${run##*:}
  fold=$((seed % 4))
  if [ "$kind" = "full" ]; then directional=1; else directional=0; fi
  name="dscc_arm1_${kind}_s${seed}"
  gpu=${GPUS[$((slot % ${#GPUS[@]}))]}
  # CUDA_VISIBLE_DEVICES by UUID: with GPU 6 off the bus, CUDA's own enumeration no longer
  # matches nvidia-smi's indices (index 7 raised "No CUDA GPUs are available").
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$gpu" '$1==g {print $2}')
  bash "$W/defect-perovskite/b3_run.sh" "${uuid:-$gpu}" "$name" \
    python "$W/defect-perovskite/dscc_train.py" --name "$name" --seed "$seed" --fold "$fold" \
      --directional "$directional" --coupling 0 --route_b 0 --epochs "$EPOCHS" --device cuda \
      --run_dir "$HOME/runs/dscc/$name"
  slot=$((slot + 1))
  sleep 2
done
