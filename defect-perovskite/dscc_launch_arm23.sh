#!/usr/bin/env bash
# Arm 2+3 launch (after the Arm-1 decision routes A or D and the entry gates): writes the
# queue (Route A only unless ROUTES is set), sends the b3 share to the b3 queue runner
# (3 per GPU on 4 5 7) and keeps LOCAL_N runs for the local GPU.
#   bash defect-perovskite/dscc_launch_arm23.sh            # ROUTES=A PER_GPU=3 LOCAL_N=2 EPOCHS=30
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
ROUTES=${ROUTES:-A}; PER_GPU=${PER_GPU:-3}; LOCAL_N=${LOCAL_N:-2}; EPOCHS=${EPOCHS:-30}; SKIP=${SKIP:-}; SEEDS=${SEEDS:-}
# SEEDS: explicit comma-separated seed list (the Arm-1 seeds whose full H0 passed the precondition).
python "$W/defect-perovskite/dscc_make_arm23_queue.py" --routes "$ROUTES" --epochs "$EPOCHS" --skip_seeds "$SKIP" --seed_list "$SEEDS" --out "$HOME/runs/arm23_queue_all.txt"
head -n "$LOCAL_N" "$HOME/runs/arm23_queue_all.txt" > "$HOME/runs/arm23_queue_local.txt"
tail -n +$((LOCAL_N + 1)) "$HOME/runs/arm23_queue_all.txt" > "$HOME/runs/arm23_queue_b3.txt"
scp -q "$HOME/runs/arm23_queue_b3.txt" b3:~/runs/arm23_queue_b3.txt
ssh b3 "cd $W && setsid nohup bash defect-perovskite/dscc_queue.sh ~/runs/arm23_queue_b3.txt '4 5 7' $PER_GPU > ~/runs/arm23_queue_b3.log 2>&1 < /dev/null & sleep 1; echo b3 queue started"
setsid nohup bash "$W/defect-perovskite/dscc_queue.sh" "$HOME/runs/arm23_queue_local.txt" "0" "$LOCAL_N" > "$HOME/runs/arm23_queue_local.log" 2>&1 < /dev/null &
echo "local queue started ($LOCAL_N runs); b3 queue: $(wc -l < "$HOME/runs/arm23_queue_b3.txt") runs at $PER_GPU per GPU"
