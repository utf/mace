#!/usr/bin/env bash
# Arm 2+3 launch (v4.3: full H0 winners admitted on the C5 gate; Route A unless ROUTES is
# set; 60 epochs, the Arm-1 protocol). Writes the queue, ships the b3 share to the b3 queue
# runner (PER_GPU per GPU on 4 5 7) and keeps LOCAL_N runs for the local GPU, which are
# started from a snapshot of HEAD (SNAP) so that edits to the worktree cannot touch running
# code. The converted H0 states (h0_state.pt) are copied to b3 first.
#   SEEDS=0,1,2,3,4,6,7,9,10 EPOCHS=60 bash defect-perovskite/dscc_launch_arm23.sh
#   LOCAL_START=0 leaves the local queue file in place without starting it.
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
B3_W=/home/alex/src/mace/.claude/worktrees/size-extensivity
ROUTES=${ROUTES:-A}; PER_GPU=${PER_GPU:-3}; LOCAL_N=${LOCAL_N:-2}; EPOCHS=${EPOCHS:-60}; SKIP=${SKIP:-}; SEEDS=${SEEDS:-}
LOCAL_START=${LOCAL_START:-1}; SNAP=${SNAP:-$HOME/runs/dscc_src_arm23}
python "$W/defect-perovskite/dscc_make_arm23_queue.py" --routes "$ROUTES" --epochs "$EPOCHS" --skip_seeds "$SKIP" --seed_list "$SEEDS" --out "$HOME/runs/arm23_queue_all.txt"
head -n "$LOCAL_N" "$HOME/runs/arm23_queue_all.txt" > "$HOME/runs/arm23_queue_local.txt"
tail -n +$((LOCAL_N + 1)) "$HOME/runs/arm23_queue_all.txt" > "$HOME/runs/arm23_queue_b3.txt"
# the admitted seeds' converted H0 states, to the same paths on b3
for s in ${SEEDS//,/ }; do
  ssh b3 "mkdir -p /home/alex/runs/dscc/dscc_arm1_full_s$s"
  scp -q "$HOME/runs/dscc/dscc_arm1_full_s$s/h0_state.pt" "b3:/home/alex/runs/dscc/dscc_arm1_full_s$s/h0_state.pt"
done
scp -q "$HOME/runs/arm23_queue_b3.txt" b3:~/runs/arm23_queue_b3.txt
ssh b3 "setsid nohup bash $B3_W/defect-perovskite/dscc_queue.sh /home/alex/runs/arm23_queue_b3.txt '4 5 7' $PER_GPU > /home/alex/runs/arm23_queue_b3.log 2>&1 < /dev/null & sleep 1; echo b3 queue started"
if [ "$LOCAL_START" = "1" ]; then
  rm -rf "$SNAP"; mkdir -p "$SNAP"
  (cd "$W" && git archive HEAD | tar -x -C "$SNAP")
  for d in dataset_pbe dataset_cf; do rm -rf "$SNAP/defect-perovskite/$d"; ln -s "$W/defect-perovskite/$d" "$SNAP/defect-perovskite/$d"; done
  setsid nohup bash "$SNAP/defect-perovskite/dscc_queue.sh" "$HOME/runs/arm23_queue_local.txt" "0" "$LOCAL_N" > "$HOME/runs/arm23_queue_local.log" 2>&1 < /dev/null &
  echo "local queue started ($LOCAL_N runs from $SNAP)"
fi
echo "b3 queue: $(wc -l < "$HOME/runs/arm23_queue_b3.txt") runs at $PER_GPU per GPU; local: $(wc -l < "$HOME/runs/arm23_queue_local.txt")"
