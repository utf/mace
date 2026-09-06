#!/usr/bin/env bash
# Wave 2 of Arm 1: waits until the b3 wave-1 runs are done (no dscc_train.py process left),
# then launches the remaining five runs on GPUs 4 5 7 (two per GPU). Run detached on b3:
#   nohup bash defect-perovskite/dscc_queue_arm1_wave2.sh > ~/runs/dscc_queue_wave2.log 2>&1 &
W=$(cd "$(dirname "$0")/.." && pwd)
while pgrep -f "[d]scc_train.py --name dscc_arm1" > /dev/null; do sleep 300; done
echo "$(date): wave 1 finished, launching wave 2"
bash "$W/defect-perovskite/dscc_wave_arm1.sh" "4 5 7" "ctrl:3 full:4 ctrl:4 full:5 ctrl:5"
