#!/usr/bin/env bash
# Plan v8 Stage 1.3 on b3, end to end, waiting on CONDITIONS (files), never on processes:
#   1. the forward-only arms' 18 model files (stage13_prepare) and the one-epoch training
#      smoke of arm (b) (s13smoke_s1.model) exist;
#   2. the gate queue on s13a, s13b, s13c in turn (GPUs 4-7 each);
#   3. the retrain queue (queue_stage13_train.sh), which starts each arm's gates itself.
#
#   nohup bash defect-perovskite/stage13_chain.sh > ~/runs/stage13_chain.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=$HOME/runs
export PYTHONPATH="$(cd "$HERE/.." && pwd)" PATH="$HOME/micromamba/envs/py13/bin:$PATH"

need=()
for arm in s13a s13b s13c; do for s in 1 2 3 4 5 6; do need+=("$R/${arm}_models/${arm}_s$s.model"); done; done
need+=("$R/s13smoke_s1/s13smoke_s1.model")
echo "=== waiting for the 18 prepared models and the smoke  $(date +%F' '%H:%M:%S) ==="
while true; do
    ok=1; for m in "${need[@]}"; do [ -f "$m" ] || ok=0; done
    [ "$ok" = 1 ] && break
    sleep 60
done
echo "=== present $(date +%F' '%H:%M:%S) ==="
grep -aE "Error|Traceback" "$R/s13smoke_s1.log" | head -3 && { echo "ABORT: the smoke logged an error"; exit 1; }

for arm in s13a s13b s13c; do
    TAG="$arm" bash "$HERE/queue_stage13_gates.sh" > "$R/${arm}_gates.log" 2>&1
    echo "=== $arm gates done $(date +%F' '%H:%M:%S) ==="
done
bash "$HERE/queue_stage13_train.sh" > "$R/stage13_train.log" 2>&1
echo "=== Stage 1.3 chain complete $(date +%F' '%H:%M:%S) ==="
