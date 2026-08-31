#!/usr/bin/env bash
# Wait for the four cross-fit bases, then run the T2 null control.
#
# CPU-bound on purpose. The GPU is running the local R2 cell, and that cell sizes its own
# concurrency from a whole-device memory reading -- an analysis process sharing the card would
# inflate the measurement and silently shrink the cell's concurrency. Threads are capped for
# the same reason on the CPU side: the training runs need cores for data loading.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$HOME/runs"
exec >> "$HOME/runs/t2_crossfit.log" 2>&1
echo "=== T2 cross-fit analysis queued $(date +%F' '%H:%M:%S) ==="

while pgrep -f "run_train --name=cf_[b]ase" > /dev/null; do sleep 60; done
echo "bases finished at $(date +%H:%M:%S)"
sleep 20

for k in 0 1 2 3; do
    m="$HOME/runs/cf_base_f${k}/cf_base_f${k}.model"
    if [ ! -f "$m" ]; then
        echo "FAILED: $m never appeared; check $HOME/runs/cf_base_f${k}.log"
        exit 1
    fi
done
echo "all four fold bases present"

cd "$WT" || exit 1
export PYTHONPATH="$WT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$HOME/micromamba/envs/py13/bin/python" defect-perovskite/t2_crossfit.py --device cpu
echo "=== T2 cross-fit analysis done $(date +%F' '%H:%M:%S) ==="
