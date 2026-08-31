#!/usr/bin/env bash
# R2, local cell: H3 x noanneal x 8 seeds, run entirely on this machine.
#
# b3's GPU3 (PCI 1E:00.0) failed and wedged CUDA init machine-wide, killing all 32 R2 runs at
# startup. The design is therefore split by CELL, not by seed: a cell never straddles two GPU
# architectures. The A4000 here is Ampere (TF32 matmuls) and b3's cards are Turing, and a
# seed-level split would put that hardware difference inside the very comparison the screen
# makes. This machine owns H3/noanneal; b3 takes the other three cells when it returns.
#
# Concurrency is MEASURED, not assumed: seed 1 runs alone first and the rest size themselves
# from its actual footprint. The 10 A Hamiltonian is heavier than the base runs, and an OOM
# discovered three hours in costs more than the probe does.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$HOME/runs"
exec >> "$HOME/runs/local_r2_cell.log" 2>&1
echo "=== local R2 cell starting $(date +%F' '%H:%M:%S) ==="

echo "waiting for the cross-fit bases to release the GPU..."
while pgrep -f "run_train --name=cf_base" > /dev/null; do sleep 60; done
echo "cross-fit bases finished at $(date +%H:%M:%S); settling"
sleep 45

cd "$WT" || exit 1

HEADS=h3 ARMS=noanneal SEEDS=1 GPUS_LIST=0 RUNS_PER_GPU=1 EPOCHS=50 \
    nohup bash defect-perovskite/run_r2.sh > "$HOME/runs/r2_local_probe.log" 2>&1 &

for _ in $(seq 1 60); do
    sleep 30
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${U:-0}" -gt 2500 ] && break
done
sleep 180
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "single-run footprint: ${USED} MiB of ${TOTAL} MiB"

if [ "${USED:-0}" -lt 500 ]; then
    echo "probe never allocated -- check $HOME/runs/r2_h3_noanneal_s1.log"
    exit 1
fi

# ~1.5 GB headroom; cap at 4 so the eigensolver stays responsive.
N=$(( (TOTAL - 1500) / USED ))
[ "$N" -gt 4 ] && N=4
[ "$N" -lt 2 ] && N=2
EXTRA=$(( N - 1 ))
echo "concurrency ${N} total -> launching seeds 2-8 at ${EXTRA} alongside the probe"

HEADS=h3 ARMS=noanneal SEEDS="2 3 4 5 6 7 8" GPUS_LIST=0 RUNS_PER_GPU="$EXTRA" EPOCHS=50 \
    bash defect-perovskite/run_r2.sh > "$HOME/runs/r2_local_rest.log" 2>&1
wait
echo "=== local R2 cell complete $(date +%F' '%H:%M:%S) ==="
