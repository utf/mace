#!/usr/bin/env bash
# R2, local cell 2: H3 x bandwidth-anneal x 8 seeds.
#
# The noanneal cell returned a clean 4/4 negative -- ratio 0.844-0.875 against a 0.80
# threshold, with the base FROZEN through epoch 30, so the head had an uncontested chance to
# localise and did not. Seeds 5-8 of that cell were skipped as unable to change the answer.
#
# Anneal is now the interesting arm: hoppings are scaled by s(e) = 4^(1 - e/20) for e <= 20,
# so the band starts FOUR TIMES WIDER than its trained width and narrows to it by epoch 20.
# A level can then separate from the band gradually, rather than having to tunnel out of an
# already-converged delocalised solution -- which is what the noanneal cell got stuck in.
#
# This arm has never actually run. The schedule was declared, passed and documented, but
# nothing assigned hop_scale, so it sat at 1.0 and the anneal arm was byte-identical to the
# plain arm. Wired and tested before this cell was launched.
#
# Run in waves of 4 (the measured footprint was 3054 MiB/run, so 4 x 3054 fits 16 GB with
# headroom). If wave 1 is another clean negative, kill the scheduler -- NOT the running
# subshells -- exactly as the noanneal cell was cut short:
#     ps -eo pid,ppid,args | grep run_r2.sh     # the scheduler is the child of THIS script
#     kill <scheduler pid>
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$HOME/runs"
exec >> "$HOME/runs/local_r2_anneal.log" 2>&1
echo "=== local R2 anneal cell queued $(date +%F' '%H:%M:%S) ==="

# Bracket expression so this script's own command line cannot match the pattern it waits on;
# a plain "noanneal" here would match the pgrep in any monitoring shell and never release.
echo "waiting for the noanneal seeds still in flight..."
while pgrep -f "name=r2_h3_noannea[l]" > /dev/null; do sleep 60; done
echo "noanneal cell clear at $(date +%H:%M:%S); settling"
sleep 45

cd "$WT" || exit 1
HEADS=h3 ARMS=anneal SEEDS="1 2 3 4 5 6 7 8" GPUS_LIST=0 RUNS_PER_GPU=4 EPOCHS=50 \
    bash defect-perovskite/run_r2.sh
echo "=== local R2 anneal cell complete $(date +%F' '%H:%M:%S) ==="
