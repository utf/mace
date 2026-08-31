#!/usr/bin/env bash
# Wait for b3's CUDA driver to come back, then submit the rest of the R2 matrix.
#
# GPU3 (PCI 0000:1E:00.0) failed and wedged driver init machine-wide; all 32 R2 runs died at
# CUDA startup and clearing it needs root, which this session does not have. So the launch is
# queued instead of abandoned: when someone reboots b3, the 24 runs go in by themselves.
#
# Three things this deliberately does NOT do:
#   * run on b3 -- a watcher there would die in the very reboot it is waiting for, so it polls
#     from the machine that owns the local cell;
#   * trust nvidia-smi -- it listed all seven surviving cards throughout the outage, so health
#     is decided by a real allocation (probe_gpus.py);
#   * hardcode GPU indices -- "0 1 2 3" is exactly how seed 8 landed on the dead card. Indices
#     renumber if GPU3 stays off the bus, so the four are chosen from the probe at launch time.
#
# The user's cap of at most 4 GPUs in use on b3 is enforced here and is not a tunable.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
REMOTE=b3
REMOTE_REPO=/home/alex/src/mace
REMOTE_PY=/home/alex/micromamba/envs/py13/bin/python
MARKER="$HOME/runs/.r2_b3_submitted"
LOG="$HOME/runs/watch_b3.log"
MAX_GPUS=4
POLL=300
DEADLINE=$(( $(date +%s) + 48 * 3600 ))

exec >> "$LOG" 2>&1
echo "=== b3 watcher started $(date +%F' '%H:%M:%S), deadline +48h ==="

if [ -f "$MARKER" ]; then
    echo "marker $MARKER exists; R2 already submitted to b3. Exiting."
    exit 0
fi

ssh_probe () {
    # Keep only the status tokens. torch prints a multi-line NVML warning on a wedged driver,
    # which would otherwise be the entire heartbeat line for 48 hours.
    timeout 180 ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" \
        "CUDA_DEVICE_ORDER=PCI_BUS_ID $REMOTE_PY -" < "${HERE}/probe_gpus.py" 2>&1 \
        | grep -E '^(HEALTHY|BAD|NO_DEVICES|DRIVER_WEDGED|NO_TORCH)' \
        || echo "UNREACHABLE"
}

while :; do
    now=$(date +%s)
    if [ "$now" -ge "$DEADLINE" ]; then
        echo "$(date +%H:%M:%S) deadline reached; giving up. Launch by hand with:"
        echo "  HEADS=h3 ARMS=anneal GPUS_LIST='<healthy>' bash defect-perovskite/run_r2.sh"
        exit 1
    fi

    out="$(ssh_probe)"
    healthy=($(printf '%s\n' "$out" | awk '$1=="HEALTHY"{print $2}'))
    echo "$(date +%H:%M:%S) heartbeat: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-140)"

    if [ "${#healthy[@]}" -ge "$MAX_GPUS" ]; then
        GPUS="${healthy[*]:0:$MAX_GPUS}"
        echo "$(date +%H:%M:%S) b3 is back; ${#healthy[@]} healthy GPUs, using: $GPUS"
        break
    fi
    sleep "$POLL"
done

echo "--- syncing the committed tree (the outage ran on a stale checkout) ---"
if ! rsync -az --info=stats1 /home/alex/src/mace/ "${REMOTE}:${REMOTE_REPO}/" ; then
    echo "rsync FAILED; not launching."; exit 1
fi

echo "--- verifying the flag chain on b3 itself, rather than trusting a commit hash ---"
if ! timeout 900 ssh -o BatchMode=yes "$REMOTE" \
        "cd ${REMOTE_REPO}/.claude/worktrees/size-extensivity && \
         $REMOTE_PY -m pytest tests/unit/test_flag_plumbing.py -q 2>&1 | tail -3"; then
    echo "flag-plumbing tests FAILED on b3; not launching -- a run whose saved artefact"
    echo "disagrees with its training config is worse than no run."
    exit 1
fi

# The two batches must be CHAINED, not both launched: separate run_r2.sh invocations each
# enforce their own concurrency, so submitting them together would put 16 runs on 4 cards.
# One remote shell runs H3/anneal to completion, then the 16 H2 runs. H3 goes first per the
# plan's sequencing -- it was the best free-attention fit in R1.
echo "--- submitting the 24-run complement on GPUs ${GPUS} ---"
timeout 600 ssh -o BatchMode=yes "$REMOTE" "cat > \$HOME/r2_complement.sh" <<REMOTE_EOF
#!/usr/bin/env bash
set -uo pipefail
cd ${REMOTE_REPO}/.claude/worktrees/size-extensivity || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "=== R2 complement on GPUs ${GPUS}, started \$(date +%F' '%H:%M:%S) ==="
HEADS=h3 ARMS=anneal SEEDS="1 2 3 4 5 6 7 8" GPUS_LIST="${GPUS}" RUNS_PER_GPU=2 EPOCHS=50 \\
    bash defect-perovskite/run_r2.sh
echo "=== H3/anneal done \$(date +%F' '%H:%M:%S); starting H2 ==="
HEADS=h2 ARMS="noanneal anneal" SEEDS="1 2 3 4 5 6 7 8" GPUS_LIST="${GPUS}" \\
    RUNS_PER_GPU=2 EPOCHS=50 bash defect-perovskite/run_r2.sh
echo "=== R2 complement complete \$(date +%F' '%H:%M:%S) ==="
REMOTE_EOF

timeout 600 ssh -o BatchMode=yes "$REMOTE" \
    "nohup bash \$HOME/r2_complement.sh > \$HOME/runs/r2_complement.log 2>&1 &
     sleep 5; echo submitted"

sleep 300
echo "--- verifying the launch ---"
timeout 120 ssh -o BatchMode=yes "$REMOTE" '
    echo "training procs: $(pgrep -c -f "python -m mace.cli.run_train")"
    echo "logs with the release scheduled: $(grep -l "Base release scheduled at epoch 30" \
        $HOME/runs/r2_h3_anneal_s*.log 2>/dev/null | wc -l)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -8
    tail -3 $HOME/runs/r2_complement.log'

date +%F' '%H:%M:%S > "$MARKER"
echo "$GPUS" >> "$MARKER"
echo "=== R2 complement submitted to b3 $(date +%F' '%H:%M:%S) ==="
