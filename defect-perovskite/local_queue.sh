#!/bin/bash
# Run a W3 queue file on the LOCAL A4000, one run at a time.
#
# One at a time on purpose: a training step peaks at 7.7-8.0 GB (measured on b3 with the
# float32 base), and two of those against the A4000's 16.4 GB leaves no headroom for the
# setup pass. The b3 runner packs two per card because those are 24 GB.
#
#   LQ_QUEUE=~/runs/w3a11_queue.txt LQ_LOG=~/runs/local_queue.log bash local_queue.sh
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
Q=${LQ_QUEUE:-$HOME/runs/w3a11_queue.txt}; L=${LQ_LOG:-$HOME/runs/local_queue.log}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  line=$(grep -v -E "^\s*(#|$)" $Q 2>/dev/null | head -1)
  if [ -z "$line" ]; then
    # WAIT, don't exit. Exiting on an empty queue has twice left this machine idle while b3
    # still had a backlog, because work appended later found no runner. LQ_IDLE_EXIT=1
    # restores the old behaviour for a one-shot batch.
    [ -n "${LQ_IDLE_EXIT:-}" ] && { echo "$(date '+%F %T') queue empty, exiting" >> $L; exit 0; }
    sleep "${LQ_POLL:-60}"; continue
  fi
  name=${line%%|*}; margs=${line#*|}
  echo "$(date '+%F %T') starting $name" >> $L
  python3 $W/defect-perovskite/dscc_train.py --name "$name" \
    --run_dir $HOME/runs/dscc/$name --device cuda $margs >> $HOME/runs/$name.log 2>&1
  echo "$(date '+%F %T') finished $name (rc $?)" >> $L
  python3 - "$Q" "$line" <<'PY'
import sys; q, line = sys.argv[1], sys.argv[2]
lines = open(q).read().splitlines(); i = next(k for k, l in enumerate(lines) if l.strip() == line.strip()); del lines[i]
open(q, "w").write("\n".join(lines) + ("\n" if lines else ""))
PY
done
