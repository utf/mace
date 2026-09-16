#!/bin/bash
# L0 queue on b3: one run per card on GPUs 5 and 7 ONLY (4 and 6 are off the bus; 0-3 are never
# used). Base v2 peaked 12-16 GB, so one per 24 GB card. One line of the queue file per run,
# `NAME|TRAIN VALID`, dropped as it is launched. Selects cards by UUID through b3_run.sh.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
Q=${W1LES_QUEUE:-$HOME/runs/w1les_queue.txt}; L=${W1LES_LOG:-$HOME/runs/w1les_queue.log}
GPUS=${W1LES_GPUS:-"5 7"}
while true; do
  line=$(grep -v -E "^\s*(#|$)" $Q 2>/dev/null | head -1)
  [ -z "$line" ] && { echo "$(date '+%F %T') queue empty, exiting" >> $L; exit 0; }
  for g in $GPUS; do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v g=$g '$1==g {print $2}')
    if [ -n "$used" ] && [ "$used" -lt 2000 ]; then
      uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v g=$g '$1==g {print $2}')
      name=${line%%|*}; args=${line#*|}
      echo "$(date '+%F %T') launching $name on GPU $g ($uuid): $args" >> $L
      bash $W/defect-perovskite/b3_run.sh "$uuid" "$name" bash $W/defect-perovskite/w1les_run.sh $name $args >> $L 2>&1
      python3 - "$Q" "$line" <<'PY'
import sys; q, line = sys.argv[1], sys.argv[2]
lines = open(q).read().splitlines(); i = next(k for k, l in enumerate(lines) if l.strip() == line.strip()); del lines[i]
open(q, "w").write("\n".join(lines) + ("\n" if lines else ""))
PY
      sleep 240; break
    fi
  done
  sleep 60
done
