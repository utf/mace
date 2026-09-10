#!/bin/bash
# Wait for b3 to return, verify its cards really run CUDA, then move queued seeds to it.
#
# WHY A CUDA PROBE AND NOT nvidia-smi: on 2026-09-10 GPU 7 fell off the bus and wedged the
# driver machine-wide. Cards 4 and 5 still enumerated perfectly in nvidia-smi while
# torch._C._cuda_init() failed, and six runs died together. Enumeration is not evidence.
#
# GPUs 0-3 are never used (standing instruction); candidates are 4, 5, 6, 7.
#
# DUPLICATION HAZARD, and how it is closed. `local_queue.sh` deletes a queue line when the
# run FINISHES, not when it starts, so the seed training right now is still line 1 of the
# file. Moving lines blindly would ship it to b3 and run the same seed twice. Any line whose
# run directory already exists locally is therefore skipped: that set is exactly {finished,
# in flight}. The local runner re-reads the file only between runs, so removing the other
# lines while a run is in flight is safe.
set -u
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
LOCAL_Q=${LOCAL_Q:-$HOME/runs/w3a11_local_queue.txt}
B3_Q=${B3_Q:-/home/alex/runs/w3a11_queue.txt}
LOG=${HANDOFF_LOG:-$HOME/runs/b3_handoff.log}
PROBE=${GPU_PROBE:-/tmp/gpu_probe.py}

echo "$(date '+%F %T') waiting for b3" >> $LOG
until timeout 8 ssh -o ConnectTimeout=5 -o BatchMode=yes b3 'echo alive' >/dev/null 2>&1; do sleep ${HANDOFF_POLL:-3600}; done
echo "$(date '+%F %T') b3 reachable; syncing and probing GPUs" >> $LOG
git -C $W archive HEAD | ssh b3 "tar -x -C $W --warning=no-timestamp"
scp -q "$PROBE" b3:/tmp/gpu_probe.py
USABLE=$(ssh b3 '/home/alex/micromamba/envs/py13/bin/python /tmp/gpu_probe.py 2>/dev/null | grep "^USABLE_GPUS=" | cut -d= -f2')
echo "$(date '+%F %T') usable GPUs on b3: [$USABLE]" >> $LOG
if [ -z "$USABLE" ]; then
  echo "$(date '+%F %T') no usable GPU on b3 -- everything stays local" >> $LOG
  exit 1
fi
NCARD=$(echo $USABLE | wc -w); SLOTS=$((NCARD * 2))
MOVED=$(python3 - "$LOCAL_Q" "$SLOTS" 2>/tmp/moved_lines.txt <<'PY'
import os, sys
q, n = sys.argv[1], int(sys.argv[2])
lines = [l for l in open(q).read().splitlines() if l.strip()]
started = [l for l in lines
           if os.path.isdir(os.path.expanduser("~/runs/dscc/" + l.split("|")[0]))]
free = [l for l in lines if l not in started]
move, keep = free[:n], started + free[n:]
open(q, "w").write("\n".join(keep) + ("\n" if keep else ""))
sys.stderr.write("\n".join(move))
print(len(move))
PY
)
echo "$(date '+%F %T') moving $MOVED runs to b3 ($NCARD cards, $SLOTS slots)" >> $LOG
if [ "$MOVED" -eq 0 ]; then echo "$(date '+%F %T') nothing free to move" >> $LOG; exit 0; fi
scp -q /tmp/moved_lines.txt b3:$B3_Q
ssh b3 "export W3_QUEUE=$B3_Q W3_LOG=/home/alex/runs/w3a11_queue.log W3_PER_GPU=2 W3_HEADROOM=8500 W3_GPUS='$USABLE'; nohup bash $W/defect-perovskite/w3_queue.sh > /home/alex/runs/w3a11_runner.log 2>&1 & sleep 5; echo started"
echo "$(date '+%F %T') b3 runner started on GPUs [$USABLE]" >> $LOG
