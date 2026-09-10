#!/bin/bash
# Wait for b3 to return, verify its cards really run CUDA, then move queued seeds to it.
#
# WHY A CUDA PROBE AND NOT nvidia-smi: on 2026-09-10 GPU 7 fell off the bus and wedged the
# driver machine-wide. Cards 4 and 5 still enumerated perfectly in nvidia-smi while
# torch._C._cuda_init() failed, and six runs died. Enumeration is not evidence.
#
# GPUs 0-3 are never used (standing instruction); candidates are 4, 5, 6, 7.
#
# The local runner re-reads its queue file only between runs, so deleting lines from it while
# a run is in flight is safe: the seed already running stays local and is never duplicated.
set -u
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
LOCAL_Q=${LOCAL_Q:-$HOME/runs/w3a11_local_queue.txt}
B3_Q=${B3_Q:-/home/alex/runs/w3a11_queue.txt}
LOG=${HANDOFF_LOG:-$HOME/runs/b3_handoff.log}

echo "$(date '+%F %T') waiting for b3" >> $LOG
until timeout 8 ssh -o ConnectTimeout=5 -o BatchMode=yes b3 'echo alive' >/dev/null 2>&1; do sleep 30; done
echo "$(date '+%F %T') b3 reachable; syncing and probing GPUs" >> $LOG
git -C $W archive HEAD | ssh b3 "tar -x -C $W --warning=no-timestamp"
scp -q $W/../../../../../tmp/gpu_probe.py b3:/tmp/gpu_probe.py 2>/dev/null || \
  scp -q ${GPU_PROBE:-/tmp/gpu_probe.py} b3:/tmp/gpu_probe.py
USABLE=$(ssh b3 '/home/alex/micromamba/envs/py13/bin/python /tmp/gpu_probe.py 2>/dev/null | grep "^USABLE_GPUS=" | cut -d= -f2')
echo "$(date '+%F %T') usable GPUs on b3: [$USABLE]" >> $LOG
if [ -z "$USABLE" ]; then
  echo "$(date '+%F %T') no usable GPU on b3 -- leaving everything local" >> $LOG
  exit 1
fi
NCARD=$(echo $USABLE | wc -w); SLOTS=$((NCARD * 2))
# Move up to SLOTS queued seeds to b3; the rest stay local. The seed currently training
# locally is already out of the queue file, so it cannot be picked up twice.
MOVED=$(python3 - "$LOCAL_Q" "$SLOTS" <<'PY'
import sys
q, n = sys.argv[1], int(sys.argv[2])
lines = [l for l in open(q).read().splitlines() if l.strip()]
move, keep = lines[:n], lines[n:]
open(q, "w").write("\n".join(keep) + ("\n" if keep else ""))
sys.stderr.write("\n".join(move))
print(len(move))
PY
2> /tmp/moved_lines.txt)
echo "$(date '+%F %T') moving $MOVED runs to b3 ($NCARD cards, $SLOTS slots)" >> $LOG
[ "$MOVED" -eq 0 ] && { echo "$(date '+%F %T') nothing left to move" >> $LOG; exit 0; }
scp -q /tmp/moved_lines.txt b3:$B3_Q
ssh b3 "export W3_QUEUE=$B3_Q W3_LOG=/home/alex/runs/w3a11_queue.log W3_PER_GPU=2 W3_HEADROOM=8500 W3_GPUS='$USABLE'; nohup bash $W/defect-perovskite/w3_queue.sh > /home/alex/runs/w3a11_runner.log 2>&1 & sleep 5; echo started"
echo "$(date '+%F %T') b3 runner started on GPUs [$USABLE]" >> $LOG
