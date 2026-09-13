#!/bin/bash
# Overnight supervisor for the W7 from-scratch runs. `--restart_latest` lets a run resume from
# its last checkpoint, but only if something restarts the PROCESS -- and on this machine a card
# that faults has killed the driver context of every job at once (2026-09-10), so an unattended
# night needs a watchdog, not just a resume flag.
#
# Relaunches NAME on the same GPU whenever its process is gone and its run has not finished
# (finished = the log's final error table has been written). Gives up after MAX_RESTARTS.
#   usage: scratch_supervise.sh NAME MODEL GPU_INDEX TRAIN VALID
#
# TO STOP A RUN, use scratch_stop.sh -- not a bare pkill on the wrapper. The trainer is a
# PYTHON CHILD of the wrapper script, so killing the wrapper alone ORPHANS it: on 2026-09-13
# that left two trainers per run writing the same log and the same checkpoint directory,
# interleaving epochs ("Epoch 16, Epoch 12, Epoch 17, Epoch 13") until it was caught.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
NAME=$1; MODEL=$2; GPU=$3; TRAIN=$4; VALID=$5
LOG=$HOME/runs/$NAME.log; SUP=$HOME/runs/$NAME.supervisor.log
MAX_RESTARTS=${MAX_RESTARTS:-8}; n=0
while true; do
  if grep -q "SCRATCH_EXIT" $LOG 2>/dev/null; then
    echo "$(date '+%F %T') $NAME finished (SCRATCH_EXIT seen), supervisor exiting" >> $SUP
    exit 0
  fi
  if ! pgrep -f "scratch_run.sh $NAME " > /dev/null 2>&1 && ! pgrep -f "work_dir.*$NAME" > /dev/null 2>&1; then
    if [ "$n" -ge "$MAX_RESTARTS" ]; then
      echo "$(date '+%F %T') $NAME gone and $n restarts used, giving up" >> $SUP
      exit 1
    fi
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v g=$GPU '$1==g {print $2}')
    if [ -z "$uuid" ]; then
      echo "$(date '+%F %T') $NAME: GPU $GPU not on the bus, waiting" >> $SUP
      sleep 300; continue
    fi
    n=$((n+1))
    echo "$(date '+%F %T') $NAME: (re)launch $n on GPU $GPU ($uuid)" >> $SUP
    cd $W
    CUDA_VISIBLE_DEVICES=$uuid nohup bash $W/defect-perovskite/scratch_run.sh \
      "$NAME" "$MODEL" "$TRAIN" "$VALID" >> $LOG 2>&1 &
    sleep 300
  fi
  sleep 120
done
