#!/bin/bash
# Overnight watchdog for the W7.1 direct fine-tune, same contract as scratch_supervise.sh.
#   usage: ft_supervise.sh NAME GPU_INDEX TRAIN VALID
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
NAME=$1; GPU=$2; TRAIN=$3; VALID=$4
LOG=$HOME/runs/$NAME.log; SUP=$HOME/runs/$NAME.supervisor.log
MAX_RESTARTS=${MAX_RESTARTS:-6}; n=0
while true; do
  if grep -q "SCRATCH_EXIT" $LOG 2>/dev/null; then
    echo "$(date '+%F %T') $NAME finished, supervisor exiting" >> $SUP; exit 0
  fi
  if ! pgrep -f "ft_les_run.sh $NAME " > /dev/null 2>&1 && ! pgrep -f "work_dir.*$NAME" > /dev/null 2>&1; then
    if [ "$n" -ge "$MAX_RESTARTS" ]; then
      echo "$(date '+%F %T') $NAME gone after $n restarts, giving up" >> $SUP; exit 1
    fi
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v g=$GPU '$1==g {print $2}')
    if [ -z "$uuid" ]; then
      echo "$(date '+%F %T') $NAME: GPU $GPU not on the bus, waiting" >> $SUP; sleep 300; continue
    fi
    n=$((n+1))
    echo "$(date '+%F %T') $NAME: (re)launch $n on GPU $GPU ($uuid)" >> $SUP
    cd $W
    CUDA_VISIBLE_DEVICES=$uuid nohup bash $W/defect-perovskite/ft_les_run.sh "$NAME" "$TRAIN" "$VALID" >> $LOG 2>&1 &
    sleep 300
  fi
  sleep 120
done
