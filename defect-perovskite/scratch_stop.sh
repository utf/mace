#!/bin/bash
# Stop a W7 run completely: supervisor first (so it cannot relaunch), then the wrapper, then
# the PYTHON TRAINER, which is the wrapper's child and survives a kill aimed at the wrapper.
#   usage: scratch_stop.sh NAME
N=$1
pkill -f "scratch_supervise.sh $N "; pkill -f "ft_supervise.sh $N "; sleep 2
pkill -f "scratch_run.sh $N "; pkill -f "ft_les_run.sh $N "; sleep 2
pkill -f "$N"; sleep 6
pkill -9 -f "$N"; sleep 2
echo "remaining for $N:"; pgrep -af "$N" || echo "  none"
