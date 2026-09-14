#!/bin/bash
# Stop a W7 run completely: supervisor first (so it cannot relaunch), then the wrapper, then
# the PYTHON TRAINER, which is the wrapper's child and survives a kill aimed at the wrapper.
#   usage: scratch_stop.sh NAME
N=$1
pkill -f "scratch_supervise.sh $N "; pkill -f "ft_supervise.sh $N "; sleep 2
pkill -f "scratch_run.sh $N "; pkill -f "ft_les_run.sh $N "; sleep 2
# ANCHOR THE NAME. `base_v3_les` is a prefix of `base_v3_les_zero`, and an unanchored
# `pkill -f base_v3_les` killed the zero-init run along with its target on 2026-09-14.
# The trainer's command line ends with the data paths, so the name is always followed by a
# space; matching "NAME " (and the end-of-word case) is enough and is exact.
pkill -f "\\b$N\\b"; sleep 6
pkill -9 -f "\\b$N\\b"; sleep 2
echo "remaining for $N:"; pgrep -af "\\b$N\\b" || echo "  none"
