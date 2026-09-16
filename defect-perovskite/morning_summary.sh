#!/bin/bash
now=$(date +%s); target=$(date -d "tomorrow 07:30" +%s); sleep $((target - now))
O=$HOME/runs/morning_summary.txt; W=/home/alex/src/mace/.claude/worktrees/size-extensivity
{ echo "== morning summary $(date '+%F %T')"; for f in d1_watch d1_guard d2_watch d2_queue2 lx_watch; do echo "-- $f.log"; tail -6 $HOME/runs/$f.log 2>/dev/null; done
  echo "-- D1/D2 final rows"; for a in d1_light2 d1_full1 d2_light2 d2_full1; do echo "$a: $(grep -a 'Epoch' $HOME/runs/$a.log 2>/dev/null | grep Default | tail -1 | cut -c30-120)"; done
  echo "-- L1 seeds"; for s in 0 1 2 3 4 6; do d=$HOME/runs/dscc/dscc_l1_s$s; [ -d $d ] && echo "s$s: $(grep -a 'epoch' $d/train.log 2>/dev/null | tail -1 | cut -c1-110)"; done
  ssh -o BatchMode=yes b3 'for s in 3 4 6; do d=~/runs/dscc/dscc_l1_s$s; [ -d $d ] && echo "b3 s$s: $(grep -a epoch $d/train.log 2>/dev/null | tail -1 | cut -c1-110)"; done; tail -3 ~/runs/dscc_l1_queue.log 2>/dev/null' 2>/dev/null
} > $O 2>&1
