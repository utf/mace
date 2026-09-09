#!/bin/bash
# W1.3 one-shot: fetch the fold bases from b3, then stage 1 -> 1b -> 2 in order.
set -u
S=/tmp/claude-1000/-home-alex-src-mace/eeca5f26-1644-4587-983a-79b6adcd9665/scratchpad
PY=/home/alex/micromamba/envs/py13/bin/python
L=$HOME/runs/w13.log
: > $L
for f in 1 2 3; do
  mkdir -p $HOME/runs/base_v2_f$f
  rsync -q b3:runs/base_v2_f$f/base_v2_f$f.model $HOME/runs/base_v2_f$f/ >> $L 2>&1 || { echo "RSYNC FAILED f$f" >> $L; exit 2; }
done
echo "$(date '+%F %T') models fetched" >> $L
$PY $S/w13_eval.py --folds 0,1,2,3 --out $HOME/runs/w1_oof_stage1.json >> $L 2>&1 || { echo "STAGE1 FAILED" >> $L; exit 3; }
echo "$(date '+%F %T') stage 1 done" >> $L
$PY $S/w13_thresholds.py --rows $HOME/runs/w1_oof_stage1.json --out $HOME/runs/w1_proxy_thresholds.json >> $L 2>&1 || { echo "STAGE1B FAILED" >> $L; exit 4; }
echo "$(date '+%F %T') thresholds written" >> $L
$PY $S/w13_charged.py --rows $HOME/runs/w1_oof_stage1.json --thresholds $HOME/runs/w1_proxy_thresholds.json --out $HOME/runs/w1_charged.json >> $L 2>&1 || { echo "STAGE2 FAILED" >> $L; exit 5; }
echo "$(date '+%F %T') W13_DONE 0" >> $L
