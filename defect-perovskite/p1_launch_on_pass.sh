#!/bin/bash
# Plan P: compute Decision 0 from the records once every input exists, write DECISION0_STATUS,
# and launch Phase 1 (base v5) on b3 GPU 5 on PASS. Polls the status files; never a process.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; P=/home/alex/runs/dscc/p0
cd $W; export PYTHONPATH=$W
while true; do
  w6_done=$(grep -c "LARGE_DONE" $P/ladder2_w6/STATUS_local 2>/dev/null)
  var_done=$(grep -c "LARGE_DONE" $P/ladder3_var/STATUS_local 2>/dev/null)
  loto=$( [ -f $P/p0_loto_r45.json ] && echo 1 || echo 0 )
  if [ "$w6_done" = "1" ] && [ "$var_done" = "1" ] && [ "$loto" = "1" ]; then break; fi
  grep -q FAILED $P/ladder2_w6/STATUS_local $P/ladder3_var/STATUS_local 2>/dev/null && { echo "a ladder FAILED $(date)" >> $P/decision.log; exit 1; }
  sleep 300
done
for a in ladder3_var; do rsync -aq b3:$P/$a/ladder.json $P/$a/ladder_small.json 2>/dev/null; done
python defect-perovskite/p0_decision.py --w6 $P/ladder2_w6 $P/ladder2_w6_large --var $P/ladder3_var $P/ladder3_var_large --loto $P/p0_loto_r45.json --out $P/DECISION0_STATUS > $P/decision.log 2>&1
head -1 $P/DECISION0_STATUS
if [ "$(head -1 $P/DECISION0_STATUS)" = "PASS" ]; then
  scp -q $W/defect-perovskite/p1_run.sh b3:$W/defect-perovskite/
  ssh -f b3 "cd /home/alex/runs/dscc && nohup bash p1_runner.sh base_v5 30 /home/alex/runs/dscc/p1/base_v5 5 --pp_r_split 4.5 --batch_size 4 --valid_batch_size 4 > /home/alex/runs/dscc/p1/base_v5_runner.out 2>&1 < /dev/null"
  echo "PHASE1_LAUNCHED $(date)" >> $P/decision.log
fi
