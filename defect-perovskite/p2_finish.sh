#!/bin/bash
# Plan P Phase 2 finisher: when all 15 arms have held_final.json (12 on b3, 3 local), pull the b3
# results and write P2_GATES.json + P2_REPORT.md via p2_report.py. Status files only.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; P2=/home/alex/runs/dscc/p2; cd $W; export PYTHONPATH=$W
B3="dscc_p2b_s0 dscc_p2b_s1 dscc_p2b_s2 dscc_p2b_s3 dscc_p2b_s4 dscc_p2b_s6 dscc_p2bm_s0 dscc_p2bm_s1 dscc_p2bm_s2 dscc_p2a_s3 dscc_p2a_s4 dscc_p2a_s6"
LOC="dscc_p2a_s0 dscc_p2a_s1 dscc_p2a_s2"
while true; do
  n=$(ssh -o ConnectTimeout=30 b3 "for r in $B3; do [ -f /home/alex/runs/dscc/\$r/held_final.json ] && echo 1; done" 2>/dev/null | wc -l)
  m=$(for r in $LOC; do [ -f /home/alex/runs/dscc/$r/held_final.json ] && echo 1; done | wc -l)
  echo "$(date '+%F %T') b3 $n/12 local $m/3" > $P2/FINISH_PROGRESS
  [ "$n" -ge 12 ] && [ "$m" -ge 3 ] && break
  sleep 1200
done
for r in $B3; do mkdir -p /home/alex/runs/dscc/$r; rsync -aq b3:/home/alex/runs/dscc/$r/held_final.json b3:/home/alex/runs/dscc/$r/held_final_avg.json b3:/home/alex/runs/dscc/$r/train.log b3:/home/alex/runs/dscc/$r/run_record.json /home/alex/runs/dscc/$r/ 2>/dev/null; done
python3 defect-perovskite/p2_report.py --runs /home/alex/runs/dscc/dscc_p2*_s* --out $P2/P2_GATES.json --md defect-perovskite/P2_REPORT.md > $P2/report.out 2>&1 && echo "REPORT_DONE $(date): $(grep -o 'Verdict: [A-Z ]*' defect-perovskite/P2_REPORT.md)" >> $P2/P2_STATUS || echo "REPORT_FAILED $(date)" >> $P2/P2_STATUS
