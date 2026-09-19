#!/bin/bash
# Plan P Phase 2b finisher: when B_s x6 and B_0 x6 (b3) and A s3/s4/s6 (local) have held_final.json,
# pull the b3 results, run the report (arms a/bs/b0, paired tables, §5 decisions), the C5
# localisation read on every checkpoint, and write P2B_REPORT.md. Status files only.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; P=/home/alex/runs/dscc/p2b; cd $W; export PYTHONPATH=$W
B3="dscc_p2bs_s0 dscc_p2bs_s1 dscc_p2bs_s2 dscc_p2bs_s3 dscc_p2bs_s4 dscc_p2bs_s6 dscc_p2b0_s0 dscc_p2b0_s1 dscc_p2b0_s2 dscc_p2b0_s3 dscc_p2b0_s4 dscc_p2b0_s6"
LOC="dscc_p2a_s3 dscc_p2a_s4 dscc_p2a_s6"
while true; do
  n=$(ssh -o ConnectTimeout=30 b3 "for r in $B3; do [ -f /home/alex/runs/dscc/\$r/held_final.json ] && echo 1; done" 2>/dev/null | wc -l)
  m=$(for r in $LOC; do [ -f /home/alex/runs/dscc/$r/held_final.json ] && echo 1; done | wc -l)
  echo "$(date '+%F %T') b3 $n/12 local $m/3" > $P/FINISH_PROGRESS
  [ "$n" -ge 12 ] && [ "$m" -ge 3 ] && break
  sleep 1200
done
for r in $B3; do mkdir -p /home/alex/runs/dscc/$r; rsync -aq b3:/home/alex/runs/dscc/$r/held_final.json b3:/home/alex/runs/dscc/$r/held_final_avg.json b3:/home/alex/runs/dscc/$r/train.log b3:/home/alex/runs/dscc/$r/run_record.json b3:/home/alex/runs/dscc/$r/model.pt b3:/home/alex/runs/dscc/$r/model_avg.pt b3:/home/alex/runs/dscc/$r/bz_readout.json /home/alex/runs/dscc/$r/ 2>/dev/null; done
python3 defect-perovskite/p2_report.py --runs /home/alex/runs/dscc/dscc_p2a_s* /home/alex/runs/dscc/dscc_p2bs_s* /home/alex/runs/dscc/dscc_p2b0_s* --out $P/P2B_GATES.json --md defect-perovskite/P2B_REPORT.md > $P/report.out 2>&1 || { echo "REPORT_FAILED $(date)" >> $P/P2B_STATUS; exit 1; }
CUDA_VISIBLE_DEVICES=0 python3 defect-perovskite/p2_c5.py --runs /home/alex/runs/dscc/dscc_p2a_s* /home/alex/runs/dscc/dscc_p2bs_s* /home/alex/runs/dscc/dscc_p2b0_s* --out $P/P2B_C5.json > $P/c5.out 2>&1 || echo "C5_FAILED $(date)" >> $P/P2B_STATUS
echo "REPORT_DONE $(date): $(grep -o 'Verdict: .*' defect-perovskite/P2B_REPORT.md | head -1 | cut -c1-200)" >> $P/P2B_STATUS
