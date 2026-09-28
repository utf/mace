#!/bin/bash
# Plan P Phase 2b on base v8 finisher: when B_s x6 and B_0 x6 (b3) and A x6 (local) have held_final.json, pull the b3 results,
# run the paired report (P2B_V8_REPORT.md, ~/runs/dscc/p2v8/P2V8_GATES.json) and the C5 localisation read. Status files only.
BASE_LABEL=${1:-base v8 (A2B provider, lr 1e-4, 30 epochs; gate 1 under override)}
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; P=/home/alex/runs/dscc/p2v8; cd $W; export PYTHONPATH=$W TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
B3="dscc_p2v8bs_s0 dscc_p2v8bs_s1 dscc_p2v8bs_s2 dscc_p2v8bs_s3 dscc_p2v8bs_s4 dscc_p2v8bs_s6 dscc_p2v8b0_s0 dscc_p2v8b0_s1 dscc_p2v8b0_s2 dscc_p2v8b0_s3 dscc_p2v8b0_s4 dscc_p2v8b0_s6"
LOC="dscc_p2v8a_s0 dscc_p2v8a_s1 dscc_p2v8a_s2 dscc_p2v8a_s3 dscc_p2v8a_s4 dscc_p2v8a_s6"
while true; do
  n=$(ssh -o ConnectTimeout=30 b3 "for r in $B3; do [ -f /home/alex/runs/dscc/\$r/held_final.json ] && echo 1; done" 2>/dev/null | wc -l)
  m=$(for r in $LOC; do [ -f /home/alex/runs/dscc/$r/held_final.json ] && echo 1; done | wc -l)
  echo "$(date '+%F %T') b3 $n/12 local $m/6" > $P/FINISH_PROGRESS
  [ "$n" -ge 12 ] && [ "$m" -ge 6 ] && break
  sleep 1200
done
for r in $B3; do mkdir -p /home/alex/runs/dscc/$r; rsync -aq b3:/home/alex/runs/dscc/$r/held_final.json b3:/home/alex/runs/dscc/$r/held_final_avg.json b3:/home/alex/runs/dscc/$r/train.log b3:/home/alex/runs/dscc/$r/run_record.json b3:/home/alex/runs/dscc/$r/model.pt b3:/home/alex/runs/dscc/$r/model_avg.pt b3:/home/alex/runs/dscc/$r/bz_readout.json /home/alex/runs/dscc/$r/ 2>/dev/null; done
python3 defect-perovskite/p2_report.py --runs /home/alex/runs/dscc/dscc_p2v8a_s* /home/alex/runs/dscc/dscc_p2v8bs_s* /home/alex/runs/dscc/dscc_p2v8b0_s* --out $P/P2V8_GATES.json --md defect-perovskite/P2B_V8_REPORT.md --base_label "$BASE_LABEL" > $P/report.out 2>&1 || { echo "REPORT_FAILED $(date)" >> $P/P2V8_STATUS; exit 1; }
CUDA_VISIBLE_DEVICES=0 python3 defect-perovskite/p2_c5.py --runs /home/alex/runs/dscc/dscc_p2v8a_s* /home/alex/runs/dscc/dscc_p2v8bs_s* /home/alex/runs/dscc/dscc_p2v8b0_s* --out $P/P2V8_C5.json > $P/c5.out 2>&1 || echo "C5_FAILED $(date)" >> $P/P2V8_STATUS
echo "REPORT_DONE $(date): $(grep -o 'Verdict: .*' defect-perovskite/P2B_V8_REPORT.md | head -1 | cut -c1-200)" >> $P/P2V8_STATUS
