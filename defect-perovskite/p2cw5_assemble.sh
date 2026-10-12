#!/bin/bash
# Waits for the third run's finisher (LARGEN_DONE in P2C5_STATUS), pulls the three step records, assembles report section 1.5.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; P=/home/alex/runs/dscc/p2c; cd $W
until grep -q "LARGEN_DONE\|REPORT_FAILED" $P/P2C5_STATUS 2>/dev/null; do sleep 300; done
mkdir -p $P/bw5_steps
for s in 0 2 6; do rsync -aq b3:/home/alex/runs/dscc/dscc_p2cwbw5_s$s/steps.jsonl $P/bw5_steps/s$s.jsonl; done
MED=$(grep -o "seed-median B_w = seed [0-9]" $P/P2C5_STATUS | grep -o "[0-9]$")
python3 defect-perovskite/p2c_bw3_section.py --g2 $P/P2C2_GATES.json --g3 $P/P2C5_GATES.json \
  --hist3 /home/alex/runs/dscc/dscc_p2cwbw5_s0/history.json /home/alex/runs/dscc/dscc_p2cwbw5_s2/history.json /home/alex/runs/dscc/dscc_p2cwbw5_s6/history.json \
  --steps3 $P/bw5_steps/s0.jsonl $P/bw5_steps/s2.jsonl $P/bw5_steps/s6.jsonl --label fifth --largen $P/bw5_s$MED/step0.json > $P/section_1_6_bw5.md 2> $P/section_1_6_bw5.err
echo "ASSEMBLED $(date) $(wc -l < $P/section_1_6_bw5.md) lines" >> $P/P2C5_STATUS
