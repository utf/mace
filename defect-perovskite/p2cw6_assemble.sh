#!/bin/bash
# Waits for the third run's finisher (LARGEN_DONE in P2C6_STATUS), pulls the three step records, assembles report section 1.5.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; P=/home/alex/runs/dscc/p2c; cd $W
until grep -q "LARGEN_DONE\|REPORT_FAILED" $P/P2C6_STATUS 2>/dev/null; do sleep 300; done
mkdir -p $P/bw6_steps
for s in 0 2 6; do rsync -aq b3:/home/alex/runs/dscc/dscc_p2cwbw6_s$s/steps.jsonl $P/bw6_steps/s$s.jsonl; done
MED=$(grep -o "seed-median B_w = seed [0-9]" $P/P2C6_STATUS | grep -o "[0-9]$")
python3 defect-perovskite/p2c_bw3_section.py --g2 $P/P2C2_GATES.json --g3 $P/P2C6_GATES.json \
  --hist3 /home/alex/runs/dscc/dscc_p2cwbw6_s0/history.json /home/alex/runs/dscc/dscc_p2cwbw6_s2/history.json /home/alex/runs/dscc/dscc_p2cwbw6_s6/history.json \
  --steps3 $P/bw6_steps/s0.jsonl $P/bw6_steps/s2.jsonl $P/bw6_steps/s6.jsonl --label fifth --largen $P/bw6_s$MED/step0.json > $P/section_p3a_1_bw6.md 2> $P/section_p3a_1_bw6.err
echo "ASSEMBLED $(date) $(wc -l < $P/section_p3a_1_bw6.md) lines" >> $P/P2C6_STATUS
