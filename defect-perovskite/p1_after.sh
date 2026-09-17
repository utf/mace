#!/bin/bash
# Plan P Phase 1 tail: when the b3 run reports DONE, pull the model, build the {Cl,Cs,Pb} base
# (checked exact), run the three gates, write P1_GATES.json and P1_STATUS. Status files only.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; P1=/home/alex/runs/dscc/p1/base_v5
cd $W; export PYTHONPATH=$W; mkdir -p $P1
while true; do
  st=$(ssh -o ConnectTimeout=30 b3 "tail -1 $P1/STATUS 2>/dev/null" 2>/dev/null)
  echo "$st" | grep -q FAILED && { echo "PHASE1_FAILED $(date)" > $P1/P1_STATUS; exit 1; }
  echo "$st" | grep -q DONE && break
  sleep 600
done
rsync -aq b3:$P1/base_v5.model b3:$P1/run.out $P1/ 2>/dev/null; rsync -aq b3:$P1/*.log $P1/ 2>/dev/null
python defect-perovskite/p1_make_base.py --model $P1/base_v5.model --out $P1/base_v5_base.pt --device cuda > $P1/make_base.out 2>&1 || { echo "MAKE_BASE_FAILED $(date)" > $P1/P1_STATUS; exit 1; }
python defect-perovskite/p1_gates.py --base $P1/base_v5_base.pt --out $P1/P1_GATES.json --device cuda > $P1/gates.out 2>&1 || { echo "GATES_SCRIPT_FAILED $(date)" > $P1/P1_STATUS; exit 1; }
tail -1 $P1/gates.out > $P1/P1_STATUS; echo "GATES_DONE $(date)" >> $P1/P1_STATUS
