#!/bin/bash
# Plan P A5 orchestration (user, 2026-09-18 02:30: "If everything looks good at the end then continue
# with the rest of the plan. If not, try a higher learning rate."): wait for base_v7's gates; ALL PASS
# -> p2_launch_on_gates.sh; otherwise launch base_v7_lr3e4 (registered fallback lr 3e-4, same recipe)
# with its gate chain, wait, and open Phase 2 on ALL PASS; else P2_STATUS BLOCKED. Status files only.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=/home/alex/runs/dscc/p1v2; P2=/home/alex/runs/dscc/p2
cd $W; export PYTHONPATH=$W; mkdir -p $P2
wait_gates() { while ! grep -q "GATES_DONE\|FAILED" $1/P1_STATUS 2>/dev/null; do sleep 600; done; head -1 $1/P1_STATUS; }
v=$(wait_gates $D/base_v7); echo "base_v7: $v" >> $P2/ORCH_LOG
if echo "$v" | grep -q "ALL PASS"; then bash defect-perovskite/p2_launch_on_gates.sh $D/base_v7 $P2 >> $P2/ORCH_LOG 2>&1; exit 0; fi
echo "base_v7 fallback: lr 3e-4 $(date)" >> $P2/ORCH_LOG
ssh -f b3 "nohup bash /home/alex/runs/dscc/p1v3_runner.sh base_v7_lr3e4 0.0003 $D/base_v7_lr3e4 5 > /dev/null 2>&1 &"
sleep 60; bash defect-perovskite/p1v2_after.sh $D/base_v7_lr3e4 b3 >> $P2/ORCH_LOG 2>&1
v=$(wait_gates $D/base_v7_lr3e4); echo "base_v7_lr3e4: $v" >> $P2/ORCH_LOG
if echo "$v" | grep -q "ALL PASS"; then bash defect-perovskite/p2_launch_on_gates.sh $D/base_v7_lr3e4 $P2 >> $P2/ORCH_LOG 2>&1; exit 0; fi
echo "BLOCKED $(date): base_v7 ($(head -1 $D/base_v7/P1_STATUS)) and base_v7_lr3e4 ($v) fail the Phase 1 gates" > $P2/P2_STATUS
