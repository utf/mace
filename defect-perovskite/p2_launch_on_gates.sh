#!/bin/bash
# Plan P: open Phase 2 when the Phase 1 gates pass (P1_STATUS "ALL PASS"); otherwise write
# P2_STATUS=BLOCKED and stop (a failing gate is the user's decision). Status files only.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; P1=/home/alex/runs/dscc/p1/base_v5; P2=/home/alex/runs/dscc/p2
mkdir -p $P2; cd $W
while true; do
  grep -q "GATES_DONE\|FAILED" $P1/P1_STATUS 2>/dev/null && break
  sleep 600
done
if ! head -1 $P1/P1_STATUS | grep -q "ALL PASS"; then echo "BLOCKED $(date): $(head -1 $P1/P1_STATUS)" > $P2/P2_STATUS; exit 0; fi
EPS_M=$(python3 -c "import json; print(round(json.load(open('$P1/P1_GATES.json'))['gate2']['eps_inf_poly'], 4))")
BASE=$P1/base_v5_base.pt
rsync -az --exclude='.git/' --exclude='__pycache__' $W/ b3:$W/
ssh b3 "mkdir -p $P1" && rsync -az $BASE b3:$BASE
bash defect-perovskite/p2_queue.sh $BASE $EPS_M > $P2/queue.out 2>&1
rsync -az /home/alex/runs/p2_queue.txt b3:/home/alex/runs/p2_queue.txt
ssh -f b3 "cd $W && W3_GPUS='5 7' W3_PER_GPU=1 W3_HEADROOM=13000 W3_QUEUE=/home/alex/runs/p2_queue.txt W3_LOG=/home/alex/runs/p2_queue.log PATH=\$HOME/micromamba/envs/py13/bin:\$PATH PYTHONPATH=$W nohup bash defect-perovskite/w3_queue.sh > /dev/null 2>&1 < /dev/null"
nohup bash defect-perovskite/p2_local.sh $BASE > $P2/local.out 2>&1 &
echo "LAUNCHED $(date) eps_M=$EPS_M" > $P2/P2_STATUS
