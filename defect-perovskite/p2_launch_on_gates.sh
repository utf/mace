#!/bin/bash
# Plan P: open Phase 2 when the Phase 1 gates pass (P1_STATUS "ALL PASS"); otherwise write
# P2_STATUS=BLOCKED and stop. Usage: p2_launch_on_gates.sh P1_RUNDIR [P2_DIR]
# A3 entry measurement: base + E_PP vs base alone on the neutral validation (p2_epp_mismatch.py,
# tau 1.7 meV/A); EXCEEDS -> registered remedy p2_readout_refit.py (readout-only, E_PP on, target
# frames only); the refit base is used only if accepted and still ALL PASS on the gates.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; P1=${1:-/home/alex/runs/dscc/p1v2/base_v7}; P2=${2:-/home/alex/runs/dscc/p2}
mkdir -p $P2; cd $W; export PYTHONPATH=$W
while true; do
  grep -q "GATES_DONE\|FAILED" $P1/P1_STATUS 2>/dev/null && break
  sleep 600
done
if ! head -1 $P1/P1_STATUS | grep -q "ALL PASS"; then echo "BLOCKED $(date): $(head -1 $P1/P1_STATUS)" > $P2/P2_STATUS; exit 0; fi
EPS_M=$(python3 -c "import json; print(round(json.load(open('$P1/P1_GATES.json'))['gate2']['eps_inf_poly'], 4))")
BASE=$P1/base.pt
python defect-perovskite/p2_epp_mismatch.py --base $BASE --out $P1/P2_EPP.json > $P1/epp.out 2>&1 || { echo "EPP_MEASURE_FAILED $(date)" > $P2/P2_STATUS; exit 1; }
if grep -q '"verdict": "EXCEEDS"' $P1/P2_EPP.json; then
  python defect-perovskite/p2_readout_refit.py --base $BASE --out $P1/base_epp.pt --json $P1/P2_REFIT.json > $P1/refit.out 2>&1 || { echo "REFIT_FAILED $(date)" > $P2/P2_STATUS; exit 1; }
  grep -q '"accepted": true' $P1/P2_REFIT.json || { echo "BLOCKED $(date): E_PP mismatch exceeds tau and the readout refit was rejected" > $P2/P2_STATUS; exit 0; }
  python defect-perovskite/p1_gates.py --base $P1/base_epp.pt --out $P1/P1_GATES_epp.json --device cuda > $P1/gates_epp.out 2>&1 || { echo "GATES_EPP_FAILED $(date)" > $P2/P2_STATUS; exit 1; }
  tail -1 $P1/gates_epp.out | grep -q "ALL PASS" || { echo "BLOCKED $(date): refit base fails the gates: $(tail -1 $P1/gates_epp.out)" > $P2/P2_STATUS; exit 0; }
  BASE=$P1/base_epp.pt
fi
rsync -az --exclude='.git/' --exclude='__pycache__' $W/ b3:$W/
ssh b3 "mkdir -p $P1" && rsync -az $BASE b3:$BASE
bash defect-perovskite/p2_queue.sh $BASE $EPS_M > $P2/queue.out 2>&1
rsync -az /home/alex/runs/p2_queue.txt b3:/home/alex/runs/p2_queue.txt
ssh -f b3 "cd $W && W3_GPUS='5 7' W3_PER_GPU=1 W3_HEADROOM=13000 W3_QUEUE=/home/alex/runs/p2_queue.txt W3_LOG=/home/alex/runs/p2_queue.log PATH=\$HOME/micromamba/envs/py13/bin:\$PATH PYTHONPATH=$W nohup bash defect-perovskite/w3_queue.sh > /dev/null 2>&1 < /dev/null"
nohup bash defect-perovskite/p2_local.sh $BASE > $P2/local.out 2>&1 &
echo "LAUNCHED $(date) eps_M=$EPS_M" > $P2/P2_STATUS
