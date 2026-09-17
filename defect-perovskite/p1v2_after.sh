#!/bin/bash
# A2 tail: when RUNDIR/STATUS says DONE, build the base and run the gates. Usage: p1v2_after.sh RUNDIR [b3|local]
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; R=$1; WHERE=${2:-b3}
cd $W; export PYTHONPATH=$W
while true; do
  if [ "$WHERE" = "b3" ]; then st=$(ssh -o ConnectTimeout=30 b3 "tail -1 $R/STATUS 2>/dev/null" 2>/dev/null); else st=$(tail -1 $R/STATUS 2>/dev/null); fi
  echo "$st" | grep -q "^FAILED" && { echo "TRAIN_FAILED $(date)" > $R/P1_STATUS 2>/dev/null; exit 1; }
  echo "$st" | grep -q "^DONE" && break
  sleep 900
done
[ "$WHERE" = "b3" ] && { mkdir -p $R; rsync -aq b3:$R/*.model b3:$R/*_run-1.log $R/ 2>/dev/null; }
M=$(ls $R/*.model | grep -v run- | head -1)
python defect-perovskite/p1_make_base.py --model $M --out $R/base.pt --device cuda > $R/make_base.out 2>&1 || { echo "MAKE_BASE_FAILED $(date)" > $R/P1_STATUS; exit 1; }
python defect-perovskite/p1_gates.py --base $R/base.pt --out $R/P1_GATES.json --device cuda > $R/gates.out 2>&1 || { echo "GATES_SCRIPT_FAILED $(date)" > $R/P1_STATUS; exit 1; }
tail -1 $R/gates.out > $R/P1_STATUS; echo "GATES_DONE $(date)" >> $R/P1_STATUS
