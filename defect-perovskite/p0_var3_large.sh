#!/bin/bash
# Plan P 0.3: W6 arm's 2159/2879 rungs on the local CPU after the b3 core (tight ladder).
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; P=/home/alex/runs/dscc/p0
cd $W; export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 PYTHONPATH=$W
while true; do
  st=$(ssh -o ConnectTimeout=30 b3 "cat $P/ladder3_var/STATUS 2>/dev/null" 2>/dev/null)
  echo "$st" | grep -q CORE_FAILED && { echo "w6 core FAILED on b3 $(date)" >> $P/var3_large.log; exit 1; }
  echo "$st" | grep -q CORE_DONE && break
  sleep 120
done
mkdir -p $P/ladder3_var_large; rsync -aq b3:$P/ladder3_var/core.json $P/ladder3_var_large/
echo "LARGE_START $(date)" >> $P/ladder3_var/STATUS_local
python defect-perovskite/p0_ladder.py --model /home/alex/runs/dscc/p0/w6_s1_pol_r45.pt --cell $P/pristine_relaxed_v2.json --stage ladder --tilings 3,3,3 3,3,4 --fmax 0.002 --steps 1000 --device cpu --out $P/ladder3_var_large > $P/ladder3_var/large.out 2>&1 && echo "LARGE_DONE $(date)" >> $P/ladder3_var/STATUS_local || echo "LARGE_FAILED $(date)" >> $P/ladder3_var/STATUS_local
