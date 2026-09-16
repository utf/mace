#!/bin/bash
# Plan P 0.3: wait for both b3 core relaxations (STATUS files), pull core.json, run the
# 2159/2879 rungs on the local CPU (48 threads each), STATUS files per arm.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
P=/home/alex/runs/dscc/p0
cd $W
for arm in w6 var; do mkdir -p $P/ladder_$arm; done
while true; do
  ok=0
  for arm in w6 var; do
    st=$(ssh -o ConnectTimeout=30 b3 "cat $P/ladder_$arm/STATUS 2>/dev/null" 2>/dev/null)
    echo "$st" | grep -q CORE_FAILED && { echo "$arm core FAILED on b3" >> $P/large_rungs.log; exit 1; }
    echo "$st" | grep -q CORE_DONE && ok=$((ok+1))
  done
  [ $ok -eq 2 ] && break
  sleep 120
done
echo "cores done $(date)" >> $P/large_rungs.log
for arm in w6 var; do rsync -aq b3:$P/ladder_$arm/core.json $P/ladder_$arm/; mkdir -p $P/ladder_${arm}_large; cp $P/ladder_$arm/core.json $P/ladder_${arm}_large/; done
MODEL_w6=/home/alex/runs/dscc/dscc_w6_s1/model.pt
MODEL_var=$P/w6_s1_polarisation.pt
for arm in w6 var; do
  m=MODEL_$arm
  ( export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48
    echo "LARGE_START $(date)" >> $P/ladder_$arm/STATUS_local
    python defect-perovskite/p0_ladder.py --model ${!m} --cell $P/pristine_relaxed_v2.json --stage ladder --tilings 3,3,3 3,3,4 --device cpu --out $P/ladder_${arm}_large > $P/ladder_$arm/large.out 2>&1 \
      && echo "LARGE_DONE $(date)" >> $P/ladder_$arm/STATUS_local || echo "LARGE_FAILED $(date)" >> $P/ladder_$arm/STATUS_local ) &
done
wait
echo "large rungs finished $(date)" >> $P/large_rungs.log
