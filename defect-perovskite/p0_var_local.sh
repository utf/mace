#!/bin/bash
# Plan P 0.3: the polarisation arm's tight ladder entirely on the local CPU (the two float64-base
# arms do not fit one 24 GB card together). core -> 639/959 -> 2159/2879, STATUS_local per stage.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; P=/home/alex/runs/dscc/p0
OUT=$P/ladder2_var_local; mkdir -p $OUT; cd $W
export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 PYTHONPATH=$W
M=$P/w6_s1_polarisation.pt; C=$P/pristine_relaxed_v2.json
echo "START $(date)" > $OUT/STATUS_local
python defect-perovskite/p0_ladder.py --model $M --cell $C --stage core --tilings 2,2,2 --fmax 0.002 --steps 1000 --device cpu --out $OUT > $OUT/core.out 2>&1 && echo "CORE_DONE $(date)" >> $OUT/STATUS_local || { echo "CORE_FAILED $(date)" >> $OUT/STATUS_local; exit 1; }
python defect-perovskite/p0_ladder.py --model $M --cell $C --stage ladder --tilings 2,2,2 2,2,3 3,3,3 3,3,4 --fmax 0.002 --steps 1000 --device cpu --out $OUT > $OUT/ladder.out 2>&1 && echo "LADDER_DONE $(date)" >> $OUT/STATUS_local || echo "LADDER_FAILED $(date)" >> $OUT/STATUS_local
