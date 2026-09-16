#!/bin/bash
# full1 (MH-1 width, 1 interaction) on b3 GPU 5 once d1_full_scratch has exited.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=$W/defect-perovskite; L=$HOME/runs/d1_queue2.log
until ssh -o BatchMode=yes b3 "grep -q D1_EXIT ~/runs/d1_full_scratch.log" 2>/dev/null; do sleep 300; done
ssh b3 "cd $W; u5=\$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' '\$1==5 {print \$2}'); rm -rf ~/runs/d1_full1; D1_BATCH=32 bash $D/b3_run.sh \$u5 d1_full1 bash $D/d1_run.sh full1 d1_full1"
echo "$(date '+%F %T') d1_full1 launched on b3 GPU 5" >> $L
