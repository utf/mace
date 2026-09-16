#!/bin/bash
# The two architecture-class arms added 2026-09-15 17:30: light2 (local, after d1_light) and full1
# (b3 GPU 5, after d1_full_scratch). Batch 32, les_fit recipe, lr 0.01, same D0 set and split.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=$W/defect-perovskite; L=$HOME/runs/d1_queue2.log
log() { echo "$(date '+%F %T') $*" >> $L; }
(
  until grep -q "D1_EXIT" $HOME/runs/d1_light.log 2>/dev/null; do sleep 300; done
  cd $W; rm -rf $HOME/runs/d1_light2; D1_BATCH=32 nohup bash $D/d1_run.sh light2 d1_light2 > $HOME/runs/d1_light2.log 2>&1 &
  log "d1_light2 launched locally"
) &
(
  until ssh -o BatchMode=yes b3 "grep -q D1_EXIT ~/runs/d1_full_scratch.log" 2>/dev/null; do sleep 300; done
  ssh b3 "cd $W; u5=\$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' '\$1==5 {print \$2}'); rm -rf ~/runs/d1_full1; D1_BATCH=32 bash $D/b3_run.sh \$u5 d1_full1 bash $D/d1_run.sh full1 d1_full1"
  log "d1_full1 launched on b3 GPU 5"
) &
wait
