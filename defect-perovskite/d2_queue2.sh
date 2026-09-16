#!/bin/bash
# D2 fine-tunes of the two added D1 classes, queued behind their students: d2_light2 on the local
# card when d1_light2 exits; d2_full1 on b3 GPU 5 when d1_full1 exits (its own card).
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=$W/defect-perovskite; L=$HOME/runs/d2_queue2.log
log() { echo "$(date '+%F %T') $*" >> $L; }
(
  until grep -q "D1_EXIT" $HOME/runs/d1_light2.log 2>/dev/null; do sleep 300; done
  M=$HOME/runs/d1_light2/d1_light2_stagetwo.model; [ -f $M ] || M=$HOME/runs/d1_light2/d1_light2.model
  cd $W; rm -rf $HOME/runs/d2_light2; nohup bash $D/d2_run.sh $M d2_light2 > $HOME/runs/d2_light2.log 2>&1 &
  log "d2_light2 launched locally from $(basename $M)"
) &
(
  until ssh -o BatchMode=yes b3 "grep -q D1_EXIT ~/runs/d1_full1.log" 2>/dev/null; do sleep 300; done
  ssh b3 "cd $W; M=~/runs/d1_full1/d1_full1_stagetwo.model; [ -f \$M ] || M=~/runs/d1_full1/d1_full1.model; u5=\$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' '\$1==5 {print \$2}'); rm -rf ~/runs/d2_full1; bash $D/b3_run.sh \$u5 d2_full1 bash $D/d2_run.sh \$M d2_full1"
  log "d2_full1 launched on b3 GPU 5"
) &
wait
