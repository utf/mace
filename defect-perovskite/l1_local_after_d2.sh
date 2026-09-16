#!/bin/bash
until grep -q "D2_EXIT" $HOME/runs/d2_light2.log 2>/dev/null; do sleep 600; done
LX_NAME=dscc_l1 LX_SEEDS="0 1 2" LX_BASE=/home/alex/runs/d2_full_scratch/d2_full_scratch_base.pt LX_EXTRA="" nohup bash /home/alex/src/mace/.claude/worktrees/size-extensivity/defect-perovskite/lx_local_queue.sh > $HOME/runs/dscc_l1_local.log 2>&1 &
echo "$(date '+%F %T') L1 local seeds 0 1 2 launched behind d2_light2" >> $HOME/runs/lx_watch.log
