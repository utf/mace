#!/bin/bash
# b3: start the L1 W6 runner only after the w1les queue has dispatched its last fold base, so the
# two runners never claim the same freed card in the same minute.
until ! grep -qE '^\s*base' /home/alex/runs/w1les_queue.txt 2>/dev/null; do sleep 300; done
sleep 120
cd /home/alex/src/mace/.claude/worktrees/size-extensivity
echo "$(date '+%F %T') w1les queue drained; starting the L1 runner" >> /home/alex/runs/l1_queue.log
W3_GPUS="5 7" W3_PER_GPU=1 W3_HEADROOM=13000 W3_QUEUE=/home/alex/runs/l1_queue.txt W3_LOG=/home/alex/runs/l1_queue.log bash defect-perovskite/w3_queue.sh
