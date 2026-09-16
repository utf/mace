#!/bin/bash
until grep -q "launching L1" /home/alex/runs/l0_watch.log 2>/dev/null; do sleep 600; done
if grep -q "L0 FAILED" /home/alex/runs/l0_watch.log; then exit 0; fi
exec bash /home/alex/src/mace/.claude/worktrees/size-extensivity/defect-perovskite/lx_watch.sh L1
