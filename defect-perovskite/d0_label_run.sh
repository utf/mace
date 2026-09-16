#!/bin/bash
# D0 labelling on the local card once the L0 gate readers have finished with it.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; PY=/home/alex/micromamba/envs/py13/bin/python
until grep -q "L0 gates (record only)" $HOME/runs/l0_watch.log 2>/dev/null; do sleep 60; done
cd $W; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY defect-perovskite/d0_build.py --stage label --device cuda --batch_size 4 > $HOME/runs/d0_label.log 2>&1
echo "D0_LABEL_EXIT $?" >> $HOME/runs/d0_label.log
