#!/bin/bash
# The L0 gate readers for the closed v4 run, sequential on the local card (record only).
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=$W/defect-perovskite; PY=/home/alex/micromamba/envs/py13/bin/python
N=base_v4_lesmh_prod; R=$HOME/runs/$N; L=$HOME/runs/l0_watch.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1; cd $W
log() { echo "$(date '+%F %T') $*" >> $L; }
$PY $D/les_gates.py --model $R/${N}_base.pt --out $R/les_gates.json > $R/les_gates.log 2>&1 || log "les_gates FAILED"
$PY $D/base_compare_n.py --model "v2=/home/alex/runs/base_v2_prod/base_v2_prod_base.pt" --model "v4=$R/${N}_base.pt" --out $R/compare_v2.json > $R/compare_v2.log 2>&1 || log "base_compare_n FAILED"
$PY $D/l2_pretest.py --base $R/${N}_base.pt --tilings 2,2,2 2,2,3 3,3,3 3,3,4 --sign $($PY -c "import json;print(json.load(open('$R/les_gates.json'))['sign_fix'])") --out $R/l2_pretest.json > $R/l2_pretest.log 2>&1 || log "l2_pretest FAILED"
$PY $D/l0_gates.py --log $HOME/runs/$N.log --compare $R/compare_v2.json --label v4 --born $R/les_gates.json --out $R/L0_GATES.json > $R/l0_gates.log 2>&1
log "L0 gates (record only): $(tail -1 $R/l0_gates.log)"
