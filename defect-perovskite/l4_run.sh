#!/bin/bash
# L4 on the model the chain selects: waits for the chain's final verdict line, picks the selected
# stage's best C5-passing seed (lowest held force RMSE; fallback lowest force), then runs the
# registered L4 readers on the local card and writes ~/runs/dscc/L4/<stage>_s<seed>/.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=$W/defect-perovskite; PY=/home/alex/micromamba/envs/py13/bin/python
L=$HOME/runs/lx_watch.log; export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $W
until grep -q "selected model: L3\|chain stops here" $L 2>/dev/null; do sleep 300; done
line=$(grep "selected model: L3\|chain stops here" $L | tail -1)
case "$line" in *"selected model: L3"*) STAGE=L3;; *"L2 NOT ADOPTED"*) STAGE=L1;; *"L3 NOT ADOPTED"*) STAGE=L2;; *) STAGE=L1;; esac
NAME=$( [ $STAGE = L1 ] && echo dscc_l1 || ([ $STAGE = L2 ] && echo dscc_l2 || echo dscc_l3) )
SEED=$($PY - $STAGE $NAME <<'PY'
import json, sys, math
stage, name = sys.argv[1:3]; R = "/home/alex/runs/dscc"
c5 = json.load(open(f"{R}/c5_{stage}.json"))
best = None
for s in (0, 1, 2, 3, 4, 6):
    try: f = json.load(open(f"{R}/{name}_s{s}/held_final.json"))["force_rmse"]
    except Exception: continue
    ok = c5.get(f"{name}_s{s}", {}).get("precondition", {}).get("passed", False)
    key = (0 if ok else 1, f)
    if best is None or key < best[0]: best = (key, s)
print(best[1] if best else 1)
PY
)
M=$HOME/runs/dscc/${NAME}_s$SEED/model.pt; O=$HOME/runs/dscc/L4/${STAGE}_s$SEED; mkdir -p $O
echo "$(date '+%F %T') [L4] selected $STAGE seed $SEED ($M)" >> $L
$PY $D/dscc_embed_ladder.py --model $M --core /home/alex/runs/dscc/relax159/relaxed_structures.json --core_key 2,2,2 --r_core 8 --tilings 2,2,3 3,3,3 3,3,4 --device cuda --out $O/embed.json > $O/embed.log 2>&1 || echo "$(date '+%F %T') [L4] embed ladder FAILED" >> $L
$PY $D/dscc_relax_ladder.py --model $M --core /home/alex/runs/dscc/relax159/relaxed_structures.json --core_key 2,2,2 --r_core 8 --tilings 2,2,3 3,3,4 --device cuda --out $O/relax.json > $O/relax.log 2>&1 || echo "$(date '+%F %T') [L4] relax ladder FAILED" >> $L
$PY $D/d1_leak.py --model $M --device cuda --out $O/leak.json > $O/leak.log 2>&1 || echo "$(date '+%F %T') [L4] leak FAILED" >> $L
SC=$($PY -c "import torch,sys; sys.path.insert(0,'$W'); m=torch.load('$M',map_location='cpu',weights_only=False); print(float(m.cross_scale()) if getattr(m,'les_cross',False) else 0.5)" 2>/dev/null)
source $D/lx_env.sh; $PY $D/l4_eps0_phonon.py --base $LX_BASE --s ${SC:-0.5} --sign $LX_SIGN --device cuda --out $O/eps0.json > $O/eps0.log 2>&1 || echo "$(date '+%F %T') [L4] eps0 FAILED" >> $L
$PY $D/l4_ms.py --model $M --out $O/ms.json > $O/ms.log 2>&1 || echo "$(date '+%F %T') [L4] ms FAILED" >> $L
echo "$(date '+%F %T') [L4] done: $O" >> $L
