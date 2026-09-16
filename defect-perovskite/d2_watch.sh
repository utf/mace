#!/bin/bash
# D2 readers, unattended: when a D2 fine-tune exits, pull it (b3 arm), convert to the single-head
# base (w1les_make_base: exactness asserted, shared readout carried), then G-D2.1 (log row, 1.2x
# base v2 = 12.3 meV/A / 1.4 meV/atom), G-D2.2 (charged shell table vs base v2, base_compare_n),
# G-D2.3 (Born, les_gates) -> ~/runs/<arm>/D2_GATES.json. Born on b3 GPU 7 when it is free (the
# local card OOMs the BEC gradient beside a training run); otherwise local with a retry.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=$W/defect-perovskite; PY=/home/alex/micromamba/envs/py13/bin/python
L=$HOME/runs/d2_watch.log; export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $W
log() { echo "$(date '+%F %T') $*" >> $L; }
declare -A DONE
while true; do
  for arm in d2_light d2_full_scratch d2_light2 d2_full1; do
    [ -n "${DONE[$arm]}" ] && continue
    case $arm in d2_full_scratch|d2_full1) rsync -aq b3:runs/$arm.log $HOME/runs/ 2>/dev/null ;; esac
    grep -q "D2_EXIT" $HOME/runs/$arm.log 2>/dev/null || continue
    case $arm in d2_full_scratch|d2_full1) rsync -aq --exclude checkpoints b3:runs/$arm/ $HOME/runs/$arm/ ;; esac
    N=$arm; R=$HOME/runs/$N
    $PY $D/w1les_make_base.py --model $R/$N.model --out $R/${N}_base.pt --device cpu > $R/convert.log 2>&1 || { log "$arm: CONVERSION FAILED"; DONE[$arm]=1; continue; }
    log "$arm converted: $(grep -a 'max |multi' $R/convert.log | cut -c1-140)"
    $PY $D/base_compare_n.py --model "v2=/home/alex/runs/base_v2_prod/base_v2_prod_base.pt" --model "d2=$R/${N}_base.pt" --out $R/compare_v2.json > $R/compare_v2.log 2>&1 || log "$arm: base_compare_n FAILED"
    rsync -aq $R/${N}_base.pt b3:runs/$N/ 2>/dev/null
    ssh -o BatchMode=yes b3 "cd $W; export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1; u7=\$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' '\$1==7 {print \$2}'); CUDA_VISIBLE_DEVICES=\$u7 $PY $D/les_gates.py --model ~/runs/$N/${N}_base.pt --out ~/runs/$N/les_gates.json > ~/runs/$N/les_gates.log 2>&1" 2>/dev/null
    rsync -aq b3:runs/$N/les_gates.json $R/ 2>/dev/null || $PY $D/les_gates.py --model $R/${N}_base.pt --out $R/les_gates.json > $R/les_gates.log 2>&1 || log "$arm: les_gates FAILED"
    GATE_F=12.3 GATE_E=1.4 $PY $D/l0_gates.py --log $HOME/runs/$N.log --compare $R/compare_v2.json --label d2 --born $R/les_gates.json --out $R/D2_GATES.json > $R/d2_gates.log 2>&1
    log "$arm gates: $(tail -1 $R/d2_gates.log) | $(python3 -c "import json; g=json.load(open('$R/D2_GATES.json')); print('G-D2.1', g['G-L0.1'].get('epoch29_E_meV_per_atom'), g['G-L0.1'].get('epoch29_F_meV_per_A'), g['G-L0.1']['pass'], '| G-D2.2 fails', g['G-L0.2']['fails'], '| Born', g['G-L0.3']['pass'], 'sumZ', round(g['G-L0.3']['sum_Z_star'],2))" 2>/dev/null)"
    DONE[$arm]=1
  done
  [ "${#DONE[@]}" = "4" ] && { log "all four D2 arms read"; exit 0; }
  sleep 900
done
