#!/bin/bash
# D1 readers, unattended: every 30 min a |q| / E_LES / readout-norm snapshot of each arm's latest
# checkpoint (raw weights) -> ~/runs/d1_traj_<arm>.jsonl; when an arm's final .model exists, the
# registered readers on the EMA model: d1_read (G-D1.1), d1_read --ablate (G-D1.3), les_gates
# (G-D1.2), d1_traj, inference ms -> ~/runs/dscc/D1_<arm>.json. No launches follow (the reading
# rule is applied by hand). Local card: d1_light; b3: full-scratch (5), full-init (7), pulled.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=$W/defect-perovskite; PY=/home/alex/micromamba/envs/py13/bin/python
L=$HOME/runs/d1_watch.log; cd $W; export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') $*" >> $L; }
declare -A DONE
while true; do
  for arm in d1_light d1_full_scratch d1_full_init d1_light2 d1_full1; do
    if [ "$arm" != "d1_light" ]; then
      mkdir -p $HOME/runs/$arm; rsync -aq --exclude 'checkpoints' b3:runs/$arm/ $HOME/runs/$arm/ 2>/dev/null
      rsync -aq b3:runs/$arm.log $HOME/runs/ 2>/dev/null; ls $HOME/runs/$arm/*.model >/dev/null 2>&1 || true
      ck=$(ssh -o BatchMode=yes b3 "ls -t ~/runs/$arm/checkpoints/*.pt 2>/dev/null | head -1")
      [ -n "$ck" ] && { mkdir -p $HOME/runs/$arm/ckpt; rsync -aq "b3:$ck" $HOME/runs/$arm/ckpt/ 2>/dev/null; }
    fi
    # template for a raw checkpoint: any saved .model of this arm (the smoke's for light, none for the full arms until saved)
    ck=$(ls -t $HOME/runs/$arm/checkpoints/*.pt $HOME/runs/$arm/ckpt/*.pt 2>/dev/null | head -1)
    # same-architecture templates for raw checkpoints (a from-scratch run saves its .model only at the end)
    case $arm in d1_light) tpl=$HOME/runs/d1_smoke_light/d1_smoke_light.model ;; d1_light2) tpl=$HOME/runs/d1_smoke_light2/d1_smoke_light2.model ;; d1_full1) tpl=$HOME/runs/d1_smoke_full1/d1_smoke_full1.model ;; *) tpl=$HOME/runs/base_v3_les_ft2/base_v3_les_ft2.model ;; esac
    ep=$(grep -a "Epoch" $HOME/runs/$arm.log 2>/dev/null | grep -a "Default" | tail -1 | grep -o "Epoch [0-9]*")
    if [ -n "$ck" ] && [ -n "$tpl" ]; then
      out=$($PY $D/d1_traj.py --model "$ck" --template "$tpl" --device cpu 2>/dev/null | tail -1)
      [ -n "$out" ] && echo "{\"time\": \"$(date '+%F %T')\", \"epoch_row\": \"$ep\", \"read\": $out}" >> $HOME/runs/d1_traj_$arm.jsonl
    fi
    final=$HOME/runs/$arm/$arm.model; [ -f $HOME/runs/$arm/${arm}_stagetwo.model ] && final=$HOME/runs/$arm/${arm}_stagetwo.model   # the stage-two final is the registered model
    [ -f $HOME/runs/dscc/D1_${arm}_born.json ] && DONE[$arm]=1      # already read (by hand or by an earlier cycle)
    if [ -f "$final" ] && [ -z "${DONE[$arm]}" ] && grep -q "D1_EXIT" $HOME/runs/$arm.log 2>/dev/null; then
      log "$arm finished: reading"
      $PY $D/d1_read.py --model $final --out $HOME/runs/dscc/D1_${arm}_read.json > $HOME/runs/dscc/D1_${arm}_read.log 2>&1
      $PY $D/d1_read.py --model $final --ablate --out $HOME/runs/dscc/D1_${arm}_ablate.json > $HOME/runs/dscc/D1_${arm}_ablate.log 2>&1
      $PY $D/les_gates.py --model $final --frames $D/dataset_d0/valid.xyz --out $HOME/runs/dscc/D1_${arm}_born.json > $HOME/runs/dscc/D1_${arm}_born.log 2>&1
      $PY $D/d1_traj.py --model $final > $HOME/runs/dscc/D1_${arm}_final_traj.json 2>/dev/null
      log "$arm read: $(tail -1 $HOME/runs/dscc/D1_${arm}_read.log | cut -c1-100) | born $(grep -o '"G-L0.3": "[A-Z]*"' $HOME/runs/dscc/D1_${arm}_born.json)"
      DONE[$arm]=1
    fi
  done
  [ "${#DONE[@]}" = "5" ] && { log "all five arms read"; exit 0; }
  sleep 1800
done
