#!/bin/bash
# Unattended chain of the LES head line after L0. Usage: lx_watch.sh STAGE  (L1 | L2 | L3)
#   waits for the stage's six seeds (0 1 2 local, 3 4 6 on b3), pulls the b3 runs, reads the stage
#   against its comparator (L1 vs W6-on-v2 [reference]; L2 vs L1; L3 vs L2) with lx_read.py, runs C5,
#   writes ~/runs/dscc/{STAGE}_GATES.json, and launches the next stage by the registered rule:
#   L1 read -> L2 launches (baseline, no blocking gate); L2 adopted -> L3 launches from L2's models;
#   otherwise the chain stops with "previous stands". Every launch is logged to ~/runs/lx_watch.log.
set -u
STAGE=$1
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=$W/defect-perovskite
PY=/home/alex/micromamba/envs/py13/bin/python; L=$HOME/runs/lx_watch.log; R=$HOME/runs/dscc
source $W/defect-perovskite/lx_env.sh 2>/dev/null; BASE=${LX_BASE:-$HOME/runs/base_v4_lesmh_prod/base_v4_lesmh_prod_base.pt}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd $W
log() { echo "$(date '+%F %T') [$STAGE] $*" >> $L; }
case $STAGE in
  L1) NAME=dscc_l1; CMP="dscc_w6_s%d"; CMP_LABEL="W6v2"; NEXT=L2 ;;
  L2) NAME=dscc_l2; CMP="dscc_l1_s%d"; CMP_LABEL="L1";   NEXT=L3 ;;
  L3) NAME=dscc_l3; CMP="dscc_l2_s%d"; CMP_LABEL="L2";   NEXT="" ;;
esac
# wait for the six runs (local files for 0 1 2; b3 for 3 4 6), pulling b3's as they finish
for s in 0 1 2 3 4 6; do
  if [ $s -le 2 ]; then
    until [ -f $R/${NAME}_s$s/held_final.json ]; do sleep 600; done
  else
    until ssh -o BatchMode=yes b3 "test -f ~/runs/dscc/${NAME}_s$s/held_final.json" 2>/dev/null; do sleep 600; done
    sleep 60; rsync -aq b3:runs/dscc/${NAME}_s$s/ $R/${NAME}_s$s/ || log "RSYNC FAILED ${NAME}_s$s"
  fi
  log "${NAME}_s$s complete"
done
$PY $D/lx_read.py --a "$CMP" --b "${NAME}_s%d" --label_a $CMP_LABEL --label_b $STAGE --out $R/${STAGE}_read.json > $R/${STAGE}_read.log 2>&1
log "read: $(grep '^avg: G1\|^final: G1' $R/${STAGE}_read.log | tail -1 | cut -c1-200)"
$PY $D/dscc_arm1_gates.py --no_ladder --device cuda --winners $R/${NAME}_s0 $R/${NAME}_s1 $R/${NAME}_s2 $R/${NAME}_s3 $R/${NAME}_s4 $R/${NAME}_s6 \
    --out $R/c5_${STAGE}.json > $R/c5_${STAGE}.log 2>&1 || log "C5 reader FAILED"
$PY $D/lx_verdict.py --stage $STAGE --read $R/${STAGE}_read.json --c5 $R/c5_${STAGE}.json --out $R/${STAGE}_GATES.json > $R/${STAGE}_verdict.log 2>&1
verdict=$(grep '^VERDICT:' $R/${STAGE}_verdict.log | cut -c10-)
log "verdict: $verdict"
SIGN=${LX_SIGN:-$($PY -c "import json;print(int(json.load(open('$HOME/runs/base_v4_lesmh_prod/L0_GATES.json'))['G-L0.3']['sign_fix']))")}
CACHE=${LX_CACHE:-$HOME/runs/base_v4_lesmh_prod/les_cache.pt}
launch_next() {   # NAME EXTRA [INIT_PAT_LOCAL INIT_PAT_B3]
  local name=$1 extra=$2 initl=${3:-} initb=${4:-}
  [ -f $CACHE ] && rsync -q $CACHE b3:$(dirname $CACHE)/
  LX_NAME=$name LX_SEEDS="0 1 2" LX_BASE=$BASE LX_EXTRA="$extra" LX_INIT_PAT="$initl" nohup bash $D/lx_local_queue.sh > $HOME/runs/${name}_local.log 2>&1 &
  ssh b3 "bash $D/lx_launch_b3.sh $name $BASE \"$extra\" \"$initb\"" >> $L 2>&1
  log "launched $name: local 0 1 2, b3 3 4 6 (extra: $extra; init: ${initl:-none})"
}
case "$verdict" in
  "L1 READ; L2 LAUNCHES")
    launch_next dscc_l2 "--les_cross 1 --les_sign $SIGN --les_cache $CACHE"
    nohup bash $D/lx_watch.sh L2 > /dev/null 2>&1 & ;;
  "L2 ADOPTED")
    launch_next dscc_l3 "--les_cross 1 --les_sign $SIGN --les_cache $CACHE --les_well 1" "$R/dscc_l2_s%d/model.pt" "/home/alex/runs/dscc/dscc_l2_s%d/model.pt"
    nohup bash $D/lx_watch.sh L3 > /dev/null 2>&1 & ;;
  "L3 ADOPTED") log "selected model: L3 -- L4 evaluation next (not automated)" ;;
  *) log "chain stops here: $verdict -- selected model is the previous stage; L4 next (not automated)" ;;
esac
