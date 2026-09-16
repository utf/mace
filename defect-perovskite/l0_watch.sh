#!/bin/bash
# L0 pipeline, unattended: wait for each base_v4_lesmh run's final model on b3, pull it, convert
# (G-L0.4/5 asserted), and for the PRODUCTION base read G-L0.1/2/3 with the registered readers and
# write L0_GATES.json. On PASS: launch L1 -- seeds 0 1 2 on the local card (sequential), seeds
# 3 4 6 queued on b3 GPUs 5/7 behind the fold bases (w3_queue.sh waits for 13 GB free, one per
# card). On FAIL: stop and leave the record. Fold bases are converted for the OOF floor only.
set -u
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=$W/defect-perovskite
PY=/home/alex/micromamba/envs/py13/bin/python
L=$HOME/runs/l0_watch.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd $W
log() { echo "$(date '+%F %T') $*" >> $L; }
pull_and_convert() {   # NAME CHECK_XYZ [DEVICE]
  local n=$1 chk=$2 dev=${3:-cuda}
  until ssh -o BatchMode=yes b3 "test -f ~/runs/$n/$n.model" 2>/dev/null; do sleep 300; done
  sleep 60
  mkdir -p $HOME/runs/$n
  rsync -q b3:runs/$n/$n.model $HOME/runs/$n/ && rsync -q b3:runs/$n.log $HOME/runs/ || { log "RSYNC FAILED $n"; return 1; }
  log "pulled $n"
  $PY $D/w1les_make_base.py --model $HOME/runs/$n/$n.model --out $HOME/runs/$n/${n}_base.pt --check $chk --device $dev \
      > $HOME/runs/$n/convert.log 2>&1 || { log "CONVERSION FAILED $n (G-L0.4/5)"; return 1; }
  log "converted $n: $(grep -a 'max |multi' $HOME/runs/$n/convert.log | cut -c1-160)"
}
N=base_v4_lesmh_prod
pull_and_convert $N $D/dataset_cf/fold0/null_oof.xyz || exit 1
$PY $D/les_live_check.py $HOME/runs/$N/${N}_base.pt > $HOME/runs/$N/live.log 2>&1; log "liveness: $(tail -1 $HOME/runs/$N/live.log)"
$PY $D/les_gates.py --model $HOME/runs/$N/${N}_base.pt --out $HOME/runs/$N/les_gates.json > $HOME/runs/$N/les_gates.log 2>&1 || log "les_gates FAILED"
$PY $D/base_compare_n.py --model "v2=/home/alex/runs/base_v2_prod/base_v2_prod_base.pt" --model "v4=$HOME/runs/$N/${N}_base.pt" \
    --out $HOME/runs/$N/compare_v2.json > $HOME/runs/$N/compare_v2.log 2>&1 || log "base_compare_n FAILED"
$PY $D/l2_pretest.py --base $HOME/runs/$N/${N}_base.pt --tilings 2,2,2 2,2,3 3,3,3 3,3,4 \
    --sign $($PY -c "import json;print(json.load(open('$HOME/runs/$N/les_gates.json'))['sign_fix'])") \
    --out $HOME/runs/$N/l2_pretest.json > $HOME/runs/$N/l2_pretest.log 2>&1 || log "l2_pretest FAILED"
$PY $D/l0_gates.py --log $HOME/runs/$N.log --compare $HOME/runs/$N/compare_v2.json --label v4 \
    --born $HOME/runs/$N/les_gates.json --out $HOME/runs/$N/L0_GATES.json > $HOME/runs/$N/l0_gates.log 2>&1
verdict=$(tail -1 $HOME/runs/$N/l0_gates.log)
log "L0 gates: $verdict"
# 2026-09-15 12:20: the distilled-base plan closes the v4 run "at epoch 29 with its gate file"; the
# L1 launch and the fold-base conversions are disabled (L0_LAUNCH_L1=0 / L0_FOLDS=0).
if [ "${L0_LAUNCH_L1:-1}" = "1" ] && [ "$verdict" = "L0: PASS" ]; then
  # L2's q_LES / Jacobian cache for this base (1.7 s per frame on the A4000, ~30 min for the
  # 1047 charged frames): built now while the local card is free, before L1 s0 takes it.
  $PY $D/l2_les_cache.py --base $HOME/runs/$N/${N}_base.pt --out $HOME/runs/$N/les_cache.pt > $HOME/runs/$N/les_cache.log 2>&1 \
      && log "L2 cache: $(tail -1 $HOME/runs/$N/les_cache.log)" || log "L2 cache build FAILED (L1 launches regardless)"
  log "launching L1: local seeds 0 1 2; b3 seeds 3 4 6 queued behind the fold bases"
  L1_SEEDS="0 1 2" nohup bash $D/l1_local_queue.sh > $HOME/runs/l1_local.log 2>&1 &
  rsync -q $HOME/runs/$N/${N}_base.pt b3:runs/$N/
  # The b3 queue file is written now; the runner is started only once the w1les queue has
  # DISPATCHED its last fold base (no `base` line left), so the two runners never see the same
  # freed card in the same minute (f3 and s3 would land on one card). The wait runs on b3.
  ssh b3 "cd $W; : > ~/runs/l1_queue.txt; for s in 3 4 6; do f=\$([ \$s = 3 ] && echo 3 || ([ \$s = 4 ] && echo 0 || echo 2));
    echo \"dscc_l1_s\$s|--seed \$s --fold \$f --epochs 60 --directional 1 --regime B --base $HOME/runs/$N/${N}_base.pt --coupling 0 --scf_free 1 --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 --base_float32 1 --setup_batch_size 4 --base_cache_batch_size 4 --gpu_memory_fraction 0\" >> ~/runs/l1_queue.txt; done"
  rsync -q $D/l1_start.sh b3:runs/l1_start.sh
  ssh b3 "nohup bash ~/runs/l1_start.sh > /dev/null 2>&1 & echo armed"
  log "b3 L1 queue: $(ssh b3 'wc -l < ~/runs/l1_queue.txt') lines; starter armed behind the w1les queue"
else
  log "L0 FAILED -- L1 not launched"
fi
# fold bases on the CPU: the local card is running L1 s0 (11.6 GB of 16) by now
for k in ${L0_FOLDS:-0 1 2 3}; do pull_and_convert base_v4_lesmh_f$k $D/dataset_cf/fold$k/null_oof.xyz cpu; done
log "all fold bases converted; watcher done"
