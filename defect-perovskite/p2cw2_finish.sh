#!/bin/bash
# Plan P Phase 2c Step 1 finisher: when the three B_w runs (b3) have held_final.json, pull them, run the paired read against B_s
# (p2c_report.py -> ~/runs/dscc/p2c/P2C2_GATES.json, step1_bw2.md), then the large-N read in the label tiling on the seed-median B_w
# checkpoint (p2c_step0.py stage 0.2 on b3 GPU 7 by UUID: rows (a), (b) core re-relaxed by that checkpoint, (a')). Status files only.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; P=/home/alex/runs/dscc/p2c; cd $W; export PYTHONPATH=$W TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
RUNS="dscc_p2cwbw2_s0 dscc_p2cwbw2_s2 dscc_p2cwbw2_s6"
while true; do
  n=$(ssh -o ConnectTimeout=30 b3 "for r in $RUNS; do [ -f /home/alex/runs/dscc/\$r/held_final_avg.json ] && echo 1; done" 2>/dev/null | wc -l)
  echo "$(date '+%F %T') b3 $n/3" > $P/FINISH2_PROGRESS
  [ "$n" -ge 3 ] && break
  sleep 1200
done
for r in $RUNS; do mkdir -p /home/alex/runs/dscc/$r; rsync -aq b3:/home/alex/runs/dscc/$r/{held_final.json,held_final_avg.json,history.json,train.log,run_record.json,model.pt,model_avg.pt,bz_readout.json} /home/alex/runs/dscc/$r/ 2>/dev/null; done
python3 defect-perovskite/p2c_report.py --bw /home/alex/runs/dscc/dscc_p2cwbw2_s0 /home/alex/runs/dscc/dscc_p2cwbw2_s2 /home/alex/runs/dscc/dscc_p2cwbw2_s6 \
  --bs /home/alex/runs/dscc/dscc_p2v8wbs_s0 /home/alex/runs/dscc/dscc_p2v8wbs_s2 /home/alex/runs/dscc/dscc_p2v8wbs_s6 --out $P/P2C2_GATES.json --md $P/step1_bw2.md > $P/report.out 2>&1 || { echo "REPORT_FAILED $(date)" >> $P/P2C2_STATUS; exit 1; }
# seed-median B_w checkpoint by held-out total force RMSE (model_avg)
MED=$(python3 -c "
import json; g=json.load(open('$P/P2C2_GATES.json')); d={int(s):v['force_rmse'] for s,v in g['per_seed']['bw'].items()}; s=sorted(d,key=d.get); print(s[len(s)//2])")
echo "REPORT_DONE $(date): seed-median B_w = seed $MED; $(grep -o 'B_w kept.*\|Step 1 force gate FAILED.*' $P/step1_bw2.md | head -1 | cut -c1-160)" >> $P/P2C2_STATUS
ssh b3 "cd $W && export PATH=\$HOME/micromamba/envs/py13/bin:\$PATH PYTHONPATH=$W TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=GPU-66ce2688-e79c-b9cd-cc5d-2d0396ba23ef && \
  nohup python defect-perovskite/p2c_step0.py --model /home/alex/runs/dscc/dscc_p2cwbw2_s$MED/model_avg.pt --tag bw2_s$MED --out_dir $P/bw2_s$MED --stages 0.2 0.3 > $P/bw2_s$MED.out 2>&1; echo LARGEN_DONE \$? \$(date) >> $P/P2C2_STATUS_B3" 
rsync -aq b3:$P/bw2_s$MED $P/ 2>/dev/null; rsync -aq b3:$P/P2C2_STATUS_B3 $P/ 2>/dev/null
echo "LARGEN_PULLED $(date)" >> $P/P2C2_STATUS
