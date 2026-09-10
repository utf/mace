#!/bin/bash
# Poll the local A1.1 arm every 300 s: which seed is running, its epoch, and any finals.
while true; do
  d=$(ls -td /home/alex/runs/dscc/dscc_w3a11_phi0_s*/ 2>/dev/null | head -1)
  n=$(basename "$d" 2>/dev/null)
  ep=$(python3 -c "
import json,sys
try:
    h=json.load(open(sys.argv[1]))
    print('%d force %.3e dEp95 %.3f'%(h[-1]['epoch'], h[-1]['force'], h[-1].get('on_site_shift_eV_p95') or -1))
except Exception: print('setup')
" "$d/history.json" 2>/dev/null)
  done_n=$(ls -d /home/alex/runs/dscc/dscc_w3a11_phi0_s*/held_final.json 2>/dev/null | wc -l)
  left=$(grep -c . /home/alex/runs/w3a11_local_queue.txt 2>/dev/null || echo 0)
  echo "$(date +%F\ %T) running $n epoch $ep | finished $done_n | queued $left"
  sleep 300
done
