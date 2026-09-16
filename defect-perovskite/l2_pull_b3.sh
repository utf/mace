#!/bin/bash
for s in 1; do until ssh -o BatchMode=yes b3 "test -f ~/runs/dscc/dscc_l2_s$s/held_final.json" 2>/dev/null; do sleep 300; done; sleep 60; rsync -aq b3:runs/dscc/dscc_l2_s$s/ $HOME/runs/dscc/dscc_l2_s$s/; echo "$(date '+%F %T') [L2] dscc_l2_s$s pulled from b3" >> $HOME/runs/lx_watch.log; done
