#!/bin/bash
until ssh -o BatchMode=yes b3 "grep -q READ_DONE ~/runs/d2_light/compare_v2.log" 2>/dev/null; do sleep 300; done
rsync -aq b3:runs/d2_light/compare_v2.json b3:runs/d2_light/les_gates.json /home/alex/runs/d2_light/ 2>/dev/null; rsync -aq b3:runs/d2_light/les_gates.json /home/alex/runs/d2_light/
cd /home/alex/src/mace/.claude/worktrees/size-extensivity; GATE_F=12.3 GATE_E=1.4 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 /home/alex/micromamba/envs/py13/bin/python /home/alex/src/mace/.claude/worktrees/size-extensivity/defect-perovskite/l0_gates.py --log /home/alex/runs/d2_light.log --compare /home/alex/runs/d2_light/compare_v2.json --label d2 --born /home/alex/runs/d2_light/les_gates.json --out /home/alex/runs/d2_light/D2_GATES.json > /home/alex/runs/d2_light/d2_gates.log 2>&1
echo "$(date '+%F %T') d2_light gates (re-read on b3): $(tail -1 /home/alex/runs/d2_light/d2_gates.log)" >> /home/alex/runs/d2_watch.log
