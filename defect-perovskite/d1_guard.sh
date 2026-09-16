#!/bin/bash
# Divergence guard for the from-scratch D1 arms (registered 2026-09-15 14:30): every 5 min read
# each arm's Default validation rows; if the force RMSE is NaN, or exceeds 2x its running minimum
# on 3 consecutive epochs, kill the arm and relaunch it from scratch at lr 0.005 (D1_LR), once.
# The event is logged to ~/runs/d1_guard.log and the relaunched run is named <arm>_lr5e3.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=$W/defect-perovskite; L=$HOME/runs/d1_guard.log
log() { echo "$(date '+%F %T') $*" >> $L; }
declare -A FIRED
check() {   # LOGFILE -> prints DIVERGED if the rule fires
  python3 - "$1" <<'PY'
import re, sys, math
rows = [l for l in open(sys.argv[1], errors="replace") if "Epoch" in l and "head: Default" in l]
f = [float(m.group(1)) if m.group(1) != "nan" else math.nan for m in (re.search(r"RMSE_F=\s*([\d.]+|nan)", r) for r in rows) if m]
if not f: sys.exit(0)
if any(math.isnan(x) for x in f[-3:]): print("DIVERGED nan"); sys.exit(0)
runmin = [min(f[:i + 1]) for i in range(len(f))]
bad = [f[i] > 2.0 * runmin[i - 1] for i in range(1, len(f))]
if len(bad) >= 3 and all(bad[-3:]): print("DIVERGED 3x above 2*min (%s)" % ", ".join("%.1f" % x for x in f[-4:]))
PY
}
while true; do
  for arm in d1_light d1_full_scratch d1_light2 d1_full1; do
    [ -n "${FIRED[$arm]}" ] && continue
    if [ $arm = d1_light ] || [ $arm = d1_light2 ]; then logf=$HOME/runs/$arm.log; else rsync -aq b3:runs/$arm.log $HOME/runs/ 2>/dev/null; logf=$HOME/runs/$arm.log; fi
    v=$(check $logf)
    if [ -n "$v" ]; then
      log "$arm: $v -- killing and relaunching at lr 0.005 as ${arm}_lr5e3"
      case $arm in d1_light) kw=light ;; d1_light2) kw=light2 ;; d1_full_scratch) kw=full-scratch ;; d1_full1) kw=full1 ;; esac
      if [ $arm = d1_light ] || [ $arm = d1_light2 ]; then
        for p in $(pgrep -f "^bash /home/alex/src/mace/.claude/worktrees/size-extensivity/defect-perovskite/d1_run.sh $kw ") $(pgrep -f "^python - $kw $arm"); do kill $p; done; sleep 3
        (cd $W; D1_BATCH=32 D1_LR=0.005 nohup bash $D/d1_run.sh $kw ${arm}_lr5e3 > $HOME/runs/${arm}_lr5e3.log 2>&1 &)
      else
        ssh b3 "for p in \$(pgrep -f '^bash /home/alex/src/mace/.claude/worktrees/size-extensivity/defect-perovskite/d1_run.sh $kw ') \$(pgrep -f '^python - $kw $arm'); do kill \$p; done; sleep 3; cd $W; u5=\$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' '\$1==5 {print \$2}'); D1_BATCH=32 D1_LR=0.005 bash $D/b3_run.sh \$u5 ${arm}_lr5e3 bash $D/d1_run.sh $kw ${arm}_lr5e3" >> $L 2>&1
      fi
      FIRED[$arm]=1
    fi
  done
  sleep 300
done
