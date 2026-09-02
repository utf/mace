#!/usr/bin/env bash
# Premise test: does Stage 3 train with ONLY the float64 fix in?
#
# The rerun in the direction block is premised on "biased gradient (frozen P) + pathological
# random init (atomic limit)". The NaN triage found a third cause that was live the whole
# time: the occupation solve was running in float32 and returning NaN intermittently, at BOTH
# cell sizes. Stage 3's seeds trained through that. So part of the failure-to-start may be
# arithmetic rather than landscape, and the premise needs checking before a day goes into the
# P-backward and the Harrison init are specified against it.
#
# Two arms, six seeds each, everything else exactly as the last Stage-3 run:
#   lr 0.01 -- the rate that previously did not train AT ALL (5.34 -> 4.81 over 40 epochs)
#   lr 0.05 -- the rate that trained 4 of 6, with 2 stuck on the initial plateau
#
# NO P-backward, NO Harrison init, NO c-shift, NO warmup. The only change since those runs is
# that the eigensolve and occupation solve are float64.
#
# Reads:
#   lr 0.01 now trains        -> the lr anomaly was arithmetic; the premise needs revising
#   lr 0.01 still flat        -> landscape, as diagnosed; sections 2 and 3 proceed unchanged
#   lr 0.05 now 6/6           -> the failure-to-start was float32, not the atomic-limit init
#   lr 0.05 still 4/6         -> the init diagnosis stands independently
#
# b3 GPUs 4-7 only.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
ARCH=$R/r2_h3_anneal_s5/r2_h3_anneal_s5.model
BASE=$R/e0_base_s1/e0_base_s1.model
cd "$W" || exit 1
for f in "$ARCH" "$BASE"; do [ -f "$f" ] || { echo "ABORT: missing $f"; exit 1; }; done
# The float64 fix must actually be on THIS machine. Four sync failures in one day.
grep -q 'H = H.double()' mace/modules/defect_counting.py || {
    echo "ABORT: stale defect_counting.py -- the float64 fix is not on this machine"; exit 1; }

cell () {  # gpu lr seedstart tag
    CUDA_VISIBLE_DEVICES=$1 python defect-perovskite/stage_run.py \
        --arch "$ARCH" --base "$BASE" --stage 3 --madelung on \
        --frames 48 --epochs 60 --lr "$2" --seeds 3 --seed-start "$3" --e-gap 2.4 \
        --save-dir "$R/premise_models" --device cuda \
        --out "$R/$4.json" > "$R/$4.log" 2>&1
    echo "  done $4 (exit $?)"
}

echo "=== premise test starting $(date +%F' '%H:%M:%S) ==="
cell 4 0.01 1 pre_lr01_a &
cell 5 0.01 4 pre_lr01_b &
cell 6 0.05 1 pre_lr05_a &
cell 7 0.05 4 pre_lr05_b &
wait
echo "=== premise test complete $(date +%F' '%H:%M:%S) ==="

python - <<'PY' | tee "$R/premise_summary.txt"
import json, glob
import numpy as np
for tag, label, prev in (("pre_lr01", "lr 0.01", "was: 0/6 trained, 5.34 -> 4.81"),
                         ("pre_lr05", "lr 0.05", "was: 4/6 trained, 2 stuck at ~3")):
    rows = []
    for f in sorted(glob.glob(f'/home/alex/runs/{tag}_[ab].json')):
        try:
            rows += [r for r in json.load(open(f)) if 'error' not in r]
        except Exception:
            pass
    if not rows:
        print(f"{label}: no rows")
        continue
    conv = [r for r in rows if r.get('force_final', 9) < 0.05]
    print(f"\n{label}  ({prev})")
    print(f"  TRAINED {len(conv)}/{len(rows)}")
    if conv:
        print(f"  axial_red {np.mean([r['axial_red'] for r in conv]):+.3f}"
              f"+-{np.std([r['axial_red'] for r in conv]):.3f}"
              f"  rmse_all {np.mean([r['rmse_all'] for r in conv]):.1f}"
              f"  N_eff {np.mean([r['neff'] for r in conv]):6.2f}")
    for r in sorted(rows, key=lambda x: x['seed']):
        stuck = r.get('force_final', 9) >= 0.05
        print(f"    seed {r['seed']}  force {r.get('force_final', float('nan')):.5f}"
              f"  axial_red {r['axial_red']:+.3f}  rmse {r['rmse_all']:6.1f}"
              f"  N_eff {r['neff']:6.2f}" + ("   STUCK" if stuck else ""))
PY
