#!/usr/bin/env bash
# Section 4: the Stage-3 rerun, replacing the old Stage 3 wholesale.
#
# The old seeds are not reseeded or extended -- a pathological init plus a float32 occupation
# solve makes that distribution uninterpretable, so it is discarded rather than repaired.
#
# WHAT IS IN: float64 eigensolve and occupation solve; Harrison term-value initialisation;
# the c-shift calibrated on the init batch; 5-epoch linear warmup; grad-norm clip 1.0; the
# initialisation gate reporting every firing; loss_gap; learned Z. Target lr 0.01.
#
# WHAT IS NOT IN, and this is the one deviation to read the results against: the P-backward
# is built and validated but is NOT yet wired into the force path. The head still evaluates
# forces with a detached P. So this rerun measures section 3's contribution alone, and the
# density-response term remains absent from the parameter gradient. F2 is therefore tested
# only in part -- if 6/6 train here, the init was sufficient by itself; if some still stall,
# that is where the P-backward is expected to act and the reading stays open.
#
# The Stage-2 control is re-run identically at lr 0.01, restoring "everything fixed but the
# edit" -- the earlier control ran at 0.05.
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
grep -q 'H = H.double()'      mace/modules/defect_counting.py || {
    echo "ABORT: stale defect_counting.py -- no float64 fix"; exit 1; }
grep -q 'HARRISON_TERMS'      mace/modules/defect_counting.py || {
    echo "ABORT: stale defect_counting.py -- no Harrison init"; exit 1; }
grep -q 'init gate'           defect-perovskite/stage_run.py || {
    echo "ABORT: stale stage_run.py -- no init gate"; exit 1; }

cell () {  # gpu stage seedstart tag
    CUDA_VISIBLE_DEVICES=$1 python defect-perovskite/stage_run.py \
        --arch "$ARCH" --base "$BASE" --stage "$2" --madelung on \
        --frames 48 --epochs 60 --lr 0.01 --seeds 3 --seed-start "$3" --e-gap 2.4 \
        --save-dir "$R/s3rerun_models" --device cuda \
        --out "$R/$4.json" > "$R/$4.log" 2>&1
    echo "  done $4 (exit $?)"
}

echo "=== Stage-3 rerun starting $(date +%F' '%H:%M:%S) ==="
cell 4 3 1 s3r_a &
cell 5 3 4 s3r_b &
cell 6 2 1 s2r_a &
cell 7 2 4 s2r_b &
wait
echo "=== training complete $(date +%F' '%H:%M:%S) ==="

python - <<'PY' | tee "$R/s3rerun_summary.txt"
import json, glob
import numpy as np
for pat, label in (("s3r_[ab]", "STAGE 3 rerun (counting, s+p)"),
                   ("s2r_[ab]", "STAGE 2 control (bounded, s-only)")):
    rows = []
    for f in sorted(glob.glob(f'/home/alex/runs/{pat}.json')):
        try:
            rows += [r for r in json.load(open(f)) if 'error' not in r]
        except Exception:
            pass
    if not rows:
        print(f"{label}: no rows")
        continue
    conv = [r for r in rows if r.get('force_final', 9) < 0.05]
    trips = [r['seed'] for r in rows if r.get('init_passed') is False]
    m = lambda k, g: np.mean([r[k] for r in g if k in r])
    sd = lambda k, g: np.std([r[k] for r in g if k in r])
    print(f"\n{label}")
    print(f"  TRAINED {len(conv)}/{len(rows)}   init-gate trips: "
          f"{trips if trips else 'none'}")
    if conv:
        print(f"  axial_red {m('axial_red', conv):+.3f}+-{sd('axial_red', conv):.3f}"
              f"  rmse_all {m('rmse_all', conv):.1f}+-{sd('rmse_all', conv):.1f}"
              f"  N_eff {m('neff', conv):6.2f}"
              f"  bandwidth {m('init_bandwidth', conv):.1f} eV")
        if 'pristine_frontier_gap' in conv[0]:
            print(f"  pristine frontier gap {m('pristine_frontier_gap', conv):.3f} eV "
                  f"(target 2.40, gate |delta| <= 0.10)")
    for r in sorted(rows, key=lambda x: x['seed']):
        stuck = r.get('force_final', 9) >= 0.05
        print(f"    seed {r['seed']}  force {r.get('force_final', float('nan')):.5f}"
              f"  axial_red {r['axial_red']:+.3f}  rmse {r['rmse_all']:6.1f}"
              f"  N_eff {r['neff']:6.2f}"
              f"  gap {r.get('pristine_frontier_gap', float('nan')):.3f}"
              + ("   STUCK" if stuck else "")
              + ("   INIT-TRIP" if r.get('init_passed') is False else ""))
PY
