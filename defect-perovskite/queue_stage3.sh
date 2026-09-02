#!/usr/bin/env bash
# Stage 3: Edit 4, the electron-counting head over a uniform s+p basis.
#
# Learned Z, per the coadvisor: Stage 2 showed the runaway was an interaction with the
# unbounded on-site term, and learned-vs-nominal being indistinguishable there is itself the
# evidence the parametrisation is sane.
#
# THE NUMBER TO BEAT is Stage 2's BANDS arm: axial_red +0.030, rmse_all 50.8. That is the
# honest s-only baseline -- what a physically-bounded Hamiltonian achieves once the superatom
# route is shut. Stage 2's arm mean (+0.16) is not the reference; it is contaminated by the
# four seeds that found the route anyway.
#
# WATCH ITEM: any seed whose pristine split fraction exceeds 0.30 is reseeded, not
# interpreted. stage_run marks them; the summary below lists them.
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
grep -q 'superatom_reseed' defect-perovskite/stage_run.py || {
    echo "ABORT: stale stage_run.py on this machine (no superatom watch)"; exit 1; }

cell () {  # gpu seedstart tag
    CUDA_VISIBLE_DEVICES=$1 python defect-perovskite/stage_run.py \
        --arch "$ARCH" --base "$BASE" --stage 3 --madelung on \
        --frames 48 --epochs 40 --seeds 3 --seed-start "$2" --e-gap 2.4 \
        --save-dir "$R/stage3_models" --device cuda \
        --out "$R/$3.json" > "$R/$3.log" 2>&1
    echo "  done $3 (exit $?)"
}

echo "=== Stage 3 starting $(date +%F' '%H:%M:%S) ==="
cell 4 1 s3_a &
cell 5 4 s3_b &
wait
echo "=== Stage 3 training complete $(date +%F' '%H:%M:%S) ==="

python - <<'PY' | tee "$R/stage3_summary.txt"
import json, glob
import numpy as np
rows = []
for f in sorted(glob.glob('/home/alex/runs/s3_*.json')):
    try:
        rows += [r for r in json.load(open(f)) if 'error' not in r]
    except Exception:
        pass
if rows:
    flagged = [r['seed'] for r in rows if r.get('superatom_reseed')]
    keep = [r for r in rows if not r.get('superatom_reseed')]
    m = lambda k, g: np.mean([r[k] for r in g if k in r])
    sd = lambda k, g: np.std([r[k] for r in g if k in r])
    print(f"ALL      n={len(rows)}  axial_red {m('axial_red', rows):+.3f}"
          f"  rmse_all {m('rmse_all', rows):.1f}  N_eff {m('neff', rows):6.2f}")
    if keep:
        print(f"KEPT     n={len(keep)}  axial_red {m('axial_red', keep):+.3f}"
              f"+-{sd('axial_red', keep):.3f}  rmse_all {m('rmse_all', keep):.1f}"
              f"+-{sd('rmse_all', keep):.1f}  N_eff {m('neff', keep):6.2f}"
              f"  null_ratio {m('null_ratio', keep):.3f}")
        print(f"         pristine split fraction {m('split_fraction', keep):.4f}")
    print(f"reseed flagged (split > 0.30): {flagged if flagged else 'none'}")
    print("Stage-2 BANDS baseline to beat: axial_red +0.030, rmse_all 50.8")
    for r in sorted(rows, key=lambda x: x['seed']):
        print(f"  seed {r['seed']}  axial_red {r['axial_red']:+.3f}  "
              f"rmse {r['rmse_all']:.1f}  split {r.get('split_fraction', float('nan')):.4f}"
              f"  gap {r.get('pristine_gap', float('nan')):.3f}"
              + ("  RESEED" if r.get('superatom_reseed') else ""))
PY

echo "=== D-2 on Stage-3 models $(date +%F' '%H:%M:%S) ==="
CUDA_VISIBLE_DEVICES=4 python defect-perovskite/d2_bins.py \
    --models "$R"/stage3_models/s3_on_s*.model --limit 24 \
    --device cuda --out "$R/d2_stage3.json" > "$R/d2_stage3.log" 2>&1
echo "=== Stage 3 complete $(date +%F' '%H:%M:%S) ==="
