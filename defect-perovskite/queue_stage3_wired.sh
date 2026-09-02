#!/usr/bin/env bash
# Section 2: the six Stage-3 seeds again, under the WIRED density response.
#
# Identical to the last rerun in every other respect -- Harrison init, c-shift, 5-epoch
# warmup, clip 1.0, lr 0.01, loss_gap at E_gap = 2.4, learned Z, init gate on, float64 solve,
# 48 charged frames, 60 epochs, same seeds 1-6. The ONLY change is that the force loss's
# parameter gradient now carries `-Tr(dD/dtheta . dH/dR)`, the density response.
#
# The Stage-2 control is NOT re-run: it has no counting head, so the wiring cannot touch it,
# and the numbers from the last rerun stand as the comparison.
#
# Forecast on record before launch: 6/6 train again, axial_red and N_eff within the spread of
# the frozen-P rerun (+0.519 +- 0.008, N_eff 2.92). The response term measured 128x the
# frozen-P term in d(force loss)/d(v0_raw), but AdamW normalises per parameter, so a large
# gradient need not mean a different trajectory. If the numbers DO move, that is the finding.
#
# Three cells of two seeds rather than two of three: one more GPU, one third less wall clock.
# b3 GPUs 4-7 only.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
ARCH=$R/r2_h3_anneal_s5/r2_h3_anneal_s5.model
BASE=$R/e0_base_s1/e0_base_s1.model
cd "$W" || exit 1
for f in "$ARCH" "$BASE"; do [ -f "$f" ] || { echo "ABORT: missing $f"; exit 1; }; done

# The wiring must be present in BOTH files. A stale sync that trained the frozen-P gradient
# under a wired banner would be indistinguishable from the real thing in the output json.
grep -q 'force_out'          mace/modules/defect_counting.py || {
    echo "ABORT: stale defect_counting.py -- no force_out plumbing"; exit 1; }
grep -q '_FermiDensitySum'   mace/modules/defect_counting.py || {
    echo "ABORT: stale defect_counting.py -- no multi-fill density Function"; exit 1; }
grep -q 'force_response'     mace/modules/defect_models.py || {
    echo "ABORT: stale defect_models.py -- the response never reaches the forces"; exit 1; }
grep -q 's/epoch'            defect-perovskite/stage_run.py || {
    echo "ABORT: stale stage_run.py -- no epoch timing"; exit 1; }

echo "=== section-1 gate on this machine, before any seed is spent ==="
CUDA_VISIBLE_DEVICES=4 python defect-perovskite/s1_wiring_check.py \
    --arch "$ARCH" --base "$BASE" > "$R/s1_wiring_check_b3.log" 2>&1
if [ $? -ne 0 ]; then
    echo "ABORT: the wiring check FAILED on b3 -- see $R/s1_wiring_check_b3.log"
    grep -E '^[A-E] |SECTION' "$R/s1_wiring_check_b3.log"
    exit 1
fi
grep -E '^[A-E] |SECTION' "$R/s1_wiring_check_b3.log"

cell () {  # gpu seedstart tag
    CUDA_VISIBLE_DEVICES=$1 python defect-perovskite/stage_run.py \
        --arch "$ARCH" --base "$BASE" --stage 3 --madelung on \
        --frames 48 --epochs 60 --lr 0.01 --seeds 2 --seed-start "$2" --e-gap 2.4 \
        --save-dir "$R/s3wired_models" --device cuda \
        --out "$R/$3.json" > "$R/$3.log" 2>&1
    echo "  done $3 (exit $?)"
}

echo "=== Stage-3 wired rerun starting $(date +%F' '%H:%M:%S) ==="
cell 4 1 s3w_a &
cell 5 3 s3w_b &
cell 6 5 s3w_c &
wait
echo "=== training complete $(date +%F' '%H:%M:%S) ==="

python defect-perovskite/summarise_stage3.py s3w > "$R/s3wired_summary.txt" 2>&1
cat "$R/s3wired_summary.txt"
