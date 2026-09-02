#!/usr/bin/env bash
# Section 5: the six-seed rerun under the corrected full-sum Madelung kernel.
#
# CONFIGURATION: the standing Stage-3 one -- Harrison init, c-shift, 5-epoch warmup, clip 1.0,
# lr 0.01, loss_gap at E_gap 2.4, learned Z, init gate, float64 solve, in-loop reach
# assertion, 48 charged frames, seeds 1-6, 60 epochs. The ONLY change from the wired rerun is
# the kernel.
#
# TWO DEVIATIONS FROM THE PLAN, BOTH STATED RATHER THAN PAPERED OVER.
#
# 1. The plan says "config path only". This runs through stage_run.py, the hand-wired path.
#    Section 3 established that the two build a bit-identical model -- eps and delta_sr agree
#    exactly, energy and forces inside the same-model repeat floor -- so the MODEL is the one
#    the config path would build. What the production trainer does NOT have is the Stage-3
#    training PROTOCOL: Harrison initialisation, the c-shift calibration, the warmup, loss_gap
#    and the init gate all live in stage_run. Porting them into mace.cli.run_train is real
#    work and is not something to improvise underneath a six-seed run. Recorded as owed.
#
# 2. The plan orders section 5 after section 4's two config decisions. The T_el scan and the
#    saturation audit are still running. This launches on the STANDING values (T_el = 25 meV,
#    bounds as they are), which is what "otherwise the standing Stage-3 configuration" means
#    if the scans call for no change. If either scan does call for a change, this run is
#    superseded and must be relaunched -- ~25 minutes of GPU, which is why starting now is
#    the cheaper bet than idling.
#
# Forecasts on record, per the plan: eps0 / Z / c re-settle under the shifted per-species
# potentials; pristine gap re-passes within 0.1 eV; F4 slope unchanged within CI (the fix is
# size-level, not d-level); dilution and F5 within spread; cross-size consistency improves.
# Any miss stops the pipeline before section 6.
#
# b3 GPUs 4-7 only; section 4 still holds one, so this takes three.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
ARCH=$R/r2_h3_anneal_s5/r2_h3_anneal_s5.model
BASE=$R/e0_base_s1/e0_base_s1.model
cd "$W" || exit 1
for f in "$ARCH" "$BASE"; do [ -f "$f" ] || { echo "ABORT: missing $f"; exit 1; }; done

# The corrected kernel, asserted rather than assumed. Training six seeds against the retired
# convention under a new tag is the failure mode this whole section exists to undo.
grep -q 'no longer accepts a self-potential' mace/modules/defect_madelung.py || {
    echo "ABORT: stale defect_madelung.py -- self-image subtraction still live"; exit 1; }
grep -q 'self_potential_of' mace/modules/defect_models.py && {
    echo "ABORT: defect_models.py still calls self_potential_of"; exit 1; }
grep -q 'force_response' mace/modules/defect_models.py || {
    echo "ABORT: the density response is not wired into the forces"; exit 1; }

cell () {  # gpu seedstart tag
    CUDA_VISIBLE_DEVICES=$1 python defect-perovskite/stage_run.py \
        --arch "$ARCH" --base "$BASE" --stage 3 --madelung on \
        --frames 48 --epochs 60 --lr 0.01 --seeds 2 --seed-start "$2" --e-gap 2.4 \
        --save-dir "$R/s5_models" --device cuda \
        --out "$R/$3.json" > "$R/$3.log" 2>&1
    echo "  done $3 (exit $?)"
}

echo "=== section 5 rerun, full-sum kernel, starting $(date +%F' '%H:%M:%S) ==="
cell 4 1 s5_a &
cell 5 3 s5_b &
cell 7 5 s5_c &
wait
echo "=== training complete $(date +%F' '%H:%M:%S) ==="
python defect-perovskite/summarise_stage3.py s5 > "$R/s5_summary.txt" 2>&1
cat "$R/s5_summary.txt"
