#!/usr/bin/env bash
# Step 2: the R1 clamps re-run with the carrier-field response channel on.
#
# R1 without the channel: every clamp at axial_red ~0 against `free` ~0.8, and flat between
# hub2 and its distance-matched controls. The head has no way to give a COMPACT state the
# ~6 A force footprint E0 measured, because its forces come from dt/dR and deps/dR and both
# die with the hopping envelope. Electrostatics has the reach.
#
# Gate, both clauses required:
#   1. axial_red(hub2) rises from ~0 to >= 0.4 with rmse_nbhd(hub2) falling toward `free`
#      -- the mechanism gives compact states the footprint;
#   2. hub2 beats lig2 and rand2 by more than the seed spread -- a physical kernel should
#      make hub and ligand distinguishable through the SHAPE of the response.
#
# Prediction recorded before running: clause 1 passes, clause 2 is the real unknown.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PYTHONPATH=$PWD
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
R=$HOME/runs
ARCH=${ARCH:-$R/r2_h3_anneal_s5/r2_h3_anneal_s5.model}
BASE=${BASE:-$R/e0_base_s1/e0_base_s1.model}

run () { local gpu=$1 out=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu python defect-perovskite/r1_matrix.py --arch "$ARCH" \
        --base "$BASE" --head h3 --response --frames 48 --epochs 40 --masks "$@" \
        --losses full nbhd --seeds-2atom 5 --seeds-other 3 --device cuda --out "$out" \
        > "$R/step2_$(basename "$out" .json).log" 2>&1
    echo "  done $* (exit $?)"
}

echo "=== step 2 (response channel) starting $(date +%F' '%H:%M:%S) ==="
run 0 $R/step2_hub.json  hub2 &
run 1 $R/step2_lig.json  lig2 &
run 2 $R/step2_rand.json rand2_0 rand2_1 rand2_2 &
run 3 $R/step2_rest.json nbhd12 free &
wait
echo "=== step 2 complete $(date +%F' '%H:%M:%S) ==="
