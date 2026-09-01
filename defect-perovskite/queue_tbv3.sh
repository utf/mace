#!/usr/bin/env bash
# T-B on V3, b3 GPUs 4-7 (0-3 are off-limits by instruction; GPU3 has failed three times).
#
#   GPU4  f_m = 0.1   E_gap = 2.2   6 seeds
#   GPU5  f_m = 0.2   E_gap = 2.2   6 seeds
#   GPU6  f_m = 0.3   E_gap = 2.2   6 seeds
#   GPU7  f_m = 0.2   E_gap = 2.0, then 2.4   6 seeds each   (the gap sensitivity)
#
# 30 cells. The E_gap = 2.4 arm doubles as a self-consistency check against the dataset's own
# band_edges.json, whose symmetric +/-1.2 eV edges imply a 2.4 eV gap -- that is a
# label-referencing convention rather than a measured gap, which is why 2.2 (literature PBE,
# scalar-relativistic, no SOC, matching the labels) is the default and 2.4 is a sensitivity
# arm rather than the headline.
#
# PREDICTIONS, recorded before launch:
#   * no breach of E_gap in any cell (V1 reached +11.9 eV; V3 should not be able to);
#   * Delta_bind >= m reached only by a level inside the candidate region;
#   * retained mass >= 0.9 in most seeds;
#   * some force-fit cost against the delocalised ceiling (axial_red +0.855, rmse_nbhd 20.6).
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
ARCH=$R/r2_h3_anneal_s5/r2_h3_anneal_s5.model
BASE=$R/e0_base_s1/e0_base_s1.model

for f in "$ARCH" "$BASE"; do
    [ -f "$f" ] || { echo "=== ABORT: missing $f ==="; exit 1; }
done
cd "$W" || exit 1

run () {   # gpu f_m e_gap tag
    local gpu=$1 fm=$2 gap=$3 tag=$4
    CUDA_VISIBLE_DEVICES=$gpu python defect-perovskite/tb_v3.py \
        --arch "$ARCH" --base "$BASE" --frames 48 --epochs 40 --seeds 6 \
        --f-m "$fm" --e-gap "$gap" --n-pristine 64 --pristine-batch 8 \
        --device cuda --out "$R/tbv3_${tag}.json" > "$R/tbv3_${tag}.log" 2>&1
    echo "  done $tag (exit $?)"
}

echo "=== T-B/V3 starting $(date +%F' '%H:%M:%S) ==="
run 4 0.1 2.2 fm0.1_gap2.2 &
run 5 0.2 2.2 fm0.2_gap2.2 &
run 6 0.3 2.2 fm0.3_gap2.2 &
( run 7 0.2 2.0 fm0.2_gap2.0 ; run 7 0.2 2.4 fm0.2_gap2.4 ) &
wait
echo "=== T-B/V3 complete $(date +%F' '%H:%M:%S) ==="
