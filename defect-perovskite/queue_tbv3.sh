#!/usr/bin/env bash
# T-B on V3, b3 GPUs 4-7 (0-3 are off-limits by instruction).
#
#   GPU4  f_m = 0.1   E_gap = 2.4   6 seeds
#   GPU5  f_m = 0.2   E_gap = 2.4   6 seeds
#   GPU6  f_m = 0.3   E_gap = 2.4   6 seeds
#   GPU7  f_m = 0.2   E_gap = 2.2   6 seeds   (the gap sensitivity arm)
#
# 24 cells. E_gap = 2.4 is now the default, from the dataset's own band_edges.json (+/-1.2 eV)
# -- the reference the energy labels are already stated against. 2.2 (literature PBE,
# scalar-relativistic, no SOC, possibly cubic-phase) is the sensitivity arm. The 0.2 eV
# between them moves m by 0.04 eV at f_m = 0.2 and the cap is soft, so no decision turns on it.
#
# SUPERSEDES the 18:45 launch, which ran at decay_init = 2.5 A. That gives
# t(10 A)/t(2.85 A) ~ 6% across ~100 neighbours -- the long-ranged nearly-complete graph whose
# lowest state is the in-phase superatom mode, which is what E2 measured at 10 A reach. Those
# cells could not have measured localisation and are archived under ~/runs/tbv3_superseded/.
# Connectivity is now set through the amplitude at a physical decay length (1.0 A), and the
# realised profile is logged at init and at the end of every cell.
#
# PREDICTIONS, recorded before launch (unchanged from the first attempt):
#   * no breach of E_gap in any cell;
#   * Delta_bind >= m reached only by a level inside the candidate region;
#   * retained mass >= 0.9 in most seeds -- now read together with the interface mass, since
#     retained ~0.5 alone cannot distinguish band-like from a straddling artefact at the join;
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
        --save-dir "$R/tbv3_models" \
        --device cuda --out "$R/tbv3_${tag}.json" > "$R/tbv3_${tag}.log" 2>&1
    echo "  done $tag (exit $?)"
}

echo "=== T-B/V3 starting $(date +%F' '%H:%M:%S) ==="
run 4 0.1 2.4 fm0.1_gap2.4 &
run 5 0.2 2.4 fm0.2_gap2.4 &
run 6 0.3 2.4 fm0.3_gap2.4 &
run 7 0.2 2.2 fm0.2_gap2.2 &
wait
echo "=== T-B/V3 complete $(date +%F' '%H:%M:%S) ==="
