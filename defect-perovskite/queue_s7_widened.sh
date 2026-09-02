#!/usr/bin/env bash
# Step 1 branch + step 2 + step 3: the widened rerun, and the forward-only trio beside it.
#
# WIDENING, from the audit: gamma 1.0 -> 3.0 eV. Every chlorine was pinned at the old bound
# in 6/6 seeds (pre-tanh -3.60 +- 0.19, 100% of Cl, 0% of Cs and Pb), so those parameters
# were frozen at sech^2 ~ 0.004. 3 eV is the scale of the missing-anion Madelung shift.
# The hop bound stays at 0.5: 19% saturated is real but secondary, and moving two things at
# once makes neither attributable.
#
# SMEARING: Gaussian sigma = 0.05 eV, the label pipeline's own convention (doped ISMEAR = 0).
#
# PRE-REGISTERED, before the run: if every Cl was pinned at the SAME value the correction is
# a rigid per-species shift and cannot carry a d-trend, so F4's slope should NOT move
# materially. If it does move, that reading is wrong and the anion offset was carrying
# d-dependence. Either way the answer is in s7_f4.json.
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
grep -q 'ON_SITE_RANGE_DEFAULT = 3.0' mace/modules/defect_counting.py || {
    echo "ABORT: stale tree -- gamma is not widened"; exit 1; }
grep -q 'SMEARING_FAMILY = "gaussian"' mace/modules/defect_counting.py || {
    echo "ABORT: stale tree -- Gaussian smearing absent"; exit 1; }
grep -q 'counting_t_el: float = 0.05' mace/modules/defect_models.py || {
    echo "ABORT: stale tree -- the smearing WIDTH is not the labels' 0.05"; exit 1; }

echo "=== step 2c: Z endpoints, and the eps0/correction degeneracy ==="
CUDA_VISIBLE_DEVICES=7 python -u defect-perovskite/s7_z_endpoints.py \
    --old "$R"/s3wired_models/s3_on_s*.model --new "$R"/s5_models/s3_on_s*.model \
    --out "$R/s7_z_endpoints.json" > "$R/s7_z_endpoints.log" 2>&1 &

# The saved model is checked BEFORE the seeds run, not after. The first attempt at this run
# trained at Gaussian 0.025 -- family switched, width left at the old k_B*300K default -- and
# would have been reported as the labels' convention. Nothing about that raises an error.
echo "=== config check on a freshly built model ==="
CHK_ARCH="$ARCH" CHK_BASE="$BASE" CHK_W="$W" python - <<'PYCHK' || exit 1
import os
import sys
sys.path.insert(0, os.environ["CHK_W"] + "/defect-perovskite")
from stage_run import build
# Paths through the environment, not through the heredoc: a quoted heredoc does not expand
# shell variables, so "$ARCH" arrived as four literal characters and torch.load failed on it.
m = build(os.environ["CHK_ARCH"], os.environ["CHK_BASE"], 1, "cpu", 3, True, 4.0, 2.861,
          lambda *a: None)
g, w = float(m.spectral.h.on_site_range), float(m.spectral.t_el)
fam = getattr(m.spectral, "smearing_family", "missing")
print(f"  gamma {g}  smearing {fam} {w}")
ok = (abs(g - 3.0) < 1e-9) and (fam == "gaussian") and (abs(w - 0.05) < 1e-9)
print("  CONFIG OK" if ok else "  CONFIG WRONG -- refusing to spend seeds")
sys.exit(0 if ok else 1)
PYCHK

echo "=== step 1 branch: six seeds, gamma = 3 eV, Gaussian 0.05 ==="
cell () {  # gpu seedstart tag
    CUDA_VISIBLE_DEVICES=$1 python defect-perovskite/stage_run.py \
        --arch "$ARCH" --base "$BASE" --stage 3 --madelung on \
        --frames 48 --epochs 60 --lr 0.01 --seeds 2 --seed-start "$2" --e-gap 2.4 \
        --save-dir "$R/s7_models" --device cuda \
        --out "$R/$3.json" > "$R/$3.log" 2>&1
    echo "  done $3 (exit $?)"
}
cell 4 1 s7_a &
cell 5 3 s7_b &
cell 6 5 s7_c &
wait
echo "=== training complete $(date +%F' '%H:%M:%S) ==="
python defect-perovskite/summarise_stage3.py s7 > "$R/s7_summary.txt" 2>&1
cat "$R/s7_summary.txt"

echo
echo "=== step 2a/2b + gates on the widened models ==="
MODELS=("$R"/s7_models/s3_on_s*.model)
CUDA_VISIBLE_DEVICES=4 python -u defect-perovskite/s3_dehead_trend.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s7_f4.json" > "$R/s7_f4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u defect-perovskite/s3_lambda_d.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s7_f5.json" > "$R/s7_f5.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 python -u defect-perovskite/s3_dilution.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s7_dilution.json" \
    > "$R/s7_dilution.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 python -u defect-perovskite/s7_smearing_scan.py \
    --models "${MODELS[@]}" --device cuda --out "$R/s7_smearing.json" \
    > "$R/s7_smearing.log" 2>&1 &
wait
echo "=== step 1-3 complete $(date +%F' '%H:%M:%S) ==="
for f in s7_z_endpoints s7_f4 s7_f5 s7_dilution s7_smearing; do
    echo; echo "--- $f ---"
    grep -vE "Warning|warn|^ *$|shape = |torch.load|openequivariance|alternative|visitor|WARNING" \
        "$R/$f.log" | tail -14
done
