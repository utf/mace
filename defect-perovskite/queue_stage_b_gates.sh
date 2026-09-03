#!/usr/bin/env bash
# Stage B gates (spec section 4), scored by the existing scripts against the Stage A'
# references. Waits on the CONDITION -- the six model files -- never on a process.
#
#   1 regression   b10_adoption with --reference-json (the A' references) and the A' folds
#   2 F4           b10's criterion 4 against the new reference (within 1.5x, >= 4/6)
#   3 c-consistency c4_stage_b_extras (report)
#   4 F10          b4_per_atom_corrections (ligand-Cl - bulk-Cl > 50 meV, >= 4/6)
#   5 gap          b10 (pristine gap 2.4 +- 0.1); pinned continuum = unit test; bandwidth
#                  from each run's init-gate log line
#   6 dilution     s3_dilution (R <= 1.3, bound fraction) and b6_depth_edges (depth, spread)
#   7 stops / L_b  c4_stage_b_extras
#   8 participation b13_participation (--on the Stage B seeds, --off the s7 cohort as the
#                  head-only comparison, --baseline the joint E_LR-off control)
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
TAG="${TAG:-stageb}"
REF="${REF:-$R/aprime_reference.json}"
cd "$W" || exit 1

MODELS=()
for s in 1 2 3 4 5 6; do MODELS+=("$R/${TAG}_s$s/${TAG}_s$s.model"); done
echo "=== waiting for the six Stage B models ==="
while true; do
    ok=1; for m in "${MODELS[@]}"; do [ -f "$m" ] || ok=0; done
    [ "$ok" = 1 ] && break
    sleep 120
done
[ -f "$REF" ] || { echo "ABORT: missing reference $REF"; exit 1; }
echo "=== models present $(date +%F' '%H:%M:%S); scoring ==="
mkdir -p "$R/${TAG}_models"
for s in 1 2 3 4 5 6; do ln -sfn "$R/${TAG}_s$s/${TAG}_s$s.model" "$R/${TAG}_models/${TAG}_s$s.model"; done

CUDA_VISIBLE_DEVICES=4 python -u defect-perovskite/b10_adoption.py \
    --models "${MODELS[@]}" --fold-prefix aprime_f --cf-runs "$R" \
    --reference-json "$REF" --device cuda --out "$R/${TAG}_adopt.json" \
    > "$R/${TAG}_adopt.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=5 python -u defect-perovskite/b4_per_atom_corrections.py \
    --models "${MODELS[@]}" --device cuda --out "$R/${TAG}_peratom.json" \
    > "$R/${TAG}_peratom.log" 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=6 python -u defect-perovskite/s3_dilution.py \
    --models "${MODELS[@]}" --device cuda --out "$R/${TAG}_dilution.json" \
    > "$R/${TAG}_dilution.log" 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=7 python -u defect-perovskite/c4_stage_b_extras.py \
    --models "${MODELS[@]}" --device cuda --out "$R/${TAG}_extras.json" \
    > "$R/${TAG}_extras.log" 2>&1 &
P4=$!
wait $P1 $P2 $P3 $P4
CUDA_VISIBLE_DEVICES=4 python -u defect-perovskite/b6_depth_edges.py \
    --arms "prejoint=$R/s7_models" "${TAG}=$R/${TAG}_models" \
    --device cuda --out "$R/${TAG}_depth.json" > "$R/${TAG}_depth.log" 2>&1 &
P5=$!
CUDA_VISIBLE_DEVICES=5 python -u defect-perovskite/b13_participation.py \
    --on "${MODELS[@]}" --off "$R"/s7_models/s3_on_s*.model \
    --baseline "$R"/nolr_s1/nolr_s1.model "$R"/nolr_s2/nolr_s2.model \
               "$R"/nolr_s3/nolr_s3.model "$R"/nolr_s4/nolr_s4.model \
    --device cuda --out "$R/${TAG}_participation.json" \
    > "$R/${TAG}_participation.log" 2>&1 &
P6=$!
wait $P5 $P6
for f in ${TAG}_adopt ${TAG}_peratom ${TAG}_dilution ${TAG}_extras ${TAG}_depth ${TAG}_participation; do
    echo; echo "--- $f ---"
    grep -avE "Warning|warn|openequivariance|falling back|visitor|shape = |_Jd|np.reshape|UNDER-DET|alternative|detach|^ *$" \
        "$R/$f.log" | tail -45
done
echo "=== gates complete $(date +%F' '%H:%M:%S) ==="
