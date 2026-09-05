#!/usr/bin/env bash
# Plan v8 Stage 1.3: the gate scorers on one arm's six models (TAG=s13a|s13b|s13c for the
# forward-only arms, s13ra|s13rb|s13rc for the retrained ones), plus stage13_forces.
# Waits on the CONDITION -- the six model files -- never on a process. b3 GPUs 4-7.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
TAG="${TAG:?TAG required}"
REF="${REF:-$R/aprime_reference.json}"
GPU0="${GPU0:-4}"
cd "$W" || exit 1

MODELS=()
for s in 1 2 3 4 5 6; do MODELS+=("$R/${TAG}_s$s/${TAG}_s$s.model"); done
echo "=== $TAG: waiting for the six models ==="
while true; do
    ok=1; for m in "${MODELS[@]}"; do [ -f "$m" ] || ok=0; done
    [ "$ok" = 1 ] && break
    sleep 120
done
[ -f "$REF" ] || { echo "ABORT: missing reference $REF"; exit 1; }
echo "=== $TAG: models present $(date +%F' '%H:%M:%S); scoring ==="
mkdir -p "$R/${TAG}_models"
for s in 1 2 3 4 5 6; do ln -sfn "$R/${TAG}_s$s/${TAG}_s$s.model" "$R/${TAG}_models/${TAG}_s$s.model"; done
g1=$GPU0; g2=$((GPU0 + 1)); g3=$((GPU0 + 2)); g4=$((GPU0 + 3))

CUDA_VISIBLE_DEVICES=$g1 python -u defect-perovskite/b10_adoption.py \
    --models "${MODELS[@]}" --fold-prefix aprime_f --cf-runs "$R" \
    --reference-json "$REF" --device cuda --out "$R/${TAG}_adopt.json" \
    > "$R/${TAG}_adopt.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=$g2 python -u defect-perovskite/stage13_forces.py \
    --models "${MODELS[@]}" --device cuda --out "$R/${TAG}_forces.json" \
    > "$R/${TAG}_forces.log" 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=$g3 python -u defect-perovskite/s3_dilution.py \
    --models "${MODELS[@]}" --device cuda --out "$R/${TAG}_dilution.json" \
    > "$R/${TAG}_dilution.log" 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=$g4 python -u defect-perovskite/c4_stage_b_extras.py \
    --models "${MODELS[@]}" --device cuda --out "$R/${TAG}_extras.json" \
    > "$R/${TAG}_extras.log" 2>&1 &
P4=$!
wait $P1 $P2 $P3 $P4
CUDA_VISIBLE_DEVICES=$g1 python -u defect-perovskite/b6_depth_edges.py \
    --arms "prejoint=$R/s7_models" "${TAG}=$R/${TAG}_models" \
    --device cuda --out "$R/${TAG}_depth.json" > "$R/${TAG}_depth.log" 2>&1 &
P5=$!
CUDA_VISIBLE_DEVICES=$g2 python -u defect-perovskite/b13_participation.py \
    --on "${MODELS[@]}" --off "$R"/s7_models/s3_on_s*.model \
    --baseline "$R"/nolr_s1/nolr_s1.model "$R"/nolr_s2/nolr_s2.model \
               "$R"/nolr_s3/nolr_s3.model "$R"/nolr_s4/nolr_s4.model \
    --device cuda --out "$R/${TAG}_participation.json" \
    > "$R/${TAG}_participation.log" 2>&1 &
P6=$!
# F10 (section 7.7): the ligand-Cl minus bulk-Cl correction on the centred channel.
CUDA_VISIBLE_DEVICES=$g3 python -u defect-perovskite/b4_per_atom_corrections.py \
    --models "${MODELS[@]}" --device cuda --out "$R/${TAG}_peratom.json" \
    > "$R/${TAG}_peratom.log" 2>&1 &
P7=$!
wait $P5 $P6 $P7
for f in ${TAG}_adopt ${TAG}_forces ${TAG}_dilution ${TAG}_extras ${TAG}_depth ${TAG}_participation ${TAG}_peratom; do
    echo; echo "--- $f ---"
    grep -avE "Warning|warn|openequivariance|falling back|visitor|shape = |_Jd|np.reshape|UNDER-DET|alternative|detach|^ *$" \
        "$R/$f.log" | tail -45
done
echo "=== $TAG gates complete $(date +%F' '%H:%M:%S) ==="
