#!/usr/bin/env bash
# Arm B and then arm C, after the original chain was stopped.
#
# WHY THIS FILE EXISTS. Arm A completed and was scored. Arm B then failed on every seed
# within a minute: `collect_pristine_centre` -- the pristine pass that MEASURES delta_L --
# ran the image term's bound switch, which raises when delta_L is unset. Arm A never hit it
# because the term is off there. The chain was stopped rather than left to hang in
# `gates armb`'s wait-for-six-models loop, the guard was fixed and unit-tested, and the
# remaining schedule lives here as a NEW file -- editing a script a running bash is reading
# is what nearly killed the chain at 10:45 this morning.
#
# Arm C is included because gate 7 fired on arm A at six seeds: ss-sigma 5/6, sp-sigma 3/6,
# pp-sigma 2/6, pp-pi 4/6, against a pre-registered trigger of "more than 1/6".
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
R=$HOME/runs
export PYTHONPATH=$REPO PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$REPO" || exit 1

NULLS="${NULLS:-$R/aprime_nulls.json}"
REF="${REF:-$R/aprime_reference.json}"
ARMS="${ARMS:-b c}"
[ -f "$REF" ]   || { echo "ABORT: missing $REF"; exit 1; }
[ -f "$NULLS" ] || { echo "ABORT: missing $NULLS"; exit 1; }

for arm in $ARMS; do
    tag="arm$arm"
    for seeds in "1 2 3 4" "5 6"; do
        echo "=== $tag, seeds $seeds  $(date +%F' '%H:%M:%S) ==="
        ARM="$arm" SEEDS="$seeds" GPU0=4 NULLS="$NULLS" bash "$HERE/queue_arms.sh"
    done

    # Refuse to score an arm that did not train. The original chain would have waited for
    # six models for ever; this says so and moves on.
    n=$(ls "$R/${tag}_s"*/"${tag}_s"*.model 2>/dev/null | wc -l)
    if [ "$n" -lt 6 ]; then
        echo "=== ABORT $tag: only $n of 6 models exist; not scoring, not continuing ==="
        exit 1
    fi

    echo "=== gates for $tag  $(date +%F' '%H:%M:%S) ==="
    TAG="$tag" REF="$REF" bash "$HERE/queue_stage_b_gates.sh"
    models=("$R/${tag}_models/${tag}_s"*.model)
    echo "  scoring ${#models[@]} models"
    echo "=== F10 on the corrected form, $tag ==="
    CUDA_VISIBLE_DEVICES=4 python -u "$HERE/c7_centred_f10.py" \
        --models "${models[@]}" --device cuda --out "$R/${tag}_f10.json" 2>&1 | tail -40 || true
    echo "=== 79-atom head slope (gate 9), $tag ==="
    CUDA_VISIBLE_DEVICES=5 python -u "$HERE/b2_size_slopes.py" \
        --models "${models[@]}" --device cuda --out "$R/${tag}_b2.json" 2>&1 | tail -20 || true
    echo "=== post-gates (section 2.2, gate 10, the table), $tag ==="
    bash "$HERE/post_gates.sh" "$tag" 2>&1 | tail -45 || true
done
echo "=== chain B/C complete $(date +%F' '%H:%M:%S) ==="
