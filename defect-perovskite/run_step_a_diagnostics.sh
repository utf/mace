#!/usr/bin/env bash
# Waits for the Step A training to finish, then runs every acceptance diagnostic and
# leaves a report. Each stage records its exit code, so a crash is distinguishable from a
# section that legitimately has nothing to say.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$(cd "${HERE}/.." && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
NAME="${NAME:-perov_lr_stepA_s1}"
MODEL="$HOME/runs/${NAME}/${NAME}.model"
REPEATS="${REPEATS:-2,2,2 3,2,2 3,3,2 3,3,3 4,3,3}"
REPORT="$HOME/runs/step_a_report.md"
STATUS="$HOME/runs/step_a_status.tsv"
: > "$STATUS"

record () {                  # record <stage> <exit-code> <logfile>
    local stage="$1" code="$2" log="$3"
    if [ "$code" -eq 0 ]; then
        printf '%s\tOK\t\n' "$stage" >> "$STATUS"
    else
        printf '%s\tFAILED(%s)\t%s\n' "$stage" "$code" \
            "$(grep -E "Error|Traceback|error:" "$log" 2>/dev/null | tail -1 \
               | tr -d '\t' | cut -c1-160)" >> "$STATUS"
    fi
}

echo "waiting for ${NAME} to finish"
while pgrep -f "cli\.run_train.*${NAME}" >/dev/null 2>&1; do sleep 120; done
sleep 30
if [ ! -f "$MODEL" ]; then
    echo "TRAINING PRODUCED NO MODEL at $MODEL" | tee "$REPORT"
    exit 1
fi

# 1. Term decomposition -- the gate on pol^2 and host.pol.
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 python -u "${HERE}/lr_term_scaling.py" \
    --model "$MODEL" --mode baseline \
    > "${HERE}/${NAME}_decomposition.txt" 2>&1
record "decomposition" "$?" "${HERE}/${NAME}_decomposition.txt"

# 2. Where the carrier actually sits: real frames, several sites, per species.
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 python -u "${HERE}/alpha_audit.py" \
    --model "$MODEL" --frames 4 --repeats 2,2,2 3,3,2 --sites 3 \
    > "$HOME/runs/step_a_alpha.log" 2>&1
record "alpha_audit" "$?" "$HOME/runs/step_a_alpha.log"

# 3. Relaxed ladders, periodic and dilute.
for variant in "" "--dilute"; do
    tag="stepA"; [ -n "$variant" ] && tag="stepA_dilute"
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=40 python -u "${HERE}/perovskite_size_test.py" \
        --model "$MODEL" --repeats ${REPEATS} --site 0 --fmax 0.05 --steps 300 \
        --out-json "${HERE}/perov_size_${tag}.json" ${variant} \
        > "$HOME/runs/step_a_ladder_${tag}.log" 2>&1
    record "ladder:${tag}" "$?" "$HOME/runs/step_a_ladder_${tag}.log"
    CUDA_VISIBLE_DEVICES="" python -u "${HERE}/plot_perov_size.py" \
        "${HERE}/perov_size_${tag}.json" --out "${HERE}/perov_size_${tag}.png" \
        >> "$HOME/runs/step_a_ladder_${tag}.log" 2>&1
    record "plot:${tag}" "$?" "$HOME/runs/step_a_ladder_${tag}.log"
done

# 4. Dilute magnitudes, now on a model whose attention should be repaired.
CUDA_VISIBLE_DEVICES="" python -u "${HERE}/dilute_impact.py" \
    "${HERE}/perov_size_stepA.json" "${HERE}/perov_size_stepA_dilute.json" \
    > "$HOME/runs/step_a_dilute.log" 2>&1
record "dilute_impact" "$?" "$HOME/runs/step_a_dilute.log"

# 5. The gate table itself.
CUDA_VISIBLE_DEVICES="" python -u "${HERE}/check_gates.py" --name "$NAME" \
    > "$HOME/runs/step_a_gates.log" 2>&1
GATE_CODE=$?
record "check_gates" "$GATE_CODE" "$HOME/runs/step_a_gates.log"

section () { echo; echo "## $1"; echo; }
{
    echo "# Step A: q^pol ablated -- acceptance report"
    echo
    echo "Long-range branch on, q^pol off, plus the k = 0 neutralising background and the"
    echo "delocalised-carrier exemption guard. Gates are on DRIFT, not absolute delta_lr:"
    echo "host.carrier is -1.906 eV, constant, and physical."
    echo
    echo "## STAGE STATUS"
    echo
    echo '```'
    if grep -q "FAILED" "$STATUS" 2>/dev/null; then
        echo "SOME STAGES FAILED -- results below are INCOMPLETE:"; echo
    fi
    column -t -s "$(printf '\t')" "$STATUS" 2>/dev/null || cat "$STATUS"
    echo '```'

    section "Acceptance gates"
    echo '```'
    sed 's/\x1b\[[0-9;]*m//g' "$HOME/runs/step_a_gates.log" 2>/dev/null || echo "(n/a)"
    echo '```'

    section "Training"
    echo '```'
    grep -A6 "Error-table on TRAIN and VALID" "$HOME/runs/${NAME}.log" 2>/dev/null \
        | tail -5 | sed 's/\x1b\[[0-9;]*m//g' || echo "(n/a)"
    echo
    grep "carrier channels" "$HOME/runs/${NAME}.log" 2>/dev/null | tail -1 \
        | sed 's/\x1b\[[0-9;]*m//g' | fold -w 110 || echo "(n/a)"
    echo '```'

    section "Term decomposition"
    echo '```'
    sed -n '/Decomposition/,$p' "${HERE}/${NAME}_decomposition.txt" 2>/dev/null \
        | head -30 || echo "(n/a)"
    echo '```'

    section "Attention: where the carrier sits"
    echo '```'
    sed -n '/REAL V_Cl+ FRAMES/,$p' "$HOME/runs/step_a_alpha.log" 2>/dev/null \
        | grep -E "partic|top5|frames|SUPERCELL|PRISTINE|->" | head -30 || echo "(n/a)"
    echo '```'

    section "Relaxed ladders"
    for tag in stepA stepA_dilute; do
        echo "### ${tag}"
        echo '```'
        sed -n '/repeat/,/^$/p' "$HOME/runs/step_a_ladder_${tag}.log" 2>/dev/null \
            | grep -vE "Warning|warn" || echo "(n/a)"
        grep -A6 "drift from the smallest" "$HOME/runs/step_a_ladder_${tag}.log" 2>/dev/null
        echo '```'
    done

    section "Dilute"
    echo "Magnitudes are only physical if the attention above is localised: the correction"
    echo "takes q_carrier = a alpha as input, so a sublattice-smeared alpha gives a"
    echo "sublattice-smeared monopole."
    echo '```'
    cat "$HOME/runs/step_a_dilute.log" 2>/dev/null || echo "(n/a)"
    echo '```'

    echo
    echo "Figures: perov_size_stepA.png, perov_size_stepA_dilute.png in defect-perovskite/."
} > "$REPORT"
echo "STEP_A_DIAGNOSTICS_DONE -- report at $REPORT (gates exit ${GATE_CODE})"
