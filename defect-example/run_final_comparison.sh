#!/usr/bin/env bash
# Three-model size-transferability comparison (HANDOFF section 29).
#
#   1. small data,  L1        -- b_128ch_L1_s1     (already measured)
#   2. larger data, L1        -- best bfull_128ch_L1_*
#   3. larger data, L1 + LR   -- efull_128ch_L1_*
#
# The question is whether more data, more capacity and the long-range term hold `alpha` on
# the defect as the cell grows, or whether softmax dilution is structural and survives all
# three.
#
# Scheduling: the extensivity sweeps run on CPU (float64) and the training on GPU, so leg
# 2's sweep overlaps leg 3's training instead of queueing behind it. float64 on this A4000
# runs at 1/32 rate, so CPU is genuinely the right place for the sweeps, not a fallback.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
OUT="$HOME/runs/final_comparison.md"
RUNS="$HOME/runs"

# ---- wait for stage B on the full dataset -------------------------------------------
while pgrep -f "run_stage_b_full\.sh" >/dev/null 2>&1; do sleep 120; done
echo "stage B (full) finished"

# ---- pick the best L1 by delta-energy RMSE on the SAVED checkpoint --------------------
# The saved artefact's error table, not the last epoch: those disagree, and by a lot
# (section 26). dE is the criterion because it carries the headline observable.
best=""; best_de=""
for log in "$RUNS"/bfull_128ch_L1_s*.log; do
    [ -f "$log" ] || continue
    de=$(grep -A6 "Error-table on TRAIN and VALID" "$log" | grep valid_Default \
         | tail -1 | awk -F'|' '{gsub(/ /,"",$5); print $5}')
    [ -n "$de" ] || continue
    echo "  $(basename "$log" .log): RMSE dE = $de meV"
    if [ -z "$best_de" ] || awk "BEGIN{exit !($de < $best_de)}"; then
        best_de=$de; best=$(basename "$log" .log)
    fi
done
[ -n "$best" ] || { echo "no finished bfull L1 run found" >&2; exit 1; }
echo "best L1 on the full dataset: ${best} (RMSE dE = ${best_de} meV)"
BEST_MODEL="$RUNS/$best/$best.model"
SEED="${best##*_s}"

# ---- leg 3 training starts now, on the GPU -------------------------------------------
# A plain background child, not setsid: this script is itself detached, and setsid would
# reparent the job so `wait` below could not follow it.
SEED="$SEED" NAME="efull_128ch_L1_s${SEED}" \
    "${HERE}/run_stage_e_full.sh" > "$RUNS/stage_e_full.log" 2>&1 &
stage_e_pid=$!
echo "stage E (full + LR) launched on the GPU, pid ${stage_e_pid}"

# ---- leg 2 sweep runs concurrently, on CPU -------------------------------------------
echo "=== extensivity: larger data, L1 (${best}) ==="
MODEL="$BEST_MODEL" THREADS=32 \
    "${HERE}/run_size_convergence.sh" > "$RUNS/size_convergence_bfull.log" 2>&1
cp -f "${HERE}/size_convergence.json" "${HERE}/size_convergence_bfull.json" 2>/dev/null
cp -f "${HERE}/size_convergence.png" "${HERE}/size_convergence_bfull.png" 2>/dev/null
python -u "${HERE}/alpha_dilution.py" --model "$BEST_MODEL" --device cpu \
    > "$RUNS/alpha_dilution_bfull.log" 2>&1

# ---- wait for leg 3, then sweep it ----------------------------------------------------
wait $stage_e_pid
e_status=$?
E_MODEL="$RUNS/efull_128ch_L1_s${SEED}/efull_128ch_L1_s${SEED}.model"
if [ $e_status -eq 0 ] && [ -f "$E_MODEL" ]; then
    echo "=== extensivity: larger data, L1 + LR ==="
    # --dilute stays OFF: the dilute term is size-independent by construction, so it
    # would make the long-range model pass this test trivially rather than be measured.
    MODEL="$E_MODEL" THREADS=32 \
        "${HERE}/run_size_convergence.sh" > "$RUNS/size_convergence_efull.log" 2>&1
    cp -f "${HERE}/size_convergence.json" "${HERE}/size_convergence_efull.json" 2>/dev/null
    cp -f "${HERE}/size_convergence.png" "${HERE}/size_convergence_efull.png" 2>/dev/null
    python -u "${HERE}/alpha_dilution.py" --model "$E_MODEL" --device cpu \
        > "$RUNS/alpha_dilution_efull.log" 2>&1
else
    echo "stage E did not produce a model (exit ${e_status}); leg 3 skipped" >&2
fi

# ---- collect ---------------------------------------------------------------------------
{
    echo "# Three-model size-transferability comparison"
    echo
    echo "Best L1 on the full dataset: \`${best}\` (RMSE dE = ${best_de} meV)."
    for leg in "small data L1:b_128ch_L1_s1:" \
               "larger data L1:${best}:_bfull" \
               "larger data L1 + LR:efull_128ch_L1_s${SEED}:_efull"; do
        label="${leg%%:*}"; rest="${leg#*:}"; name="${rest%%:*}"; suffix="${rest##*:}"
        echo
        echo "## ${label} — \`${name}\`"
        echo '```'
        sed -n '/repeat/,/correction spread/p' \
            "$RUNS/size_convergence${suffix}.log" 2>/dev/null \
            | grep -vE "Warning|warn" || echo "(not available)"
        echo '```'
        echo '```'
        grep -E "alpha on shell|implied gap" "$RUNS/alpha_dilution${suffix}.log" \
            2>/dev/null || echo "(not available)"
        echo '```'
    done
} > "$OUT"
echo "wrote $OUT"
