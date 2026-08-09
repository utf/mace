#!/usr/bin/env bash
# Queue the remaining stages and analyse each as it lands, so results are summarised
# without waiting on a human. Stages run in sequence because they share one GPU.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
OUT="$HOME/runs/pipeline_results.md"

stage () {
    local name="$1"; shift
    local script="$1"; shift
    echo "=== $(date -u +%H:%M) starting ${name} ===" | tee -a "$OUT"
    if "$script" "$@"; then
        echo "=== ${name} finished ===" | tee -a "$OUT"
    else
        echo "=== ${name} FAILED (exit $?) — continuing to the next stage ===" | tee -a "$OUT"
    fi
}

analyse () {
    { echo; echo "## $1"; echo '```'; "${HERE}/analyse_runs.sh" "${@:2}"; echo '```'; } >> "$OUT"
}

stage "D8.4 (stage C on the seeded variant)" "${HERE}/run_d84_stagec.sh"
analyse "D8.4 — does the linear N-scaling survive seeding?" d84_n286 d84_n398

stage "Stage E (long-range, frozen a)" "${HERE}/run_stage_e.sh"
analyse "Stage E — long-range branch enabled" e_lr

stage "Stage B (capacity)" "${HERE}/run_stage_b.sh"
analyse "Stage B — capacity sweep" b_128ch

echo "PIPELINE COMPLETE" | tee -a "$OUT"
