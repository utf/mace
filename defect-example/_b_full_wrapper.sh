#!/usr/bin/env bash
# Wait for the running stage B continuation, then run stage B on the full dataset and
# append its analysis. A file, not an inline `bash -c`: nested quoting there has twice
# produced a syntax error that ran the analysis against stale results.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/runs/pipeline_results.md"
while pgrep -f "cli\.run_train" >/dev/null 2>&1; do sleep 60; done
"${HERE}/run_stage_b_full.sh"
status=$?
{
    echo
    echo "## Stage B on the FULL 1341-structure dataset, 140 epochs (exit ${status})"
    echo "Probe floor on this split: 46.66 meV (v4 800-frame split was 53.72 — not comparable)."
    echo '```'
    "${HERE}/analyse_runs.sh" bfull_128ch
    echo '```'
} >> "$OUT"
echo "full-dataset stage B done (exit ${status})"
