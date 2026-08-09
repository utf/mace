#!/usr/bin/env bash
# Wrapper: run the stage B continuation, then append its analysis. Kept as a file rather
# than an inline `bash -c` string -- nested quoting in the inline form has silently
# produced a syntax error and run the analysis against stale results.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/runs/pipeline_results.md"
"${HERE}/run_stage_b_continue.sh"
status=$?
{
    echo
    echo "## Stage B — continued to 140 total epochs (exit ${status})"
    echo '```'
    "${HERE}/analyse_runs.sh" b_128ch
    echo '```'
} >> "$OUT"
echo "wrapper done (exit ${status})"
