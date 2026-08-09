#!/usr/bin/env bash
# Summarise a set of runs: escape epoch, final and best held-out delta metrics, and the
# final per-channel attention state.
#
# Escape is the first epoch whose RMSE_dF falls 10% below that run's OWN epoch-0 value.
# An absolute threshold would be wrong across cell sizes: RMSE_dF averages over every
# atom while the delta forces sit on six, so a larger cell starts lower by dilution alone.
#
#   ./analyse_runs.sh a4_v4 e1c_anneal
set -euo pipefail
RUNS="${RUNS_DIR:-$HOME/runs}"

printf "%-20s %8s %10s %10s %10s\n" run escape final_dE final_dF best_dE
for prefix in "$@"; do
    for f in "$RUNS/${prefix}"*.log; do
        [ -f "$f" ] || continue
        name=$(basename "$f" .log)
        # Deduplicate by epoch, keeping the LAST occurrence, then sort numerically.
        # A restarted run appends a fresh segment to the same log, so the raw file can
        # hold several lines for the same epoch from different attempts; taking them in
        # file order would interleave two trajectories and report a meaningless "final".
        rows=$(grep -E "Epoch [0-9]+: head" "$f" 2>/dev/null \
            | sed -E 's/.*Epoch ([0-9]+).*RMSE_dE= *([0-9.]+).*RMSE_dF= *([0-9.]+).*/\1 \2 \3/' \
            | awk '{seen[$1]=$0} END {for (e in seen) print seen[e]}' \
            | sort -n -k1,1) || true
        [ -n "$rows" ] || { printf "%-20s %8s\n" "$name" "no-epochs"; continue; }
        plateau=$(echo "$rows" | head -1 | awk '{print $3}')
        escape=$(echo "$rows" | awk -v p="$plateau" '$3 < 0.9*p {print $1; exit}')
        final=$(echo "$rows" | tail -1)
        best=$(echo "$rows" | sort -k2 -n | head -1 | awk '{print $2}')
        printf "%-20s %8s %10s %10s %10s\n" "$name" "${escape:-none}" \
            "$(echo "$final" | awk '{print $2}')" "$(echo "$final" | awk '{print $3}')" "$best"
    done
done

echo
echo "final per-channel state (e_maj e_min h_maj h_min):"
for prefix in "$@"; do
    for f in "$RUNS/${prefix}"*.log; do
        [ -f "$f" ] || continue
        line=$(grep "carrier channels" "$f" 2>/dev/null | tail -1 | sed 's/.*channels ([^)]*)://') || true
        [ -n "$line" ] && printf "  %-20s%s\n" "$(basename "$f" .log)" "$line"
    done
done

echo
echo "final logit-seed gain (must be exactly zero on an annealed run):"
for prefix in "$@"; do
    for f in "$RUNS/${prefix}"*.log; do
        [ -f "$f" ] || continue
        g=$(grep "anneal gamma" "$f" 2>/dev/null | tail -1 | sed -E 's/.*gamma=(\[[^]]*\]).*/\1/') || true
        [ -n "$g" ] && printf "  %-20s%s\n" "$(basename "$f" .log)" "$g"
    done
done
