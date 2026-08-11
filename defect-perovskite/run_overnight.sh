#!/usr/bin/env bash
# Overnight: finish the LR training, run every ladder, and leave a report.
#
# Ladder starts at 640 atoms. The 160-atom cell is dropped on the receptive-field
# criterion, not for being small: its edges are 16.2 x 16.2 x 22.9 A against the 2 * r_max *
# n_layers = 20 A needed for periodic images to fall outside the model's own cutoff, so two
# of three edges are below threshold and it measures self-interaction rather than size.
#
# Three evaluations, chosen so each difference is attributable:
#   nolr            short-range only -- cannot represent the q^2/L image term at all
#   lr              long-range, PERIODIC -- includes the compensating background
#   lr + dilute     long-range, ISOLATED limit
# The lr/dilute difference IS the finite-size correction the branch applies, so measuring
# both quantifies it directly rather than inferring it.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$(cd "${HERE}/.." && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
REPEATS="${REPEATS:-2,2,2 3,2,2 3,3,2 3,3,3 4,3,3}"
REPORT="$HOME/runs/perovskite_report.md"

ladder () {          # name model extra-flags...
    local name="$1" model="$2"; shift 2
    [ -f "$model" ] || { echo "missing $model"; return 1; }
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=40 python -u "${HERE}/perovskite_size_test.py" \
        --model "$model" --repeats ${REPEATS} --site 0 --fmax 0.05 --steps 300 \
        --out-json "${HERE}/perov_size_${name}.json" "$@" \
        > "$HOME/runs/perov_ladder_${name}.log" 2>&1
    CUDA_VISIBLE_DEVICES="" python -u "${HERE}/plot_perov_size.py" \
        "${HERE}/perov_size_${name}.json" --out "${HERE}/perov_size_${name}.png" \
        >> "$HOME/runs/perov_ladder_${name}.log" 2>&1
    echo "  ladder ${name} done"
}

echo "waiting for the long-range training to finish"
while pgrep -f "cli\.run_train.*perov_lr" >/dev/null 2>&1; do sleep 120; done
sleep 30
LR_MODEL="$HOME/runs/perov_lr_s1/perov_lr_s1.model"
NOLR_MODEL="$HOME/runs/perov_nolr_s1/perov_nolr_s1.model"

ladder "nolr" "$NOLR_MODEL"
ladder "lr" "$LR_MODEL"
ladder "lr_dilute" "$LR_MODEL" --dilute

# Attention stability, cheap single points, for both models.
for tag in nolr lr; do
    model="$HOME/runs/perov_${tag}_s1/perov_${tag}_s1.model"
    [ -f "$model" ] || continue
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 python -u "${HERE}/perov_alpha_check.py" \
        --model "$model" --repeats 2,2,2 3,2,2 3,3,2 3,3,3 \
        > "$HOME/runs/perov_alpha_${tag}.log" 2>&1
done

section () { echo; echo "## $1"; echo; }
{
    echo "# CsPbCl3 V_Cl: overnight results"
    echo
    echo "Generated automatically. Ladder from 640 atoms up; the 160-atom cell is excluded"
    echo "because two of its three edges are below the 20 A needed for periodic images to"
    echo "fall outside the model's receptive field."
    echo
    echo "Counter convention: NEUTRAL reference per composition."
    echo '```'
    echo "pristine  n=(0,0,0,0) 2M_s_ref 0  mult 1  q  0"
    echo "V_Cl0     n=(0,0,0,0) 2M_s_ref 1  mult 2  q  0   <- reference"
    echo "V_Cl+     n=(0,0,1,0) 2M_s_ref 1  mult 1  q +1"
    echo '```'

    section "Training"
    for tag in nolr lr; do
        echo "### perov_${tag}_s1"
        echo '```'
        grep -A6 "Error-table on TRAIN and VALID" "$HOME/runs/perov_${tag}_s1.log" 2>/dev/null \
            | tail -5 | sed 's/\x1b\[[0-9;]*m//g' || echo "(not available)"
        echo '```'
        echo "Final attention state:"
        echo '```'
        grep "carrier channels" "$HOME/runs/perov_${tag}_s1.log" 2>/dev/null | tail -1 \
            | sed 's/\x1b\[[0-9;]*m//g' | fold -w 110 || echo "(not available)"
        echo '```'
    done

    section "Size ladders"
    for tag in nolr lr lr_dilute; do
        echo "### ${tag}"
        echo '```'
        sed -n '/repeat/,/^$/p' "$HOME/runs/perov_ladder_${tag}.log" 2>/dev/null \
            | grep -vE "Warning|warn" || echo "(not available)"
        grep -A6 "drift from the smallest" "$HOME/runs/perov_ladder_${tag}.log" 2>/dev/null
        echo '```'
    done

    section "Impact of the dilute option"
    echo "The periodic evaluation carries the compensating background and the q^2/L image"
    echo "term; the dilute one is the isolated limit. Their difference is the finite-size"
    echo "correction the long-range branch is applying to a charged cell."
    echo '```'
    CUDA_VISIBLE_DEVICES="" python - <<'PY' 2>/dev/null || echo "(not available)"
import json
from pathlib import Path
here = Path("/home/alex/src/mace/.claude/worktrees/size-extensivity/defect-perovskite")
try:
    a = json.loads((here / "perov_size_lr.json").read_text())
    b = json.loads((here / "perov_size_lr_dilute.json").read_text())
except FileNotFoundError:
    raise SystemExit(1)
ra = {r["n_host"]: r for r in a["rows"]}
rb = {r["n_host"]: r for r in b["rows"]}
print(f"{'N':>6s} {'eps_opt periodic':>17s} {'dilute':>10s} {'difference':>12s}")
for n in sorted(set(ra) & set(rb)):
    d = (ra[n]["eps_opt_eV"] - rb[n]["eps_opt_eV"]) * 1e3
    print(f"{n:6d} {ra[n]['eps_opt_eV']:17.4f} {rb[n]['eps_opt_eV']:10.4f} {d:10.1f} meV")
print()
print(f"{'N':>6s} {'eps(+/0) periodic':>18s} {'dilute':>10s} {'difference':>12s}")
for n in sorted(set(ra) & set(rb)):
    d = (ra[n]["transition_level_eV"] - rb[n]["transition_level_eV"]) * 1e3
    print(f"{n:6d} {ra[n]['transition_level_eV']:18.4f} "
          f"{rb[n]['transition_level_eV']:10.4f} {d:10.1f} meV")
PY
    echo '```'

    section "Attention stability"
    for tag in nolr lr; do
        echo "### ${tag}"
        echo '```'
        grep -E "alpha on shell|<u> pooled|^ +[0-9]+ +[0-9]" \
            "$HOME/runs/perov_alpha_${tag}.log" 2>/dev/null | tail -12 || echo "(n/a)"
        echo '```'
    done

    section "How to read this"
    echo "* A charged cell has a genuine q^2 alpha_M / 2 eps L image tail. A short-range"
    echo "  model cannot represent it, so a FLAT nolr ladder is self-consistency, not"
    echo "  physical correctness. The lr ladder is the one that can be right."
    echo "* eps(+/0) is measured from the FITTED gauge VBM (gap 2.4 eV at extraction), not"
    echo "  a DFT one. It is not comparable to a published transition level until a real"
    echo "  E_VBM per level of theory replaces it."
    echo "* a is frozen at an INTERIM eps_inf = 4.0. 1/a^2 must not be quoted as a fitted"
    echo "  screening constant until DFPT replaces it."
    echo "* The MD snapshots are thermally disordered and every Cl site is inequivalent, so"
    echo "  a single site is one sample of a distribution, not 'the' V_Cl level."
    echo
    echo "Figures: perov_size_{nolr,lr,lr_dilute}.png in defect-perovskite/."
} > "$REPORT"
cp "${HERE}"/perov_size_*.png /home/alex/src/mace/defect-perovskite/ 2>/dev/null
cp "$REPORT" /home/alex/src/mace/defect-perovskite/ 2>/dev/null
echo "OVERNIGHT_DONE -- report at $REPORT"
