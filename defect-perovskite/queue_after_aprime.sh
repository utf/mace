#!/usr/bin/env bash
# The chain after Stage A' (spec section 5, steps 4 and 5), on b3. Every wait is on a FILE.
#
#   1. wait for the four A' fold models and the production model (trained locally, copied
#      here when it finishes)
#   2. b1 against the A' folds and production base  -> aprime_b1.json
#   3. c5: the reference JSON in b10's format, F17     -> aprime_reference.json
#   4. c1 on the A' folds: w_E for Stage B (section 1) -> c1_ood_aprime.json
#   5. Stage B wave 1 (seeds 1-4), wave 2 (seeds 5-6)
#   6. the gates
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$W" || exit 1
B=defect-perovskite

need=("$R/aprime_prod/aprime_prod.model")
for k in 0 1 2 3; do need+=("$R/aprime_f$k/aprime_f$k.model"); done
echo "=== waiting for the five A' models $(date +%F' '%H:%M:%S) ==="
while true; do
    ok=1; for m in "${need[@]}"; do [ -f "$m" ] || ok=0; done
    [ "$ok" = 1 ] && break
    sleep 120
done
echo "=== A' models present $(date +%F' '%H:%M:%S) ==="

CUDA_VISIBLE_DEVICES=4 python -u $B/b1_label_slope_by_size.py \
    --runs "$R" --fold-prefix aprime_f --production-base aprime_prod \
    --device cuda --out "$R/aprime_b1.json" > "$R/aprime_b1.log" 2>&1
echo "b1 exit $?"
CUDA_VISIBLE_DEVICES=4 python -u $B/c5_references.py --b1-json "$R/aprime_b1.json" \
    --production "$R/aprime_prod/aprime_prod.model" --device cuda \
    --out "$R/aprime_reference.json" > "$R/aprime_reference.log" 2>&1
echo "c5 exit $?"
CUDA_VISIBLE_DEVICES=5 python -u $B/c1_ood_indicator.py \
    --folds "$R"/aprime_f0/aprime_f0.model "$R"/aprime_f1/aprime_f1.model \
            "$R"/aprime_f2/aprime_f2.model "$R"/aprime_f3/aprime_f3.model \
    --production "$R/aprime_prod/aprime_prod.model" --batch-size 4 --device cuda \
    --out "$R/c1_ood_aprime.json" > "$R/c1_ood_aprime.log" 2>&1
echo "c1 (A') exit $?"
for f in aprime_b1 aprime_reference c1_ood_aprime; do
    echo; echo "--- $f ---"
    grep -avE "Warning|warn|openequivariance|falling back|visitor|shape = |_Jd|np.reshape|UNDER-DET|alternative|detach|scored|^ *$" \
        "$R/$f.log" | tail -30
done
[ -s "$R/aprime_reference.json" ] && [ -s "$R/c1_ood_aprime.json" ] || {
    echo "ABORT: references or w_E missing; Stage B not launched"; exit 1; }

echo; echo "=== Stage B wave 1 $(date +%F' '%H:%M:%S) ==="
SEEDS="1 2 3 4" GPU0=4 bash $B/queue_stage_b.sh
echo "=== Stage B wave 2 $(date +%F' '%H:%M:%S) ==="
SEEDS="5 6" GPU0=4 bash $B/queue_stage_b.sh
echo "=== Stage B gates $(date +%F' '%H:%M:%S) ==="
bash $B/queue_stage_b_gates.sh
echo "=== chain complete $(date +%F' '%H:%M:%S) ==="
