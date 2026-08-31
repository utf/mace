#!/usr/bin/env bash
# Four diagnostic Stage-A bases, one per cross-fit fold (T2 null).
#
# Each trains on the neutral frames of the other three folds, so its held-out fold can be
# scored as a genuine generalisation residual. Together they give ~1191 out-of-fold neutral
# residuals across the whole d(Pb-Pb) range, where the original null control had 60 frames
# topping out at 5.77 A -- which is why T2 could not run at all.
#
# DIAGNOSTIC ONLY. The production Stage-A base (e0_base_s1/s2) is untouched and remains what
# T1 and R2 use; forking that lineage for a diagnostic would make the arms incomparable.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA="${DATA:-$HERE/dataset_cf}"
CONCURRENCY="${CONCURRENCY:-4}"

mkdir -p "$HOME/runs"
echo "code: $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l) files"

run () {
    local k="$1" name="cf_base_f$1"
    rm -rf "$HOME/runs/$name" "$HOME/runs/${name}.log"
    NAME="$name" WORK_DIR="$HOME/runs/$name" DATA_DIR="$DATA/fold$k" \
    MACE_REPO="$REPO" \
    MAX_NUM_EPOCHS="${EPOCHS:-140}" NUM_CHANNELS="${CHANNELS:-128}" MAX_L=1 \
    NUM_RADIAL_BASIS=8 R_MAX="${R_MAX:-5.0}" \
    BATCH_SIZE="${BATCH:-8}" VALID_BATCH_SIZE="${BATCH:-8}" DEVICE=cuda \
    DEFAULT_DTYPE=float32 USE_EMA=False PATIENCE=250 SEED=1 \
    ENABLE_CUEQ=True \
    ENERGY_WEIGHT=10.0 TOTAL_ENERGY_WEIGHT=10.0 \
    DELTA_ENERGY_WEIGHT=0.0 DELTA_FORCES_WEIGHT=0.0 \
    USE_LONG_RANGE=False \
    DEFECT_SEED_ANNEAL=False DEFECT_LOGIT_SEED_GAMMA=0.0 \
    DEFECT_SIZE_WEIGHT=0.0 DEFECT_GAUGE_WEIGHT=0.0 \
    DEFECT_U_L2=0.0 DEFECT_ZN_L2=0.0 DEFECT_P_L2=0.0 DEFECT_QHOST_L2=0.0 \
    "${HERE}/../defect-example/train_defect_model.sh" > "$HOME/runs/${name}.log" 2>&1
    echo "  ${name} done (exit $?)"
}

pids=()
for k in 0 1 2 3; do
    run "$k" &
    pids+=($!)
    sleep 20
    while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do sleep 20; done
done
wait "${pids[@]}"
echo "cross-fit bases complete"
