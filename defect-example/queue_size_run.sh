#!/usr/bin/env bash
# Fourth leg of the size comparison: full L1 + long range, WITH the size hinge at 1e-4.
#
# Compared against efull_128ch_L1_s2, which is the same configuration with the hinge off,
# so the size term is the only difference between them and its effect is attributable.
#
# Queued rather than launched: efull is mid-run and holding 15.3 of 16.4 GB, so two
# 128-channel models will not co-exist on this card. Waits on the training process itself
# rather than a fixed sleep.
#
# 1e-4 is four times the weight calibrated on the beta set, chosen deliberately. That
# calibration was on an 8-channel model where |c| is small and the constraint correspondingly
# tight; at production width and with 140 epochs there is more room, and the beta runs showed
# the hinge is invisible to stability well beyond the weight that starts to act.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while pgrep -f "cli\.run_train.*efull_128ch" >/dev/null 2>&1; do sleep 120; done
echo "efull finished; starting the size run"

SEED=2 NAME="esize_128ch_L1_s2" \
DEFECT_SIZE_WEIGHT="${WEIGHT:-1e-4}" DEFECT_SIZE_WARMUP_EPOCHS=20 \
    "${HERE}/run_stage_e_full.sh"
echo "size run finished (exit $?)"
