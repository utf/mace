#!/usr/bin/env bash
# Plan v8 Stage 1.3, the retrained half: three Madelung arms x six seeds on the Stage B
# recipe (queue_stage_b.sh, unchanged apart from DEFECT_MADELUNG_RANGE), b3 GPUs 4-7, four
# runs at a time. Arm tags: s13ra (full), s13rb (long_range), s13rc (off). Each arm's gate
# queue is started once its six models exist (queue_stage13_gates waits on the files).
#
#   nohup bash defect-perovskite/queue_stage13_train.sh > ~/runs/stage13_train.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=$HOME/runs
export PYTHONPATH="$(cd "$HERE/.." && pwd)" PATH="$HOME/micromamba/envs/py13/bin:$PATH"
EPOCHS="${EPOCHS:-24}"
# Standing rule 2: charged energies enter the loss only for a size class with a neutral
# null; the A' null file establishes which (a guard added to the trainer after Stage B ran).
export DEFECT_NULL_REFERENCE="${DEFECT_NULL_REFERENCE:-$R/aprime_nulls.json}"
[ -f "$DEFECT_NULL_REFERENCE" ] || { echo "ABORT: missing null file $DEFECT_NULL_REFERENCE"; exit 1; }

wave () {   # arm mode seeds
    local arm="$1" mode="$2" seeds="$3"
    echo "=== $arm ($mode) seeds $seeds  $(date +%F' '%H:%M:%S) ==="
    TAG="$arm" SEEDS="$seeds" GPU0=4 EPOCHS="$EPOCHS" DEFECT_MADELUNG_RANGE="$mode" \
        bash "$HERE/queue_stage_b.sh"
}

# 18 runs in waves of four (the last wave is two): each wave is one arm's seeds so an arm
# completes as early as possible and its gates start while the next trains.
wave s13ra full       "1 2 3 4"
wave s13ra full       "5 6"
TAG=s13ra nohup bash "$HERE/queue_stage13_gates.sh" > "$R/s13ra_gates.log" 2>&1 &
wave s13rb long_range "1 2 3 4"
wave s13rb long_range "5 6"
TAG=s13rb nohup bash "$HERE/queue_stage13_gates.sh" > "$R/s13rb_gates.log" 2>&1 &
wave s13rc off        "1 2 3 4"
wave s13rc off        "5 6"
TAG=s13rc nohup bash "$HERE/queue_stage13_gates.sh" > "$R/s13rc_gates.log" 2>&1 &
wait
echo "=== Stage 1.3 retrains and gates complete $(date +%F' '%H:%M:%S) ==="
