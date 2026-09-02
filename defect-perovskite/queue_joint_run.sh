#!/usr/bin/env bash
# The joint run. Two arms, eight seeds, from config.
#
# ARM A (6 seeds): Stage-A base loaded and trained at 0.1x the head's rate. This is the
#   configuration the programme has been building toward.
# ARM B (2 seeds): everything jointly from scratch at equal rates. The staging CONTROL --
#   the spectral-era argument for staging was made about a head that no longer exists, so
#   either this arm retires the question or it revives it, and both are results.
#
# WHY THE UPWEIGHTS COME IN PAIRS. The charged 159-atom frames carry the only measurement
# that distinguishes a bound carrier from a band-like one, and at natural weight they are
# 1.6% of the charged force loss. The NEUTRAL 159-atom frames are the base's only direct
# constraint at large d -- the rest of the neutral set stops near 6.0 A, and above that the
# base extrapolates, which is the measured +0.132 eV/A error in the carrier-free 79-atom
# residual. Upweighting the charged seventeen while leaving the neutral seventeen at natural
# weight would ask the correction to absorb a base error the base was never given the chance
# to fix. That is exactly the leakage the adoption rule tests for, so both are raised.
#
# THE ADOPTION RULE IS A LEAKAGE DETECTOR FIRST. "M1b's +0.36 artefact removed" is passable
# by a base that removes it by absorbing the carrier. The detector is the null-cleared -0.134
# staying put: if the 159-atom charged residual slope shrinks toward -0.06, the base ate the
# carrier and the run is not adopted regardless of total fit.
#
# b3 GPUs 4-7 only, four at a time. Two waves.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
DATA="${DATA:-$W/defect-perovskite/dataset_pbe}"
BASE=$R/e0_base_s1/e0_base_s1.model
TAG="${TAG:-joint}"
EPOCHS="${EPOCHS:-40}"
LR_START="${LR_START:-25}"          # E_LR staged re-enable, mid-run
HOP_FORM="${HOP_FORM:-linear}"      # section 2's branch writes this
HOP_BETA="${HOP_BETA:-1.0986122886681098}"
cd "$W" || exit 1
[ -f "$BASE" ] || { echo "ABORT: missing $BASE"; exit 1; }
[ -f "$DATA/train.xyz" ] || { echo "ABORT: missing $DATA/train.xyz"; exit 1; }

# The knobs read off a BUILT model before any seed is spent. Every config check in this
# programme that trusted the argument chain has eventually been wrong about something; the
# most recent cost six seeds trained at Gaussian 0.025 under a 0.05 banner.
echo "=== pre-seed config read-back ==="
CHK_W="$W" CHK_FORM="$HOP_FORM" CHK_BETA="$HOP_BETA" python - <<'PYCHK' || exit 1
import os
import sys

sys.path.insert(0, os.environ["CHK_W"] + "/defect-perovskite")
import torch  # noqa: E402

import mace  # noqa: F401,E402
from mace.modules.defect_models import MACEDefect  # noqa: E402
from mace.tools.scripts_utils import extract_config_mace_model  # noqa: E402

arch = torch.load(os.path.expanduser("~/runs/r2_h3_anneal_s5/r2_h3_anneal_s5.model"),
                  map_location="cpu", weights_only=False)
cfg = extract_config_mace_model(arch)
cfg.update(counting_head=True, madelung_on_site=True,
           madelung_composition=[3.0, 1.0, 1.0], madelung_z_init=[-1.0, 1.0, 2.0],
           counting_hop_form=os.environ["CHK_FORM"],
           counting_hop_log_beta=float(os.environ["CHK_BETA"]))
m = MACEDefect(**cfg)
h = m.spectral.h
live = dict(gamma=h.on_site_range, hop_range=h.hop_range,
            envelope=getattr(h, "envelope", "exp"), decay=h.decay_length,
            hop_form=getattr(h, "hop_form", "linear"),
            beta=round(float(getattr(h, "hop_log_beta", 0.0)), 4),
            family=getattr(m.spectral, "smearing_family", "?"), width=m.spectral.t_el)
print("  built: " + "  ".join(f"{k} {v}" for k, v in live.items()))
# And the same knobs after a config round trip, which is what a reloaded model trains with.
back = extract_config_mace_model(m)
rb = MACEDefect(**back).spectral.h
ok = (getattr(rb, "hop_form", "linear") == live["hop_form"]
      and abs(float(getattr(rb, "hop_log_beta", 0.0)) - float(live["beta"])) < 1e-3
      and getattr(rb, "envelope", "exp") == live["envelope"]
      and abs(float(rb.on_site_range) - 3.0) < 1e-9
      and live["family"] == "gaussian" and abs(float(live["width"]) - 0.05) < 1e-9)

# THE TRUNK NORMALISATION, checked against the Stage-A base itself. `avg_num_neighbors` is a
# plain float on each interaction block dividing every message -- not a parameter, not a
# buffer, so absent from state_dict and invisible to the loader's name-and-shape check. Stage
# A trained at r_max = 5.0 without a carrier head and got 14.08; a run with the spectral head
# builds its graph at the 10 A carrier cutoff and computes 112.5 on the same data. Loading
# Stage A's weights into that trunk divides every message by eight times too much, and the
# loader reports success. It cost the first joint launch.
from mace.modules.defect_stage import load_stage_a_base  # noqa: E402

base_path = os.path.expanduser("~/runs/e0_base_s1/e0_base_s1.model")
stage_a = torch.load(base_path, map_location="cpu", weights_only=False)
want = [round(float(b.avg_num_neighbors), 4) for b in stage_a.interactions]
m2 = MACEDefect(**cfg)
for blk, v in zip(m2.interactions, [112.5] * len(want)):
    blk.avg_num_neighbors = v          # the wrong value a real run would have computed
load_stage_a_base(m2, base_path, device="cpu")
got = [round(float(b.avg_num_neighbors), 4) for b in m2.interactions]
print(f"  avg_num_neighbors: Stage A {want}, after load {got}")
ok = ok and (got == want)
print("  CONFIG OK" if ok else "  CONFIG WRONG -- refusing to spend seeds")
sys.exit(0 if ok else 1)
PYCHK

# ONE EPOCH ON A SMALL SET BEFORE ANY SEED IS SPENT. The config read-back above proves the
# MODEL is right; it says nothing about the training loop, and the first attempt at this run
# lost four seeds at epoch 0 to an UnboundLocalError in the epoch hook -- a per-epoch logging
# block inserted above the line that defines the variable it reads. Every unit test passed;
# the two-epoch smoke had been run before that patch existed. A model that builds is not a
# run that trains, which is the same lesson as "a model is not a run" one level down.
preflight () {
    local out="$R/${TAG}_preflight"
    rm -rf "$out" "$out.log"
    NAME="${TAG}_preflight" WORK_DIR="$out" \
    DATA_DIR="$W/defect-perovskite/dataset_cf/fold0" MACE_REPO="$W" \
    CUDA_VISIBLE_DEVICES=4 \
    MAX_NUM_EPOCHS=1 NUM_CHANNELS=128 MAX_L=1 NUM_RADIAL_BASIS=8 R_MAX=5.0 \
    BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float64 \
    EVAL_INTERVAL=1 USE_EMA=False PATIENCE=250 SEED=99 ENABLE_CUEQ=False \
    LR=0.005 BASE_LR_FACTOR=0.1 DEFECT_BASE_INIT="$BASE" \
    DEFECT_SPECTRAL_HEAD=True DEFECT_COUNTING_HEAD=True DEFECT_SPECTRAL_R_CUT=10.0 \
    DEFECT_MADELUNG_ON_SITE=True DEFECT_MADELUNG_COMPOSITION="3,1,1" \
    DEFECT_MADELUNG_Z_INIT="-1,1,2" DEFECT_MADELUNG_EPS_INF=4.0 \
    DEFECT_COUNTING_ON_SITE_RANGE=3.0 DEFECT_COUNTING_HOP_RANGE=0.5 \
    DEFECT_COUNTING_SMEARING=gaussian DEFECT_COUNTING_T_EL=0.05 \
    DEFECT_COUNTING_ENVELOPE=exp DEFECT_COUNTING_DECAY_LENGTH=1.0 \
    DEFECT_COUNTING_HOP_FORM="$HOP_FORM" DEFECT_COUNTING_HOP_BETA="$HOP_BETA" \
    DEFECT_PROTOCOL=True DEFECT_PROTOCOL_BOND_LENGTH=2.861 \
    DEFECT_PROTOCOL_WARMUP=5 DEFECT_PROTOCOL_HEAD_ONLY=False \
    DEFECT_PROTOCOL_ZERO_ON_SITE=True \
    DEFECT_GAP_WEIGHT=1.0 DEFECT_E_GAP=2.4 DEFECT_GAP_COMPOSITION="3,1,1" \
    DEFECT_TWO_SIZE_UPWEIGHT=0.25 DEFECT_NEUTRAL_SIZE_UPWEIGHT=0.25 \
    USE_LONG_RANGE=True LR_START_EPOCH=0 \
    "$W/defect-example/train_defect_model.sh" > "$out.log" 2>&1
    local code=$?
    if [ "$code" -ne 0 ]; then
        echo "  PREFLIGHT FAILED (exit $code) -- refusing to spend eight seeds"
        grep -avE "Warning|warn|openequivariance|falling back" "$out.log" | tail -12
        return 1
    fi
    # The epoch hook must actually have RUN, not merely not crashed: an exception swallowed
    # somewhere upstream would leave the log clean and the gauge silent.
    if ! grep -aq "Gauge: epoch 0" "$out.log"; then
        echo "  PREFLIGHT: no gauge line at epoch 0 -- the epoch hook did not run"
        return 1
    fi
    echo "  preflight OK: one epoch, gauge logged, exit 0"
    grep -aE "Gauge: epoch 0|Stage-3 protocol: c-shift" "$out.log" \
        | sed 's/^.*INFO: /    /' | head -3
    return 0
}

echo "=== preflight: one epoch on the small set ==="
preflight || exit 1

run () {   # gpu name seed arm
    local gpu="$1" name="$2" seed="$3" arm="$4"
    rm -rf "$R/$name" "$R/$name.log"
    local base_init="" base_factor="1.0"
    if [ "$arm" = "stagea" ]; then
        base_init="$BASE"
        base_factor="0.1"          # the base moves, slowly, rather than being frozen
    fi
    NAME="$name" WORK_DIR="$R/$name" DATA_DIR="$DATA" MACE_REPO="$W" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    MAX_NUM_EPOCHS="$EPOCHS" NUM_CHANNELS=128 MAX_L=1 NUM_RADIAL_BASIS=8 R_MAX=5.0 \
    BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float64 \
    EVAL_INTERVAL="${EVAL_EVERY:-3}" \
    USE_EMA=False PATIENCE=250 SEED="$seed" ENABLE_CUEQ=False \
    LR=0.005 BASE_LR_FACTOR="$base_factor" DEFECT_BASE_INIT="$base_init" \
    DEFECT_SPECTRAL_HEAD=True DEFECT_COUNTING_HEAD=True DEFECT_SPECTRAL_R_CUT=10.0 \
    DEFECT_MADELUNG_ON_SITE=True DEFECT_MADELUNG_COMPOSITION="3,1,1" \
    DEFECT_MADELUNG_Z_INIT="-1,1,2" DEFECT_MADELUNG_EPS_INF=4.0 \
    DEFECT_COUNTING_ON_SITE_RANGE=3.0 DEFECT_COUNTING_HOP_RANGE=0.5 \
    DEFECT_COUNTING_SMEARING=gaussian DEFECT_COUNTING_T_EL=0.05 \
    DEFECT_COUNTING_ENVELOPE=exp DEFECT_COUNTING_DECAY_LENGTH=1.0 \
    DEFECT_COUNTING_HOP_FORM="$HOP_FORM" DEFECT_COUNTING_HOP_BETA="$HOP_BETA" \
    DEFECT_PROTOCOL=True DEFECT_PROTOCOL_BOND_LENGTH=2.861 \
    DEFECT_PROTOCOL_WARMUP=5 DEFECT_PROTOCOL_HEAD_ONLY=False \
    DEFECT_PROTOCOL_ZERO_ON_SITE=True \
    DEFECT_GAP_WEIGHT=1.0 DEFECT_E_GAP=2.4 DEFECT_GAP_COMPOSITION="3,1,1" \
    DEFECT_TWO_SIZE_UPWEIGHT=0.25 DEFECT_NEUTRAL_SIZE_UPWEIGHT=0.25 \
    USE_LONG_RANGE=True LR_START_EPOCH="$LR_START" \
    "$W/defect-example/train_defect_model.sh" > "$R/$name.log" 2>&1
    echo "  $name (arm $arm, seed $seed) exit $?"
}

wave () {   # "name:seed:arm" x4, on GPUs 4-7
    local gpu=4
    local pids=()
    for spec in "$@"; do
        IFS=: read -r n s a <<< "$spec"
        run "$gpu" "$n" "$s" "$a" &
        pids+=($!)
        gpu=$((gpu + 1))
        sleep 15
    done
    wait "${pids[@]}"
}

echo "=== joint run: $EPOCHS epochs, hop_form $HOP_FORM, E_LR from epoch $LR_START ==="
echo "=== wave 1: four Stage-A seeds  $(date +%F' '%H:%M:%S) ==="
wave "${TAG}_a1:1:stagea" "${TAG}_a2:2:stagea" "${TAG}_a3:3:stagea" "${TAG}_a4:4:stagea"
echo "=== wave 2: two Stage-A seeds + the two from-scratch controls  $(date +%F' '%H:%M:%S) ==="
wave "${TAG}_a5:5:stagea" "${TAG}_a6:6:stagea" "${TAG}_b1:11:scratch" "${TAG}_b2:12:scratch"
echo "=== joint run complete $(date +%F' '%H:%M:%S) ==="

for f in "$R/${TAG}"_*.log; do
    echo; echo "--- $(basename "$f") ---"
    grep -aE "Stage-3 protocol|Realised large-cell|Gauge: epoch (0|10|20|30|39) " "$f" \
        | sed 's/^.*INFO: //' | head -8
    grep -aE "Epoch [0-9]+:" "$f" | tail -2 | sed 's/^.*INFO: //'
done
