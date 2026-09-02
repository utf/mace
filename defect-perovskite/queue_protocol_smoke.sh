#!/usr/bin/env bash
# Section 3's close-out: run the PRODUCTION trainer with --defect_protocol on, twice.
#
# WHAT THIS ESTABLISHES, and what it deliberately does not. It does NOT claim step-level
# weight identity with defect-perovskite/stage_run.py: the two drivers batch differently --
# the harness holds a fixed list of graphs, the trainer shuffles a DataLoader -- so their
# weights after one epoch differ for reasons that have nothing to do with the protocol.
# Chasing that number would be measuring the shuffle.
#
# What it does establish:
#   * every protocol call site FIRES in the real trainer, on real data, and is visible in the
#     log: Harrison initialisation, the c-shift, the initialisation gate, the warmup, and the
#     protocol summary read off the head;
#   * running the identical command twice reproduces itself to the same-model repeat floor,
#     so any later difference is a change and not the scatter-atomics noise;
#   * the SAVED model, reloaded and interrogated, agrees with the summary the run logged.
#     That last check is the one that matters: a run trained at Gaussian 0.025 under a 0.05
#     banner because the family and the width came from different places, and only the
#     assembled object knew.
#
# b3 GPUs 4-7 only.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
DATA="${DATA:-$W/defect-perovskite/dataset_cf/fold0}"
BASE=$R/e0_base_s1/e0_base_s1.model
cd "$W" || exit 1
[ -f "$BASE" ] || { echo "ABORT: missing $BASE"; exit 1; }
[ -f "$DATA/train.xyz" ] || { echo "ABORT: missing $DATA/train.xyz"; exit 1; }

run () {   # gpu name
    rm -rf "$R/$2" "$R/$2.log"
    NAME="$2" WORK_DIR="$R/$2" DATA_DIR="$DATA" MACE_REPO="$W" \
    CUDA_VISIBLE_DEVICES="$1" \
    MAX_NUM_EPOCHS=2 NUM_CHANNELS=32 MAX_L=1 NUM_RADIAL_BASIS=8 R_MAX=5.0 \
    BATCH_SIZE=4 VALID_BATCH_SIZE=4 DEVICE=cuda DEFAULT_DTYPE=float64 \
    USE_EMA=False PATIENCE=250 SEED=1 ENABLE_CUEQ=False \
    DEFECT_SPECTRAL_HEAD=True DEFECT_COUNTING_HEAD=True \
    DEFECT_SPECTRAL_R_CUT=10.0 \
    DEFECT_MADELUNG_ON_SITE=True DEFECT_MADELUNG_COMPOSITION="3,1,1" \
    DEFECT_MADELUNG_Z_INIT="-1,1,2" DEFECT_MADELUNG_EPS_INF=4.0 \
    DEFECT_COUNTING_ON_SITE_RANGE=3.0 DEFECT_COUNTING_HOP_RANGE=0.5 \
    DEFECT_COUNTING_SMEARING=gaussian DEFECT_COUNTING_T_EL=0.05 \
    DEFECT_COUNTING_ENVELOPE=exp DEFECT_COUNTING_DECAY_LENGTH=1.0 \
    DEFECT_PROTOCOL=True DEFECT_PROTOCOL_BOND_LENGTH=2.861 \
    DEFECT_PROTOCOL_WARMUP=5 DEFECT_PROTOCOL_HEAD_ONLY=True \
    DEFECT_GAP_WEIGHT=1.0 DEFECT_E_GAP=2.4 DEFECT_GAP_COMPOSITION="3,1,1" \
    DEFECT_BASE_INIT="$BASE" BASE_LR_FACTOR=0.0 \
    USE_LONG_RANGE=False \
    "$W/defect-example/train_defect_model.sh" > "$R/$2.log" 2>&1
    echo "  $2 exit $?"
}

echo "=== two identical runs of the production trainer, protocol ON ==="
run 4 protosmoke_a
run 5 protosmoke_b

echo
echo "=== did every call site fire? ==="
grep -E "Stage-3 protocol|Stage-3 warmup" "$R/protosmoke_a.log" | sed 's/^.*INFO: //' | head -12

echo
echo "=== the saved model, read back and compared to what the run logged ==="
SMOKE_MODEL="$R/protosmoke_a/protosmoke_a.model" SMOKE_LOG="$R/protosmoke_a.log" \
    python - <<'PYCHK'
import json
import os
import re

import torch

import mace  # noqa: F401
from mace.modules.defect_protocol import protocol_summary

m = torch.load(os.environ["SMOKE_MODEL"], map_location="cpu", weights_only=False)
head = m.spectral
live = dict(gamma=float(head.h.on_site_range), hop_range=float(head.h.hop_range),
            envelope=getattr(head.h, "envelope", "exp"),
            decay_length=float(head.h.decay_length),
            family=getattr(head, "smearing_family", "?"), width=float(head.t_el))
print("  read off the saved model: " + "  ".join(f"{k} {v}" for k, v in live.items()))

logged = None
for line in open(os.environ["SMOKE_LOG"]):
    if "Stage-3 protocol: {" in line:
        logged = json.loads(line[line.index("{"):].strip())
print(f"  logged by the run:        {logged}")
ok = (logged is not None
      and logged["smearing_family"] == live["family"]
      and abs(logged["smearing_width"] - live["width"]) < 1e-12)
print("  SUMMARY MATCHES THE SAVED HEAD" if ok else
      "  MISMATCH -- the artefact does not describe the run that produced it")

# The config round trip: rebuild from the extracted config and check the knobs survive.
from mace.tools.scripts_utils import extract_config_mace_model

cfg = extract_config_mace_model(m)
rebuilt = m.__class__(**cfg)
rb = rebuilt.spectral.h
same = (getattr(rb, "envelope", "exp") == live["envelope"]
        and abs(float(rb.decay_length) - live["decay_length"]) < 1e-12
        and abs(float(rb.on_site_range) - live["gamma"]) < 1e-12
        and abs(float(rebuilt.spectral.t_el) - live["width"]) < 1e-12)
print("  CONFIG ROUND TRIP OK" if same else "  CONFIG ROUND TRIP LOST A KNOB")
PYCHK

echo
echo "=== the repeat floor: two identical commands ==="
python - <<'PYFLOOR'
import os
import re

R = os.path.expanduser("~/runs")


def losses(path):
    out = []
    for line in open(path):
        m = re.search(r"loss=([0-9.eE+-]+)", line)
        if m and "Epoch" in line:
            out.append(float(m.group(1)))
    return out


a, b = losses(f"{R}/protosmoke_a.log"), losses(f"{R}/protosmoke_b.log")
print(f"  run a: {a}")
print(f"  run b: {b}")
if a and b and len(a) == len(b):
    d = max(abs(x - y) for x, y in zip(a, b))
    print(f"  max |difference| {d:.3e}  -> "
          f"{'within the repeat floor' if d < 1e-8 else 'NOT bit-identical'}")
    print("  Scatter-atomics on GPU are nondeterministic at ~1e-6 eV/A in forces, so a "
          "small\n  non-zero difference here is the floor and not a disagreement.")
else:
    print("  could not compare: no epoch losses parsed from one of the logs")
PYFLOOR
