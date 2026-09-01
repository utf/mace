#!/usr/bin/env bash
# C1 (clamped 2x2) then, gated on its result, the A/B. b3 GPUs 4-7 only.
#
# C1: hub2 clamp x {ON, OFF} x 2 seeds. With mass fixed by the clamp the only free lever is
# t(d), so this tests the energy-channel claim in isolation from placement.
#
# GATE (pre-registered, evaluated automatically because nobody is watching):
#   ON-clamped mean axial_red >= 0.40  AND  exceeds OFF-clamped by >= 0.20
# The spec also asks for autograd |t'| >= 0.1 eV/A. That machinery (M3) is NOT built, so the
# gate runs on axial_red alone and this file records that the t' half was not evaluated --
# it is not silently treated as passed.
#
# If the gate fails the A/B does not run: the spec says go to s+p directly rather than spend
# 12 cells.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
ARCH=$R/r2_h3_anneal_s5/r2_h3_anneal_s5.model
BASE=$R/e0_base_s1/e0_base_s1.model
cd "$W" || exit 1
for f in "$ARCH" "$BASE"; do [ -f "$f" ] || { echo "ABORT: missing $f"; exit 1; }; done

cell () {  # gpu sign clamp seeds seedstart tag extra...
    local gpu=$1 sign=$2 clamp=$3 seeds=$4 s0=$5 tag=$6; shift 6
    local cl=""; [ "$clamp" != "none" ] && cl="--clamp $clamp"
    CUDA_VISIBLE_DEVICES=$gpu python defect-perovskite/tb_v3.py \
        --arch "$ARCH" --base "$BASE" --frames 48 --epochs 40 \
        --seeds "$seeds" --seed-start "$s0" --f-m 0.2 --e-gap 2.4 \
        --n-pristine 64 --pristine-batch 8 --sign "$sign" $cl \
        --save-dir "$R/ab_models" --device cuda \
        --out "$R/${tag}.json" > "$R/${tag}.log" 2>&1
    echo "  done $tag (exit $?)"
}

echo "=== C1 starting $(date +%F' '%H:%M:%S) ==="
cell 4 on  hub2 2 1 c1_on  &
cell 5 off hub2 2 1 c1_off &
wait
echo "=== C1 complete $(date +%F' '%H:%M:%S) ==="

python - <<'PY' > "$R/c1_gate.txt" 2>&1
import json, numpy as np
def m(p):
    try:
        r=[x for x in json.load(open(p)) if 'error' not in x]
        return float(np.mean([x['axial_red'] for x in r])), len(r)
    except Exception as e:
        return float('nan'), 0
on,non   = m('/home/alex/runs/c1_on.json')
off,noff = m('/home/alex/runs/c1_off.json')
print(f"C1 ON  axial_red {on:+.4f}  (n={non})")
print(f"C1 OFF axial_red {off:+.4f}  (n={noff})")
ok = (on >= 0.40) and (on - off >= 0.20)
print("GATE", "PASS" if ok else "FAIL")
print("NOTE: the autograd |t'| >= 0.1 eV/A half of the gate was NOT evaluated (M3 unbuilt).")
open('/home/alex/runs/c1_gate_verdict','w').write('PASS' if ok else 'FAIL')
PY
cat "$R/c1_gate.txt"

if [ "$(cat "$R/c1_gate_verdict" 2>/dev/null)" != "PASS" ]; then
    echo "=== C1 gate FAILED -- A/B not launched, per the decision tree (s+p branch) ==="
    exit 0
fi

echo "=== A/B starting $(date +%F' '%H:%M:%S) ==="
cell 4 on  none 3 1 ab_on_a  &
cell 5 on  none 3 4 ab_on_b  &
cell 6 off none 3 1 ab_off_a &
cell 7 off none 3 4 ab_off_b &
wait
echo "=== A/B complete $(date +%F' '%H:%M:%S) ==="
