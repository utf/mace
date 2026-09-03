#!/usr/bin/env bash
# Two scorings the post-run chain did not cover, run after the E_LR-off control has finished.
#
# 1. b6_depth_edges on ALL SIX joint models. The post-run chain passed only joint_a1.model, so
#    joint_depth.json is n = 1 and its "F9 holds" line rests on one seed. This rewrites it at
#    n = 6 (joint_depth6.json; the n = 1 file is kept).
# 2. b10_adoption on the four E_LR-off control models. An ADDITION to the plan, labelled as
#    one: it says whether E_LR contributes to the energy leak. The control was launched for
#    participation; scoring it by the adoption rule costs one GPU for about ten minutes.
#
# It waits on the CONDITION -- the participation JSON the control queue writes last -- and not
# on a process (LEDGER.md entry 10 and the header of queue_joint_nolr.sh say why).
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
TAG="${TAG:-nolr}"
cd "$W" || exit 1

echo "=== waiting for $R/${TAG}_participation.json ==="
while [ ! -s "$R/${TAG}_participation.json" ]; do sleep 60; done
echo "=== present $(date +%F' '%H:%M:%S) ==="

# b6 takes one directory (or one-level glob) per arm; the joint models sit two levels deep, so
# collect them behind symlinks.
mkdir -p "$R/joint_a_models"
for s in 1 2 3 4 5 6; do ln -sfn "$R/joint_a$s/joint_a$s.model" "$R/joint_a_models/joint_a$s.model"; done
OFF=();  for s in 1 2 3 4; do f="$R/${TAG}_s$s/${TAG}_s$s.model"; [ -f "$f" ] && OFF+=("$f"); done

CUDA_VISIBLE_DEVICES=4 python -u defect-perovskite/b6_depth_edges.py \
    --arms "prejoint=$R/s7_models" "joint=$R/joint_a_models" \
    --device cuda --out "$R/joint_depth6.json" > "$R/joint_depth6.log" 2>&1 &
P1=$!
if [ "${#OFF[@]}" -gt 0 ]; then
    CUDA_VISIBLE_DEVICES=5 python -u defect-perovskite/b10_adoption.py \
        --models "${OFF[@]}" --device cuda --out "$R/${TAG}_adopt.json" \
        > "$R/${TAG}_adopt.log" 2>&1 &
    P2=$!
else
    echo "no control models found; skipping b10 on the control"; P2=""
fi
wait $P1; echo "b6 (six seeds) exit $?"
[ -n "$P2" ] && { wait $P2; echo "b10 (E_LR off) exit $?"; }
for f in joint_depth6 ${TAG}_adopt; do
    echo; echo "--- $f ---"
    grep -avE "Warning|warn|openequivariance|falling back|visitor|shape = |_Jd|np.reshape|UNDER-DET|alternative|^ *$" \
        "$R/$f.log" | tail -40
done
echo "=== follow-up complete $(date +%F' '%H:%M:%S) ==="
