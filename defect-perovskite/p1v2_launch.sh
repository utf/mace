#!/bin/bash
# A2: when the data sets exist, sync them and the tree to b3 and launch the three learning rates:
# lr 1e-3 on b3 GPU 5, lr 1e-4 on b3 GPU 7, lr 1e-2 on the local card (batch 4; falls back to
# b3 after the others if it does not fit). Status files only.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=/home/alex/runs/dscc/p1v2
cd $W
while true; do
  [ -f $D/replay_v2.xyz ] && [ -f $D/train_v2_iso.xyz ] && [ -f $D/valid_v2.xyz ] && [ -f $D/e0s_estimated.json ] && break
  grep -q FAILED $D/LABEL_STATUS 2>/dev/null && { echo "LABEL_FAILED $(date)" > $D/LAUNCH_STATUS; exit 1; }
  sleep 120
done
rsync -az --exclude='.git/' --exclude='__pycache__' $W/ b3:$W/
ssh b3 "mkdir -p $D" && rsync -az $D/replay_v2.xyz $D/train_v2_iso.xyz $D/valid_v2.xyz $D/e0s_estimated.json b3:$D/
scp -q $W/defect-perovskite/p1v2_run.sh b3:$W/defect-perovskite/
cat > /tmp/claude-1000/-home-alex-src-mace/eeca5f26-1644-4587-983a-79b6adcd9665/scratchpad/b3_p1v2_runner.sh <<'EOR'
#!/bin/bash
NAME=$1; LR=$2; OUT=$3; GPU=$4; shift 4
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -v g=$GPU -F', ' '$1==g{print $2}')
export CUDA_VISIBLE_DEVICES=$UUID PATH="$HOME/micromamba/envs/py13/bin:$PATH"
mkdir -p $OUT; echo "START $(date) gpu $GPU lr $LR" > $OUT/STATUS
bash $W/defect-perovskite/p1v2_run.sh $NAME $LR $OUT "$@" > $OUT/run.out 2>&1 && echo "DONE $(date)" >> $OUT/STATUS || echo "FAILED $(date)" >> $OUT/STATUS
EOR
scp -q /tmp/claude-1000/-home-alex-src-mace/eeca5f26-1644-4587-983a-79b6adcd9665/scratchpad/b3_p1v2_runner.sh b3:/home/alex/runs/dscc/p1v2_runner.sh
ssh -f b3 "cd /home/alex/runs/dscc && nohup bash p1v2_runner.sh base_v6_lr3 0.001 $D/base_v6_lr3 5 > $D/base_v6_lr3_runner.out 2>&1 < /dev/null"
ssh -f b3 "cd /home/alex/runs/dscc && nohup bash p1v2_runner.sh base_v6_lr4 0.0001 $D/base_v6_lr4 7 > $D/base_v6_lr4_runner.out 2>&1 < /dev/null"
mkdir -p $D/base_v6_lr2; echo "START $(date) local lr 0.01" > $D/base_v6_lr2/STATUS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( bash $W/defect-perovskite/p1v2_run.sh base_v6_lr2 0.01 $D/base_v6_lr2 > $D/base_v6_lr2/run.out 2>&1 && echo "DONE $(date)" >> $D/base_v6_lr2/STATUS || echo "FAILED $(date)" >> $D/base_v6_lr2/STATUS ) &
echo "LAUNCHED $(date)" > $D/LAUNCH_STATUS
wait
