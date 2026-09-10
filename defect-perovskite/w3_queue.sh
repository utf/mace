#!/bin/bash
# W3 queue on b3: ONE run per GPU on 4, 5, 7 (a training step on base v2 peaks at 11.6 GB, so two
# do not share a 23.5 GB card). Never touches GPUs 0-3. Selects by UUID (GPU 6 is off the bus).
# One line of the queue file per run, dropped as it is launched.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
Q=${W3_QUEUE:-$HOME/runs/w3_queue.txt}; L=${W3_LOG:-$HOME/runs/w3_queue.log}
PER_GPU=${W3_PER_GPU:-1}; HEADROOM=${W3_HEADROOM:-13000}
# One run per card: a training step on base v2 peaks at 11.6 GB (measured on the full data set,
# both arms). expandable_segments keeps the reserved pool at the live size -- without it the
# many small setup batches fragment it to 14.2 GB against 6.5 GB live.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  line=$(grep -v -E "^\s*(#|$)" $Q 2>/dev/null | head -1)
  [ -z "$line" ] && { echo "$(date '+%F %T') queue empty, exiting" >> $L; exit 0; }
  for g in 4 5 7; do
    read -r used total < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v g=$g '$1==g {print $2, $3}')
    [ -z "$used" ] && continue
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v g=$g '$1==g {print $2}')
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$uuid")
    if [ "$n" -lt "$PER_GPU" ] && [ $((total - used)) -gt "$HEADROOM" ]; then
      name=${line%%|*}; margs=${line#*|}
      echo "$(date '+%F %T') launching $name on GPU $g ($n running, $((total-used)) MiB free)" >> $L
      bash $W/defect-perovskite/b3_run.sh "$uuid" "$name" \
        $HOME/micromamba/envs/py13/bin/python $W/defect-perovskite/dscc_train.py \
        --name "$name" --run_dir $HOME/runs/dscc/$name --device cuda $margs >> $L 2>&1
      python3 - "$Q" "$line" <<'PY'
import sys; q, line = sys.argv[1], sys.argv[2]
lines = open(q).read().splitlines(); i = next(k for k, l in enumerate(lines) if l.strip() == line.strip()); del lines[i]
open(q, "w").write("\n".join(lines) + ("\n" if lines else ""))
PY
      sleep 180; break
    fi
  done
  sleep 60
done
