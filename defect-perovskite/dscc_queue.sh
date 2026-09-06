#!/usr/bin/env bash
# Generic run queue: launches the commands listed one per line in a file, keeping at most
# PER_GPU runs on each GPU of GPUS (by nvidia-smi index; CUDA_VISIBLE_DEVICES by UUID).
# A line is "name|args..." where args are passed to dscc_train.py; --run_dir and --name are
# added here. Detached with setsid so it survives the launching ssh session.
#   setsid nohup bash defect-perovskite/dscc_queue.sh QUEUEFILE "4 5 7" 3 > ~/runs/queue.log 2>&1 < /dev/null &
set -u
W=$(cd "$(dirname "$0")/.." && pwd)
QUEUE=$1; GPUS=($2); PER_GPU=${3:-2}
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$W" || exit 1
declare -A PIDS
count_on_gpu() {   # running queue jobs on a GPU index
  local n=0
  for key in "${!PIDS[@]}"; do
    local pid=${PIDS[$key]%%:*} gpu=${PIDS[$key]##*:}
    if kill -0 "$pid" 2>/dev/null; then [ "$gpu" = "$1" ] && n=$((n + 1)); else unset "PIDS[$key]"; fi
  done
  echo $n
}
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  name=${line%%|*}; args=${line#*|}
  while :; do
    for gpu in "${GPUS[@]}"; do
      if [ "$(count_on_gpu "$gpu")" -lt "$PER_GPU" ]; then
        uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v g="$gpu" '$1==g {print $2}')
        CUDA_VISIBLE_DEVICES="${uuid:-$gpu}" nohup python "$W/defect-perovskite/dscc_train.py" --name "$name" \
          --run_dir "$HOME/runs/dscc/$name" --device cuda $args > "$HOME/runs/$name.log" 2>&1 < /dev/null &
        PIDS[$name]="$!:$gpu"
        echo "$(date '+%F %T') launched $name on GPU $gpu (pid $!)"
        sleep 5
        continue 3
      fi
    done
    sleep 60
  done
done < "$QUEUE"
echo "$(date '+%F %T') all queued runs launched; waiting"
for key in "${!PIDS[@]}"; do wait "${PIDS[$key]%%:*}" 2>/dev/null; done
echo "$(date '+%F %T') queue finished"
