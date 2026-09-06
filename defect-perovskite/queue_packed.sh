#!/bin/bash
# Packed training waves (v8.1 operating rule): the jobs run on the HEALTHY GPUs among 4-7,
# up to PER_GPU runs per GPU (default 2), each process capped at 32 / n_slots threads, new
# jobs started as slots free, starts staggered by STAGGER seconds.
#   JOBS="s14a:full:1 s14a:full:2 ... s14c:off:6" bash queue_packed.sh
# A job is TAG:MODE:SEED -> run name TAG_sSEED with DEFECT_MADELUNG_RANGE=MODE.
# GPUs: `nvidia-smi` must report the device AND torch must initialise CUDA on it; a node
# whose CUDA is down (a faulted GPU takes the runtime down for every process) aborts here
# rather than queueing runs that would die at startup.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTHONPATH="$(cd "$HERE/.." && pwd)"
export DEFECT_NULL_REFERENCE="${DEFECT_NULL_REFERENCE-$HOME/runs/aprime_nulls.json}"
RECIPE="${RECIPE:-stage_b_run}"      # stage_b_run (Stage B) or stage_v81_run (plan v8.1)
source "$HERE/stage_b_recipe.sh"
PER_GPU="${PER_GPU:-2}"; STAGGER="${STAGGER:-60}"; CANDIDATES="${CANDIDATES:-4 5 6 7}"
JOBS="${JOBS:?JOBS='tag:mode:seed ...' required}"

healthy=()
for g in $CANDIDATES; do
    if nvidia-smi -i "$g" --query-gpu=index --format=csv,noheader >/dev/null 2>&1 \
       && CUDA_VISIBLE_DEVICES="$g" python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)" 2>/dev/null; then
        healthy+=("$g")
    else
        echo "GPU $g unavailable"
    fi
done
[ "${#healthy[@]}" -gt 0 ] || { echo "ABORT: no healthy GPU among $CANDIDATES"; exit 1; }
n_slots=$(( ${#healthy[@]} * PER_GPU ))
threads=$(( $(nproc) / n_slots )); [ "$threads" -lt 2 ] && threads=2
echo "=== packed queue ($RECIPE): GPUs ${healthy[*]}, $n_slots slots, $threads threads each, jobs: $JOBS  $(date +%F' '%H:%M:%S) ==="

slots=()          # pid per slot ("" = free)
slot_gpu=()
for ((s = 0; s < n_slots; s++)); do slots+=(""); slot_gpu+=("${healthy[$((s % ${#healthy[@]}))]}"); done

free_slot () {    # echo the index of a free slot, reaping finished jobs first
    for ((s = 0; s < n_slots; s++)); do
        local pid="${slots[$s]}"
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then wait "$pid" 2>/dev/null; slots[$s]=""; fi
        [ -z "${slots[$s]}" ] && { echo "$s"; return 0; }
    done
    return 1
}

for job in $JOBS; do
    IFS=: read -r tag mode seed <<< "$job"
    name="${tag}_s${seed}"
    while ! s=$(free_slot); do sleep 30; done
    gpu="${slot_gpu[$s]}"
    echo "start $name on GPU $gpu (slot $s)  $(date +%F' '%H:%M:%S)"
    "$RECIPE" "$gpu" "$seed" "$name" "$mode" "$threads" &
    slots[$s]=$!
    sleep "$STAGGER"
done
for pid in "${slots[@]}"; do [ -n "$pid" ] && wait "$pid"; done
echo "=== packed queue complete  $(date +%F' '%H:%M:%S) ==="
