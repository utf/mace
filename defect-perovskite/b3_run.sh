#!/usr/bin/env bash
# Launch one command on b3 from the worktree, detached, on one GPU, logging to ~/runs.
#   usage: b3_run.sh GPU LOGNAME command args...
# Exists because `ssh b3 '...'` starts in $HOME and a relative script path silently fails;
# every launch goes through here so the cwd, the env and the log location are one thing.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$W" || exit 1
gpu="$1"; log="$2"; shift 2
CUDA_VISIBLE_DEVICES="$gpu" nohup "$@" > "$R/$log.log" 2>&1 < /dev/null &
echo "launched $log on GPU $gpu, pid $!"
