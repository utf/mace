#!/usr/bin/env bash
# Run one command from the worktree on the local box with the worktree on PYTHONPATH.
# The mirror of b3_run.sh: /home/alex/src/mace is a real checkout of the same package, so a
# bare `python defect-perovskite/x.py` imports the WRONG mace. Foreground, unlike b3_run.sh.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
export PYTHONPATH="$W"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$W" || exit 1
exec "$@"
