#!/usr/bin/env python
"""Route the Stage-2 experiment exhaustively (addendum section 8, "Stages 2 and 3").

    python stage2_route.py --readouts <verdicts.json>          # {"1": true, ..., "4": false}
    python stage2_route.py --c1 pass --c2 pass --c3 fail --c4 pass
    python stage2_route.py --table                             # all sixteen rows

The verdicts are booleans the Stage-2 scorers write against plan v8 section 8's numerical
definitions (they are not recomputed here: a routing script that also scored would be
retyping numbers, which is what `c12_gate_table.py` exists to stop). `--state <file>` keeps
the router's history across invocations so outcome C's single registered bound release
cannot be taken twice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mace.modules.defect_routing import (ACTIONS, Criteria, Stage2Router,  # noqa: E402
                                         all_verdicts, route)


def _verdict(text: str) -> bool:
    t = text.strip().lower()
    if t in ("pass", "true", "1", "yes"):
        return True
    if t in ("fail", "false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"a verdict is pass/fail, not {text!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--readouts", help="JSON with the four verdicts")
    for i, name in enumerate(("hopping_stop", "delta_b_active", "splitting_sign",
                              "participation_spread"), start=1):
        p.add_argument(f"--c{i}", type=_verdict, help=f"criterion {i}: {name}")
    p.add_argument("--table", action="store_true", help="print all sixteen rows")
    p.add_argument("--state", help="router history file (outcome C's single repeat)")
    args = p.parse_args(argv)

    if args.table:
        print("c1 c2 c3 c4  outcome  action")
        for c in all_verdicts():
            o = route(c)
            print("  ".join("P" if v else "F" for v in c.as_tuple()) + f"   {o.value}       "
                  f"{ACTIONS[o]}")
        return 0

    if args.readouts:
        verdicts = json.loads(Path(args.readouts).read_text())
        criteria = Criteria.from_mapping(verdicts)
    else:
        given = {f"c{i}": getattr(args, f"c{i}") for i in range(1, 5)}
        if any(v is None for v in given.values()):
            p.error("give --readouts or all four of --c1..--c4 (the rule is exhaustive; a "
                    "missing verdict is not a verdict)")
        criteria = Criteria.from_mapping(given)

    router = Stage2Router()
    if args.state and Path(args.state).exists():
        for row in json.loads(Path(args.state).read_text()):
            router.decide(Criteria(*row["criteria"]))
    record = router.decide(criteria)
    if args.state:
        Path(args.state).write_text(json.dumps(
            [{"criteria": list(c.as_tuple()), "outcome": o.value} for c, o in router.history],
            indent=2) + "\n")
    print(f"criteria (1..4): {['pass' if v else 'fail' for v in criteria.as_tuple()]}")
    print(f"outcome {record['outcome']}: {record['action']}")
    if record["stop"]:
        print("STOP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
