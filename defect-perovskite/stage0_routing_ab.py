#!/usr/bin/env python
"""Did the v8.1 Tier-1 routing change (06f4fd1) alter the class integers?

The gauge A/B (`stage0_gauge_ab.py`) compares two Hamiltonian conventions under ONE routing,
so it cannot see a routing-induced change; and the v6 golden compare is not an instrument
here, because its capture predates Stage 1.3's Madelung range work and was already failing
before either change. The direct question -- does the constructor produce the same integers
before and after the routing change -- was therefore left open.

This closes it. It builds the composition-class table on the golden frames with whichever
tree it is run from, and writes the integers to JSON. Run it twice, once from a checkout of
`06f4fd1^` (the last commit with the VBM-proximity Tier 1) and once from HEAD, then compare:

    python stage0_routing_ab.py --model <base.model> --out pre.json      # from 06f4fd1^
    python stage0_routing_ab.py --model <base.model> --out post.json     # from HEAD
    python stage0_routing_ab.py --compare pre.json post.json

`tier` is EXPECTED to differ: the point of the change is that the first size of a homologous
family now runs the continuation and later sizes are verified by transport. What must not
differ is any integer -- `m_vb`, `n_e`, `n_h`, `q_core`, `d_sigma`, `n_sigma` -- because those
are what every downstream energy depends on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The integers the comparison is about. `tier`, `nearest`, `tier1_reason`, `tier1_record`,
# `gamma` and the Tier-2 agreement flags are routing DIAGNOSTICS and may legitimately move.
INTEGER_FIELDS = ("m_vb", "n_e", "n_h", "q_core", "n_sigma", "d_sigma", "m_f", "ambiguous")


def collect(model_path: str) -> dict:
    import torch

    import stage0_golden as G

    args = argparse.Namespace(model=model_path)
    model, ctx, frames, _ = G.setup(args)
    # Without the gauge: this asks about the ROUTING only, and the pre-change tree has no
    # gauge to register anyway, so registering one here would confound the comparison.
    table = G.ensure_table(model, ctx, frames, log=False, gauge=False)

    out = {}
    for key, rec in sorted(table.get("classes", {}).items()):
        out[key] = {f: rec.get(f) for f in INTEGER_FIELDS if f in rec}
        out[key]["_tier"] = rec.get("tier")          # recorded, not compared
        out[key]["_n_atoms"] = rec.get("n_atoms")
    return {"classes": out, "git_sha": G.git_sha(), "model": model_path,
            "torch": torch.__version__}


def compare(pre_path: str, post_path: str) -> int:
    pre = json.loads(Path(pre_path).read_text())
    post = json.loads(Path(post_path).read_text())
    pre_c, post_c = pre["classes"], post["classes"]

    print(f"pre  {pre['git_sha']}   {len(pre_c)} classes")
    print(f"post {post['git_sha']}   {len(post_c)} classes")

    diffs, added, compared = [], set(), set()
    if set(pre_c) != set(post_c):
        diffs.append(f"class SET differs: only-pre {sorted(set(pre_c) - set(post_c))}, "
                     f"only-post {sorted(set(post_c) - set(pre_c))}")

    for key in sorted(set(pre_c) & set(post_c)):
        a, b = pre_c[key], post_c[key]
        for field in INTEGER_FIELDS:
            # A field absent from the PRE record is a schema addition, not a changed
            # integer: `d_sigma` and `m_f` were introduced by this programme, so comparing
            # them against a record written before they existed would report every class
            # as changed and hide any real difference in the noise.
            if field not in a:
                if field in b:
                    added.add(field)
                continue
            if field not in b:
                diffs.append(f"{key}: {field} was dropped ({a[field]!r} -> absent)")
            elif a[field] != b[field]:
                diffs.append(f"{key}: {field}  {a[field]!r} -> {b[field]!r}")
            else:
                compared.add(field)
        if a.get("_tier") != b.get("_tier"):
            print(f"  routing moved (expected): {key}  tier {a.get('_tier')} -> "
                  f"{b.get('_tier')}")

    print()
    if added:
        print(f"fields new since the pre record (not comparable): {', '.join(sorted(added))}")
    if diffs:
        print(f"INTEGERS CHANGED -- {len(diffs)} difference(s):")
        for d in diffs:
            print("  " + d)
        return 1
    print(f"INTEGERS UNCHANGED across the routing change: "
          f"{len(set(pre_c) & set(post_c))} classes, fields compared: "
          f"{', '.join(sorted(compared))}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model")
    p.add_argument("--out")
    p.add_argument("--compare", nargs=2, metavar=("PRE", "POST"))
    args = p.parse_args(argv)

    if args.compare:
        return compare(*args.compare)
    if not (args.model and args.out):
        p.error("--model and --out are required unless --compare is given")
    record = collect(args.model)
    Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {len(record['classes'])} classes at {record['git_sha']}")
    for key, rec in sorted(record["classes"].items()):
        print(f"  {key}  tier={rec.get('_tier')}  m_vb={rec.get('m_vb')}  "
              f"q_core={rec.get('q_core')}  n_e={rec.get('n_e')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
