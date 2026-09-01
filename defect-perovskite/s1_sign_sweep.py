#!/usr/bin/env python3
"""S1: does the head's axial hub force oppose the residual it is fitting?

The diagnosis is that the head's level is `lambda_min` of a bonding-signed H -- an ADDED
ELECTRON's level, eps - t(d), which falls as the Pb pair closes. The carrier here is a hole
removed from the bonding gap state, whose energy is -eps + t(d) and RISES as the pair closes.
If that is right, the head's dominant force term has the wrong sign at the hub: attractive
where DFT is repulsive. And the hub on-site term cannot compensate, because the partner Pb at
5.3-6.8 A lies outside the 5 A block-1 descriptor, so eps_hub cannot depend on d at all.

This is the gate for the whole sign-fix plan: if the signs are NOT systematically opposite,
the diagnosis is wrong and nothing downstream should be built.

PREDICTIONS, recorded before running:
  * sign agreement well below 0.5 on localised cells (target pushes apart, correction pulls
    together);
  * |H_ab| across the vacancy at the t_min floor (~0.01 eV at 5.6 A by construction), against
    the 0.097 eV the distance-matched pair carried in the old head -- "~0" means "at the
    floor", not literally zero;
  * the decomposition showing the on-site terms straining against the hopping term.

Stratified by localisation, because the prediction is specifically about cells where occupancy
has been forced onto a compact state -- pooling all 24 would dilute it.

d1_sign.py and d1_leak.py are invoked as they stand rather than reimplemented here: they are
the scripts that produced the D1 and sign results this diagnosis rests on, and a second copy
of either comparison is the reimplementation pattern that has produced four passing guards
over broken paths on this project.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def run_json(script, model, data, limit, device, out, extra=None):
    cmd = [sys.executable, str(HERE / script), "--model", str(model), "--data", str(data),
           "--limit", str(limit), "--device", device, "--out", str(out)]
    if extra:
        cmd += extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not Path(out).exists():
        return None, r.stderr[-400:]
    try:
        return json.loads(Path(out).read_text()), None
    except Exception as exc:                                # noqa: BLE001
        return None, f"{exc}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True, help="directory of .model files")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--results", type=Path, nargs="*", default=[],
                    help="tbv3_*.json, joined on saved_model for the N_eff stratification")
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/s1_tmp"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.tmp.mkdir(parents=True, exist_ok=True)

    # N_eff / gate status per trained cell, so the sweep can be stratified.
    meta = {}
    for f in args.results:
        try:
            for r in json.loads(Path(f).read_text()):
                if "saved_model" in r:
                    meta[Path(r["saved_model"]).name] = r
        except Exception:                                   # noqa: BLE001
            continue

    models = sorted(Path(args.models).glob("*.model"))
    print(f"  {len(models)} models, {args.limit} charged frames each, "
          f"{len(meta)} joined to training results")
    rows = []
    for i, m in enumerate(models, 1):
        sign, err1 = run_json("d1_sign.py", m, args.data, args.limit, args.device,
                              args.tmp / f"{m.stem}_sign.json")
        leak, err2 = run_json("d1_leak.py", m, args.data, args.limit, args.device,
                              args.tmp / f"{m.stem}_leak.json")
        info = meta.get(m.name, {})
        row = {"model": m.name,
               "neff": info.get("neff"), "gate_neff": info.get("gate_neff"),
               "axial_red": info.get("axial_red"), "f_m": info.get("f_m"),
               "e_gap": info.get("e_gap"), "seed": info.get("seed")}
        if sign:
            row.update(agree=sign.get("sign_agreement"), pearson=sign.get("pearson"),
                       target_mean=sign.get("target_mean"),
                       correction_mean=sign.get("corr_mean"),
                       target_median=sign.get("target_median"),
                       correction_median=sign.get("corr_median"), n=sign.get("n"))
        else:
            row["sign_error"] = err1
        if leak and isinstance(leak.get("summary"), dict):
            row.update({f"leak_{k}": v for k, v in leak["summary"].items()
                        if not isinstance(v, (dict, list))})
        else:
            row["leak_error"] = err2
        rows.append(row)
        a = row.get("agree")
        print(f"  [{i:2d}/{len(models)}] {m.name:32s} "
              f"N_eff {str(row['neff'])[:6]:>6s}  agree "
              f"{('%.3f' % a) if a is not None else '  --  '}  "
              f"r {('%+.3f' % row['pearson']) if row.get('pearson') is not None else '  --  '}")
        args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    ok = [r for r in rows if r.get("agree") is not None]
    if ok:
        loc = [r for r in ok if r.get("gate_neff")]
        rest = [r for r in ok if not r.get("gate_neff")]
        print()
        for name, grp in (("localised (N_eff <= region)", loc), ("the rest", rest),
                          ("all", ok)):
            if not grp:
                continue
            print(f"  {name:28s} n={len(grp):2d}  "
                  f"agree {np.mean([r['agree'] for r in grp]):.3f}  "
                  f"r {np.mean([r['pearson'] for r in grp]):+.3f}  "
                  f"target {np.mean([r['target_mean'] for r in grp]):+.4f}  "
                  f"correction {np.mean([r['correction_mean'] for r in grp]):+.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
