#!/usr/bin/env python3
"""Evaluate the Step A acceptance gates and print an explicit pass/fail table.

The gates are on **drift**, not on absolute ``delta_lr``. ``host.carrier`` sits at
-1.906 eV, constant to five decimals, and it is physical -- the carrier monopole in the
host lattice potential -- so ``|delta_lr|`` cannot and should not fall to the scale of the
image term. An exponent alone is insufficient in either direction: a wrong-magnitude term
can carry the right scaling, and a tiny-prefactor term can carry a bad exponent harmlessly.
Every gate below therefore pairs a scaling with a magnitude.

Reads whatever artefacts exist and reports the rest as SKIP rather than failing, so a
partial run still produces a usable table.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RESET, GREEN, RED, YELLOW = "\033[0m", "\033[32m", "\033[31m", "\033[33m"


class Gates:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, name: str, before: str, measured: str, verdict: str) -> None:
        self.rows.append((name, before, measured, verdict))

    def check(self, name: str, before: str, measured: str, ok: bool) -> None:
        self.add(name, before, measured, "PASS" if ok else "FAIL")

    def skip(self, name: str, before: str, why: str) -> None:
        self.add(name, before, why, "SKIP")

    def report(self) -> int:
        width = max(len(r[0]) for r in self.rows) + 2
        print(f"\n{'gate':<{width}} {'before':>16} {'measured':>26}  verdict")
        print("-" * (width + 55))
        for name, before, measured, verdict in self.rows:
            colour = {"PASS": GREEN, "FAIL": RED, "SKIP": YELLOW}[verdict]
            print(f"{name:<{width}} {before:>16} {measured:>26}  {colour}{verdict}{RESET}")
        failed = sum(1 for r in self.rows if r[3] == "FAIL")
        skipped = sum(1 for r in self.rows if r[3] == "SKIP")
        print()
        if failed:
            print(f"{RED}{failed} gate(s) FAILED{RESET}", end="")
        else:
            print(f"{GREEN}all evaluated gates passed{RESET}", end="")
        print(f"; {skipped} skipped" if skipped else "")
        return 1 if failed else 0


def strip(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_decomposition(path: Path) -> dict:
    """Pull the exponent row and the per-N table out of lr_term_scaling output."""
    text = strip(path.read_text())
    block = text[text.index("Decomposition of delta_lr"):]
    lines = [line for line in block.splitlines() if line.strip()]
    header = next(line for line in lines if line.strip().startswith("N"))
    keys = header.split()[1:]
    rows, scaling = [], {}
    for line in lines[lines.index(header) + 1:]:
        parts = line.split()
        if parts and parts[0] == "scaling":
            scaling = dict(zip(keys, (float(v) for v in parts[1:])))
            break
        if parts and parts[0].isdigit():
            rows.append(dict(zip(["N"] + keys, (float(v) for v in parts))))
    ss = {}
    if "Charge bookkeeping" in text:
        tail = text[text.index("Charge bookkeeping"):]
        tail_lines = [line for line in tail.splitlines() if line.strip()]
        tail_header = next(line for line in tail_lines if line.strip().startswith("N"))
        tail_keys = tail_header.split()[1:]
        for line in tail_lines[tail_lines.index(tail_header) + 1:]:
            parts = line.split()
            if parts and parts[0] == "scaling":
                break
            if parts and parts[0].isdigit():
                ss[int(float(parts[0]))] = dict(
                    zip(tail_keys, (float(v) for v in parts[1:]))
                )
    return {"rows": rows, "scaling": scaling, "bookkeeping": ss}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--name", default="perov_lr_a4_s1")
    parser.add_argument("--runs", type=Path, default=Path.home() / "runs")
    args = parser.parse_args()
    gates = Gates()

    # ---- 1. training accuracy -------------------------------------------------------
    log = args.runs / f"{args.name}.log"
    if log.exists():
        text = strip(log.read_text())
        # The row is "| valid_Default |   2.7  |   9.7  | None | None |", so the config
        # type carries a suffix and an anchored "valid|" never matches.
        matches = re.findall(
            r"\|\s*valid\w*\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", text
        )
        if matches:
            energy, force = (float(v) for v in matches[-1])
            gates.check(
                "valid RMSE E (meV/atom)", "2.7", f"{energy:.2f}", energy <= 3.1 * 1.15
            )
            gates.check(
                "valid RMSE F (meV/A)", "9.7", f"{force:.2f}", force <= 12.0 * 1.15
            )
        else:
            gates.skip("valid RMSE", "2.7 / 9.7", "no error table in log")

        channels = [line for line in text.splitlines() if "carrier channels" in line]
        if channels:
            last = channels[-1]
            partic = re.search(r"partic=\[([^\]]+)\]", last)
            size_f = re.search(r"size_f=\[([^\]]+)\]", last)
            if partic:
                live = float(partic.group(1).split()[2])  # h_maj
                gates.check("partic (h_maj)", "16.07", f"{live:.2f}", live < 6.0)
            if size_f:
                value = float(size_f.group(1).split()[2])
                gates.check(
                    "size_f (h_maj)", "1.000 (exempt)", f"{value:.3f}", value < 0.1
                )
        else:
            gates.skip("attention", "16.07", "no channel diagnostics in log")
    else:
        gates.skip("training", "-", f"{log} not found")

    # ---- 2. decomposition ------------------------------------------------------------
    decomposition = here / f"{args.name}_decomposition.txt"
    if decomposition.exists():
        data = parse_decomposition(decomposition)
        rows, scaling = data["rows"], data["scaling"]
        for key, before_scale, before_drift in (
            ("pol2", "+1.002", "21.5 eV"),
            ("host_pol", "+1.022", "52.3 eV"),
        ):
            if key not in scaling:
                gates.skip(f"{key}", before_scale, "column missing")
                continue
            drift = abs(rows[-1][key] - rows[0][key])
            gates.check(
                f"{key} drift",
                before_drift,
                f"{drift * 1e3:.2f} meV (exp {scaling[key]:+.3f})",
                drift < 0.05,
            )
        if "delta_lr" in scaling:
            drift = abs(rows[-1]["delta_lr"] - rows[0]["delta_lr"])
            gates.check(
                "delta_lr drift",
                "52.4 eV",
                f"{drift * 1e3:.2f} meV (exp {scaling['delta_lr']:+.3f})",
                drift < 0.05,
            )
        book = data["bookkeeping"]
        if book:
            values = [v["ss_pol"] for v in book.values()]
            gates.check(
                "ss_pol", "2.22 -> 7.41", f"{max(values):.4f}", max(values) < 0.05
            )
            carriers = {v["sum_carrier"] for v in book.values()}
            gates.check(
                "sum q_i = a q",
                "0.50000",
                f"{sorted(carriers)[-1]:.5f}",
                all(abs(c - 0.5) < 1e-6 for c in carriers),
            )
    else:
        gates.skip("decomposition", "+1.002 / +1.022", f"{decomposition.name} missing")

    # ---- 3. relaxed ladder -----------------------------------------------------------
    for suffix, label in (("", "eps_opt drift"), ("_dilute", "eps_opt (dilute)")):
        tag = f"{args.name}{suffix}"
        path = here / f"perov_size_{tag}.json"
        if not path.exists():
            gates.skip(label, "77 eV", f"perov_size_{tag}.json missing")
            continue
        rows = sorted(json.loads(path.read_text())["rows"], key=lambda r: r["n_host"])
        drift = abs(rows[-1]["eps_opt_eV"] - rows[0]["eps_opt_eV"])
        gates.check(label, "77 eV", f"{drift * 1e3:.1f} meV", drift < 0.20)

    raise SystemExit(gates.report())


if __name__ == "__main__":
    main()
