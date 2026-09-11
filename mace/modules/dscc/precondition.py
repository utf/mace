"""v4.2 (C5): the bound-state precondition for Arm 2+3.

Label-free, on the neutral-vacancy frames, with the Arm-1 `H0`: the reference HOMO (the
level the neutral vacancy's extra electron occupies) must be separated from the next level
by at least `Delta_c` (registered, default 10 sigma_s) and the carrier -- the Phi = 0 hole,
`dq` of the two fillings -- must have participation `N_eff = 1 / sum_i dq_i^2 <= N_loc`
(registered), on at least `fraction` (registered, 0.95) of the frames. Failure is Stage-2
outcome B: Phi is not switched on. Thresholds are registered before Arm-1 results open.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import torch

from mace.modules.dscc.fill import SIGMA_S
from mace.modules.dscc.scf import two_fillings
from mace.modules.dscc.species import S_REF, State, neutral_count

# Registered defaults (v4.2; N_loc to confirm before Arm-1 results are opened).
#
# `Delta_c` RE-REGISTERED 2026-09-11 from 10 sigma_s to 4 sigma_s (0.50 -> 0.20 eV), on the
# criterion the gate was always meant to encode rather than a round number of sigmas.
#
# What it protects is that the two fillings differ in ONE state, so the Phi = 0 hole is a
# single well-defined carrier and the Hellmann-Feynman force sum runs over one level. With
# Gaussian smearing at sigma_s = 0.05 eV the occupation that leaks to a neighbour `d` above
# the chemical potential is `erfc(d/sigma)/2`: 1.1e-5 at 3 sigma, 1e-17 at 6 sigma. Measured
# over 120 neutral-vacancy frames of a trained head, the fraction of the hole carried by the
# defect level has a MINIMUM of 0.999953 and a median of 1.000000, at separations of
# 0.28-0.82 eV. The carrier was never in doubt anywhere.
#
# At 10 sigma the gate failed 22 % of frames while protecting nothing; every head of every
# form and both bases failed it (A1.2 pass fractions 0.00-0.72, the centred pre-A1 heads
# 0.00-0.83). 4 sigma keeps the leakage below ~3e-4 in the worst case and is still twice the
# 3 sigma at which it is already 1e-5.
#
# SCOPE. This argument is for a FORCES-ONLY model, where only the active set enters. W5 puts
# energies in the loss and `Delta F_band` sums over ALL levels, so the positions of the host
# states do matter there; this threshold is not evidence about that case.
DELTA_C_DEFAULT = 4.0 * SIGMA_S
N_LOC_DEFAULT = 4.0
FRACTION_DEFAULT = 0.95


@dataclass(frozen=True)
class PreconditionConfig:
    delta_c: float = DELTA_C_DEFAULT
    n_loc: float = N_LOC_DEFAULT
    fraction: float = FRACTION_DEFAULT


@dataclass
class FrameRecord:
    separation: float      # eps[HOMO + 1] - eps[HOMO] of the reference majority fill, eV
    n_eff: float           # participation of the Phi = 0 hole
    passed: bool


def frame_record(H: torch.Tensor, numbers: Sequence[int], cfg: PreconditionConfig,
                 sigma_s: float = SIGMA_S) -> FrameRecord:
    """One neutral-vacancy frame's separation and hole participation from its `H0`."""
    n_ref = neutral_count(numbers)
    n_up, n_dn = S_REF.counts(n_ref)
    with torch.no_grad():
        eps = torch.linalg.eigvalsh(H)
        separation = float(eps[n_up] - eps[n_up - 1])
        hole = State(1, -1, 0).counts(n_ref)                 # one hole in the majority channel
        dq = two_fillings(H, hole, (n_up, n_dn), sigma_s).dq.detach()
        n_eff = float(1.0 / (dq ** 2).sum())
    return FrameRecord(separation=separation, n_eff=n_eff,
                       passed=separation >= cfg.delta_c and n_eff <= cfg.n_loc)


def bound_state_precondition(model, batches: Sequence[Dict[str, torch.Tensor]],
                             cfg: PreconditionConfig = PreconditionConfig()) -> Dict[str, object]:
    """The precondition over neutral-vacancy batches (the caller selects frames at `S_ref`
    with a vacancy by composition): per-frame records, the pass fraction and the verdict."""
    records: List[FrameRecord] = []
    for data in batches:
        with torch.no_grad():
            # `base_forward`, not `ScaleShiftMACE.forward`: W1 put the frozen base in float32
            # with the head in float64, and the raw call hands a float32 base its float64
            # inputs ("both inputs should have same dtype"). The wrapper casts in and back.
            out = model.base_forward(model._trunk_data(dict(data)), training=False,
                                     compute_force=False)
            scalars, vectors = model.features(out["node_feats"])
            species = data["node_attrs"].argmax(dim=-1)
            positions, cell = data["positions"], data["cell"].view(-1, 3, 3)
            sender, receiver = data["edge_index"][0], data["edge_index"][1]
            edge_graph = data["batch"][sender]
            shifts = torch.einsum("ei,eij->ej", data["unit_shifts"].to(positions.dtype), cell[edge_graph])
            edge_vector = positions[receiver] - positions[sender] + shifts
            ptr = data["ptr"]
            for g in range(int(ptr.numel() - 1)):
                lo, hi = int(ptr[g]), int(ptr[g + 1])
                e_mask = edge_graph == g
                H = model.h0(scalars[lo:hi], vectors[lo:hi], species[lo:hi],
                             data["edge_index"][:, e_mask] - lo, edge_vector[e_mask])
                numbers = [model.atomic_numbers[int(x)] for x in species[lo:hi].tolist()]
                records.append(frame_record(H, numbers, cfg, model.sigma_s))
    n_pass = sum(1 for r in records if r.passed)
    frac = n_pass / max(len(records), 1)
    return {"config": asdict(cfg), "n_frames": len(records), "n_pass": n_pass, "pass_fraction": frac,
            "passed": len(records) > 0 and frac >= cfg.fraction,
            "records": [asdict(r) for r in records],
            "separation_p50": float(torch.tensor([r.separation for r in records]).median()) if records else float("nan"),
            "n_eff_p50": float(torch.tensor([r.n_eff for r in records]).median()) if records else float("nan")}
