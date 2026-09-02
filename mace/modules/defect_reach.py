"""In-loop assertion that the neighbour list the LOSS uses spans the carrier Hamiltonian.

Four times now a guard has measured a reimplementation rather than the thing it guarded, and
the last one voided a 20-seed screen: `graph_cutoff` was applied only to the validation
fallback path, so every training run built its graph at r_max and the two vacancy-sharing Pb
had no edge in 0 of 40 charged frames. A pre-flight on a path the loss never executes is
worse than no check at all, because it produces a passed check.

So this runs on a real batch drawn from the training DataLoader itself -- the same object the
optimiser iterates -- and aborts the run if the graph is short.

LABEL-FREE BY CONSTRUCTION. The plan phrases the criterion as "fraction of charged frames in
which the two hub Pb share an edge", which needs the vacancy assignment inside the training
loop. The equivalent structural fact does not: if the graph was built at `graph_cutoff` then
edges longer than r_max exist and the longest edge approaches the cutoff, whereas a graph
built at r_max cannot contain them at all. That distinguishes the two cases exactly -- a 5 A
graph has max edge <= 5.0 by construction -- without telling the training process anything
about where the defect is. The hub-pair fraction remains available as an evaluation-time
diagnostic, where the assignment is allowed.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

__all__ = ["reach_report", "assert_carrier_reach", "count_edges_within"]


def _edge_lengths(batch) -> Optional[torch.Tensor]:
    try:
        idx = batch["edge_index"] if isinstance(batch, dict) else batch.edge_index
        pos = batch["positions"] if isinstance(batch, dict) else batch.positions
    except (KeyError, AttributeError):
        return None
    if idx is None or idx.numel() == 0:
        return None
    vec = pos[idx[1]] - pos[idx[0]]
    shifts = batch["shifts"] if isinstance(batch, dict) else getattr(batch, "shifts", None)
    if shifts is not None and shifts.numel() == vec.numel():
        vec = vec + shifts
    return torch.linalg.norm(vec, dim=-1).detach()


def count_edges_within(batch, r_max: float):
    """`(total_edges, edges_inside r_max)` for one batch. Both counts, so the caller divides.

    WHY IT IS NEEDED. `avg_num_neighbors` divides every message in the trunk, and the trunk's
    messages only run over edges inside `r_max`. With a carrier head the loader builds its
    graph at the CARRIER cutoff instead, so a neighbour count taken off that graph counts
    edges the trunk never uses -- 10 A against r_max 5.0 gives roughly eight times too many
    here. The ratio of the two counts is the correction, and it is measured on the same batch
    the loader produced rather than assumed from a volume argument, because the graph is
    built per frame with periodic images and a `(10/5)^3` estimate is not the same number.
    """
    lengths = _edge_lengths(batch)
    if lengths is None or lengths.numel() == 0:
        return 0, 0
    return int(lengths.numel()), int((lengths <= float(r_max)).sum())


def reach_report(batch, r_max: float, cutoff: float,
                 window=(5.2, 5.8)) -> Dict[str, float]:
    """Structural facts about the graph the loss is about to consume."""
    lengths = _edge_lengths(batch)
    if lengths is None or lengths.numel() == 0:
        return {}
    long_edges = (lengths > float(r_max)).float().mean().item()
    in_window = ((lengths >= window[0]) & (lengths < window[1])).sum().item()
    return {
        "max_edge": float(lengths.max()),
        "median_edge": float(lengths.median()),
        "frac_beyond_r_max": float(long_edges),
        "edges_in_hub_window": float(in_window),
        "n_edges": float(lengths.numel()),
        "configured_cutoff": float(cutoff),
        "r_max": float(r_max),
    }


def envelope_at(r: float, r_couple: float, use_decay: bool = True) -> float:
    """The head's own coupling envelope, as a pure function of distance.

    Duplicated from the head deliberately at exactly one point -- here -- so the reach test
    can be evaluated without constructing a head, and kept honest by
    `test_envelope_matches_the_head` rather than by inspection.
    """
    x = min(float(r) / float(r_couple), 1.0)
    if use_decay:
        return (1.0 - x ** 6) ** 2
    return (1.0 - x) ** 3 * (1.0 + 3.0 * x + 6.0 * x * x)


def assert_coupling_envelope(batch, r_couple: float, window=(5.2, 6.8),
                             floor: float = 0.3, frame_fraction: float = 0.99,
                             strict: bool = True) -> Dict[str, float]:
    """V3 test 4: the vacancy-flanking separations are genuinely coupled, label-free.

    The plan phrases this as "the flanking pair coupled with f_env >= 0.3 in >= 99% of charged
    frames", which as written needs the vacancy assignment inside the training loop and would
    breach hard rule 1. The structural equivalent does not: the flanking pair sits at
    5.2-6.8 A, so it is enough that (a) the envelope at the FAR end of that window clears the
    floor, which is a pure function of r_couple and needs no graph at all, and (b) essentially
    every frame actually has an edge in the window. Neither statement mentions where the
    defect is.
    """
    lengths = _edge_lengths(batch)
    if lengths is None or lengths.numel() == 0:
        logging.warning("Coupling envelope check: batch carried no edges")
        return {}
    f_far = envelope_at(window[1], r_couple)
    in_win = (lengths >= window[0]) & (lengths < window[1])

    idx = batch["batch"] if isinstance(batch, dict) else getattr(batch, "batch", None)
    if idx is None:
        frac_frames = float(in_win.any())
    else:
        src = batch["edge_index"][0] if isinstance(batch, dict) else batch.edge_index[0]
        gid = idx[src]
        n_g = int(gid.max()) + 1 if gid.numel() else 1
        has = torch.zeros(n_g, dtype=torch.bool, device=lengths.device)
        has[gid[in_win]] = True
        frac_frames = float(has.float().mean())

    report = {"f_env_at_window_far_end": f_far,
              "edges_in_window": float(in_win.sum()),
              "frames_with_a_window_edge": frac_frames,
              "r_couple": float(r_couple)}
    logging.info(
        f"Coupling envelope: f_env({window[1]} A; r_couple={r_couple:.1f}) = {f_far:.3f}, "
        f"{int(in_win.sum())} edges in {window[0]}-{window[1]} A, "
        f"{frac_frames:.1%} of frames carry one")

    if f_far < floor:
        msg = (f"COUPLING FAILURE: the envelope at {window[1]} A is {f_far:.3f}, below the "
               f"{floor} floor, so a flanking pair at that separation is effectively "
               f"uncoupled and the two-site state cannot be represented. Raise r_couple.")
        if strict:
            raise RuntimeError(msg)
        logging.error(msg)
    if frac_frames < frame_fraction:
        msg = (f"COUPLING FAILURE: only {frac_frames:.1%} of frames have an edge in the "
               f"{window[0]}-{window[1]} A window, below the required {frame_fraction:.0%}. "
               f"The graph is too short for the flanking separations this system has.")
        if strict:
            raise RuntimeError(msg)
        logging.error(msg)
    return report


def assert_carrier_reach(batch, r_max: float, cutoff: float, tolerance: float = 0.95,
                         strict: bool = True) -> Dict[str, float]:
    """Log the reach of the loss's own graph; abort if it was built at r_max.

    `cutoff` is what graph_cutoff() asked for. If the realised graph does not get within
    `tolerance` of it, the loader ignored it -- which is precisely the failure that voided
    R1, R2 and Test 1 -- and nothing downstream would be interpretable.
    """
    report = reach_report(batch, r_max, cutoff)
    if not report:
        logging.warning("Carrier reach check: batch carried no edges to measure")
        return report

    logging.info(
        "Carrier reach (from the training loader's own batch): "
        f"max edge {report['max_edge']:.2f} A, median {report['median_edge']:.2f} A, "
        f"{report['frac_beyond_r_max']:.1%} of edges beyond r_max={r_max:.1f}, "
        f"{int(report['edges_in_hub_window'])} edges in the 5.2-5.8 A window, "
        f"configured cutoff {cutoff:.1f} A")

    if cutoff <= float(r_max) + 1e-6:
        return report                                  # no long-range head; nothing to check

    if report["max_edge"] < tolerance * float(cutoff):
        msg = (f"CARRIER REACH FAILURE: the graph the loss consumes reaches only "
               f"{report['max_edge']:.2f} A but graph_cutoff asked for {cutoff:.1f} A. "
               f"The neighbour list was built at r_max={r_max:.1f}, so the two "
               f"vacancy-sharing Pb have no edge and the two-site state cannot be "
               f"represented. This is the fault that voided the previous screen; the run is "
               f"aborting rather than producing another uninterpretable result.")
        if strict:
            raise RuntimeError(msg)
        logging.error(msg)
    elif report["frac_beyond_r_max"] <= 0.0:
        msg = ("CARRIER REACH FAILURE: no edge in the batch is longer than r_max, so the "
               "carrier Hamiltonian has exactly the trunk's range.")
        if strict:
            raise RuntimeError(msg)
        logging.error(msg)
    return report
