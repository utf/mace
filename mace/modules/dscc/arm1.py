"""Plan section 7, Arm 1: the decision experiment on the response decomposition (not RMSE),
with the thresholds to be registered before results are opened:

  (i)   vacancy-spanning `pp` stop fraction <= 1 of 6 seeds: a seed "stops" when the
        learned modulation of `pp_sigma` on the flanking Pb-Pb pair sits at its bound
        (|tanh g| >= SATURATION) on more than half of the neutral-vacancy frames;
  (ii)  the flanking-Pb tensor coefficient (|b Q| on the flanking Pb) exceeds 3x its bulk
        spread (the standard deviation of |b Q| over the other Pb), and the scalar-only
        control loses >= 30 % of the modelled level-vs-bond response (the slope of the
        reference HOMO level against d);
  (iii) the Cl p_sigma / p_pi splitting has the analytically frozen sign: the axial
        Pb-Cl-Pb field lowers p_sigma below p_pi, so `b[Cl] < 0` with `Q_zz > 0` along
        the bond (registered sign: negative splitting `eps_sigma - eps_pi`);
  (iv)  the participation-ratio seed spread halves (full vs scalar-only) at unchanged or
        better force quality without coefficient saturation.

Every quantity is label-free: the flanking Pb are the Pb with a first shell of five Cl,
`d` their distance, the hole is the Phi = 0 two-fillings `dq`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from mace.modules.dscc.admission import slope_with_se
from mace.modules.dscc.hamiltonian import quadrupole_descriptor
from mace.modules.dscc.kernels import minimum_image_distances
from mace.modules.dscc.scf import two_fillings
from mace.modules.dscc.species import S_REF, State, neutral_count

SATURATION = 0.98


@dataclass
class Arm1Frame:
    d: float
    homo: float                    # reference HOMO level, eV
    separation: float              # eps[HOMO+1] - eps[HOMO]
    n_eff: float                   # participation of the Phi = 0 hole
    pp_modulation: float           # exp(beta tanh g) on the flanking pair's pp_sigma, 1 = Harrison
    pp_saturated: bool
    flank_tensor: float            # mean |b Q| on the two flanking Pb, eV
    bulk_tensor_spread: float      # std of |b Q| over the other Pb, eV
    cl_splitting: float            # mean over bridging Cl of eps_sigma - eps_pi from the directional block, eV
    coefficient_saturation: float  # fraction of |tanh| >= SATURATION over the directional coefficients


def _flanking(pos: torch.Tensor, cell: torch.Tensor, numbers: Sequence[int]):
    z = torch.as_tensor([int(x) for x in numbers], device=pos.device)
    r = minimum_image_distances(pos, cell)
    pb = torch.nonzero(z == 82).reshape(-1)
    cl = torch.nonzero(z == 17).reshape(-1)
    dist = torch.sort(r[pb][:, cl], dim=1).values
    flank = pb[dist[:, 5] > 4.0]
    return pb, cl, flank, r


def frame_diagnostics(model, H: torch.Tensor, numbers: Sequence[int], species: torch.Tensor,
                      pos: torch.Tensor, cell: torch.Tensor, edge_index: torch.Tensor,
                      edge_vector: torch.Tensor) -> Optional[Arm1Frame]:
    """Arm-1 quantities of one neutral-vacancy frame from its `H0`."""
    pb, cl, flank, r = _flanking(pos, cell, numbers)
    if flank.numel() != 2:
        return None
    n_ref = neutral_count(numbers)
    n_up, n_dn = S_REF.counts(n_ref)
    with torch.no_grad():
        eps = torch.linalg.eigvalsh(H)
        homo, separation = float(eps[n_up - 1]), float(eps[n_up] - eps[n_up - 1])
        dq = two_fillings(H, State(1, -1, 0).counts(n_ref), (n_up, n_dn), model.sigma_s).dq
        n_eff = float(1.0 / (dq ** 2).sum())
        d = float(r[flank[0], flank[1]])
        # pp_sigma modulation on the flanking pair: the head's hopping MLP output g on that
        # edge (log form: exp(beta tanh g)); the pair may be beyond r_cut -> None edge.
        sk = model.h0.sk
        src, dst = edge_index[0], edge_index[1]
        on_pair = ((src == flank[0]) & (dst == flank[1])) | ((src == flank[1]) & (dst == flank[0]))
        pp_mod, pp_sat = float("nan"), False
        if bool(on_pair.any()):
            e = torch.nonzero(on_pair).reshape(-1)[0]
            # Recompute the pre-activation the way SlaterKosterH.integrals does.
            scalars = model._last_scalars                       # set by the caller
            feats_i, feats_j = scalars[src[e]].unsqueeze(0), scalars[dst[e]].unsqueeze(0)
            e_i, e_j = sk.elem(species[src[e]].unsqueeze(0)), sk.elem(species[dst[e]].unsqueeze(0))
            sym = torch.cat([feats_i + feats_j, (feats_i - feats_j).abs(), e_i + e_j, (e_i - e_j).abs()], dim=-1)
            g = sk.hop(sym)[0, 2]                               # pp_sigma slot of BOND_TYPES
            pp_mod = float(torch.exp(sk.hop_log_beta * torch.tanh(g)))
            pp_sat = bool(torch.tanh(g).abs() >= SATURATION)
        # Directional block magnitudes.
        a, b = model.h0.coefficients()
        Q = quadrupole_descriptor(edge_index, edge_vector, len(numbers), model.h0.q_cut)
        bq = (b[species].reshape(-1, 1, 1) * Q).reshape(len(numbers), -1).norm(dim=-1)
        flank_tensor = float(bq[flank].mean())
        others = pb[~torch.isin(pb, flank)]
        bulk_spread = float(bq[others].std()) if others.numel() > 1 else float("nan")
        # Cl splitting from the directional block: eigenvalues of b Q on bridging Cl
        # (axis along the bond): eps_sigma - eps_pi = b (Q_axis - Q_perp).
        cl_split = []
        for i in cl.tolist():
            w, v = torch.linalg.eigh(b[species[i]] * Q[i])
            cl_split.append(float(w[-1] - 0.5 * (w[0] + w[1])) if float(b[species[i]]) >= 0 else float(w[0] - 0.5 * (w[1] + w[2])))
        cl_splitting = float(np.mean(cl_split)) if cl_split else float("nan")
        coeffs = torch.cat([torch.tanh(model.h0.alpha), torch.tanh(model.h0.beta)])
        saturation = float((coeffs.abs() >= SATURATION).to(torch.float64).mean())
    return Arm1Frame(d=d, homo=homo, separation=separation, n_eff=n_eff, pp_modulation=pp_mod,
                     pp_saturated=pp_sat, flank_tensor=flank_tensor, bulk_tensor_spread=bulk_spread,
                     cl_splitting=cl_splitting, coefficient_saturation=saturation)


def run_diagnostics(model, batches: Sequence[Dict[str, torch.Tensor]]) -> List[Arm1Frame]:
    from mace.modules.models import ScaleShiftMACE
    records: List[Arm1Frame] = []
    for data in batches:
        with torch.no_grad():
            out = ScaleShiftMACE.forward(model.base, model._trunk_data(dict(data)), training=False, compute_force=False)
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
                ei = data["edge_index"][:, e_mask] - lo
                ev = edge_vector[e_mask]
                model._last_scalars = scalars[lo:hi]
                H = model.h0(scalars[lo:hi], vectors[lo:hi], species[lo:hi], ei, ev)
                numbers = [model.atomic_numbers[int(x)] for x in species[lo:hi].tolist()]
                rec = frame_diagnostics(model, H, numbers, species[lo:hi], positions[lo:hi], cell[g], ei, ev)
                if rec is not None:
                    records.append(rec)
    return records


def summarise(records: Sequence[Arm1Frame]) -> Dict[str, object]:
    """Per-seed summary: level-vs-bond slope, N_eff statistics, stop flag, tensor ratio,
    splitting sign, saturation."""
    d = np.array([r.d for r in records]); homo = np.array([r.homo for r in records])
    slope, se, _ = slope_with_se(d, homo)
    n_eff = np.array([r.n_eff for r in records])
    stop = float(np.mean([r.pp_saturated for r in records])) > 0.5
    ratio = np.nanmean([r.flank_tensor / r.bulk_tensor_spread if r.bulk_tensor_spread else np.nan for r in records])
    return {"n_frames": len(records), "level_vs_bond_slope": slope, "level_vs_bond_se": se,
            "n_eff_p50": float(np.median(n_eff)), "n_eff_mean": float(n_eff.mean()),
            "separation_p50": float(np.median([r.separation for r in records])),
            "pp_modulation_p50": float(np.nanmedian([r.pp_modulation for r in records])),
            "pp_stop": bool(stop), "flank_tensor_over_bulk_spread": float(ratio),
            "cl_splitting_mean": float(np.nanmean([r.cl_splitting for r in records])),
            "coefficient_saturation": float(np.mean([r.coefficient_saturation for r in records]))}


def decide(full: Sequence[Dict[str, object]], control: Sequence[Dict[str, object]],
           force_rmse_full: Sequence[float], force_rmse_control: Sequence[float],
           registered_splitting_sign: float = -1.0) -> Dict[str, object]:
    """The four criteria over seeds (registered thresholds); returns each verdict and the
    routing letter: A proceed; D conditional (edge residual); B identifiability; C repeat."""
    stops = sum(1 for s in full if s["pp_stop"])
    crit_i = stops <= 1
    ratio_ok = np.median([s["flank_tensor_over_bulk_spread"] for s in full]) > 3.0
    slope_full = np.median([abs(s["level_vs_bond_slope"]) for s in full])
    slope_ctrl = np.median([abs(s["level_vs_bond_slope"]) for s in control])
    control_loss = 1.0 - slope_ctrl / slope_full if slope_full > 0 else float("nan")
    crit_ii = bool(ratio_ok and control_loss >= 0.30)
    crit_iii = all(np.sign(s["cl_splitting_mean"]) == registered_splitting_sign for s in full)
    spread_full = np.std([s["n_eff_p50"] for s in full]); spread_ctrl = np.std([s["n_eff_p50"] for s in control])
    sat = max(s["coefficient_saturation"] for s in full)
    crit_iv = bool(spread_full <= 0.5 * spread_ctrl and np.median(force_rmse_full) <= np.median(force_rmse_control) and sat < 0.5)
    n_pass = sum([crit_i, crit_ii, crit_iii, crit_iv])
    route = "A" if n_pass == 4 else ("D" if n_pass == 3 else ("C" if crit_iii and crit_i else "B"))
    return {"i_stop_fraction": stops, "i": crit_i, "ii": crit_ii, "ii_ratio": float(np.median([s["flank_tensor_over_bulk_spread"] for s in full])),
            "ii_control_loss": float(control_loss), "iii": crit_iii, "iv": crit_iv,
            "iv_spread_full": float(spread_full), "iv_spread_control": float(spread_ctrl), "iv_saturation": float(sat),
            "route": route}
