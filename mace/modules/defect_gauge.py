###########################################################################################
# The frozen-pristine spectral gauge (transition plan v8.1 addendum, section 3.1)
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
"""One scalar that fixes the runtime one-electron energy zero.

The problem this removes is an exact flat direction. A common shift of the learned
one-electron Hamiltonian, ``H -> H + aI``, leaves every density, occupation, force, stress
and gap unchanged, but it moves the band contribution to a *charged* state by ``a`` per
electron of imbalance -- which the fitted energy constant then compensates. Two seeds could
therefore differ by several eV in their spectra and in their constants while being the same
model, and the constant is refitted rarely enough that the compensation goes stale.

The gauge is defined on the frozen pristine reference cell, not per frame::

    mu_g(theta) = sum_sigma Tr[P_V,sigma^pris(theta) Htilde_fix,sigma^pris(theta)]
                  / sum_sigma M_V,sigma^pristine

    H_fix,sigma(R; theta) = Htilde_fix,sigma(R; theta) - mu_g(theta) I

and the same ``mu_g`` is subtracted from every aligned spectral edge. Note what it is *not*:
it is a potential-zero convention, never fitted per frame, defect, charge, composition class
or cell size, and it has no runtime coordinate or strain dependence -- so it contributes
nothing to forces or stress. It is differentiated with respect to ``theta``, because the
pristine spectrum moves as the head trains.

Rank normalisation is essential and is the reason the trace is divided by
``sum_sigma M_V,sigma``: an unnormalised occupied-manifold trace is extensive, so it would
grow with the reference cell and make the gauge depend on the tiling used to define it.

**Representation.** The addendum requires one registered orthonormalised orbital
representation, and prohibits subtracting ``mu_g I`` in a nonorthogonal basis (the correct
operation there is ``Htilde - mu_g S``). This head's Slater-Koster basis is orthogonal by
construction -- overlap matrices are explicitly out of scope in plan v8 -- so ``mu_g I`` is
the exactly equivalent operation and no overlap is stored. :func:`assert_orthonormal`
records that assumption at the one place it could silently stop being true.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

# Representation tag written into the record; a future overlap-based basis must not reuse it.
REPRESENTATION = "orthonormal_sk_v1"

# The pristine occupied manifold must stay separated from the rest of the spectrum, or the
# trace above is not the trace of a well-defined subspace and the gauge record is void.
DEFAULT_GAP_FLOOR = 1e-3  # eV


class GaugeError(RuntimeError):
    """The gauge could not be established, so no charged energy may be reported."""


@dataclass(frozen=True)
class GaugeRecord:
    """Everything needed to reproduce and invalidate a gauge value.

    Enters the checkpoint and every energy cache key: two checkpoints with different
    ``mu_g`` are different models, and a cached charged energy computed under one is not
    valid under the other.
    """

    mu_g: float
    ranks: Tuple[int, ...]          # M_V,sigma on the pristine reference
    n_valence: int                  # sum_sigma M_V,sigma, the normalising rank
    gaps: Tuple[float, ...]         # separating gap at each rank (eV)
    gap_floor: float
    representation: str
    pristine_key: str               # which reference cell the gauge was read off
    basis_version: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ranks"] = list(self.ranks)
        d["gaps"] = list(self.gaps)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GaugeRecord":
        d = dict(d)
        d["ranks"] = tuple(int(x) for x in d["ranks"])
        d["gaps"] = tuple(float(x) for x in d["gaps"])
        return cls(**d)

    @property
    def fingerprint(self) -> str:
        """Content hash for cache keys. Rounded so float noise is not a cache miss."""
        payload = {
            "mu_g": round(float(self.mu_g), 12),
            "ranks": list(self.ranks),
            "representation": self.representation,
            "pristine_key": self.pristine_key,
            "basis_version": self.basis_version,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def assert_orthonormal(overlap: Optional[torch.Tensor] = None) -> None:
    """Refuse to gauge a nonorthogonal representation.

    ``mu_g I`` and ``mu_g S`` differ whenever ``S != I``, and the difference is a
    site-dependent shift rather than a constant -- it would change densities and forces, not
    only the energy zero. Rather than silently apply the wrong one, this raises.
    """
    if overlap is None:
        return
    identity = torch.eye(overlap.shape[-1], dtype=overlap.dtype, device=overlap.device)
    if not torch.allclose(overlap, identity, atol=1e-10, rtol=0.0):
        raise GaugeError(
            "the spectral gauge requires an orthonormal representation: subtracting "
            "mu_g * I in a nonorthogonal basis is not a constant shift. Use "
            "Htilde - mu_g * S, or orthonormalise first (addendum section 3.1)")


def valence_trace(eigenvalues: torch.Tensor, ranks: Sequence[int]) -> torch.Tensor:
    """``sum_sigma Tr[P_V,sigma H]`` for a spin-restricted spectrum.

    The head shares one Hamiltonian between spins and differs only in the fill, so the trace
    of the occupied projector is the sum of the lowest ``M_sigma`` eigenvalues in each spin
    channel. Written as a subspace sum rather than a sum over individually selected
    eigenvectors: it stays differentiable through degeneracies *inside* the occupied
    manifold, which is exactly the case a per-eigenvector formulation would break on. Only
    the gap at the manifold boundary matters, and :func:`gauge_scalar` guards that.
    """
    eigenvalues = torch.sort(eigenvalues).values
    total = eigenvalues.new_zeros(())
    for rank in ranks:
        rank = int(rank)
        if rank < 1 or rank > eigenvalues.numel():
            raise GaugeError(
                f"valence rank {rank} outside 1..{eigenvalues.numel()} on the pristine "
                "reference spectrum")
        total = total + eigenvalues[:rank].sum()
    return total


def gauge_scalar(eigenvalues: torch.Tensor, ranks: Sequence[int], *,
                 pristine_key: str, basis_version: str = REPRESENTATION,
                 gap_floor: float = DEFAULT_GAP_FLOOR,
                 overlap: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, GaugeRecord]:
    """``(mu_g, record)`` from the frozen pristine reference spectrum.

    ``eigenvalues`` is the raw (ungauged) pristine spectrum and ``ranks`` the per-spin
    occupied valence ranks ``M_V,sigma^pristine``, which are exact by electron count on a
    pristine cell. The returned tensor keeps its graph, so ``mu_g`` is differentiated with
    respect to the head parameters; the record carries the detached value for serialisation.
    """
    assert_orthonormal(overlap)
    ranks = tuple(int(r) for r in ranks)
    if not ranks:
        raise GaugeError("the gauge needs at least one spin channel's valence rank")

    ordered = torch.sort(eigenvalues).values
    gaps = []
    for rank in ranks:
        if rank >= ordered.numel():
            raise GaugeError(
                f"valence rank {rank} leaves no state above the occupied manifold; the "
                "pristine reference cannot define a separating gap")
        gap = float(ordered[rank].detach() - ordered[rank - 1].detach())
        if gap < gap_floor:
            # Without the gap the occupied manifold is not a well-defined subspace, so the
            # trace is not reproducible and every charged energy derived under it is void.
            raise GaugeError(
                f"the pristine occupied manifold lost its separating gap at rank {rank}: "
                f"{gap:.3e} eV < floor {gap_floor:.3e} eV. The gauge record and any "
                "checkpoint carrying it are invalid (addendum section 3.1)")
        gaps.append(gap)

    n_valence = int(sum(ranks))
    mu_g = valence_trace(eigenvalues, ranks) / float(n_valence)
    record = GaugeRecord(
        mu_g=float(mu_g.detach()), ranks=ranks, n_valence=n_valence,
        gaps=tuple(gaps), gap_floor=float(gap_floor), representation=REPRESENTATION,
        pristine_key=str(pristine_key), basis_version=str(basis_version))
    return mu_g, record


def apply_gauge(hamiltonian: torch.Tensor, mu_g) -> torch.Tensor:
    """``Htilde - mu_g I``, for every runtime geometry, size, state and boundary.

    Applied to the raw Hamiltonian before any eigensolve, so occupations, densities and the
    band free energy all see the gauged spectrum.
    """
    if not isinstance(mu_g, torch.Tensor):
        mu_g = torch.as_tensor(mu_g, dtype=hamiltonian.dtype, device=hamiltonian.device)
    mu_g = mu_g.to(dtype=hamiltonian.dtype, device=hamiltonian.device)
    n = hamiltonian.shape[-1]
    identity = torch.eye(n, dtype=hamiltonian.dtype, device=hamiltonian.device)
    return hamiltonian - mu_g * identity


def shift_edges(edges, mu_g) -> Any:
    """Subtract the same ``mu_g`` from aligned spectral edges.

    The edges are read off the same raw spectra, so leaving them ungauged while gauging the
    Hamiltonian would move every window relative to the spectrum by ``mu_g`` -- which is the
    one thing the gauge must not do.
    """
    value = float(mu_g.detach()) if isinstance(mu_g, torch.Tensor) else float(mu_g)
    if isinstance(edges, torch.Tensor):
        return edges - value
    if isinstance(edges, dict):
        return {k: (v - value if isinstance(v, (int, float)) else v) for k, v in edges.items()}
    if isinstance(edges, (list, tuple)):
        return type(edges)(e - value for e in edges)
    return edges - value
