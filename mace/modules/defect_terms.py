"""Plan v8 sections 2.5 and 2.6: the term registry and the two-kernel interface.

WHAT A TERM REGISTERS. Every energy term of the charge functional declares, in one place:
its name; where the forward reports its energy; whether it depends on the density `P` and,
if so, what its `dE/dP` is as an orbital-space potential (a term that depends on `P` and
contributes no potential to `H` is not a functional, it is a bolt-on -- E_resp was one, and
the plan retires that class); whether it depends on the atomic positions at fixed `P` and on
the trunk features at fixed `P`; and which electrostatic kernel it is evaluated under, if
any. The finite-difference harness of section 7.2 walks this registry, term by term and
assembled, and records a status per term; the acceptance for Stage 0 is that the status is
RECORDED, and for Stage 1 that it passes.

The registry describes the forward that exists. It does not yet restructure it: the terms
below are the v6 functional's terms, reported by `MACEDefect.forward` under the keys named
here, and the `potential` column records how each one reaches `H` today. Stages 4 and 5
replace entries (Madelung -> V_static, E_LR -> Phi_FF with V_FF) rather than add columns.

TWO KERNELS, ONE FUNCTIONAL. `Kernel.PBC` is the periodic Ewald sum the labels were computed
under and training runs under; `Kernel.ISOLATED` is the same smeared-charge evaluator with
no images and no background, for inference in the dilute limit. Every gauge-dependent term
is evaluated through `evaluate_kernel`, so the choice is one argument and never a second
code path. Which terms are gauge-dependent is a column of the registry, and the
section-2.6 list (V_static, Phi_FF, Phi_F,ind, Phi_S,ind, Phi_ind,ind) is what the column
will read once those terms exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import torch

__all__ = ["Kernel", "GAUGES", "TermSpec", "registry", "term_energies", "evaluate_kernel"]


class Kernel(str, Enum):
    PBC = "periodic"
    ISOLATED = "isolated"


#: The model's `gauge` config value: which kernel the gauge-dependent terms run under.
GAUGES = tuple(k.value for k in Kernel)


def evaluate_kernel(ewald, kernel: Kernel, charges: torch.Tensor, positions: torch.Tensor,
                    cell: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """The electrostatic energy of `charges` under `kernel`, one value per graph.

    The single entry point of section 2.6: `G_PBC` is `LatentEwald.energy` (periodic, with
    the jellium background), `G_inf` is `LatentEwald.isolated_energy` (no images, no
    background, no cell). `G_img = G_PBC - G_inf` is their difference and nothing else.
    """
    kernel = Kernel(kernel)
    if kernel is Kernel.PBC:
        return ewald.energy(charges, positions, cell, batch)
    return ewald.isolated_energy(charges, positions, batch, num_graphs)


@dataclass(frozen=True)
class TermSpec:
    """One term of the functional, as registered. Every field is a statement to be tested."""

    name: str
    #: The key of `MACEDefect.forward`'s output dict that carries this term's energy, `[B]`.
    output_key: str
    #: Does the energy depend on the density matrix `P`?
    depends_on_P: bool
    #: How `dE/dP` reaches `H` as an orbital-space potential. "band" for the band term
    #: itself (`dF/dH = P`); "in_H" for a potential added to the on-site energies before the
    #: solve; "none" for a P-independent term; "absent" for a P-dependent term that
    #: contributes no potential to H (the frontier term before Stage 5).
    potential: str
    #: How `dE/dP` reaches the FORCES (section 2.5) -- a different question from `potential`,
    #: which the registry used to conflate. "band" for the band term (Hellmann-Feynman plus
    #: the head's own response); "divided_difference" for a term whose P-dependence is
    #: built from matrix functions of H and differentiated by the Daleckii-Krein route
    #: (`Tr(P~ dH/dR)` by autograd through `channel_matrix`); "none" for a P-independent
    #: term; "absent" for a P-dependent term whose response is NOT in the forces -- the
    #: class the plan retires, listed so the harness reports it rather than it being
    #: invisible. No term of the Stage 1.2 functional is in that class.
    response: str
    #: `dE/dR` at fixed `P` is non-zero.
    depends_on_R: bool
    #: `dE/dh` at fixed `P` (through the trunk features) is non-zero.
    depends_on_features: bool
    #: Which kernel the term runs under; None for a term with no electrostatics.
    kernel: Optional[str]
    #: The plan section that owns the term's current form.
    section: str

    @property
    def gauge_dependent(self) -> bool:
        return self.kernel is not None


def registry(model) -> List[TermSpec]:
    """The terms live on THIS model configuration, in the order the energy sums them."""
    terms = [
        TermSpec(name="base", output_key="base_trunk_energy", depends_on_P=False,
                 potential="none", response="none", depends_on_R=True,
                 depends_on_features=True, kernel=None, section="E_base"),
    ]
    if getattr(model, "spectral", None) is not None:
        # The band term: F_band(H_Q) - F_band(H_0). Everything that enters H -- the SK
        # hoppings, the on-site levels, the Madelung shift, the c table -- is inside this
        # one term, and its potential IS the density matrix.
        terms.append(TermSpec(
            name="band", output_key="delta_sr_energy", depends_on_P=True, potential="band",
            response="band", depends_on_R=True, depends_on_features=True,
            kernel=(Kernel.PBC.value if getattr(model, "madelung", None) is not None
                    else None),
            section="2.2 (F_band; V_static^B inside H before Stage 4 as the Madelung term)"))
    if getattr(model, "frontier_active", False):
        # STAGE 1.2. Phi_FF = 1/2 B_img[w rho_loc, w rho_loc] on the channel-normalised
        # frontier density (section 2.8), as `Phi_FF(S) - Phi_FF(S_ref)`. P-dependent
        # through the channel projectors, the normalisations and w; its response reaches
        # the forces by the divided-difference route; its POTENTIAL is absent from H until
        # Stage 5 (V_FF = dPhi_FF/dP). Feature-dependent through H. Zero under the
        # isolated gauge.
        terms.append(TermSpec(
            name="frontier", output_key="frontier_energy", depends_on_P=True,
            potential="absent", response="divided_difference", depends_on_R=True,
            depends_on_features=True, kernel=getattr(model, "gauge", Kernel.PBC.value),
            section="2.8 (Phi_FF; V_FF = dPhi_FF/dP from Stage 5)"))
    return terms


def term_energies(model, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """`{term name: energy [B]}` read off one forward's output, plus `assembled`."""
    energies = {}
    for spec in registry(model):
        value = out.get(spec.output_key)
        if value is None:
            raise KeyError(f"the forward reports no {spec.output_key!r} for term "
                           f"{spec.name!r}; the registry and the forward disagree")
        energies[spec.name] = value
    energies["assembled"] = out["energy"]
    return energies
