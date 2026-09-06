###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import math
from typing import Optional, Sequence

import torch

from mace.tools.scatter import scatter_sum
import torch.distributed as dist

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch

from .defect_blocks import NUM_CARRIER_CHANNELS
from .defect_size import size_extensivity_probe


# ------------------------------------------------------------------------------
# Helper function for loss reduction that handles DDP correction
# ------------------------------------------------------------------------------
def is_ddp_enabled():
    return dist.is_initialized() and dist.get_world_size() > 1


def reduce_loss(raw_loss: torch.Tensor, ddp: Optional[bool] = None) -> torch.Tensor:
    """
    Reduces an element-wise loss tensor.

    If ddp is True and distributed is initialized, the function computes:

        loss = (local_sum * world_size) / global_num_elements

    Otherwise, it returns the regular mean.
    """
    ddp = is_ddp_enabled() if ddp is None else ddp
    if ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        n_local = raw_loss.numel()
        loss_sum = raw_loss.sum()
        total_samples = torch.tensor(
            n_local, device=raw_loss.device, dtype=raw_loss.dtype
        )
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        return loss_sum * world_size / total_samples
    return raw_loss.mean()


# ------------------------------------------------------------------------------
# Energy Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    raw_loss = torch.square(ref["energy"] - pred["energy"])
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Calculate per-graph number of atoms.
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # shape: [n_graphs]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_absolute_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.abs((ref["energy"] - pred["energy"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Stress and Virials Loss Functions
# ------------------------------------------------------------------------------


def weighted_mean_squared_stress(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
    raw_loss = (
        configs_weight
        * configs_stress_weight
        * torch.square(ref["stress"] - pred["stress"])
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_virials(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_virials_weight = ref.virials_weight.view(-1, 1, 1)
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    raw_loss = (
        configs_weight
        * configs_virials_weight
        * torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Forces Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    raw_loss = (
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )
    return reduce_loss(raw_loss, ddp)


def mean_normed_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    raw_loss = torch.linalg.vector_norm(ref["forces"] - pred["forces"], ord=2, dim=-1)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Dipole Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_dipole(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    raw_loss = torch.square((ref["dipole"] - pred["dipole"]) / num_atoms)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Polarizability Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_polarizability(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[
        bool
    ] = None,  # ,mean: Optional[torch.Tensor] = None , std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # polarizability: [n_graphs, ]
    # ref_polar = ref["polarizability"].view(-1, 3, 3) * std.view(1, 3, 3) + mean.view(1, 3, 3) if mean is not None and std is not None else ref["polarizability"]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  # [n_graphs,1]
    raw_loss = torch.square(
        (ref["polarizability"].view(-1, 3, 3) - pred["polarizability"]) / num_atoms
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Conditional Losses for Forces
# ------------------------------------------------------------------------------


def conditional_mse_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    # Define multiplication factors for different regimes.
    factors = torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref["forces"].device, dtype=ref["forces"].dtype
    )
    err = ref["forces"] - pred["forces"]
    se = torch.zeros_like(err)
    norm_forces = torch.norm(ref["forces"], dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    se[c1] = torch.square(err[c1]) * factors[0]
    se[c2] = torch.square(err[c2]) * factors[1]
    se[c3] = torch.square(err[c3]) * factors[2]
    se[~(c1 | c2 | c3)] = torch.square(err[~(c1 | c2 | c3)]) * factors[3]
    raw_loss = configs_weight * configs_forces_weight * se
    return reduce_loss(raw_loss, ddp)


def conditional_huber_forces(
    ref_forces: torch.Tensor,
    pred_forces: torch.Tensor,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    factors = huber_delta * torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref_forces.device, dtype=ref_forces.dtype
    )
    norm_forces = torch.norm(ref_forces, dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    c4 = ~(c1 | c2 | c3)
    se = torch.zeros_like(pred_forces)
    se[c1] = torch.nn.functional.huber_loss(
        ref_forces[c1], pred_forces[c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        ref_forces[c2], pred_forces[c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        ref_forces[c3], pred_forces[c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        ref_forces[c4], pred_forces[c4], reduction="none", delta=factors[3]
    )
    return reduce_loss(se, ddp)


# ------------------------------------------------------------------------------
# Loss Modules Combining Multiple Quantities
# ------------------------------------------------------------------------------


class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


class WeightedForcesLoss(torch.nn.Module):
    def __init__(self, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.forces_weight * loss_forces

    def __repr__(self):
        return f"{self.__class__.__name__}(forces_weight={self.forces_weight:.3f})"


class WeightedEnergyForcesStressLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_stress = weighted_mean_squared_stress(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedHuberEnergyForcesStressLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        # We store the huber_delta rather than a loss with fixed reduction.
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="none", delta=self.huber_delta
            )
            loss_forces = reduce_loss(loss_forces, ddp)
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="none", delta=self.huber_delta
            )
            loss_stress = reduce_loss(loss_stress, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="mean", delta=self.huber_delta
            )
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="mean", delta=self.huber_delta
            )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class UniversalLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        stress_weight=1.0,
        magforces_weight=1.0,
        huber_delta=0.01,
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "magforces_weight",
            torch.tensor(magforces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
        configs_energy_weight = ref.energy_weight
        configs_forces_weight = torch.repeat_interleave(
            ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
        ).unsqueeze(-1)
        configs_magforces_weight = torch.repeat_interleave(
            ref.magforces_weight, ref.ptr[1:] - ref.ptr[:-1]
        ).unsqueeze(-1)
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="none",
                delta=self.huber_delta,
            )
            loss_stress = reduce_loss(loss_stress, ddp)
            loss_magforces = 0
            if "magforces" in pred.keys() and (
                pred["magforces"] is not None and ref["magforces"] is not None
            ):
                loss_magforces = torch.nn.functional.huber_loss(
                    configs_magforces_weight * ref["magforces"],
                    configs_magforces_weight * pred["magforces"],
                    reduction="none",
                    delta=self.huber_delta,
                )
                loss_magforces = reduce_loss(loss_magforces, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_magforces = 0
            if "magforces" in pred.keys() and (
                pred["magforces"] is not None and ref["magforces"] is not None
            ):
                loss_magforces = torch.nn.functional.huber_loss(
                    configs_magforces_weight * ref["magforces"],
                    configs_magforces_weight * pred["magforces"],
                    reduction="mean",
                    delta=self.huber_delta,
                )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
            + self.magforces_weight * loss_magforces
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f}, magforces_weight={self.magforces_weight:.3f})"
        )


class WeightedEnergyForcesVirialsLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, virials_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_virials = weighted_mean_squared_virials(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.virials_weight * loss_virials
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, virials_weight={self.virials_weight:.3f})"
        )


class DipoleSingleLoss(torch.nn.Module):
    def __init__(self, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss = (
            weighted_mean_squared_error_dipole(ref, pred, ddp) * 100.0
        )  # scale adjustment
        return self.dipole_weight * loss

    def __repr__(self):
        return f"{self.__class__.__name__}(dipole_weight={self.dipole_weight:.3f})"


class DipolePolarLoss(torch.nn.Module):
    def __init__(
        self, dipole_weight=1.0, polarizability_weight=1.0
    ) -> (
        None
    ):  # dipole_mean=None,dipole_std=None,polarizability_mean=None,polarizability_std=None
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_dipole = weighted_mean_squared_error_dipole(
            ref, pred, ddp
        )  # ,self.dipole_mean,self.dipole_std) #* 100.0  # scale adjustment

        loss_polarizability = weighted_mean_squared_error_polarizability(
            ref, pred, ddp
        )  # ,self.polarizability_mean,self.polarizability_std) #* 100.0  # scale adjustment
        return (
            self.dipole_weight * loss_dipole
            + self.polarizability_weight * loss_polarizability
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dipole_weight={self.dipole_weight:.3f}, polarizability_weight={self.polarizability_weight:.3f})"
        )


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_dipole = weighted_mean_squared_error_dipole(ref, pred, ddp) * 100.0
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.dipole_weight * loss_dipole
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipole_weight:.3f})"
        )


class WeightedEnergyForcesL1L2Loss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_absolute_error_energy(ref, pred, ddp)
        loss_forces = mean_normed_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


# ------------------------------------------------------------------------------
# Charge-aware defect loss (see charge_aware_defect_mlip_implementation.md)
# ------------------------------------------------------------------------------


def _per_atom_counts(ref: Batch) -> torch.Tensor:
    return ref.ptr[1:] - ref.ptr[:-1]


def _spread_to_atoms(values: torch.Tensor, ref: Batch) -> torch.Tensor:
    return torch.repeat_interleave(values, _per_atom_counts(ref)).unsqueeze(-1)


def weighted_mean_squared_error_field(
    ref: Batch,
    pred: TensorDict,
    field: str,
    weight_field: str,
    per_atom: bool = True,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    """Per-configuration squared error on a scalar field with its own weight column."""
    scale = _per_atom_counts(ref) if per_atom else torch.ones_like(ref.weight)
    raw_loss = (
        ref.weight
        * ref[weight_field]
        * torch.square((ref[field] - pred[field]) / scale)
    )
    return reduce_loss(raw_loss, ddp)


def mean_squared_error_forces_field(
    ref: Batch,
    pred: TensorDict,
    field: str,
    weight_field: str,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    raw_loss = (
        _spread_to_atoms(ref.weight, ref)
        * _spread_to_atoms(ref[weight_field], ref)
        * torch.square(ref[field] - pred[field])
    )
    return reduce_loss(raw_loss, ddp)


class DefectLoss(torch.nn.Module):
    """Loss for carrier-conditioned defect models.

    Three terms on three subsets, selected by the data rather than by config type
    (plan section 4):

    (a) ``L_base`` on configurations that supervise the base branch -- one per paired
        geometry -- against the reference-state labels;
    (b) ``L_delta`` on paired configurations. This term carries the headline
        observables, since fixed-geometry charge-state differences are algebraically
        free of base-model error, so it is weighted highest;
    (c) ``L_tot`` on charged configurations with no reference-state partner, with the
        base branch **detached**: without that, unpaired charged data reshapes the base
        potential and the residual decomposition stops meaning what it is supposed to.
        It is down-weighted, since base-model error at those geometries is absorbed into
        the correction.

    The delta terms are *not* normalised per atom: a charge-state difference is
    intensive, of order the carrier binding energy, and dividing it by the atom count
    would make it vanish from the objective exactly in the large cells that matter.

    Regularisation. The L2 on ``u`` was expected to be load-bearing -- the mechanism that
    makes the optimiser buy localisation, and hence the logit gap that keeps the
    correction extensive (plan section 3.2). It is **off by default**, because that
    expectation did not survive measurement: the 4H-SiC divacancy reaches a logit gap of
    about 11.7, comfortably meeting the section 3.2 requirement, with no regulariser at
    all. The data prefers localisation here without being paid to.

    Two further reasons not to switch it back on casually. It is a *mean* over sites, so
    it is intensive and cannot penalise a bulk-wide level of ``u`` in the first place.
    And it reaches carrier channels that no frame occupies, which the counter prefactor
    would otherwise leave untouched -- breaking the dead-channel invariant that makes
    live-channel behaviour interpretable (see ``test_dead_channels_are_never_updated``).
    """

    def __init__(
        self,
        energy_weight: float = 1.0,
        forces_weight: float = 100.0,
        delta_energy_weight: float = 10.0,
        delta_forces_weight: float = 100.0,
        total_energy_weight: float = 0.1,
        stress_weight: float = 0.0,
        pressure_weight: float = 0.0,
        u_l2: float = 0.0,
        zn_l2: float = 0.0,
        p_l2: float = 1e-4,
        qhost_l2: float = 1e-4,
        detach_base_in_totals: bool = False,
        eps_inf: Optional[float] = None,
        eps_inf_prior_weight: float = 0.0,
        size_weight: float = 0.0,
        size_ratio: float = 1e4,
        size_tol: float = 1e-3,
        size_warmup_epochs: int = 20,
        size_ema_decay: float = 0.95,
        size_contrast_eps: float = 1e-9,
        size_delocalised_fraction: float = 0.05,
        size_delocalised_min_atoms: float = 8.0,
        size_delocalised_leak: float = 0.05,
        gauge_weight: float = 0.0,
        eps_gauge_weight: float = 0.0,
        gap_weight: float = 0.0,
        e_gap: float = 0.0,
        energy_shape_weight: float = 0.0,
        energy_pair_slots: int = 0,
        energy_scale: float = 1.0,
        gap_composition: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("energy_weight", energy_weight),
            ("forces_weight", forces_weight),
            ("delta_energy_weight", delta_energy_weight),
            ("delta_forces_weight", delta_forces_weight),
            ("total_energy_weight", total_energy_weight),
            ("stress_weight", stress_weight),
            ("pressure_weight", pressure_weight),
        ):
            self.register_buffer(
                name, torch.tensor(value, dtype=torch.get_default_dtype())
            )
        self.u_l2 = u_l2
        self.zn_l2 = zn_l2
        self.p_l2 = p_l2
        self.qhost_l2 = qhost_l2
        self.detach_base_in_totals = detach_base_in_totals
        self.eps_inf = eps_inf
        self.eps_inf_prior_weight = eps_inf_prior_weight
        self.size_weight = size_weight
        self.size_ratio = size_ratio
        self.size_tol = size_tol
        self.size_warmup_epochs = size_warmup_epochs
        self.size_ema_decay = size_ema_decay
        self.size_contrast_eps = size_contrast_eps
        # A channel is refused the delocalised-carrier exemption when its participation
        # exceeds BOTH an absolute floor and a fraction of the cell. Both are needed: the
        # fraction alone misfires on small cells, where a genuinely localised carrier
        # occupies a non-trivial share (6 shell atoms of 66 is 0.10), while the floor alone
        # would stop discriminating once cells get large. Calibrated against measured runs:
        # the healthy short-range model sits at participation 2.0 and the collapsed
        # long-range one at 16.1 in a 79-atom cell, where a whole sublattice is 16.
        self.size_delocalised_fraction = size_delocalised_fraction
        self.size_delocalised_min_atoms = size_delocalised_min_atoms
        self.size_delocalised_leak = size_delocalised_leak
        self.gauge_weight = gauge_weight
        self.eps_gauge_weight = eps_gauge_weight
        self.gap_weight = float(gap_weight)
        self.e_gap = float(e_gap)
        # PLAN v8.1 SECTION 8: the charged-energy objective of Stages 1-4. Every batch built
        # by `WithinStratumPairSampler` ends with `energy_pair_slots` registered pairs; their
        # total-cell-eV residuals (one path each, by provenance) enter as the mean over
        # pairs of (xi_i - xi_j)^2 / 2, whose expectation is the within-stratum shape loss.
        # When this term is on, the per-atom totals term and the paired delta-energy term
        # are refused for charged energies: a charged label enters exactly one path.
        self.energy_shape_weight = float(energy_shape_weight)
        self.energy_pair_slots = int(energy_pair_slots)
        self.energy_scale = float(energy_scale)
        if self.energy_shape_weight > 0:
            if self.total_energy_weight > 0 or self.delta_energy_weight > 0:
                raise ValueError(
                    "the v8.1 energy-shape objective replaces the per-atom totals term and "
                    "the delta-energy term for charged energies (addendum section 8: one "
                    "registered path per observation); set total_energy_weight and "
                    "delta_energy_weight to zero")
            if self.energy_pair_slots < 1:
                raise ValueError("energy_shape_weight > 0 needs energy_pair_slots >= 1")
        self.last_energy_shape_value = 0.0
        # Stoichiometry in the model's own species order, e.g. (3, 1, 1) for CsPbCl3 with
        # the AtomicNumberTable sorted (Cl, Cs, Pb). Only the RATIO is used, so the same
        # numbers describe every supercell of the host.
        self.gap_composition = (
            None if gap_composition is None
            else tuple(float(v) for v in gap_composition))
        # Diagnostic: how many steps actually carried the term. A gap penalty that never
        # fires because no batch happened to contain a stoichiometric cell would otherwise
        # be indistinguishable from one that is working.
        self.gap_steps_with_term = 0
        self.gap_steps_total = 0
        # Running |c| per channel, used only to place the (detached) threshold. A buffer so
        # it survives checkpointing: restarting with a cold EMA would put every channel in
        # the exempt branch for the first few batches and briefly switch the term off.
        self.register_buffer(
            "contrast_ema",
            torch.zeros(NUM_CARRIER_CHANNELS, dtype=torch.get_default_dtype()),
        )
        self.last_size_exempt = 0.0
        self.last_size_spread: Optional[torch.Tensor] = None
        self.last_size_x: Optional[torch.Tensor] = None
        self.last_size_threshold: Optional[torch.Tensor] = None
        # Set by the trainer's epoch hook from the *absolute* epoch, so a run restarted
        # past the warmup resumes with the term already active. Getting this from a
        # counter local to the loss would silently re-serve the warmup on every restart --
        # the same trap that made the gamma anneal restart from scratch.
        self.current_epoch = 0

    def _totals_mask(self, ref: Batch) -> torch.Tensor:
        """Every configuration carrying carriers, i.e. all ``n != 0`` frames.

        This used to select *unpaired* charged frames only, because the base branch was
        detached here and paired frames already had delta supervision. Under corrected
        labels that left total forces at every defect geometry supervised by nothing at
        all, which is the gap this term now closes (plan A5.3).
        """
        return (ref["carrier_counts"].sum(dim=-1) > 0).to(ref.weight.dtype)

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        self.last_size_value = 0.0
        self.last_delta_energy_value = 0.0
        loss = self.energy_weight * weighted_mean_squared_error_field(
            ref, pred, "base_energy", "base_energy_weight", per_atom=True, ddp=ddp
        )
        loss = loss + self.forces_weight * mean_squared_error_forces_field(
            ref, pred, "base_forces", "base_forces_weight", ddp=ddp
        )
        delta_energy_term = self.delta_energy_weight * weighted_mean_squared_error_field(
            ref, pred, "delta_energy", "delta_energy_weight", per_atom=False, ddp=ddp
        )
        self.last_delta_energy_value = float(delta_energy_term.detach())
        loss = loss + delta_energy_term
        loss = loss + self.delta_forces_weight * mean_squared_error_forces_field(
            ref, pred, "delta_forces", "delta_forces_weight", ddp=ddp
        )

        # (c) Totals at n != 0, with the base branch receiving gradient.
        #
        # E_base at a defect geometry is a gauge, not an observable: shifting it by f(R)
        # and letting the correction absorb -f leaves every label unchanged. Detaching it
        # was the original protection against that. It is not needed, and it cost the
        # base branch all supervision at defect geometries: the correction is
        # sum_i alpha_i u_i with sum_i alpha_i = 1, so it is *intensive*, while E_base is
        # a sum of local energies and so *extensive*. A bulk-wide base error therefore
        # cannot be absorbed by the correction at any localisation of alpha -- the
        # normalisation does that, not the localisation. The residual gauge freedom is
        # confined to the defect-localised intensive part (plan A5.2).
        #
        # Note this must use the total energy at *this* frame's counter, not
        # `delta_energy`, which is now a paired difference against n_ref and only equals
        # the correction when n_ref = 0.
        mask = self._totals_mask(ref)
        if bool(mask.any()):
            num_atoms = _per_atom_counts(ref)
            total = pred["energy"]
            if self.detach_base_in_totals:
                total = pred["base_energy"].detach() + pred["correction_energy"]
            raw_energy = (
                ref.weight
                * ref.energy_weight
                * mask
                * torch.square((ref["energy"] - total) / num_atoms)
            )
            loss = loss + self.total_energy_weight * reduce_loss(raw_energy, ddp)
            # Forces carry no referencing constant, so the raw total force is a valid
            # label here even though the energy scale differs.
            raw_forces = (
                _spread_to_atoms(ref.weight, ref)
                * _spread_to_atoms(ref.forces_weight * mask, ref)
                * torch.square(ref["forces"] - pred["forces"])
            )
            loss = loss + self.forces_weight * reduce_loss(raw_forces, ddp)

        if self.stress_weight > 0:
            loss = loss + self.stress_weight * weighted_mean_squared_stress(
                ref, pred, ddp
            )
        if self.pressure_weight > 0 and pred.get("stress") is not None:
            # Hydrostatic component only: its purpose is identification of the screening
            # amplitude, and it is quadratic in q, which separates it from the
            # deformation-potential response. Shear stays masked in charged cells.
            pressure_pred = -pred["stress"].diagonal(dim1=-2, dim2=-1).mean(dim=-1)
            pressure_ref = -ref["stress"].diagonal(dim1=-2, dim2=-1).mean(dim=-1)
            raw_pressure = (
                ref.weight * ref.stress_weight * torch.square(pressure_ref - pressure_pred)
            )
            loss = loss + self.pressure_weight * reduce_loss(raw_pressure, ddp)

        loss = loss + self.energy_shape(ref, pred)
        loss = loss + self.size_penalty(ref, pred, ddp)
        loss = loss + self.gap_penalty(ref, pred, ddp)
        loss = loss + self.gauge_penalty(ref, pred, ddp)
        loss = loss + self.eps_gauge(pred, ref, ddp)
        loss = loss + self.regularisation(pred)
        return loss

    def energy_shape(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        """The registered pair term (plan v8.1 section 8) on the batch's pair slots.

        A batch built by `defect_objective.pair_loader` carries `pair_slots`; its last
        `2 * energy_pair_slots` graphs are the sampler's pairs, in order, each pair's members
        of one `stratum_id` (asserted) and scored on the residual path of their
        `provenance`. Those graphs carry `weight` zero, so they enter no other term; the
        base graphs contribute nothing here. A batch without the stamp (validation, a plain
        loader) scores zero rather than being mis-read by position.
        """
        zero = torch.zeros((), dtype=ref.weight.dtype, device=ref.weight.device)
        self.last_energy_shape_value = 0.0
        if self.energy_shape_weight <= 0.0:
            return zero
        from mace.modules.defect_objective import residual_paths, sampled_pair_term

        slots = getattr(ref, "pair_slots", None)
        if slots is None:
            # Not a pair batch (validation, or any plain loader): the term is not scored.
            # The held-out shape diagnostic is the recalibration tool's, on whole strata.
            return zero
        if int(slots) != self.energy_pair_slots:
            raise ValueError(f"the batch carries {int(slots)} pair slots, the loss expects "
                             f"{self.energy_pair_slots}")
        n_pair = 2 * self.energy_pair_slots
        n_graphs = int(ref.num_graphs)
        if n_graphs <= n_pair:
            raise ValueError(f"the batch holds {n_graphs} graphs, no more than its "
                             f"{n_pair} pair slots: it was not built by the pair sampler")
        sid = ref["stratum_id"].reshape(-1)[-n_pair:]
        if not bool((sid[0::2] == sid[1::2]).all()) or bool((sid < 0).any()):
            raise ValueError("pair slots hold graphs of different strata (or a neutral "
                             "frame); the batch was not built by WithinStratumPairSampler")
        xi, path = residual_paths(pred, ref, self.energy_scale)
        xi_pairs = xi[-n_pair:].reshape(-1, 2)
        if bool((path[-n_pair:] < 0).any()):
            raise ValueError("a pair slot holds a frame with no charged-energy residual")
        term = self.energy_shape_weight * sampled_pair_term(xi_pairs)
        self.last_energy_shape_value = float(term.detach())
        return term

    def size_threshold(
        self,
        counts: torch.Tensor,
        delocalised: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``x*_c = ln(t/(1-t))`` with ``t = (tol/n_c) / |c|_EMA``; ``+inf`` when ``t >= 1``.

        The constraint being expressed is ``sigma(x)|c| <= tol_c``. Since ``sigma`` is
        monotone that is exactly ``x <= x*``, so this threshold defines the *same* feasible
        set as the superseded energy-space hinge -- only the coordinate differs.

        ``t >= 1`` means the contrast is already inside tolerance, so no ``x`` can violate
        the constraint and the channel is exempt for any amount of dilution. That is the
        delocalised-carrier exemption, and it arrives continuously as ``|c| -> tol`` rather
        than as a branch: ``x* -> +inf`` smoothly from below.

        ``n_c`` enters through the tolerance rather than as an outer weight. The old
        ``n_c^2`` factor was justified by the penalised quantity being an energy; ``x`` is a
        dimensionless log-ratio, so that argument does not carry over. A per-channel drift
        budget does: a channel carrying two carriers may drift half as far.
        """
        counts = counts.to(self.contrast_ema.dtype)
        per_channel_tol = self.size_tol / counts.clamp_min(1.0)
        magnitude = self.contrast_ema.clamp_min(self.size_contrast_eps)
        t = (per_channel_tol / magnitude).clamp_max(1.0)
        satisfied = t >= 1.0
        if delocalised is not None:
            satisfied = satisfied & ~delocalised
        # Floored as well as capped. Only ``t >= 1`` -- contrast already inside tolerance --
        # means exempt, and that is the branch the caller skips. ``t -> 0`` is the opposite
        # extreme, a maximally strict constraint, and it must stay *finite*: an unfloored
        # ``log(0) = -inf`` is non-finite too, so a caller testing ``isfinite`` would read
        # the strictest possible setting as "no constraint" and switch the term off exactly
        # when it was asked for most. Observed with ``--defect_size_tol 0``.
        safe = t.clamp(min=1e-12, max=1.0 - 1e-12)
        threshold = torch.log(safe / (1.0 - safe))
        if delocalised is not None:
            # GUARD. The exemption is self-reinforcing and traps the fit: flat attention
            # drives the contrast to zero, |c| falls below tol, the channel is exempted,
            # the hinge switches off, and nothing pulls the attention back -- the
            # constraint degenerates exactly at the state it exists to prevent. Observed
            # end to end: the perovskite long-range run spent its whole life at
            # size_f = 1.000 on all four channels with |c| = 0.000 and finished with its
            # live channel uniform over the Cs sublattice and zero weight on the vacancy
            # shell, while the short-range run kept |c| = 0.090, stayed live, and put the
            # hole on exactly the two under-coordinated Pb.
            #
            # Refusing the exemption is NOT enough on its own, and the near-miss is worth
            # recording. With `satisfied` merely flipped, the threshold still comes from
            # |c| -- and |c| ~ 0 sends t to its clamp, giving x* = ln((1-1e-12)/1e-12)
            # = 27.6 while x is structurally capped at ln(R-1) = 9.21. The violation is
            # then max(0, 9.21 - 27.6) = 0: a guard that changes a label and no gradient.
            # In float32, which is what training uses, 1 - 1e-12 rounds to 1.0 and the
            # threshold becomes +inf, so the channel is silently exempt again.
            #
            # So for a collapsed channel the threshold must not be derived from |c| at
            # all -- its normalisation is the thing that degenerated. Use a fixed
            # localisation target instead: cap the leaked fraction at
            # `size_delocalised_leak`, i.e. x* = ln(f*/(1-f*)). Same units as `size_f`,
            # so the target reads directly off the diagnostic.
            leak = self.size_delocalised_leak
            target = math.log(leak / (1.0 - leak))
            resolved = torch.where(
                satisfied, torch.full_like(threshold, float("inf")), threshold
            )
            # `minimum`, not a replacement. A smaller x* is a stricter constraint, and
            # while |c| is healthy the ordinary path can already be stricter than the
            # fixed target -- at |c| = 0.033 it gives -3.46 against the target's -2.94, so
            # substituting would LOOSEN a channel that was being held. Taking the minimum
            # makes the guard a floor: it binds only when the |c| path has gone slack, and
            # `minimum(+inf, target) = target` covers the exempted case for free.
            return torch.where(
                delocalised.expand_as(resolved),
                torch.minimum(resolved, torch.full_like(resolved, target)),
                resolved,
            )
        return torch.where(
            satisfied, torch.full_like(threshold, float("inf")), threshold
        )

    def size_penalty(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        """Hinge in log-ratio space (size plan section 4, as amended).

        ``L_size = sum_{c: n_c > 0} max(0, x_c - x*_c)^2``

        Same zero-set as penalising the energy drift ``sigma(x)|c|`` directly, but the
        gradient is ``-2 max(0, x - x*)`` in ``ln A`` -- linear in the violation and never
        vanishing. The energy-space form's gradient carried a factor ``f(1-f)``, which is
        ~0 exactly where the term is most needed, and its only reachable optimum at
        ``f -> 1`` was to flatten the defect contrast. Here ``c`` is detached throughout, so
        that route does not exist at all.

        Badly-violated behaviour is gentler too: the penalty grows like ``(ln|c|)^2``
        rather than ``|c|^2``, which is better conditioned.

        Still silent early in training: before ``u`` develops contrast ``|c| < tol``, so
        every channel is exempt. The warmup remains a backstop, and matters for a second
        reason -- on a uniform state the gradient is identically zero, so the term can only
        deepen a gap that already exists, never create one.
        """
        zero = torch.zeros((), dtype=ref.weight.dtype, device=ref.weight.device)
        logits = pred.get("carrier_logits")
        if logits is None or pred.get("carrier_readouts") is None:
            return zero
        num_graphs = int(ref.num_graphs)
        probe = size_extensivity_probe(
            logits=logits,
            readouts=pred["carrier_readouts"],
            alpha=pred["carrier_alpha"],
            batch=ref["batch"],
            node_attrs=ref["node_attrs"],
            num_graphs=num_graphs,
            ratio=self.size_ratio,
        )
        counts = ref["carrier_counts"].view(num_graphs, -1)
        live = counts > 0

        # The threshold is detached by construction, so smoothing it across batches costs
        # nothing and stops it chasing per-batch noise in |c|.
        if torch.is_grad_enabled() and bool(live.any()):
            observed = (probe.contrast.abs() * live).sum(dim=0) / live.sum(dim=0).clamp_min(1)
            self.contrast_ema.mul_(self.size_ema_decay).add_(
                observed.to(self.contrast_ema.dtype) * (1.0 - self.size_ema_decay)
            )

        # Participation ratio 1 / sum_i alpha_i^2 per channel. A localised carrier gives a
        # handful of atoms at any N; a whole sublattice pins to a fixed fraction of the cell
        # (1/5 for Cs in CsPbCl3) no matter how large the supercell, which is the signature
        # being caught.
        alpha = pred["carrier_alpha"]
        inverse = scatter_sum(
            alpha.pow(2), ref["batch"], dim=0, dim_size=num_graphs
        ).clamp_min(1e-30)
        sizes = scatter_sum(
            torch.ones_like(alpha[:, :1]), ref["batch"], dim=0, dim_size=num_graphs
        ).clamp_min(1.0)
        spread = (1.0 / inverse) / sizes
        floor = torch.clamp(
            sizes * self.size_delocalised_fraction, min=self.size_delocalised_min_atoms
        )
        delocalised = ((1.0 / inverse) > floor).any(dim=0)
        self.last_size_spread = spread.detach().amax(dim=0)
        threshold = self.size_threshold(counts, delocalised=delocalised)
        self.last_size_exempt = float((~torch.isfinite(threshold) & live).sum())
        self.last_size_x = probe.x.detach()
        self.last_size_threshold = threshold
        if self.size_weight <= 0.0 or self.current_epoch < self.size_warmup_epochs:
            return zero
        violation = torch.where(
            torch.isfinite(threshold) & live,
            (probe.x - threshold).clamp_min(0.0),
            torch.zeros_like(probe.x),
        )
        raw = ref.weight * violation.pow(2).sum(dim=-1)
        value = self.size_weight * reduce_loss(raw, ddp)
        # The plan asks for lambda_size calibrated to ~5-10% of the delta-energy loss and
        # for the *realised* ratio to be logged, not the intended one -- the two diverge as
        # the fit moves, and the loss is now dimensionless so the previous weight carries
        # no meaning at all.
        self.last_size_value = float(value.detach())
        return value

    def stoichiometric_mask(self, ref: Batch) -> Optional[torch.Tensor]:
        """Which graphs in this batch are defect-free, decided by COMPOSITION alone.

        LABEL-FREE BY CONSTRUCTION, and that is the whole reason the gap term is allowed in
        the loss at all. The test is `n_species(g) proportional to gap_composition`, a
        property of the formula unit; it never consults the vacancy assignment, the carrier
        counters or any defect annotation. A 79-atom V_Cl cell fails it because one Cl is
        missing, which is arithmetic and not a label.

        Not by atom count: the largest cells in this training set are 159-atom DEFECT
        supercells, so "the big frames are the pristine ones" picks defect cells and builds
        the bulk reference out of the structures it is meant to distinguish.
        """
        if self.gap_composition is None:
            return None
        counts = scatter_sum(ref.node_attrs, ref.batch, dim=0,
                             dim_size=int(ref.num_graphs))
        target = torch.as_tensor(self.gap_composition, dtype=counts.dtype,
                                 device=counts.device)
        if target.numel() != counts.shape[1]:
            raise ValueError(
                f"gap_composition has {target.numel()} entries but the model has "
                f"{counts.shape[1]} species; the two must be in the same order")
        # Units of formula per cell from the TOTAL atom count, not from one species'.
        # Pivoting on a single species is wrong in exactly the case that matters: a 79-atom
        # V_Cl cell divided by the Cl coefficient gives 15.667 units, whose expected counts
        # are (47, 15.667, 15.667) -- within a third of an atom of the real (47, 16, 16),
        # so a half-atom tolerance calls the vacancy cell pristine. The defect hides in the
        # species you pivot on.
        #
        # The tolerance is tight rather than generous for the same reason. These counts are
        # exact integers, so a stoichiometric cell matches to floating-point round-off and
        # anything looser only admits near-misses -- which is what a vacancy is.
        units = counts.sum(dim=-1) / target.sum()
        expected = units.unsqueeze(-1) * target.unsqueeze(0)
        return (counts - expected).abs().max(dim=-1).values < 1e-3

    def gap_penalty(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        """`w * (gap_pristine - E_gap)^2`. A SPECTRUM constraint, not an energy label.

        The frontier gap of a defect-free cell must be the host band gap. That is a property
        of the material, so this term carries none of M1b's base-extrapolation slope -- which
        is exactly why it is the one supervision the head-only stages are allowed besides
        forces.

        THE DIFFERENCE FROM `stage_run.py`, stated because it is real. The harness draws a
        dedicated pristine batch per step; here the term is applied to whichever
        stoichiometric graphs the batch already contains, which costs no extra forward and
        keeps the loss a pure function of `(pred, ref)`. Steps whose batch contains none
        contribute zero, so `gap_steps_with_term / gap_steps_total` is recorded and belongs
        in the run's report -- the realised coverage is a property of the shuffle, not a
        constant.
        """
        zero = torch.zeros((), dtype=ref.weight.dtype, device=ref.weight.device)
        gap = pred.get("logit_gap")
        if self.gap_weight <= 0.0 or gap is None:
            return zero
        mask = self.stoichiometric_mask(ref)
        self.gap_steps_total += 1
        if mask is None or not bool(mask.any()):
            return zero
        self.gap_steps_with_term += 1
        residual = gap[mask, 0].mean() - self.e_gap
        return self.gap_weight * residual ** 2

    def gauge_penalty(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        """Pin the level mode of ``u`` on pristine cells (forward plan, stage D-opt).

        The softmax is shift-invariant, so a uniform offset in ``u^c`` moves the energy
        without moving ``alpha`` -- one unidentified direction per channel. With the
        counters present here, three of the four are free. This penalises the pooled
        readout evaluated at each training counter vector on **pristine** frames, where a
        carrier is a band-to-band transition and band-edge referencing therefore requires
        the pooled value to be zero.

        It does not fix extensivity and never claimed to: a uniform offset contributes
        ``n_c * const`` independent of ``N`` because ``sum_i alpha_i = 1``. What it buys is
        that the *diluted* limit becomes physically interpretable instead of arbitrary,
        which is what makes a size measurement readable at all.
        """
        zero = torch.zeros((), dtype=ref.weight.dtype, device=ref.weight.device)
        gauge = pred.get("gauge_mean_u")
        if self.gauge_weight <= 0.0 or gauge is None:
            return zero
        num_graphs = int(ref.num_graphs)
        # Pristine frames are exactly the complement of the totals mask; no new flag.
        pristine = 1.0 - self._totals_mask(ref)
        raw = ref.weight * pristine * gauge.pow(2).sum(dim=(-1, -2))
        return self.gauge_weight * reduce_loss(raw, ddp)

    def eps_gauge(self, pred: TensorDict, ref, ddp: bool) -> torch.Tensor:
        """T5: pin the site-energy gauge with a penalty instead of a subtraction.

        eps has an exact uniform mode -- adding a constant to every site shifts every
        eigenvalue and hence dE_SR, and a trainable base can partly absorb it, which is how
        eps reached 12-30 eV before any gauge control existed.

        Subtracting the per-frame mean removes the mode exactly but makes lambda depend on
        cell size: for a localised well the mean is ~depth*k/N, giving an O(1/N) term
        (measured -3*(1 - 1/N) in a toy, ~2.6 meV across the 640-5120 ladder against a
        <= 1 meV gate). Penalising the mean instead pins the gauge without touching the
        N-dependence of the eigenvalue, and mu_c continues to carry the level position.

        Applied per frame and only on ACTIVE channels: a channel with n_c = 0 contributes
        nothing to the energy, so its site energies are unconstrained by the data and
        penalising them would be regularising noise.
        """
        zero = torch.zeros((), dtype=ref.weight.dtype, device=ref.weight.device)
        mean_eps = pred.get("carrier_eps_mean")
        if self.eps_gauge_weight <= 0.0 or mean_eps is None:
            return zero
        counts = ref.carrier_counts.view(int(ref.num_graphs), -1).to(mean_eps.dtype)
        active = (counts.abs() > 0).to(mean_eps.dtype)
        raw = ref.weight * (active * mean_eps.pow(2)).sum(dim=-1)
        return self.eps_gauge_weight * reduce_loss(raw, ddp)

    def regularisation(self, pred: TensorDict) -> torch.Tensor:
        """Penalties on the correction readouts, plus the optional prior on ``a``."""
        total = torch.zeros((), dtype=torch.get_default_dtype())
        for weight, key in (
            (self.u_l2, "carrier_readouts"),
            (self.p_l2, "polarisation"),
            (self.qhost_l2, "latent_charges_host"),
        ):
            value = pred.get(key)
            if weight > 0 and value is not None:
                total = total.to(value.device) + weight * torch.mean(
                    torch.square(value)
                )
        # Already a squared norm of a weight slice, not a per-site quantity, so it is
        # added directly rather than averaged.
        counter_input = pred.get("counter_input_l2")
        if self.zn_l2 > 0 and counter_input is not None:
            total = total.to(counter_input.device) + self.zn_l2 * counter_input
        amplitude = pred.get("screening_amplitude")
        if (
            self.eps_inf_prior_weight > 0
            and self.eps_inf is not None
            and amplitude is not None
        ):
            target = 1.0 / math.sqrt(self.eps_inf)
            total = total.to(amplitude.device) + self.eps_inf_prior_weight * torch.mean(
                torch.square(amplitude - target)
            )
        return total

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"delta_energy_weight={self.delta_energy_weight:.3f}, "
            f"delta_forces_weight={self.delta_forces_weight:.3f}, "
            f"total_energy_weight={self.total_energy_weight:.3f}, "
            f"u_l2={self.u_l2}, p_l2={self.p_l2}, qhost_l2={self.qhost_l2})"
        )
