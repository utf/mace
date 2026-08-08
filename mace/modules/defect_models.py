###########################################################################################
# Charge-aware defect models
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
"""``MACEDefect``: a carrier-conditioned residual on a charge-blind base potential.

    ``E_total(R, n) = E_base(R) + Delta E_SR(R, n) + E_LR(R, n)``

The base branch sees geometry alone; the correction branch sees the carrier counters.
Every ``n``-dependent term carries an explicit counter factor, so at ``n = 0`` the
correction vanishes identically and ``E_total(R, 0) == E_base(R)`` holds for *any*
parameters -- a unit test, not a metric (plan section 8.1).

The long-range branch of section 3.4 is added on top of this skeleton; the short-range
correction here is complete on its own.
"""

from typing import Any, Dict, List, Optional

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock
from mace.modules.defect_blocks import (
    CarrierAttentionPooling,
    CounterEmbedding,
    StructuredLatentCharges,
)
from mace.modules.latent_ewald import LatentEwald
from mace.modules.models import ScaleShiftMACE
from mace.modules.utils import get_atomic_virials_stresses, get_outputs, prepare_graph
from mace.tools.scatter import scatter_sum


def _readout_input_irreps(block: torch.nn.Module) -> o3.Irreps:
    if isinstance(block, LinearReadoutBlock):
        return o3.Irreps(str(block.linear.irreps_in))  # type: ignore
    if isinstance(block, NonLinearReadoutBlock):
        return o3.Irreps(str(block.linear_1.irreps_in))  # type: ignore
    raise TypeError(f"Unsupported readout type: {type(block)}")


def _energy_gradient(
    energy: torch.Tensor, positions: torch.Tensor, create_graph: bool
) -> torch.Tensor:
    """Forces from a scalar branch, keeping the graph for the total-energy pass."""
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],
        inputs=[positions],
        grad_outputs=grad_outputs,
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=True,
    )[0]
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


@compile_mode("script")
class MACEDefect(ScaleShiftMACE):
    """MACE with a carrier-conditioned short-range correction.

    Args:
        carrier_feature_dim: width of the invariant features extracted per layer for the
            correction heads.
        counter_embedding_dim: width of the carrier counter embedding.
        carrier_mlp_hidden: hidden width of the per-channel logit and energy readouts.
        share_logits_across_spin: share the logit network between the two spin channels
            of a carrier type (a parameter economy; the plan asks for both to be tested).
        correction_trunk: ``shared`` reuses the base trunk's features. ``separate`` is
            reserved for the case where charged-data contamination of the base
            representation proves to be a problem.
    """

    def __init__(
        self,
        carrier_feature_dim: int = 32,
        counter_embedding_dim: int = 32,
        carrier_mlp_hidden: int = 64,
        share_logits_across_spin: bool = False,
        high_precision_softmax: bool = True,
        correction_trunk: str = "shared",
        use_long_range: bool = True,
        les_arguments: Optional[Dict[str, Any]] = None,
        eps_inf_init: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if correction_trunk != "shared":
            raise NotImplementedError(
                f"correction_trunk='{correction_trunk}' is not implemented; only "
                "'shared' is available"
            )
        self.correction_trunk = correction_trunk
        self.carrier_feature_dim = carrier_feature_dim
        self.counter_embedding_dim = counter_embedding_dim
        self.carrier_mlp_hidden = carrier_mlp_hidden
        self.share_logits_across_spin = share_logits_across_spin

        cueq_config = kwargs.get("cueq_config", None)
        feature_irreps = o3.Irreps(f"{carrier_feature_dim}x0e")
        self.defect_feature_readouts = torch.nn.ModuleList(
            [
                LinearReadoutBlock(
                    irreps_in=_readout_input_irreps(readout),
                    irrep_out=feature_irreps,
                    cueq_config=cueq_config,
                )
                for readout in self.readouts  # type: ignore
            ]
        )
        feature_dim = carrier_feature_dim * len(self.defect_feature_readouts)
        self.counter_embedding = CounterEmbedding(counter_embedding_dim)
        self.carrier_pooling = CarrierAttentionPooling(
            feature_dim=feature_dim,
            counter_dim=counter_embedding_dim,
            hidden_dim=carrier_mlp_hidden,
            share_logits_across_spin=share_logits_across_spin,
            high_precision_softmax=high_precision_softmax,
        )

        # Long-range branch. E_LR[q_host] is part of the *base* branch: it is a function
        # of geometry alone and is present at n = 0.
        self.use_long_range = use_long_range
        self.eps_inf_init = eps_inf_init
        self.les_arguments = dict(les_arguments) if les_arguments else None
        if use_long_range:
            self.latent_ewald = LatentEwald(les_arguments)
            self.latent_charges = StructuredLatentCharges(
                feature_dim=feature_dim,
                counter_dim=counter_embedding_dim,
                hidden_dim=carrier_mlp_hidden,
                eps_inf_init=eps_inf_init,
            )

    def forward(  # pylint: disable=too-many-branches
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        dilute: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(vectors.dtype)

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        if hasattr(self, "joint_embedding"):
            embedding_features: Dict[str, torch.Tensor] = {}
            for name, _ in self.embedding_specs.items():
                embedding_features[name] = data[name]
            node_feats += self.joint_embedding(data["batch"], embedding_features)
            if hasattr(self, "embedding_readout"):
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                e0 += scatter_sum(
                    src=embedding_node_energy,
                    index=data["batch"],
                    dim=0,
                    dim_size=num_graphs,
                )

        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_list.append(node_feats)

        # Base branch readouts, and the invariant features the correction heads see.
        defect_feats_list: List[torch.Tensor] = []
        for i, (readout, defect_readout) in enumerate(
            zip(self.readouts, self.defect_feature_readouts)
        ):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es = readout(node_feats_list[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]
            node_es_list.append(node_es)
            defect_feats_list.append(defect_readout(node_feats_list[feat_idx]))

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        base_energy = e0 + inter_e
        node_energy = node_e0.clone().double() + node_inter_es.clone().double()

        # Carrier correction. The counters are canonicalised at data loading and at every
        # inference entry point, so the network never sees a non-canonical vector.
        counts = data["carrier_counts"].view(num_graphs, -1).to(vectors.dtype)
        counter_emb = self.counter_embedding(counts)
        delta_sr, alpha, carrier_readouts, logit_gap = self.carrier_pooling(
            node_feats=torch.cat(defect_feats_list, dim=-1),
            counter_emb=counter_emb,
            counts=counts,
            batch=data["batch"],
            num_graphs=num_graphs,
        )

        # Long-range branch (plan section 3.4).
        latent_charge: Optional[torch.Tensor] = None
        q_host: Optional[torch.Tensor] = None
        q_carrier: Optional[torch.Tensor] = None
        polarisation: Optional[torch.Tensor] = None
        amplitude: Optional[torch.Tensor] = None
        dilute_correction: Optional[torch.Tensor] = None
        delta_lr = torch.zeros_like(base_energy)
        # The host long-range term is part of the base branch but does depend on the
        # positions, so it must reach the force/stress derivative.
        energy_lr_host = torch.zeros_like(base_energy)

        if self.use_long_range:
            # A null cell selects the isolated evaluator inside LES, which is how
            # non-periodic configurations are handled.
            cell_les = cell.clone()
            pbc_tensor = data["pbc"].to(device=cell.device)
            no_pbc_rows = (~pbc_tensor.any(dim=-1)).repeat_interleave(3)
            cell_les[no_pbc_rows] = torch.zeros(
                (int(no_pbc_rows.sum()), 3),
                dtype=cell_les.dtype,
                device=cell_les.device,
            )

            (
                latent_charge,
                q_host,
                q_carrier,
                polarisation,
                amplitude,
            ) = self.latent_charges(
                node_feats=torch.cat(defect_feats_list, dim=-1),
                counter_emb=counter_emb,
                counts=counts,
                alpha=alpha,
                batch=data["batch"],
                num_graphs=num_graphs,
            )
            energy_lr_total = self.latent_ewald.energy(
                latent_charge, positions, cell_les, data["batch"]
            )
            energy_lr_host = self.latent_ewald.energy(
                q_host, positions, cell_les, data["batch"]
            )
            # At n = 0 the polarisation and carrier channels vanish identically, so the
            # two evaluations see the same charges and this difference is exactly zero.
            base_energy = base_energy + energy_lr_host
            delta_lr = energy_lr_total - energy_lr_host

            if dilute:
                dilute_correction = self.latent_ewald.dilute_correction(
                    q_carrier, positions, cell_les, data["batch"], num_graphs
                )
                delta_lr = delta_lr + dilute_correction

        delta_energy = delta_sr + delta_lr
        total_energy = base_energy + delta_energy

        # Delta forces first: they differentiate the same graph the total-energy pass
        # below consumes, and get_outputs is free to release it.
        delta_forces: Optional[torch.Tensor] = None
        if compute_force:
            delta_forces = _energy_gradient(delta_energy, positions, training)

        forces, virials, stress, hessian, edge_forces, _ = get_outputs(
            energy=inter_e + energy_lr_host + delta_energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )

        # The base-branch forces are what L_base is trained against. They come for free
        # as the difference: the total energy is the sum of the two branches, and the
        # gradient is linear.
        base_forces: Optional[torch.Tensor] = None
        if forces is not None and delta_forces is not None:
            base_forces = forces - delta_forces

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )

        return {
            "energy": total_energy,
            "base_energy": base_energy,
            "delta_energy": delta_energy,
            "delta_sr_energy": delta_sr,
            "node_energy": node_energy,
            "forces": forces,
            "base_forces": base_forces,
            "delta_forces": delta_forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
            "carrier_alpha": alpha,
            "carrier_readouts": carrier_readouts,
            "logit_gap": logit_gap,
            "latent_charges": latent_charge,
            "latent_charges_host": q_host,
            "latent_charges_carrier": q_carrier,
            "polarisation": polarisation,
            "screening_amplitude": amplitude,
            "dilute_correction": dilute_correction,
        }
