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
    NUM_CARRIER_CHANNELS,
    CarrierAttentionPooling,
    CounterEmbedding,
    StructuredLatentCharges,
)
from mace.modules.latent_ewald import LatentEwald
from mace.modules.models import ScaleShiftMACE
from mace.modules.utils import get_atomic_virials_stresses, get_outputs, prepare_graph
from mace.tools.scatter import scatter_sum
from mace.tools.torch_tools import to_high_precision


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
        counter_embedding_dim: int = 16,
        carrier_mlp_hidden: int = 32,
        share_logits_across_spin: bool = False,
        high_precision_softmax: bool = True,
        zero_u_init: bool = True,
        alpha_mode: str = "logits",
        beta: float = 10.0,
        logit_seed_gamma: float = 0.0,
        correction_trunk: str = "shared",
        use_long_range: bool = True,
        les_arguments: Optional[Dict[str, Any]] = None,
        eps_inf_init: float = 1.0,
        freeze_amplitude: bool = False,
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
        self.alpha_mode = alpha_mode
        self.beta = beta

        # Novelty logit seeding (plan D7.1, logit route). gamma is trainable per channel,
        # so a shallow or effective-mass state -- for which the localisation prior is
        # simply wrong -- can drive it to zero rather than having to unlearn a seeded u.
        #
        # NOTE this is an architecture term, not an initialisation: it is present at
        # inference and is part of the energy. The descriptor is therefore recomputed
        # from the trunk's edge basis on every forward so that forces differentiate it,
        # and its scale constants are buffers that travel with the model.
        self.logit_seed = logit_seed_gamma > 0.0
        if self.logit_seed:
            num_channels = (
                self.radial_embedding.out_dim
                * (int(kwargs["max_ell"]) + 1)
                * int(kwargs["num_elements"])
            )
            self.register_buffer(
                "novelty_channel_scale", torch.ones(num_channels)
            )
            self.register_buffer("novelty_global_scale", torch.ones(()))
            self.logit_seed_gamma = torch.nn.Parameter(
                torch.full((NUM_CARRIER_CHANNELS,), float(logit_seed_gamma))
            )

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
            zero_u_init=zero_u_init,
            alpha_mode=alpha_mode,
            beta=beta,
        )

        # Long-range branch. E_LR[q_host] is part of the *base* branch: it is a function
        # of geometry alone and is present at n = 0.
        self.use_long_range = use_long_range
        self.eps_inf_init = eps_inf_init
        self.freeze_amplitude = freeze_amplitude
        self.les_arguments = dict(les_arguments) if les_arguments else None
        if use_long_range:
            self.latent_ewald = LatentEwald(les_arguments)
            self.latent_charges = StructuredLatentCharges(
                feature_dim=feature_dim,
                counter_dim=counter_embedding_dim,
                hidden_dim=carrier_mlp_hidden,
                eps_inf_init=eps_inf_init,
                freeze_amplitude=freeze_amplitude,
            )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Fill in attributes added after a checkpoint was written.

        Whole `MACEDefect` objects are pickled, so a model saved before a new flag
        existed comes back without it and `forward` fails on attribute access -- not at
        load time, but on the first evaluation, which is a confusing place to find out.
        Defaults here must reproduce the behaviour the checkpoint was trained with.
        """
        super().__setstate__(state)
        for name, default in (
            ("logit_seed", False),
            ("alpha_mode", "logits"),
            ("beta", 10.0),
            ("freeze_amplitude", False),
            ("correction_trunk", "shared"),
        ):
            if not hasattr(self, name):
                object.__setattr__(self, name, default)

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
        node_energy = to_high_precision(node_e0.clone()) + to_high_precision(
            node_inter_es.clone()
        )

        # Carrier correction. The counters are canonicalised at data loading and at every
        # inference entry point, so the network never sees a non-canonical vector.
        defect_feats = torch.cat(defect_feats_list, dim=-1)
        counts = data["carrier_counts"].view(num_graphs, -1).to(vectors.dtype)
        # The counter the paired difference is measured from. Absent (zeros) is the
        # closed-shell reference, for which the correction vanishes identically -- so a
        # dataset without reference counters behaves exactly as before.
        if "carrier_counts_ref" in data:
            counts_ref = data["carrier_counts_ref"].view(num_graphs, -1).to(vectors.dtype)
        else:
            counts_ref = torch.zeros_like(counts)

        logit_bias: Optional[torch.Tensor] = None
        if self.logit_seed:
            from mace.modules.defect_seed import novelty_from_basis

            novelty = novelty_from_basis(
                edge_feats=edge_feats,
                edge_sh=edge_attrs,
                edge_index=data["edge_index"],
                node_attrs=data["node_attrs"],
                batch=data["batch"],
                num_graphs=num_graphs,
                channel_scale=self.novelty_channel_scale,
                global_scale=self.novelty_global_scale,
            )
            logit_bias = novelty.unsqueeze(-1) * self.logit_seed_gamma.unsqueeze(0)

        counter_emb = self.counter_embedding(counts)
        (
            delta_sr,
            alpha,
            carrier_readouts,
            logit_gap,
            delta_u,
            carrier_logits,
        ) = self.carrier_pooling(
            node_feats=defect_feats,
            counter_emb=counter_emb,
            counts=counts,
            batch=data["batch"],
            num_graphs=num_graphs,
            logit_bias=logit_bias,
        )
        # Intrinsic gap: the same pooling with the seed switched off, so the logged gap
        # separates what MLP_l has learned from what the seed is supplying. The dead
        # channel is a decent proxy for the seed baseline (its logit weights are frozen
        # by n_c = 0) but not an exact one, since its trunk inputs still move; this is
        # the exact version. Costs one MLP pass, no trunk work.
        logit_gap_intrinsic: Optional[torch.Tensor] = None
        if self.logit_seed:
            _, _, _, logit_gap_intrinsic, _, _ = self.carrier_pooling(
                node_feats=defect_feats,
                counter_emb=counter_emb,
                counts=counts,
                batch=data["batch"],
                num_graphs=num_graphs,
                logit_bias=None,
            )

        counter_emb_ref = self.counter_embedding(counts_ref)
        delta_sr_ref, alpha_ref, _, _, _, _ = self.carrier_pooling(
            node_feats=defect_feats,
            counter_emb=counter_emb_ref,
            counts=counts_ref,
            batch=data["batch"],
            num_graphs=num_graphs,
            logit_bias=logit_bias,
        )

        # Long-range branch (plan section 3.4).
        latent_charge: Optional[torch.Tensor] = None
        q_host: Optional[torch.Tensor] = None
        q_carrier: Optional[torch.Tensor] = None
        polarisation: Optional[torch.Tensor] = None
        amplitude: Optional[torch.Tensor] = None
        dilute_correction: Optional[torch.Tensor] = None
        delta_lr = torch.zeros_like(base_energy)
        delta_lr_ref = torch.zeros_like(base_energy)
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
                node_feats=defect_feats,
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

            # The same at the reference counter. This branch is *not* inert at q = 0:
            # q^carrier is a compensated but pointwise non-zero charge whose self-term is
            # the electron-hole interaction, and q^pol carries a factor sum_c n_c. Both
            # are live whenever the reference state itself carries carriers.
            latent_charge_ref, _, _, _, _ = self.latent_charges(
                node_feats=defect_feats,
                counter_emb=counter_emb_ref,
                counts=counts_ref,
                alpha=alpha_ref,
                batch=data["batch"],
                num_graphs=num_graphs,
            )
            delta_lr_ref = (
                self.latent_ewald.energy(
                    latent_charge_ref, positions, cell_les, data["batch"]
                )
                - energy_lr_host
            )

            if dilute:
                dilute_correction = self.latent_ewald.dilute_correction(
                    q_carrier, positions, cell_les, data["batch"], num_graphs
                )
                delta_lr = delta_lr + dilute_correction

        # The correction at this frame's own counter is what the total energy carries;
        # the paired difference is what the delta labels supervise. They coincide only
        # when the reference is the closed-shell state.
        correction_energy = delta_sr + delta_lr
        correction_energy_ref = delta_sr_ref + delta_lr_ref
        delta_energy = correction_energy - correction_energy_ref
        total_energy = base_energy + correction_energy

        # Correction forces first: they differentiate the same graph the total-energy pass
        # below consumes, and get_outputs is free to release it.
        correction_forces: Optional[torch.Tensor] = None
        delta_forces: Optional[torch.Tensor] = None
        if compute_force:
            correction_forces = _energy_gradient(correction_energy, positions, training)
            correction_forces_ref = _energy_gradient(
                correction_energy_ref, positions, training
            )
            delta_forces = correction_forces - correction_forces_ref

        forces, virials, stress, hessian, edge_forces, _ = get_outputs(
            energy=inter_e + energy_lr_host + correction_energy,
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
        # gradient is linear. Note this subtracts the correction at *this* frame's
        # counter, not the paired difference -- the two agree only when the reference is
        # the closed-shell state, which is exactly when base labels exist.
        base_forces: Optional[torch.Tensor] = None
        if forces is not None and correction_forces is not None:
            base_forces = forces - correction_forces

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
            "correction_energy": correction_energy,
            "counter_input_l2": self.carrier_pooling.counter_input_l2(),
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
            # The exact inputs the correction readouts consume, exposed so that seeding
            # and diagnostics do not have to re-derive the trunk (defect_seed.py).
            "defect_features": defect_feats,
            "counter_embedding": counter_emb,
            "carrier_readouts": carrier_readouts,
            # Post-clamp, post-bias: the tensor the softmax actually consumed. The size
            # term re-forms the normalisation for a hypothetically larger cell, which
            # alpha cannot support because it has already divided the denominator out.
            "carrier_logits": carrier_logits,
            "logit_gap": logit_gap,
            "delta_u": delta_u,
            "logit_gap_intrinsic": logit_gap_intrinsic,
            "latent_charges": latent_charge,
            "latent_charges_host": q_host,
            "latent_charges_carrier": q_carrier,
            "polarisation": polarisation,
            "screening_amplitude": amplitude,
            "dilute_correction": dilute_correction,
        }
