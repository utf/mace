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

import logging
from typing import Any, Dict, List, Optional, Sequence

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
from mace.modules.defect_profile import mark
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


#: Section 2.5 of the Stage A' spec. "uniform" is the historical single-dtype model;
#: "mixed" runs the trunk in the process default (float32) and the carrier head, the
#: Madelung term and the long-range branch in float64, with explicit casts at the boundary.
PRECISION_POLICIES = ("uniform", "mixed")


RETIRED_STATE = ("pristine_level_spacing", "pristine_spacing_set")


def _drop_retired_state(module, state_dict, prefix, local_metadata, strict, missing_keys,
                        unexpected_keys, error_msgs):
    """load_state_dict pre-hook: keys of deleted paths are dropped and logged.

    A checkpoint written before plan v8 carries the depth sigmoid's two buffers. Loading
    it must neither fail on them (they are not an architecture mismatch) nor keep them
    (nothing reads them); they are removed here, and the log says so.
    """
    for name in RETIRED_STATE:
        key = prefix + name
        if key in state_dict:
            del state_dict[key]
            logging.info("state dict: dropped retired key %s (plan v8 section 3)", key)


def _sync_trunk_constants_into_state_dict(module, state_dict, prefix, local_metadata):
    """state_dict hook: the buffer copies the blocks' CURRENT plain-float values.

    The blocks are the source of truth while the model runs (the trainer rescales them, the
    Stage-A loader resets them); the buffer is only ever a serialisation of them. Refreshing
    it here, at write time, is what makes the state dict a faithful record.
    """
    key = prefix + "trunk_avg_num_neighbors"
    if key in state_dict and hasattr(module, "interactions"):
        state_dict[key] = torch.tensor(
            [float(b.avg_num_neighbors) for b in module.interactions], dtype=torch.float64)
    return state_dict


def _restore_trunk_constants_from_buffer(module, incompatible_keys):
    """load_state_dict post-hook: the loaded buffer is written back onto the blocks.

    Without this a state dict carries the value but nothing consumes it, which is the
    original fault in a different coat.
    """
    buf = getattr(module, "trunk_avg_num_neighbors", None)
    blocks = getattr(module, "interactions", None)
    if buf is None or blocks is None or buf.numel() != len(blocks):
        return
    with torch.no_grad():
        for block, value in zip(blocks, buf.tolist()):
            block.avg_num_neighbors = float(value)


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
        spectral_head: bool = False,
        spectral_num_states: int = 6,
        spectral_smearing: float = 0.020,
        spectral_r_cut: float = 0.0,
        spectral_local: bool = False,
        spectral_r_couple: float = 0.0,
        spectral_single_manifold: bool = False,
        spectral_decay: bool = False,
        spectral_t_min: float = 0.02,
        spectral_decay_init: float = 1.0,
        spectral_first_shell: bool = False,
        spectral_sigma: bool = False,
        spectral_gauge_penalty: bool = False,
        counting_head: bool = False,
        counting_on_site_range: float = 3.0,
        counting_hop_range: float = 0.5,
        counting_smearing_family: str = "gaussian",
        counting_envelope: str = "exp",
        counting_decay_length: float = 1.0,
        counting_hop_form: str = "linear",
        counting_hop_log_beta: float = 1.0986122886681098,
        # Stage A' spec (section 2.4, 2.2, 2.3, 2.5, 2.1). Every knob is a constructor
        # argument from the day it exists, so the config round trip carries it (section 5.1).
        counting_decay_learned: bool = False,
        counting_decay_log_beta: float = 0.6931471805599453,
        # Speed-cycle spec section 2.1: where the pristine centre is subtracted.
        counting_centre_form: str = "argument",
        # Section 2.2: the per-site charge deviation's bound, in e. Zero disables the
        # channel, which is the Stage-B model exactly.
        madelung_site_zeta: float = 0.0,
        lr_detach_density: bool = False,
        lr_freeze: bool = False,
        image_compensation: bool = False,
        precision_policy: str = "uniform",
        on_site_centred: bool = False,
        # The LABELS' width, 0.05 eV, not 25 meV. The old default was a k_B * 300 K
        # coincidence, and leaving it here while switching the family to Gaussian
        # would have trained six seeds at Gaussian 0.025 and reported them as the
        # labels' convention -- which is what the saved-model check caught.
        counting_t_el: float = 0.05,
        madelung_on_site: bool = False,
        madelung_eps_inf: float = 4.0,
        madelung_composition: Optional[Sequence[float]] = None,
        madelung_z_init: Optional[Sequence[float]] = None,
        logit_seed_gamma: float = 0.0,
        correction_trunk: str = "shared",
        use_long_range: bool = True,
        les_arguments: Optional[Dict[str, Any]] = None,
        eps_inf_init: float = 1.0,
        freeze_amplitude: bool = False,
        use_polarisation: bool = True,
        host_charge_detached: bool = False,
        pol_gate: bool = False,
        pol_gate_lambda: float = 6.0,
        pol_gate_hops: int = 2,
        host_carrier_coupling: bool = True,
        carrier_self_isolated: bool = False,
        lr_start_epoch: int = 0,
        gauge_counters: Optional[List[List[int]]] = None,
        # Transition plan v8, section 2.1: the occupation policy and the designated
        # reference state are config. Only `count_fill` is accepted; `reference_state` is
        # the serialised `ElectronicStateSpec` of S_ref, None meaning the programme's
        # count-filled neutral reference.
        occupation_policy: str = "count_fill",
        reference_state: Optional[Dict[str, Any]] = None,
        # Section 2.6: which kernel the gauge-dependent terms run under. "periodic" is
        # G_PBC, the training gauge; "isolated" is G_inf, the inference gauge.
        gauge: str = "periodic",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        from mace.modules.defect_state import (PRODUCTION_POLICIES, ElectronicStateSpec,
                                               reference_state as default_reference)
        from mace.modules.defect_terms import GAUGES

        if str(gauge) not in GAUGES:
            raise ValueError(f"gauge {gauge!r} is not one of {GAUGES}")
        self.gauge = str(gauge)

        if str(occupation_policy) not in PRODUCTION_POLICIES:
            raise ValueError(
                f"occupation_policy {occupation_policy!r} is not a production policy; "
                f"Stages 0-6 reach exactly {PRODUCTION_POLICIES}")
        self.occupation_policy = str(occupation_policy)
        ref = (default_reference() if reference_state is None
               else ElectronicStateSpec.from_dict(reference_state))
        if ref.delta_n != (0, 0) or ref.occupation_policy != self.occupation_policy:
            raise ValueError(f"reference_state {ref} is not a valid S_ref for policy "
                             f"{self.occupation_policy!r}: delta_n must be (0, 0) and the "
                             "policy must match")
        self.reference_state = ref.to_dict()
        if correction_trunk != "shared":
            raise NotImplementedError(
                f"correction_trunk='{correction_trunk}' is not implemented; only "
                "'shared' is available"
            )
        self.correction_trunk = correction_trunk
        # Counter vectors the level-mode gauge probe is evaluated at, populated from the
        # training set. A buffer rather than a plain attribute so it moves with the model
        # across devices and dtypes and lands in the state dict; empty by default, which
        # is what makes the probe cost exactly nothing unless it is configured.
        self.register_buffer(
            "gauge_counters",
            torch.as_tensor(
                gauge_counters if gauge_counters else [], dtype=torch.long
            ).reshape(-1, NUM_CARRIER_CHANNELS),
            persistent=True,
        )
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

        # Component H. When on, carrier energy is the lowest eigenvalue of a learned
        # short-ranged Hamiltonian rather than a softmax-weighted mean of site energies, so
        # localisation is a bound-state threshold instead of an amplitude contest against N.
        # Both heads expose the same six outputs, so every existing diagnostic keeps working:
        # `alpha` becomes sum_k w_k psi^2 and the site energy plays the role of u.
        self.spectral_head = bool(spectral_head)
        self.spectral_first_shell = bool(spectral_first_shell)
        self.spectral_feature_dim = (carrier_feature_dim if spectral_first_shell
                                     else feature_dim)
        # Edit 1, and Edit 2 in the same breath. The response channel is gone: it measured
        # that electrostatics has the reach (clause 1) but does not select a site (clause 2),
        # which is exactly what an energy-readout term cannot do -- V could change the energy
        # but never the Hamiltonian, so it could never change where the carrier goes. The
        # Madelung potential of learnable per-species charges goes on the ON-SITE ENERGIES
        # instead, where it is variational.
        #
        # The two must never both exist: they are the same physics counted twice.
        self.madelung_on_site = bool(madelung_on_site)
        self.madelung_eps_inf = float(madelung_eps_inf)
        self.madelung = None
        if madelung_on_site:
            from mace.modules.defect_madelung import MadelungOnSite

            if madelung_composition is None:
                raise ValueError(
                    "madelung_on_site needs madelung_composition: the pristine stoichiometry "
                    "in the model's own species order, which is what the neutrality "
                    "projection is defined against")
            self.madelung = MadelungOnSite(
                num_elements=int(kwargs["num_elements"]),
                composition=madelung_composition,
                z_init=madelung_z_init,
                site_zeta=float(madelung_site_zeta),
                feature_dim=int(self.spectral_feature_dim))
        self.counting_head = bool(counting_head)
        self.counting_on_site_range = float(counting_on_site_range)
        self.counting_hop_range = float(counting_hop_range)
        self.counting_smearing_family = str(counting_smearing_family)
        self.counting_envelope = str(counting_envelope)
        self.counting_decay_length = float(counting_decay_length)
        self.counting_hop_form = str(counting_hop_form)
        self.counting_hop_log_beta = float(counting_hop_log_beta)
        self.counting_decay_learned = bool(counting_decay_learned)
        self.counting_decay_log_beta = float(counting_decay_log_beta)
        self.counting_centre_form = str(counting_centre_form)
        self.madelung_site_zeta = float(madelung_site_zeta)
        if precision_policy not in PRECISION_POLICIES:
            raise ValueError(f"unknown precision_policy {precision_policy!r}; expected one "
                             f"of {sorted(PRECISION_POLICIES)}")
        # PLAN v8 SECTION 3 retires three switches. They stay in the signature so a config
        # written before the plan still constructs a model, with the meaning each now has:
        #   image_compensation  the term is deleted; True is refused, not ignored;
        #   lr_detach_density   the density is never detached; True is ignored and logged;
        #   lr_freeze           the long-range branch is ALWAYS frozen at its eps_inf-only
        #                       values (host charges zero, polarisation off, amplitude
        #                       1/sqrt(eps_inf)); False is ignored and logged.
        if bool(image_compensation):
            raise ValueError(
                "image_compensation is retired by plan v8 (section 1: the one-shot "
                "compensation becomes V_F inside frontier stationarity at Stage 5); the "
                "term and its depth sigmoid are deleted, and a model cannot be built with it")
        if bool(lr_detach_density):
            logging.warning("lr_detach_density=True is ignored: plan v8 retires detached "
                            "densities, and the long-range charges are never detached")
        if use_long_range and not bool(lr_freeze):
            logging.warning("lr_freeze=False is ignored: plan v8 freezes the long-range "
                            "branch at its eps_inf-only values in Stages 0-6")
        self.lr_detach_density = False
        self.lr_freeze = True
        self.image_compensation = False
        self.precision_policy = str(precision_policy)
        self.on_site_centred = bool(on_site_centred)
        # Section 5.1 of the Stage A' spec: every non-parameter float that touches the forward
        # travels in the state dict. `avg_num_neighbors` is a plain float on each interaction
        # block, absent from state_dict, and a Stage-B trunk once normalised every message by
        # eight times Stage A's value with the loader reporting success. The buffer is
        # refreshed from the blocks whenever a state dict is written and written back onto
        # them whenever one is loaded, so the two cannot disagree across a stage boundary.
        self.register_buffer(
            "trunk_avg_num_neighbors",
            torch.tensor([float(b.avg_num_neighbors) for b in self.interactions],
                         dtype=torch.float64),
            persistent=True)
        # Section 2.5: the checksum of the frozen base whose outputs a training run cached,
        # so a checkpoint records which base its head was trained against. SHA-256 digest
        # bytes; all zero means no cache was ever built for this model.
        self.register_buffer("base_cache_checksum", torch.zeros(32, dtype=torch.uint8),
                             persistent=True)
        # Section 2.1: the species-mean output of the FIRST interaction block over the
        # pristine reference cell, set by the trainer from a stoichiometric frame once the
        # base is loaded and frozen. The trunk feature, not the readout of it: the readout
        # is a correction parameter and trains, so it is applied live in forward. All-zero
        # rows with `pristine_centre_set` False mean "not set", and the centred correction
        # then refuses to run rather than centring on zero.
        block0_dim = o3.Irreps(kwargs["hidden_irreps"]).dim
        self.register_buffer("pristine_block0_mean",
                             torch.zeros(int(kwargs["num_elements"]), block0_dim),
                             persistent=True)
        self.register_buffer("pristine_centre_set", torch.zeros((), dtype=torch.bool),
                             persistent=True)
        # Plan v8 section 3: the depth sigmoid's state (`pristine_level_spacing`,
        # `pristine_spacing_set`) is deleted; a state dict written with it is loaded
        # with those keys dropped and logged, never silently consumed.
        self._register_load_state_dict_pre_hook(_drop_retired_state, with_module=True)
        self._register_state_dict_hook(_sync_trunk_constants_into_state_dict)
        self.register_load_state_dict_post_hook(_restore_trunk_constants_from_buffer)
        self.counting_t_el = float(counting_t_el)
        self.spectral = None
        # 0.0 means "the trunk's receptive field", r_max * num_interactions. That is the
        # natural scale: eps_i and t_ij are functions of node features that already aggregate
        # everything within the receptive field, so a shorter Hamiltonian discards locality
        # the model has already computed. It also covers the whole shell Pb-Pb distribution
        # (median 5.32 A, worst case 6.80 A), which 5.0 A misses entirely and 8.0 A only half
        # covers.
        self.spectral_r_cut = float(spectral_r_cut) if spectral_r_cut > 0 else (
            float(self.r_max) * int(self.num_interactions))
        # V3: matrix elements from the detached block-1 descriptor, two length constants.
        # Built here rather than only by surgery so a saved V3 model rebuilt from its own
        # extracted config comes back as V3 -- the round trip the cuEq conversion performs.
        self.spectral_local = bool(spectral_local)
        self.spectral_single_manifold = bool(spectral_single_manifold)
        self.spectral_r_couple = (float(spectral_r_couple) if spectral_r_couple > 0
                                  else self.spectral_r_cut)
        if self.spectral_head and self.spectral_local:
            from mace.modules.defect_spectral_v3 import LocalSpectralHead

            if not self.spectral_first_shell:
                raise ValueError(
                    "spectral_local needs spectral_first_shell=True: otherwise the head is "
                    "handed every block's features concatenated, and the descriptor is not "
                    "local at r_max.")
            self.spectral = LocalSpectralHead(
                feature_dim=self.spectral_feature_dim,
                counter_dim=counter_embedding_dim,
                num_elements=int(kwargs["num_elements"]),
                r_max=float(self.r_max),
                r_couple=float(self.spectral_r_couple),
                hidden=carrier_mlp_hidden,
                num_states=spectral_num_states,
                smearing=spectral_smearing,
                gauge_penalty=bool(spectral_gauge_penalty),
                t_min=float(spectral_t_min),
                single_manifold=self.spectral_single_manifold,
            )
        elif self.spectral_head and self.counting_head:
            # Edit 4 replaces the spectral head rather than sitting beside it: two heads
            # would both write `delta_sr` and the model would double count.
            from mace.modules.defect_counting import CountingHead

            self.spectral = CountingHead(
                num_elements=int(kwargs["num_elements"]),
                feature_dim=self.spectral_feature_dim,
                atomic_numbers=[int(z) for z in kwargs["atomic_numbers"]],
                r_cut=float(self.spectral_r_cut),
                centre_form=str(counting_centre_form),
                on_site_range=float(counting_on_site_range),
                hop_range=float(counting_hop_range),
                smearing_family=str(counting_smearing_family),
                envelope=str(counting_envelope),
                decay_length=float(counting_decay_length),
                hop_form=str(counting_hop_form),
                hop_log_beta=float(counting_hop_log_beta),
                decay_learned=bool(counting_decay_learned),
                decay_log_beta=float(counting_decay_log_beta),
                t_el=float(counting_t_el),
                occupation_policy=self.occupation_policy,
                reference_state=self.reference_state)
        elif self.spectral_head:
            from mace.modules.defect_spectral import SpectralCarrierHead

            self.spectral = SpectralCarrierHead(
                feature_dim=self.spectral_feature_dim,
                counter_dim=counter_embedding_dim,
                hidden=carrier_mlp_hidden,
                num_states=spectral_num_states,
                smearing=spectral_smearing,
                r_cut=self.spectral_r_cut,
                num_elements=int(kwargs["num_elements"]),
                use_decay=bool(spectral_decay),
                use_sigma=bool(spectral_sigma),
                gauge_penalty=bool(spectral_gauge_penalty),
                t_min=float(spectral_t_min),
                decay_init=float(spectral_decay_init),
            )
        self.spectral_decay = bool(spectral_decay)
        self.spectral_sigma = bool(spectral_sigma)
        self.spectral_gauge_penalty = bool(spectral_gauge_penalty)

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
        self.use_polarisation = use_polarisation
        self.host_charge_detached = host_charge_detached
        self.pol_gate = pol_gate
        self.pol_gate_lambda = pol_gate_lambda
        self.pol_gate_hops = pol_gate_hops
        self.host_carrier_coupling = host_carrier_coupling
        self.carrier_self_isolated = carrier_self_isolated
        # Hold the long-range branch out of the energy AND the loss until this epoch. Not a
        # ramp on `a`: the branch is absent entirely, so the model IS the short-range model
        # until it fires. The point is that the short-range model reliably finds the
        # vacancy shell within a few epochs (participation 10.1 -> 4.3 -> 2.6 by epoch 4),
        # while every long-range run to date has had attention captured before it could.
        # Starting from a settled, correct attention asks a different question: whether the
        # long-range branch DESTROYS a right answer, rather than whether it can find one.
        self.lr_start_epoch = int(lr_start_epoch)
        # A buffer, so it travels with the model and survives checkpointing and the cuEq
        # round trip. Set by the trainer's epoch hook from the ABSOLUTE epoch.
        self.register_buffer(
            "current_epoch", torch.zeros((), dtype=torch.long), persistent=True
        )
        if use_long_range:
            self.latent_ewald = LatentEwald(les_arguments)
            self.latent_charges = StructuredLatentCharges(
                feature_dim=feature_dim,
                counter_dim=counter_embedding_dim,
                hidden_dim=carrier_mlp_hidden,
                eps_inf_init=eps_inf_init,
                freeze_amplitude=freeze_amplitude,
                use_polarisation=use_polarisation,
                host_charge_detached=host_charge_detached,
                pol_gate=pol_gate,
                pol_gate_lambda=pol_gate_lambda,
                pol_gate_hops=pol_gate_hops,
            )
        elif self.madelung_on_site:
            # Edit 1 needs an Ewald evaluator whether or not the long-range ENERGY branch is
            # enabled -- Stages 1 to 3 retrain the head with E_LR still staged off. Same
            # module and the same `les_arguments`, so phi_LR and E_LR cannot end up on
            # different smearings or different G = 0 conventions, which is the whole point of
            # "same code, same convention" in the spec.
            self.latent_ewald = LatentEwald(les_arguments)
        # Section 2.5: the per-frame base cache is attached by the trainer, never pickled
        # (see __getstate__), and absent means "compute everything".
        self._base_cache = None
        self._apply_precision_policy()
        self._apply_long_range_policy()

    # ------------------------------------------------------------ Stage A' policies

    def _apply_precision_policy(self) -> None:
        """Section 2.5: trunk float32, everything the carrier sees float64.

        The process default dtype is whatever the trainer set (float64 for every production
        run); under "mixed" the trunk -- embeddings, interactions, products, base readouts --
        is cast to float32 and the head, its readouts, the Madelung term and the long-range
        branch to float64. The casts at the two boundaries live in `forward`. Under "uniform"
        nothing is touched and the model runs at the process default, as it always did.
        """
        if getattr(self, "precision_policy", "uniform") != "mixed":
            return
        for name in ("node_embedding", "radial_embedding", "interactions", "products",
                     "readouts"):
            module = getattr(self, name, None)
            if module is not None:
                module.float()
        for name in ("defect_feature_readouts", "counter_embedding", "carrier_pooling",
                     "spectral", "madelung", "latent_charges", "latent_ewald"):
            module = getattr(self, name, None)
            if module is not None:
                module.double()

    def _apply_long_range_policy(self) -> None:
        """Section 2.2: `lr_freeze` pins the long-range branch at PHYSICAL values.

        Physical, not "whatever the initialisation drew". The branch's charge is
        `q_host + q_pol + q_carrier`; the host and polarisation MLPs are random at
        construction, and a Stage-B base trained without E_LR cannot absorb a
        geometry-dependent term nobody fitted -- so under the freeze the host charges are
        zeroed (the ion lattice is already in H through the Madelung term), the
        polarisation channel is off, and the amplitude stays at its 1/sqrt(eps_inf)
        initialisation. What survives is exactly the spec's `E_LR = E_per(q) - E_iso(q)`
        with `q = a * alpha` from the density-matrix difference, and every parameter of the
        branch has `requires_grad = False`.
        """
        charges = getattr(self, "latent_charges", None)
        if charges is not None:
            with torch.no_grad():
                last = [m for m in charges.host_charge.modules()
                        if isinstance(m, torch.nn.Linear)][-1]
                last.weight.zero_()
                if last.bias is not None:
                    last.bias.zero_()
            charges.use_polarisation = False
        for name in ("latent_charges", "latent_ewald"):
            module = getattr(self, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(False)

    def set_base_cache(self, cache) -> None:
        """Attach (or detach, with None) the per-frame base cache of section 2.5."""
        self._base_cache = cache

    def terms(self):
        """Plan v8 section 2.5: the registered terms of this model's functional."""
        from mace.modules.defect_terms import registry

        return registry(self)

    @torch.no_grad()
    def set_pristine_centre(self, block0_feats: torch.Tensor, species: torch.Tensor) -> None:
        """Section 2.1: record the per-species mean of the first block's features over a
        pristine cell. `block0_feats` is the `trunk_block0` output of a forward on that cell."""
        n_el = int(self.pristine_block0_mean.shape[0])
        mean = torch.zeros_like(self.pristine_block0_mean)
        for s in range(n_el):
            sel = species == s
            if bool(sel.any()):
                mean[s] = block0_feats[sel].to(mean.dtype).mean(dim=0)
        self.pristine_block0_mean.copy_(mean)
        self.pristine_centre_set.fill_(True)

    def pristine_centre(self, head_dtype: torch.dtype) -> Optional[torch.Tensor]:
        """The centre in the head's feature space, through the LIVE first readout."""
        if not getattr(self, "on_site_centred", False) or getattr(
                self, "_collecting_centre", False):
            return None
        if not bool(self.pristine_centre_set):
            raise RuntimeError(
                "on_site_centred is set but no pristine centre has been recorded; call "
                "collect_pristine_centre on the stoichiometric frames before the first "
                "forward")
        feats = self.defect_feature_readouts[0](self.pristine_block0_mean.to(head_dtype))
        return feats[:, : self.spectral_feature_dim]

    @torch.no_grad()
    def collect_pristine_centre(self, batches, device=None) -> int:
        """Section 2.1, the one entry point: forward every batch of stoichiometric frames
        with the centre switched OFF for the duration, then record the per-species mean of
        the first block's features over all of them. Returns the number of atoms averaged.

        One method rather than a trainer-side loop, so the trainer and the tests set the
        centre the same way -- and so the probing forward cannot trip the "no centre yet"
        guard, which is what a bare forward with the flag on does."""
        feats, species, sizes_seen = [], [], []
        self._collecting_centre = True
        try:
            for batch in batches:
                if device is not None:
                    batch = batch.to(device)
                out = self(batch.to_dict(), training=False, compute_force=False)
                feats.append(out["trunk_block0"].detach())
                species.append(batch.node_attrs.argmax(dim=-1))
                ptr = getattr(batch, "ptr", None)
                if ptr is not None:
                    sizes_seen.extend(int(x) for x in (ptr[1:] - ptr[:-1]).tolist())
        finally:
            self._collecting_centre = False
        if not feats:
            raise RuntimeError("no stoichiometric frames to centre on")
        self.set_pristine_centre(torch.cat(feats), torch.cat(species))
        # Standing rule 1: the pristine cell's atom count is what the c table's size class
        # is a multiple of, and it comes from the same pristine geometry the centre does.
        head = getattr(self, "spectral", None)
        if head is not None and hasattr(head, "pristine_atoms") and sizes_seen:
            head.pristine_atoms.fill_(int(min(sizes_seen)))
        return int(sum(f.shape[0] for f in feats))

    def __getstate__(self) -> Dict[str, Any]:
        # The cache is a training-time object keyed to one dataset; a checkpoint carries the
        # checksum buffer that names it, never the values.
        state = self.__dict__.copy()
        state["_base_cache"] = None
        state["_collecting_centre"] = False
        return state

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
            ("use_polarisation", True),
            ("host_charge_detached", False),
            ("pol_gate", False),
            ("pol_gate_lambda", 6.0),
            ("pol_gate_hops", 2),
            ("host_carrier_coupling", True),
            ("carrier_self_isolated", False),
            ("lr_start_epoch", 0),
            ("precision_policy", "uniform"),
            ("counting_decay_learned", False),
            ("lr_detach_density", False),
            ("lr_freeze", False),
            ("image_compensation", False),
            ("counting_centre_form", "output"),
            ("madelung_site_zeta", 0.0),
            ("skip_neutral_reference", True),
            ("on_site_centred", False),
            ("_base_cache", None),
            ("_collecting_centre", False),
            # Plan v8 section 3: legacy checkpoints load with DETERMINISTIC defaults -- the
            # count fill and the neutral reference, which is what every one of them ran.
            ("occupation_policy", "count_fill"),
            ("reference_state", None),
            ("gauge", "periodic"),
        ):
            if not hasattr(self, name):
                object.__setattr__(self, name, default)
        if getattr(self, "image_compensation", False):
            logging.warning(
                "this checkpoint was trained WITH the image compensation term, which "
                "plan v8 deletes; it now evaluates without it, and its numbers are not "
                "the numbers it was trained to produce")
            object.__setattr__(self, "image_compensation", False)
        if getattr(self, "reference_state", None) is None:
            from mace.modules.defect_state import reference_state as default_reference

            object.__setattr__(self, "reference_state", default_reference().to_dict())
        # Buffers need registering, not just setting, or they stay out of the state dict
        # and out of `.to()`. A model pickled before the gauge probe existed has none, and
        # an empty buffer is exactly the "probe disabled" state, so old checkpoints keep
        # behaving as they did.
        if not hasattr(self, "current_epoch"):
            # Old checkpoints predate the gate. lr_start_epoch defaults to 0 for them, so
            # any value leaves the branch on, which is the behaviour they were trained with.
            self.register_buffer(
                "current_epoch", torch.zeros((), dtype=torch.long), persistent=True
            )
        if not hasattr(self, "gauge_counters"):
            self.register_buffer(
                "gauge_counters",
                torch.zeros((0, NUM_CARRIER_CHANNELS), dtype=torch.long),
                persistent=True,
            )
        # The charge assembly is pickled as a submodule, so it needs the same treatment.
        charges = getattr(self, "latent_charges", None)
        if charges is not None:
            for name, default in (
                ("use_polarisation", True),
                ("host_charge_detached", False),
                ("pol_gate", False),
                ("pol_gate_lambda", 6.0),
                ("pol_gate_hops", 2),
                ("pol_gate_eps", 1e-8),
            ):
                if not hasattr(charges, name):
                    object.__setattr__(charges, name, default)


    def _isolated_carrier_self(
        self,
        alpha: torch.Tensor,  # [n_nodes, 4]
        counts: torch.Tensor,  # [n_graphs, 4]
        amplitude: torch.Tensor,  # [n_graphs]
        positions: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """``sum_c E_isolated[Q^c]`` -- the in-cell electrostatics of each carrier channel
        with itself, evaluated with no images.

        Subtracting this from the periodic energy leaves the finite-size correction and
        nothing else. The in-cell part is spurious: a single hole has no Hartree
        self-repulsion, so the whole in-cell electrostatic energy of ONE channel is
        self-interaction error. Measured at the training cell it pays +0.104 eV to spread
        the attention out, and removing it is what let two of three control seeds reach the
        correct vacancy-shell solution.

        Per channel, not on the summed charge: the CROSS-channel terms are real physics
        (the electron-hole interaction on 4H-SiC) and must survive. Only the within-channel
        self-energy is removed.

        The i = j cancellation is exact regardless of the minimum-image convention, because
        both evaluators build the same on-site term from the same smeared charge -- which is
        the robust part of this construction.
        """
        signed = self.latent_charges.carrier_signs.unsqueeze(0) * counts
        total = torch.zeros(num_graphs, dtype=positions.dtype, device=positions.device)
        for channel in range(NUM_CARRIER_CHANNELS):
            charge = amplitude[batch] * alpha[:, channel] * signed[batch, channel]
            total = total + self.latent_ewald.isolated_energy(
                charge, positions, batch, num_graphs
            )
        return total

    def _carrier_head(
        self,
        node_feats: torch.Tensor,
        counter_emb: torch.Tensor,
        counts: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        edge_index: torch.Tensor,
        edge_length: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        node_species: Optional[torch.Tensor] = None,
        clamp_mask: Optional[torch.Tensor] = None,
        edge_vector: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        force_out: Optional[Dict[str, torch.Tensor]] = None,
        centre: Optional[torch.Tensor] = None,
        graph_sizes: Optional[torch.Tensor] = None,
        state=None,
    ):
        """Either carrier head, behind one signature.

        Returns the attention head's six outputs in every case, so the three call sites and
        every downstream diagnostic stay head-agnostic. The spectral head's analogues:

            alpha             sum_k w_k psi_k^2, a genuine per-cell probability like softmax
            carrier_readouts  the diagonal site energy eps, which is what u always meant
            logit_gap         the SPECTRAL gap lambda_band - lambda_bound; large means a
                              split-off bound level rather than a large logit difference
            carrier_logits    eps again. Only the size hinge consumes this, and the hinge is
                              retired under the spectral head -- a bound state's weight is
                              N-independent by construction, which the size ladder verifies
                              directly rather than a loss term enforcing.

        `logit_bias` is the novelty seed. It is applied to the DIAGONAL here, where it means
        "this site looks unusual, so lower its on-site energy" -- the same intent as biasing a
        logit, expressed in energy. Spectral arms run with the seed off, so it is normally
        None; honouring it keeps the two heads comparable if a seeded spectral arm is wanted.
        """
        # getattr, not attribute access: models are persisted by pickling the module, so a
        # checkpoint written before this attribute existed restores without it. Every model
        # trained up to now -- A0, the Stage-A bases, every historical run -- is in that
        # category, and plain access raises AttributeError on all of them.
        if not getattr(self, "spectral_head", False):
            # Same six outputs as always, plus an empty extras dict so both heads return the
            # same arity. The extras carry eps_mean for the gauge penalty; returning it beats
            # stashing it on the module, which put a graph-connected tensor in __dict__ and
            # broke deepcopy -- and therefore the cuEq conversion at the end of every run.
            return tuple(self.carrier_pooling(
                node_feats=node_feats,
                counter_emb=counter_emb,
                counts=counts,
                batch=batch,
                num_graphs=num_graphs,
                logit_bias=logit_bias,
            )) + ({},)

        head_feats = node_feats
        if getattr(self, "spectral_first_shell", False):
            head_feats = node_feats[:, : self.spectral_feature_dim]

        # Edit 1: the host Madelung potential on the on-site energies. Computed here, where
        # positions and the cell are in scope, and handed to the head as a finished shift --
        # the head owns the eigenproblem, not the electrostatics, and the sign lives in
        # MadelungOnSite.on_site_shift so there is exactly one place to get it wrong.
        madelung: Optional[torch.Tensor] = None
        if getattr(self, "madelung", None) is not None:
            if positions is None or cell is None or node_species is None:
                raise ValueError(
                    "madelung_on_site needs positions, cell and node_species; a call site "
                    "that omits them would silently drop the term and train a different "
                    "model than the one configured")
            # `cell`, never `cell_les`. H carries the periodic ion-lattice potential in
            # every mode -- the isolated switch belongs to E_LR alone, and there is no
            # branch here to reach it. Full lattice sum: no self-image subtraction.
            with mark("ewald/madelung"):
                    madelung = self.madelung.on_site_shift(
                    self.latent_ewald, node_species, positions, cell, batch,
                    eps_inf=self.madelung_eps_inf,
                    feats=head_feats, centre=centre, num_graphs=num_graphs)

        head_kwargs = dict(
            node_feats=head_feats,
            counter_emb=counter_emb,
            counts=counts,
            batch=batch,
            num_graphs=num_graphs,
            edge_index=edge_index,
            edge_length=edge_length,
            site_bias=None if logit_bias is None else -logit_bias,
            node_species=node_species,
            clamp_mask=clamp_mask,
            edge_vector=edge_vector,
            madelung=madelung,
        )
        # Only the counting head knows the centred correction, the c table and the state
        # batch; the older spectral heads keep their signature.
        if getattr(self.spectral, "accepts_centre", False):
            head_kwargs["centre"] = centre
            head_kwargs["graph_sizes"] = graph_sizes
        if getattr(self.spectral, "accepts_state", False):
            head_kwargs["state"] = state
        # Section 1. `force_out` is both the request and the reply: a head that can supply the
        # density response advertises `wants_positions`, and gets asked only when the caller
        # is in a training force pass. Absent here, no second backward is built -- which is
        # how "create_graph off in evaluation" is enforced structurally rather than by
        # inspecting the grad mode, since eval-with-forces has grad enabled too.
        #
        # A dict the head writes into, NOT an attribute on the head: a graph-connected tensor
        # in a module's __dict__ broke deepcopy, and with it the cuEq conversion at the end of
        # every run. Same reason `eps_mean` travels back in `head_extras`.
        if force_out is not None and getattr(self.spectral, "wants_positions", False):
            head_kwargs["positions"] = positions
            head_kwargs["force_out"] = force_out
        out = self.spectral(**head_kwargs)
        # delta_u is the spread of the occupied state's site energy over its own support: the
        # spectral analogue of "how much does u vary where alpha lives".
        #
        # PER GRAPH, shape [n_graphs, C], matching the attention head. Summing over all nodes
        # instead gave [C], which the training diagnostics then tried to reshape to
        # [n_graphs, -1] and failed on -- a batch of 8 against 4 values.
        weighted = scatter_sum(out.alpha * out.site_energy, batch, dim=0,
                               dim_size=num_graphs)
        node_count = scatter_sum(torch.ones_like(out.site_energy), batch, dim=0,
                                 dim_size=num_graphs).clamp_min(1.0)
        mean_eps = scatter_sum(out.site_energy, batch, dim=0,
                               dim_size=num_graphs) / node_count
        delta_u = weighted - mean_eps
        return (out.delta_sr, out.alpha, out.site_energy, out.gap, delta_u,
                out.site_energy, {"eps_mean": out.eps_mean})

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

        # Section 2.5 of the Stage A' spec. Under the mixed policy the trunk runs in float32
        # and the head in float64, with the casts here: trunk inputs down at the embedding,
        # trunk outputs up at the head boundary. Under "uniform" both are the data dtype and
        # every cast below is a no-op.
        mixed = getattr(self, "precision_policy", "uniform") == "mixed"
        trunk_dtype = torch.float32 if mixed else vectors.dtype
        head_dtype = torch.float64 if mixed else vectors.dtype
        # The base cache: present only when the trainer attached one AND the batch carries
        # frame keys. Any scorer that builds its own batches gets the full forward.
        base_cache = getattr(self, "_base_cache", None)
        cache_hit = base_cache is not None and "frame_key" in data
        cached_energy = cached_forces = cached_feats_rest = None
        # PLAN v8 SECTION 2.1: the electronic state of every graph, built once here from
        # the counter and used by the cache key, the reference question and the head. A
        # stated fill (`occupations`) is an alternate policy and is refused: `count_fill`
        # is the sole policy reachable from any Stage 0-6 configuration.
        if data.get("occupations") is not None:
            raise ValueError(
                "data['occupations'] is an occupation override, which is an alternate "
                "occupation policy; none is reachable in Stages 0-6 (plan v8 section 2.4)")
        from mace.modules.defect_state import ElectronicStateSpec, StateBatch

        policy_key = getattr(self, "occupation_policy", "count_fill")
        state = StateBatch.from_counts(data["carrier_counts"].view(num_graphs, -1),
                                       policy_key)
        if cache_hit:
            # Keyed by the state as well as the geometry (section 3): a frame requested
            # under a state it was not cached with is a miss that raises.
            cached_energy, cached_forces, cached_feats_rest, _ = base_cache.lookup(
                data["frame_key"], state.key_digests())
            cached_energy = cached_energy.to(head_dtype)
            cached_forces = cached_forces.to(head_dtype)

        # The carrier Hamiltonian needs a LONGER range than the trunk's message passing.
        #
        # Measured on this dataset: the two under-coordinated Pb that share the hole sit a
        # median 5.32 A apart, and the atom that would bridge them in bulk IS the vacancy. At
        # the trunk's 5.0 A cutoff only 36% of frames even have an edge between them, and the
        # smooth envelope gives those a median weight of 0.00000 -- so the physically correct
        # two-site state was not representable at all. The head's range is a property of the
        # carrier, not of the message passing; they coincided only because both reused r_max.
        #
        # The graph is therefore built at the head's cutoff and the trunk is filtered back to
        # r_max here. Filtering is an optimisation rather than a correctness fix: the radial
        # cutoff already sends contributions beyond r_max to exactly zero, so the trunk's
        # result is identical either way -- it just avoids paying for 8x the edges.
        head_species = data["node_attrs"].argmax(dim=-1)
        # R1 clamped-state probe. DIAGNOSTIC ONLY -- it consumes the vacancy assignment,
        # so it travels in the data dict under a private key rather than as a model
        # setting, and is absent from every production path.
        head_clamp = data.get("_clamp_mask", None)
        head_edge_index, head_lengths, head_vectors = (
            data["edge_index"], lengths, vectors)
        # getattr for the same reason as in _carrier_head: checkpoints pickled before these
        # attributes existed restore without them. Plain access here reintroduced exactly the
        # regression that broke loading every historical model an hour ago.
        if (getattr(self, "spectral_head", False)
                and float(getattr(self, "spectral_r_cut", 0.0)) > float(self.r_max)):
            keep = (lengths.reshape(-1) <= float(self.r_max)).nonzero(as_tuple=True)[0]
            data = dict(data)
            data["edge_index"] = head_edge_index[:, keep]
            vectors = vectors[keep]
            lengths = lengths[keep]
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

        # Embeddings, in the trunk's dtype
        trunk_attrs = data["node_attrs"].to(trunk_dtype)
        node_feats = self.node_embedding(trunk_attrs)
        edge_attrs = self.spherical_harmonics(vectors.to(trunk_dtype))
        edge_feats, cutoff = self.radial_embedding(
            lengths.to(trunk_dtype), trunk_attrs, data["edge_index"], self.atomic_numbers
        )

        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths.to(trunk_dtype), trunk_attrs, data["edge_index"],
                self.atomic_numbers
            ).to(vectors.dtype)
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

        _trunk_mark = mark("trunk")
        _trunk_mark.__enter__()
        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            if cache_hit and i > 0:
                # Everything after the first block is a function of the frozen base and the
                # geometry alone, and the head only differentiates through block 0.
                break
            node_attrs_slice = trunk_attrs
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
        _trunk_mark.__exit__(None, None, None)

        # Base branch readouts, and the invariant features the correction heads see.
        defect_feats_list: List[torch.Tensor] = []
        node_energy: Optional[torch.Tensor] = None
        if cache_hit:
            # Block 0's invariant readout is fresh (it carries the position gradient the
            # head's forces need); the later blocks' readouts come from the cache, and so
            # do the base energy and forces.
            defect_feats_list.append(
                self.defect_feature_readouts[0](node_feats_list[0].to(head_dtype)))
            if cached_feats_rest is not None:
                defect_feats_list.append(cached_feats_rest.to(head_dtype))
            inter_e = cached_energy - e0.to(head_dtype)
            base_energy = cached_energy
        else:
            for i, (readout, defect_readout) in enumerate(
                zip(self.readouts, self.defect_feature_readouts)
            ):
                feat_idx = -1 if len(self.readouts) == 1 else i
                node_es = readout(node_feats_list[feat_idx], node_heads)[
                    num_atoms_arange, node_heads
                ].to(vectors.dtype)
                node_es_list.append(node_es)
                defect_feats_list.append(
                    defect_readout(node_feats_list[feat_idx].to(head_dtype)))

            node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
            node_inter_es = self.scale_shift(node_inter_es, node_heads)
            inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1,
                                  dim_size=num_graphs)

            base_energy = (e0 + inter_e).to(head_dtype)
            node_energy = to_high_precision(node_e0.clone()) + to_high_precision(
                node_inter_es.clone()
            )
        node_feats_out = torch.cat(node_feats_list, dim=-1)
        trunk_block0 = node_feats_list[0].detach()
        graph_sizes = data["ptr"][1:] - data["ptr"][:-1]
        centre = self.pristine_centre(head_dtype)
        # The head's geometry, in the head's dtype. A no-op under "uniform".
        # The head sees the geometry in its own dtype through a DIFFERENTIABLE cast; the
        # gradient leaf stays `positions`, whatever the data dtype. Rebinding the name to the
        # cast copy made every force derivative below blind to the trunk on float32 batches.
        head_positions = (positions if positions.dtype == head_dtype
                          else positions.to(head_dtype))
        head_cell = data["cell"].to(head_dtype)
        head_lengths = head_lengths.to(head_dtype)
        head_vectors = head_vectors.to(head_dtype)

        # Carrier correction. The counters are canonicalised at data loading and at every
        # inference entry point, so the network never sees a non-canonical vector.
        defect_feats = torch.cat(defect_feats_list, dim=-1)
        counts = data["carrier_counts"].view(num_graphs, -1).to(head_dtype)
        # The counter the paired difference is measured from. Absent (zeros) is the
        # closed-shell reference, for which the correction vanishes identically -- so a
        # dataset without reference counters behaves exactly as before.
        if "carrier_counts_ref" in data:
            counts_ref = data["carrier_counts_ref"].view(num_graphs, -1).to(head_dtype)
        else:
            counts_ref = torch.zeros_like(counts)
        # The reference counter's state batch (section 2.1); `state` was built above. The
        # reference question below is asked of the PHYSICAL KEY against S_ref -- not of
        # the counter being zero.
        state_ref = StateBatch.from_counts(counts_ref, policy_key)
        s_ref = ElectronicStateSpec.from_dict(self.reference_state)

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
        # Section 1. The head's force response is requested only on a TRAINING force pass:
        # it is exactly zero in value, so an evaluation pass would pay a create_graph backward
        # for nothing. `None` means "not requested"; an empty dict that comes back empty means
        # "requested, and the fill equals the reference", which is a real zero.
        want_response = bool(training and compute_force)
        force_out: Optional[Dict[str, torch.Tensor]] = {} if want_response else None
        force_out_ref: Optional[Dict[str, torch.Tensor]] = {} if want_response else None
        (
            delta_sr,
            alpha,
            carrier_readouts,
            logit_gap,
            delta_u,
            carrier_logits,
            head_extras,
        ) = self._carrier_head(
            node_feats=defect_feats,
            counter_emb=counter_emb,
            counts=counts,
            batch=data["batch"],
            num_graphs=num_graphs,
            edge_index=head_edge_index,
            edge_length=head_lengths,
            node_species=head_species,
            clamp_mask=head_clamp,
            edge_vector=head_vectors,
            logit_bias=logit_bias,
            positions=head_positions,
            cell=head_cell,
            force_out=force_out,
            centre=centre,
            graph_sizes=graph_sizes,
            state=state,
        )
        # Intrinsic gap: the same pooling with the seed switched off, so the logged gap
        # separates what MLP_l has learned from what the seed is supplying. The dead
        # channel is a decent proxy for the seed baseline (its logit weights are frozen
        # by n_c = 0) but not an exact one, since its trunk inputs still move; this is
        # the exact version. Costs one MLP pass, no trunk work.
        logit_gap_intrinsic: Optional[torch.Tensor] = None
        logits_intrinsic: Optional[torch.Tensor] = None
        if self.logit_seed:
            (_, _, _, logit_gap_intrinsic, _, logits_intrinsic,
             _) = self._carrier_head(
                node_feats=defect_feats,
                counter_emb=counter_emb,
                counts=counts,
                batch=data["batch"],
                num_graphs=num_graphs,
                edge_index=data["edge_index"],
                edge_length=lengths,
                logit_bias=None,
            )

        # Level-mode gauge probe (forward plan, stage D-opt). The softmax is
        # shift-invariant, so a uniform offset in u^c is a free direction; on a pristine
        # cell the pooled readout at a training counter must be zero, because a carrier
        # there is a band-to-band transition. Evaluated at every training counter from the
        # trunk output already computed -- a few readout MLP passes, no extra message
        # passing, and no new frames. The loss masks this down to pristine cells; it is
        # computed for all of them so the diagnostic is available whether or not the
        # penalty is switched on.
        gauge_mean_u: Optional[torch.Tensor] = None
        if self.gauge_counters.numel() > 0:
            gauge_list: List[torch.Tensor] = []
            for index in range(self.gauge_counters.shape[0]):
                probe_counts = self.gauge_counters[index].to(counts.dtype).unsqueeze(0)
                probe_counts = probe_counts.expand(num_graphs, -1).contiguous()
                _, probe_alpha, probe_u, _, _, _ = self.carrier_pooling(
                    node_feats=defect_feats,
                    counter_emb=self.counter_embedding(probe_counts),
                    counts=probe_counts,
                    batch=data["batch"],
                    num_graphs=num_graphs,
                    logit_bias=logit_bias,
                )
                gauge_list.append(
                    scatter_sum(
                        src=probe_alpha * probe_u,
                        index=data["batch"],
                        dim=0,
                        dim_size=num_graphs,
                    )
                )
            gauge_mean_u = torch.stack(gauge_list, dim=1)  # [n_graphs, K, 4]

        # SECTION 1.1, THE REFERENCE BRANCH. When the reference counter is the neutral one
        # -- which it is unless the caller supplied `carrier_counts_ref` -- every term of
        # the reference branch is EXACTLY zero for the counting head, and skipping it is an
        # identity rather than an approximation:
        #
        #   * `delta_sr_ref`: the head's energy is a difference between the fill and the
        #     neutral reference fill. At `counts_ref = 0` the two fills are the same numbers,
        #     so the occupations subtract to exactly zero, `D = 0`, and both the trace and
        #     the entropy difference vanish identically. The head declares this as
        #     `zero_at_neutral_counts`; `test_neutral_reference_skip.py` measures it.
        #   * `q^pol` at zero counts is zero in all three of its branches (the ungated one
        #     carries a `total_carriers` factor, the gated one a gate built from the same
        #     counts, and the disabled one is zero by construction), and `q^carrier` carries
        #     `carrier_signs * counts`. So `latent_charge_ref = q_host` exactly, and
        #     `delta_lr_ref` is `E[0]` in the uncoupled branch and `E[q_host] - E[q_host]`
        #     in the coupled one. Both are zero, with zero position gradient -- E is
        #     quadratic in q, so `dE/dR` at `q = 0` is zero too.
        #   * `_isolated_carrier_self` multiplies by the same signed counts.
        #
        # What it saves is a whole head pass (eigensolve, fills, Madelung), an Ewald
        # evaluation, and the `correction_energy_ref` force gradient -- 18% of the step.
        #
        # THE GATE IS ON THE REFERENCE STATE, NOT ON WHETHER THE KEY IS PRESENT.
        # `prepare_defect_configurations` writes `carrier_counts_ref` onto EVERY paired
        # frame -- it records which member of the pair the difference is taken against --
        # so keying the skip on the field's absence never fires on real training data. The
        # condition the proof above needs is that every graph's reference state IS S_ref,
        # and under plan v8 that is equality of the physical key (section 2.1): the same
        # policy, the same payload, `Q_formal = Q_ref` and `delta_n = 0`. For the counter
        # adapter that is `e_maj = h_maj` and `e_min = h_min`, which the old `counts == 0`
        # test implied but did not state. One host synchronisation per forward.
        skip_reference = (
            bool(getattr(self, "skip_neutral_reference", True))
            and getattr(getattr(self, "spectral", None), "zero_at_neutral_counts", False)
            and state_ref.all_reference(s_ref)
        )
        counter_emb_ref = self.counter_embedding(counts_ref)
        if skip_reference:
            delta_sr_ref = torch.zeros_like(delta_sr)
            alpha_ref = torch.zeros_like(alpha)
        else:
            delta_sr_ref, alpha_ref, _, _, _, _, _ = self._carrier_head(
                node_feats=defect_feats,
                counter_emb=counter_emb_ref,
                counts=counts_ref,
                batch=data["batch"],
                num_graphs=num_graphs,
                edge_index=head_edge_index,
                edge_length=head_lengths,
                node_species=head_species,
                clamp_mask=head_clamp,
                edge_vector=head_vectors,
                logit_bias=logit_bias,
                positions=head_positions,
                cell=head_cell,
                force_out=force_out_ref,
                centre=centre,
                graph_sizes=graph_sizes,
                state=state_ref,
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
        # Section 2.5's registry reads the trunk's own energy as the `base` term, before
        # the host long-range term is folded into `base_energy` below.
        base_trunk_energy = base_energy
        # Section 2.6: the model's gauge selects the kernel; `dilute=True` is the call-site
        # form of the isolated gauge and is kept for the scorers that pass it.
        dilute = bool(dilute) or getattr(self, "gauge", "periodic") == "isolated"

        if self.use_long_range and int(self.current_epoch) >= self.lr_start_epoch:
            _lr_mark = mark("ewald/lr")
            _lr_mark.__enter__()
            # A null cell selects the isolated evaluator inside LES, which is how
            # non-periodic configurations are handled.
            cell_les = head_cell.clone()
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
                edge_index=data["edge_index"],
                edge_lengths=lengths.to(head_dtype),
            )
            # PLAN v8 SECTION 3: NOTHING HERE IS DETACHED. The v6 branch detached the
            # charges under `lr_detach_density` so that E_LR's forces were dE_LR/dR at
            # fixed q; the plan retires detached densities outright (section 2.5, "no
            # detached quantity anywhere"). What still cuts the density's gradient is the
            # head's own detached D, which Stage 1's response-density forces replace.
            #
            # The neutrality check is a CHECK, not a term, and runs on detached copies:
            # the density-matrix difference sums to -Delta_n exactly, so the carrier charge
            # sums to -a * Delta_n; the spec's own `|sum q + Delta_n|` holds only at a = 1
            # and is kept as `lr_neutrality_residual` for the run log.
            delta_n = (counts[:, 0] + counts[:, 1] - counts[:, 2] - counts[:, 3])
            q_sum = scatter_sum(latent_charge.detach(), data["batch"], dim=0,
                                dim_size=num_graphs)
            self._lr_neutrality_residual = (q_sum + delta_n).abs().max().detach()
            if amplitude is not None:
                dens_sum = scatter_sum(q_carrier.detach(), data["batch"], dim=0,
                                       dim_size=num_graphs) / amplitude.detach().clamp_min(1e-12)
                bad = (dens_sum + delta_n).abs() > 1e-6
                if bool(bad.any()):
                    raise RuntimeError(
                        "the carrier density does not sum to -Delta_n: "
                        f"max |sum q_carrier/a + Delta_n| = "
                        f"{float((dens_sum + delta_n).abs().max()):.3e}")
            energy_lr_host = self.latent_ewald.energy(
                q_host, head_positions, cell_les, data["batch"]
            )
            # At n = 0 the polarisation and carrier channels vanish identically, so the
            # two evaluations see the same charges and this difference is exactly zero.
            base_energy = base_energy + energy_lr_host
            if self.host_carrier_coupling:
                delta_lr = (
                    self.latent_ewald.energy(
                        latent_charge, head_positions, cell_les, data["batch"]
                    )
                    - energy_lr_host
                )
            else:
                # Drop the cross term between q^host and the carrier cloud. Since E_LR is
                # quadratic, E(host + d) - E(host) = E(d) + 2B(host, d), so keeping only
                # E(d) removes exactly that coupling. Both branches cost two Ewald
                # evaluations; the total-charge one is raised inside the coupled branch so
                # the uncoupled path does not pay for a result it discards.
                #
                # The measurement behind this: pooling both energies under a hand-set
                # attention gives d(host.carrier) = -1.25 eV in favour of the Cs
                # sublattice against 0.18 eV of short-range difference, size-independent
                # to 18 meV over a 2.25x range. The cross term has the same pooling form
                # as Delta E_SR = sum_c n_c <u>_alpha -- same alpha, one field learned and
                # one fixed-shape -- so the two are degenerate and the electrostatic one
                # wins by 7x. Worse, a classical point-charge potential is deepest at
                # CATION sites, so it drags the hole onto Cs; no rescaling of q^host or a
                # can fix a term that points the wrong way.
                #
                # The carrier's interaction with the host's own short-ranged Madelung
                # field belongs in u_i, which is a learned per-site energy on the same
                # sites and can represent it. The long-range branch keeps only what
                # Delta E_SR structurally cannot do: the monopole self-interaction and
                # interactions between separated carriers.
                delta_lr = self.latent_ewald.energy(
                    latent_charge - q_host, head_positions, cell_les, data["batch"]
                )

            # The same at the reference counter. This branch is *not* inert at q = 0:
            # q^carrier is a compensated but pointwise non-zero charge whose self-term is
            # the electron-hole interaction, and q^pol carries a factor sum_c n_c. Both
            # are live whenever the reference state itself carries carriers -- which is
            # exactly what `skip_reference` above tests for, and why the skip is gated on
            # the caller NOT having supplied a reference counter of its own.
            if skip_reference:
                # `latent_charge_ref = q_host` exactly (see above), so both branches below
                # evaluate the Ewald energy of an identically zero charge. Written as the
                # zero it is, rather than paying an Ewald call to arrive at it.
                latent_charge_ref = q_host
                delta_lr_ref = torch.zeros_like(base_energy)
            else:
                latent_charge_ref, _, _, _, _ = self.latent_charges(
                    node_feats=defect_feats,
                    counter_emb=counter_emb_ref,
                    counts=counts_ref,
                    alpha=alpha_ref,
                    batch=data["batch"],
                    num_graphs=num_graphs,
                    edge_index=data["edge_index"],
                    edge_lengths=lengths,
                )
            if skip_reference:
                pass
            elif self.host_carrier_coupling:
                delta_lr_ref = (
                    self.latent_ewald.energy(
                        latent_charge_ref, head_positions, cell_les, data["batch"]
                    )
                    - energy_lr_host
                )
            else:
                delta_lr_ref = self.latent_ewald.energy(
                    latent_charge_ref - q_host, head_positions, cell_les, data["batch"]
                )

            if self.carrier_self_isolated:
                # E_LR = E_periodic[Q] - sum_c E_isolated[Q^c]: only the image interaction
                # survives, with 1/L monopole scaling by construction.
                delta_lr = delta_lr - self._isolated_carrier_self(
                    alpha, counts, amplitude, head_positions, data["batch"], num_graphs
                )
                if not skip_reference:
                    # Zero at zero counts by the same `carrier_signs * counts` factor.
                    delta_lr_ref = delta_lr_ref - self._isolated_carrier_self(
                        alpha_ref, counts_ref, amplitude, head_positions,
                        data["batch"], num_graphs
                    )

            if dilute:
                dilute_correction = self.latent_ewald.dilute_correction(
                    q_carrier, head_positions, cell_les, data["batch"], num_graphs
                )
                delta_lr = delta_lr + dilute_correction
            _lr_mark.__exit__(None, None, None)

        # The correction at this frame's own counter is what the total energy carries;
        # the paired difference is what the delta labels supervise. They coincide only
        # when the reference is the closed-shell state.
        # Edit 2. The carrier-field response channel is GONE, deleted in the same commit that
        # landed Edit 1 -- they are the same physics, and running both would count the
        # carrier's electrostatics twice.
        #
        # What it measured is kept in the ledger: it confirmed that electrostatics has the
        # ~6 A reach the force footprint needs (clause 1), and that it does not select a site
        # (clause 2). Clause 2 is not a shortcoming of the implementation. E_resp was an
        # energy readout, so V could change the energy but never H, and a term that cannot
        # change the Hamiltonian cannot change where the carrier goes. Edit 1 puts the same
        # potential on the on-site energies, where the eigenproblem sees it.
        correction_energy = delta_sr + delta_lr
        correction_energy_ref = delta_sr_ref + delta_lr_ref
        delta_energy = correction_energy - correction_energy_ref
        total_energy = base_energy + correction_energy

        # Correction forces first: they differentiate the same graph the total-energy pass
        # below consumes, and get_outputs is free to release it.
        correction_forces: Optional[torch.Tensor] = None
        delta_forces: Optional[torch.Tensor] = None
        # Section 1. Exactly zero in value, so every force below is bit-identical with and
        # without it; what it carries is the density response in the parameter gradient. Added
        # to BOTH branches -- the reference counter need not be the closed shell (a caller can
        # supply `carrier_counts_ref`), and a response added only to the near side would put
        # an asymmetric term into `delta_forces`.
        response = None if force_out is None else force_out.get("force_response")
        response_ref = (None if force_out_ref is None
                        else force_out_ref.get("force_response"))
        if compute_force:
            with mark("model/grad_corr"):
                correction_forces = _energy_gradient(correction_energy, positions,
                                                     training)
            with mark("model/grad_corr_ref"):
                # `correction_energy_ref` is an exact zero with no graph under the skip, so
                # its gradient is the zero vector; taking it through autograd would either
                # cost a full backward or fail for an unused input.
                correction_forces_ref = (
                    torch.zeros_like(positions) if skip_reference
                    else _energy_gradient(correction_energy_ref, positions, training))
            if response is not None:
                correction_forces = correction_forces + response
            if response_ref is not None:
                correction_forces_ref = correction_forces_ref + response_ref
            delta_forces = correction_forces - correction_forces_ref

        _out_mark = mark("model/outputs")
        _out_mark.__enter__()
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
        _out_mark.__exit__(None, None, None)

        # The base-branch forces are what L_base is trained against. They come for free
        # as the difference: the total energy is the sum of the two branches, and the
        # gradient is linear. Note this subtracts the correction at *this* frame's
        # counter, not the paired difference -- the two agree only when the reference is
        # the closed-shell state, which is exactly when base labels exist.
        if forces is not None and response is not None:
            forces = forces + response
        base_forces: Optional[torch.Tensor] = None
        if forces is not None and correction_forces is not None:
            # Both terms gained the same response, so the base branch is untouched -- which is
            # correct: the response belongs to the head, and L_base must not see it.
            base_forces = forces - correction_forces
        if cache_hit and forces is not None:
            # `inter_e` was a constant above, so the derivative pass carried only the
            # long-range host term and the correction; the trunk's own forces are the
            # cached ones.
            forces = forces + cached_forces
            if base_forces is not None:
                base_forces = base_forces + cached_forces

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
            "delta_lr_ref_energy": delta_lr_ref,
            # Section 2.5: the registered terms, each under its own key. `base_trunk_energy`
            # is E_base without the host long-range term; `energy_lr_host` is that term;
            # `delta_lr_energy` is the carrier's long-range energy. The band term is
            # `delta_sr_energy` above.
            "base_trunk_energy": base_trunk_energy,
            "energy_lr_host": energy_lr_host,
            "delta_lr_energy": delta_lr,
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
            # T5: the per-frame mean site energy, so the loss can pin the gauge with a
            # penalty rather than by subtraction (which made lambda size-dependent).
            "carrier_eps_mean": head_extras.get("eps_mean"),
            # Section 2.1: the first block's trunk features, for `set_pristine_centre`.
            "trunk_block0": trunk_block0,
            # The exact inputs the correction readouts consume, exposed so that seeding
            # and diagnostics do not have to re-derive the trunk (defect_seed.py).
            "defect_features": defect_feats,
            "counter_embedding": counter_emb,
            "carrier_readouts": carrier_readouts,
            # Post-clamp, post-bias: the tensor the softmax actually consumed. The size
            # term re-forms the normalisation for a hypothetically larger cell, which
            # alpha cannot support because it has already divided the denominator out.
            "carrier_logits": carrier_logits,
            # [n_graphs, K, 4]: pooled u at each training counter. Present for every frame;
            # the loss restricts it to pristine ones, where band-edge referencing pins it
            # to zero. None unless gauge counters were configured.
            "gauge_mean_u": gauge_mean_u,
            "logit_gap": logit_gap,
            "delta_u": delta_u,
            "logit_gap_intrinsic": logit_gap_intrinsic,
            # Needed to gate the seed anneal on SITE structure rather than on
            # gap magnitude, which cannot tell a species gap from a site gap.
            "carrier_logits_intrinsic": logits_intrinsic,
            "latent_charges": latent_charge,
            "latent_charges_host": q_host,
            "latent_charges_carrier": q_carrier,
            "polarisation": polarisation,
            "screening_amplitude": amplitude,
            "dilute_correction": dilute_correction,
        }
