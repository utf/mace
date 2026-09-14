import ast
import logging

import numpy as np
import torch
from e3nn import o3

from mace import modules
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.finetuning_utils import load_foundations_elements, load_foundations_mdp
from mace.tools.scripts_utils import extract_config_mace_model, resolve_m_max
from mace.tools.torch_tools import dtype_dict
from mace.tools.utils import AtomicNumberTable



def _scale_les_readouts(model, scale: float) -> None:
    """Multiply the last layer of each latent-charge readout by `scale` at construction.

    **ZERO DOES NOT MEAN "START SMALL AND GROW".** The Ewald energy is QUADRATIC in the latent
    charges, so `q = 0` is a stationary point of it: `dE_lr/dq = 0` there, and therefore
    `dL/dW = dL/dE_lr . dE_lr/dq . dq/dW` vanishes identically however large the loss gradient
    is. Measured on a trained model: with these weights at zero the gradient reaching them is
    exactly 0.000000e+00 while the energy readouts receive 9.03; perturbed by 1e-3 it becomes
    9.48e-03. A zeroed latent-charge readout is PINNED -- two models trained that way for 300
    and 320 epochs came out with their charges still exactly zero and LES contributing nothing.

    Use a SMALL NON-ZERO scale to start the long-range term small and LIVE. Zero is correct
    only when the intent is to disable LES outright.
    """
    with torch.no_grad():
        for readout in model.les_readouts:
            final = getattr(readout, "linear_2", None) or getattr(readout, "linear", None)
            if final is None:
                continue
            for param in final.parameters():
                param.mul_(scale)

def configure_model(
    args,
    train_loader,
    atomic_energies,
    model_foundation=None,
    heads=None,
    z_table=None,
    head_configs=None,
):
    # Selecting outputs
    compute_virials = args.loss == "virials"
    compute_stress = args.loss in ("stress", "huber", "universal")

    if compute_virials:
        args.compute_virials = True
    elif compute_stress:
        args.compute_stress = True

    if compute_virials or compute_stress:
        if args.error_table in ["PerAtomRMSE", "PerAtomMAE", "TotalRMSE", "TotalMAE"]:
            args.error_table = (
                "PerAtomRMSEstressvirials"
                if "RMSE" in args.error_table
                else "PerAtomMAEstressvirials"
            )

    output_args = {
        "energy": args.compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": compute_stress,
        "dipoles": args.compute_dipole,
        "polarizabilities": args.compute_polarizability,
        "magforces": args.compute_magforces,
    }
    logging.info(
        f"During training the following quantities will be reported: {', '.join([f'{report}' for report, value in output_args.items() if value])}"
    )
    logging.info("===========MODEL DETAILS===========")

    if args.scaling == "no_scaling":
        args.std = 1.0
        if head_configs is not None:
            for head_config in head_configs:
                head_config.std = 1.0
        logging.info("No scaling selected")

    if (
        head_configs is not None
        and args.std is not None
        and not isinstance(args.std, list)
    ):
        atomic_inter_scale = []
        for head_config in head_configs:
            if hasattr(head_config, "std") and head_config.std is not None:
                atomic_inter_scale.append(head_config.std)
            elif args.std is not None:
                atomic_inter_scale.append(
                    args.std if isinstance(args.std, float) else 1.0
                )
        args.std = atomic_inter_scale

    elif (args.mean is None or args.std is None) and (
        args.model not in ("AtomicDipolesMACE", "AtomicDielectricMACE")
    ):
        args.mean, args.std = modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
        )
    if args.embedding_specs is not None:
        logging.info("Using embedding specifications from command line arguments")
        logging.info(f"Embedding specifications: {args.embedding_specs}")
    # Build model
    if model_foundation is not None and args.model in [
        "MACE",
        "ScaleShiftMACE",
        "MACELES",
        "PolarMACE",
    ]:
        logging.info("Loading FOUNDATION model")
        model_config_foundation = extract_config_mace_model(model_foundation)
        model_config_foundation["atomic_energies"] = atomic_energies

        if args.foundation_model_elements:
            foundation_z_table = AtomicNumberTable(
                [int(z) for z in model_foundation.atomic_numbers]
            )
            model_config_foundation["atomic_numbers"] = foundation_z_table.zs
            model_config_foundation["num_elements"] = len(foundation_z_table)
            z_table = foundation_z_table
            logging.info(
                f"Using all elements from foundation model: {foundation_z_table.zs}"
            )
        else:
            model_config_foundation["atomic_numbers"] = z_table.zs
            model_config_foundation["num_elements"] = len(z_table)
            logging.info(f"Using filtered elements: {z_table.zs}")

        args.max_L = model_config_foundation["hidden_irreps"].lmax

        if args.model in (
            "ScaleShiftMACE",
            "PolarMACE",
            "MagneticScaleShiftMACE",
        ) or model_foundation.__class__.__name__ in (
            "ScaleShiftMACE",
            "PolarMACE",
            "MagneticScaleShiftMACE",
        ):
            model_config_foundation["atomic_inter_shift"] = (
                _determine_atomic_inter_shift(args.mean, heads)
            )
        else:
            model_config_foundation["atomic_inter_shift"] = [0.0] * len(heads)
        model_config_foundation["atomic_inter_scale"] = [1.0] * len(heads)
        args.avg_num_neighbors = model_config_foundation["avg_num_neighbors"]
        if args.model == "MACELES":
            args.model = "FoundationMACELES"
        elif args.model in ("MACE", "ScaleShiftMACE"):
            args.model = "FoundationMACE"
        model_config_foundation["heads"] = heads
        model_config = model_config_foundation

        logging.info("Model configuration extracted from foundation model")
        logging.info(f"Using {args.loss} loss function for fine-tuning")
        logging.info(
            f"Message passing with hidden irreps {model_config_foundation['hidden_irreps']})"
        )
        logging.info(
            f"{model_config_foundation['num_interactions']} layers, each with correlation order: {model_config_foundation['correlation']} (body order: {model_config_foundation['correlation']+1}) and spherical harmonics up to: l={model_config_foundation['max_ell']}"
        )
        logging.info(
            f"Radial cutoff: {model_config_foundation['r_max']} A (total receptive field for each atom: {model_config_foundation['r_max'] * model_config_foundation['num_interactions']} A)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {model_config_foundation['distance_transform']}"
        )
    else:
        logging.info("Building model")
        logging.info(
            f"Message passing with {args.num_channels} channels and max_L={args.max_L} ({args.hidden_irreps})"
        )
        logging.info(
            f"{args.num_interactions} layers, each with correlation order: {args.correlation} (body order: {args.correlation+1}) and spherical harmonics up to: l={args.max_ell}"
        )
        logging.info(
            f"{args.num_radial_basis} radial and {args.num_cutoff_basis} basis functions"
        )
        logging.info(
            f"Radial cutoff: {args.r_max} A (total receptive field for each atom: {args.r_max * args.num_interactions} A)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {args.distance_transform}"
        )

        assert (
            len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
        ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

        logging.info(f"Hidden irreps: {args.hidden_irreps}")

        cueq_config = None
        if args.only_cueq:
            logging.info("Using only the backend of the model")
            cueq_config = CuEquivarianceConfig(
                enabled=True,
                layout="ir_mul",
                group="O3_e3nn",
                optimize_all=True,
                conv_fusion=(args.device == "cuda"),
            )

        model_config = dict(
            r_max=args.r_max,
            num_bessel=args.num_radial_basis,
            num_polynomial_cutoff=args.num_cutoff_basis,
            max_ell=args.max_ell,
            interaction_cls=modules.interaction_classes[args.interaction],
            num_interactions=args.num_interactions,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps(args.hidden_irreps),
            edge_irreps=o3.Irreps(args.edge_irreps) if args.edge_irreps else None,
            atomic_energies=atomic_energies,
            apply_cutoff=args.apply_cutoff,
            avg_num_neighbors=args.avg_num_neighbors,
            atomic_numbers=z_table.zs,
            use_reduced_cg=args.use_reduced_cg,
            use_so3=args.use_so3,
            use_edge_irreps_first=args.use_edge_irreps_first,
            cueq_config=cueq_config,
        )
        model_config_foundation = None

    model = _build_model(args, model_config, model_config_foundation, heads)

    if model_foundation is not None:
        if getattr(args, "finetune_dipoles_polarizabilities", False):
            # MDP fine-tuning: dedicated loader that handles higher-order irreps
            load_foundations_mdp(model, model_foundation, z_table, max_L=args.max_L)
        else:
            model = load_foundations_elements(
                model,
                model_foundation,
                z_table,
                load_readout=args.foundation_filter_elements,
                max_L=args.max_L,
                default_dtype=dtype_dict.get(args.default_dtype, torch.float64),
            )

    return model, output_args


def _determine_atomic_inter_shift(mean, heads):
    if isinstance(mean, np.ndarray):
        if mean.size == 1:
            return mean.item()
        if mean.size == len(heads):
            return mean.tolist()
        logging.info("Mean not in correct format, using default value of 0.0")
        return [0.0] * len(heads)
    if isinstance(mean, list) and len(mean) == len(heads):
        return mean
    if isinstance(mean, float):
        return [mean] * len(heads)
    logging.info("Mean not in correct format, using default value of 0.0")
    return [0.0] * len(heads)


def _parse_literal_or_none(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in ("none", ""):
            return None
        return ast.literal_eval(stripped)
    return value


def _build_model(
    args, model_config, model_config_foundation, heads
):  # pylint: disable=too-many-return-statements

    if args.model == "MagneticScaleShiftMACE":
        m_max = resolve_m_max(args.m_max, list(model_config["atomic_numbers"]))
        return modules.MagneticScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=_determine_atomic_inter_shift(args.mean, heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            m_max=m_max,
            max_m_ell=args.max_m_ell,
            num_mag_radial_basis=args.num_mag_radial_basis,
            num_mag_radial_basis_one_body=args.num_mag_radial_basis_one_body,
            use_magmom_one_body=args.use_magmom_one_body,
        )
    if args.model == "MACE":
        if args.interaction_first not in [
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticResidualNonLinearInteractionBlock",
        ]:
            args.interaction_first = "RealAgnosticInteractionBlock"
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=[0.0] * len(heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
        )
    if args.model == "ScaleShiftMACE":
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=args.mean,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
        )
    if args.model == "PolarMACE" and model_config_foundation is not None:
        return modules.PolarMACE(**model_config_foundation)
    if args.model == "PolarMACE":
        field_feature_widths = _parse_literal_or_none(args.field_feature_widths)
        field_feature_norms = _parse_literal_or_none(args.field_feature_norms)
        fixedpoint_update_config = _parse_literal_or_none(args.fixedpoint_update_config)
        field_readout_config = _parse_literal_or_none(args.field_readout_config)
        return modules.PolarMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=_determine_atomic_inter_shift(args.mean, heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
            kspace_cutoff_factor=args.kspace_cutoff_factor,
            atomic_multipoles_max_l=args.atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
            field_feature_max_l=args.field_feature_max_l,
            field_feature_widths=(
                field_feature_widths if field_feature_widths is not None else [1.0]
            ),
            num_recursion_steps=args.num_recursion_steps,
            field_si=args.field_si,
            include_electrostatic_self_interaction=args.include_electrostatic_self_interaction,
            add_local_electron_energy=args.add_local_electron_energy,
            quadrupole_feature_corrections=args.quadrupole_feature_corrections,
            return_electrostatic_potentials=args.return_electrostatic_potentials,
            field_feature_norms=field_feature_norms,
            field_norm_factor=args.field_norm_factor,
            fixedpoint_update_config=fixedpoint_update_config,
            field_readout_config=field_readout_config,
        )
    if args.model == "FoundationMACE":
        return modules.ScaleShiftMACE(**model_config_foundation)
    if args.model == "FoundationMACELES":
        from mace.modules.extensions import MACELES

        model = MACELES(
            les_arguments=args.les_arguments,
            **model_config_foundation,
        )
        # ZERO THE LATENT-CHARGE READOUTS when starting from a foundation model.
        #
        # `load_foundations_elements` transfers the backbone and the ENERGY readouts but not
        # `les_readouts`, which are new modules with a fresh initialisation -- so the Ewald
        # term starts as a random long-range energy on top of an already-good potential.
        # Measured on MH-1 + CsPbCl3: the initial validation error was 8332 meV/atom and
        # 1474 meV/A, against ~44 meV/A for the same backbone without LES. That is not the
        # E0 mismatch the fine-tuning guide's ">500 meV/atom" rule is about (the same E0s
        # give ~200 meV/atom from scratch); it is the untrained charges.
        #
        # `--les_readout_scale` controls it: 1.0 (default) leaves the library initialisation
        # alone; a SMALL value starts the long-range term small and live. ZERO PINS IT -- see
        # `_scale_les_readouts` -- and was the mistake that made two whole runs no-ops.
        _scale_les_readouts(model, float(getattr(args, "les_readout_scale", 1.0)))
        return model
    if args.model == "ScaleShiftBOTNet":
        # say it is deprecated
        raise RuntimeError("ScaleShiftBOTNet is deprecated, use MACE instead")
    if args.model == "BOTNet":
        raise RuntimeError("BOTNet is deprecated, use MACE instead")
    if args.model == "AtomicDipolesMACE":
        assert args.loss == "dipole", "Use dipole loss with AtomicDipolesMACE model"
        assert (
            args.error_table == "DipoleRMSE"
        ), "Use error_table DipoleRMSE with AtomicDipolesMACE model"
        return modules.AtomicDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )

    if args.model == "AtomicDielectricMACE":
        args.error_table = "DipolePolarRMSE"
        # std_df = modules.scaling_classes["rms_dipoles_scaling"](train_loader)
        assert (
            args.loss == "dipole_polar"
        ), "Use dipole_polar loss with AtomicDielectricMACE model"
        assert args.error_table in (
            "DipoleRMSE",
            "DipolePolarRMSE",
        ), "Use error_table DipoleRMSE with AtomicDielectricMACE model"
        return modules.AtomicDielectricMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            use_polarizability=True,
        )

    if args.model == "EnergyDipolesMACE":
        assert (
            args.loss == "energy_forces_dipole"
        ), "Use energy_forces_dipole loss with EnergyDipolesMACE model"
        assert (
            args.error_table == "EnergyDipoleRMSE"
        ), "Use error_table EnergyDipoleRMSE with AtomicDipolesMACE model"
        return modules.EnergyDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    if args.model == "MACELES":
        from mace.modules.extensions import MACELES

        model = MACELES(
            les_arguments=args.les_arguments,
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            # The FITTED shift, as the ScaleShiftMACE branch uses, not a hard 0. With
            # `rms_forces_scaling` the shift is the training set's mean per-atom interaction
            # energy; discarding it starts the model that far from its target -- measured on
            # this data, initial validation energy RMSE 1870 meV/atom against 205 for the same
            # architecture WITH the shift -- which is a handicap the LES term has nothing to
            # do with, and which confounds any LES-versus-plain comparison at equal epochs.
            atomic_inter_shift=args.mean,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
        )
        scale = 0.0 if getattr(args, "les_zero_readouts", False) else float(
            getattr(args, "les_readout_scale", 1.0))
        if scale != 1.0:
            _scale_les_readouts(model, scale)
        return model
    raise RuntimeError(f"Unknown model: '{args.model}'")
