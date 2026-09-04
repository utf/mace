###########################################################################################
# Parsing functionalities
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import os
from typing import Dict, Optional

from .default_keys import DefaultKeys


def build_default_arg_parser() -> argparse.ArgumentParser:
    try:
        import configargparse

        parser = configargparse.ArgumentParser(
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add(
            "--config",
            type=str,
            is_config_file=True,
            help="config file to aggregate options",
        )
    except ImportError:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    # Name and seed
    parser.add_argument("--name", help="experiment name", required=True)
    parser.add_argument("--seed", help="random seed", type=int, default=123)

    # Directories
    parser.add_argument(
        "--work_dir",
        help="set directory for all files and folders",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--log_dir", help="directory for log files", type=str, default=None
    )
    parser.add_argument(
        "--model_dir", help="directory for final model", type=str, default=None
    )
    parser.add_argument(
        "--checkpoints_dir",
        help="directory for checkpoint files",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--results_dir", help="directory for results", type=str, default=None
    )
    parser.add_argument(
        "--downloads_dir", help="directory for downloads", type=str, default=None
    )

    # Device and logging
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument(
        "--distributed",
        help="train in multi-GPU data parallel mode",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--launcher",
        default="slurm",
        choices=["slurm", "torchrun", "mpi", "none"],
        help="How the job was launched",
    )
    parser.add_argument("--log_level", help="log level", type=str, default="INFO")

    parser.add_argument(
        "--plot",
        help="Plot results of training",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--plot_frequency",
        help="Set plotting frequency: '0' for only at the end or an integer N to plot every N epochs.",
        type=int,
        default="0",
    )

    parser.add_argument(
        "--plot_interaction_e",
        help="Whether to plot energy without E0s",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--error_table",
        help="Type of error table produced at the end of the training",
        type=str,
        choices=[
            "PerAtomRMSE",
            "TotalRMSE",
            "PerAtomRMSEstressvirials",
            "PerAtomMAEstressvirials",
            "PerAtomMAE",
            "TotalMAE",
            "DipoleRMSE",
            "DipoleMAE",
            "DipolePolarRMSE",
            "EnergyDipoleRMSE",
            "DefectRMSE",
        ],
        default="PerAtomRMSE",
    )

    # Model
    parser.add_argument(
        "--model",
        help="model type",
        default="MACE",
        choices=[
            "BOTNet",
            "MACE",
            "ScaleShiftMACE",
            "PolarMACE",
            "MACELES",
            "MACEDefect",
            "ScaleShiftBOTNet",
            "AtomicDipolesMACE",
            "AtomicDielectricMACE",
            "EnergyDipolesMACE",
            "MagneticScaleShiftMACE",
        ],
    )
    parser.add_argument(
        "--r_max", help="distance cutoff (in Ang)", type=float, default=5.0
    )
    parser.add_argument(
        "--radial_type",
        help="type of radial basis functions",
        type=str,
        default="bessel",
        choices=["bessel", "gaussian", "chebyshev"],
    )
    parser.add_argument(
        "--num_radial_basis",
        help="number of radial basis functions",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_cutoff_basis",
        help="number of basis functions for smooth cutoff",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--pair_repulsion",
        help="use pair repulsion term with ZBL potential",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--distance_transform",
        help="use distance transform for radial basis functions",
        default="None",
        choices=["None", "Agnesi", "Soft"],
    )
    parser.add_argument(
        "--apply_cutoff",
        help="apply cutoff to the radial basis functions before MLP",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--use_last_readout_only",
        help="use only the last readout for the final output",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--use_embedding_readout",
        help="use embedding readout for the final output",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--interaction",
        help="name of interaction block",
        type=str,
        default="RealAgnosticResidualInteractionBlock",
        choices=[
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticAttResidualInteractionBlock",
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
            "RealAgnosticResidualNonLinearInteractionBlock",
            "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
            "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
        ],
    )
    parser.add_argument(
        "--interaction_first",
        help="name of interaction block",
        type=str,
        default="RealAgnosticResidualInteractionBlock",
        choices=[
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
            "RealAgnosticResidualNonLinearInteractionBlock",
            "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
            "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
        ],
    )
    parser.add_argument(
        "--max_ell", help=r"highest \ell of spherical harmonics", type=int, default=3
    )
    parser.add_argument(
        "--correlation", help="correlation order at each layer", type=int, default=3
    )
    parser.add_argument(
        "--use_reduced_cg",
        help="use reduced generalized Clebsch-Gordan coefficients",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--use_so3",
        help="use SO(3) irreps instead of O(3) irreps",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--use_agnostic_product",
        help="use element agnostic product",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--num_interactions", help="number of interactions", type=int, default=2
    )
    parser.add_argument(
        "--MLP_irreps",
        help="hidden irreps of the MLP in last readout",
        type=str,
        default="16x0e",
    )
    parser.add_argument(
        "--radial_MLP",
        help="width of the radial MLP",
        type=str,
        default="[64, 64, 64]",
    )
    parser.add_argument(
        "--hidden_irreps",
        help="irreps for hidden node states",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--edge_irreps",
        help="irreps for edge states",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_edge_irreps_first",
        help="use edge irreps in the first interaction block",
        type=str2bool,
        default=False,
    )
    # add option to specify irreps by channel number and max L
    parser.add_argument(
        "--num_channels",
        help="number of embedding channels",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_L",
        help="max L equivariance of the message",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gate",
        help="non linearity for last readout",
        type=str,
        default="silu",
        choices=["silu", "tanh", "abs", "None"],
    )
    parser.add_argument(
        "--kspace_cutoff_factor",
        help="k-space cutoff factor used by PolarMACE",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--atomic_multipoles_max_l",
        help="maximum l for atomic multipoles in PolarMACE",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--atomic_multipoles_smearing_width",
        help="Gaussian smearing width for atomic multipoles in PolarMACE",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--field_feature_max_l",
        help="maximum l for projected field features in PolarMACE",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--field_feature_widths",
        help="list of field feature widths for PolarMACE",
        type=str,
        default="[1.0]",
    )
    parser.add_argument(
        "--field_feature_norms",
        help="optional list of field feature norms for PolarMACE",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_recursion_steps",
        help="number of fixed-point recursion steps in PolarMACE",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--field_si",
        help="include self-interaction when projecting local fields in PolarMACE",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--include_electrostatic_self_interaction",
        help="include electrostatic self interaction in PolarMACE",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--add_local_electron_energy",
        help="add local electron energy correction in PolarMACE",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--quadrupole_feature_corrections",
        help="enable quadrupole feature corrections in PolarMACE",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--return_electrostatic_potentials",
        help="return electrostatic potentials from PolarMACE forward pass",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--field_norm_factor",
        help="global normalization factor for field features in PolarMACE",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--fixedpoint_update_config",
        help="dict-like config for PolarMACE fixed-point update block",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--field_readout_config",
        help="dict-like config for PolarMACE field readout block",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--scaling",
        help="type of scaling to the output",
        type=str,
        default="rms_forces_scaling",
        choices=["std_scaling", "rms_forces_scaling", "no_scaling"],
    )
    parser.add_argument(
        "--avg_num_neighbors",
        help="normalization factor for the message",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--compute_avg_num_neighbors",
        help="normalization factor for the message",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--compute_stress",
        help="Select True to compute stress",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--compute_forces",
        help="Select True to compute forces",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--compute_polarizability",
        help="Select True to compute polarizability",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--compute_atomic_dipole",
        help="Select True to compute dipoles",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--compute_magforces",
        help="Select True to compute magnetic forces",
        type=str2bool,
        default=False,
    )

    # Dataset
    parser.add_argument(
        "--train_file",
        help="Training set file, format is .xyz or .h5",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--valid_file",
        help="Validation set .xyz or .h5 file",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--valid_fraction",
        help="Fraction of training set used for validation",
        type=float,
        default=0.1,
        required=False,
    )
    parser.add_argument(
        "--test_file",
        help="Test set .xyz pt .h5 file",
        type=str,
    )
    parser.add_argument(
        "--test_dir",
        help="Path to directory with test files named as test_*.h5",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--multi_processed_test",
        help="Boolean value for whether the test data was multiprocessed",
        type=str2bool,
        default=False,
        required=False,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers for data loading",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pin_memory",
        help="Pin memory for data loading",
        default=True,
        type=str2bool,
    )
    parser.add_argument(
        "--atomic_numbers",
        help="List of atomic numbers",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--mean",
        help="Mean energy per atom of training set",
        type=float,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--std",
        help="Standard deviation of force components in the training set",
        type=float,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--statistics_file",
        help="json file containing statistics of training set",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--les_arguments",
        help="Path to the LES arguments file",
        type=read_yaml,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--band_edges_file",
        help="JSON file mapping host name to {'e_cbm_cell', 'e_vbm_cell'}, used to "
        "reference charged-cell energy labels (MACEDefect)",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--E0s",
        help="Dictionary of isolated atom energies",
        type=str,
        default=None,
        required=False,
    )

    # Fine-tuning
    parser.add_argument(
        "--pseudolabel_replay",
        help="Use pseudolabels from foundation model for replay data in multihead finetuning",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--pseudolabel_replay_compute_stress",
        help="When replay pseudolabels are generated, always generate stress labels even if the original replay data lacked stress",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--foundation_filter_elements",
        help="Filter element during fine-tuning",
        type=str2bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--heads",
        help="Dict of heads: containing individual files and E0s",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--multiheads_finetuning",
        help="Boolean value for whether the model is multiheaded",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--foundation_head",
        help="Name of the head to use for fine-tuning",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--weight_pt_head",
        help="Weight of the pretrained head in the loss function",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--real_pt_data_ratio_threshold",
        help="threshold of real data to replay data below which real data (sum over all real heads) is duplicated",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--num_samples_pt",
        help="Number of samples in the pretrained head",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--force_mh_ft_lr",
        help="Force the multiheaded fine-tuning to use arg_parser lr",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--subselect_pt",
        help="Method to subselect the configurations of the pretraining set",
        choices=["fps", "random"],
        default="random",
    )
    parser.add_argument(
        "--filter_type_pt",
        help="Filtering method for collecting the pretraining set",
        choices=["none", "combinations", "inclusive", "exclusive"],
        default="none",
    )
    parser.add_argument(
        "--disallow_random_padding_pt",
        help="do not allow random padding of the configurations to match the number of samples",
        action="store_false",
        dest="allow_random_padding_pt",
    )
    parser.add_argument(
        "--pt_train_file",
        help="Training set file for the pretrained head",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pt_valid_file",
        help="Validation set file for the pretrained head",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--foundation_model_elements",
        help="Keep all elements of the foundation model during fine-tuning",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--keep_isolated_atoms",
        help="Keep isolated atoms in the dataset, useful for transfer learning",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--lora",
        help="Use Low-Rank Adaptation for the fine-tuning",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--lora_rank",
        help="Rank of the LoRA matrices",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lora_alpha",
        help="Scaling factor for LoRA",
        type=float,
        default=1.0,
    )

    # Keys
    # Charge-aware defect keys (MACEDefect). Reading them costs nothing for ordinary
    # datasets, where the properties are simply absent and their weights stay at zero.
    parser.add_argument(
        "--carrier_counts_key",
        help="Key of the carrier counter vector (n_e_maj n_e_min n_h_maj n_h_min) in "
        "the training xyz",
        type=str,
        default=DefaultKeys.CARRIER_COUNTS.value,
    )
    parser.add_argument(
        "--host_key",
        help="Key of the host label, used for band-edge lookup and consistency checks",
        type=str,
        default=DefaultKeys.HOST.value,
    )
    parser.add_argument(
        "--pair_id_key",
        help="Key grouping configurations computed at the same geometry, which is what "
        "enables the paired charge-state difference loss",
        type=str,
        default=DefaultKeys.PAIR_ID.value,
    )
    parser.add_argument(
        "--multiplicity_key",
        help="Key of the spin multiplicity, cross-checked against the counter vector",
        type=str,
        default=DefaultKeys.MULTIPLICITY.value,
    )
    parser.add_argument(
        "--m_s_ref_doubled_key",
        help="Key of 2*M_s for the NEUTRAL reference state of this composition. Zero for "
        "an even-electron composition, where the neutral and closed-shell references "
        "coincide; 1 for an odd-electron one such as the CsPbCl3 chloride vacancy. "
        "Non-zero also disables the time-reversal canonicalisation, which is only a "
        "symmetry when the reference is itself unpolarised",
        type=str,
        default="m_s_ref_doubled",
    )
    parser.add_argument(
        "--cell_charge_key",
        help="Key of the absolute cell charge. Cross-checked against the counter charge "
        "q = holes - electrons, which must agree frame by frame under a neutral "
        "reference. This is the assertion that catches a reference-state error, which "
        "spin bookkeeping alone cannot see",
        type=str,
        default="cell_charge",
    )
    parser.add_argument(
        "--e_cbm_cell_key",
        help="Key of the supercell CBM reference energy",
        type=str,
        default=DefaultKeys.E_CBM_CELL.value,
    )
    parser.add_argument(
        "--e_vbm_cell_key",
        help="Key of the supercell VBM reference energy",
        type=str,
        default=DefaultKeys.E_VBM_CELL.value,
    )
    parser.add_argument(
        "--base_energy_key",
        help="Key of an explicitly supplied reference-state (n = 0) energy; normally "
        "produced by the pair join instead",
        type=str,
        default=DefaultKeys.BASE_ENERGY.value,
    )
    parser.add_argument(
        "--base_forces_key",
        help="Key of an explicitly supplied reference-state (n = 0) force array",
        type=str,
        default=DefaultKeys.BASE_FORCES.value,
    )
    parser.add_argument(
        "--base_stress_key",
        help="Key of an explicitly supplied reference-state (n = 0) stress",
        type=str,
        default=DefaultKeys.BASE_STRESS.value,
    )
    parser.add_argument(
        "--energy_key",
        help="Key of reference energies in training xyz",
        type=str,
        default=DefaultKeys.ENERGY.value,
    )
    parser.add_argument(
        "--forces_key",
        help="Key of reference forces in training xyz",
        type=str,
        default=DefaultKeys.FORCES.value,
    )
    parser.add_argument(
        "--virials_key",
        help="Key of reference virials in training xyz",
        type=str,
        default=DefaultKeys.VIRIALS.value,
    )
    parser.add_argument(
        "--stress_key",
        help="Key of reference stress in training xyz",
        type=str,
        default=DefaultKeys.STRESS.value,
    )
    parser.add_argument(
        "--dipole_key",
        help="Key of reference dipoles in training xyz",
        type=str,
        default=DefaultKeys.DIPOLE.value,
    )
    parser.add_argument(
        "--polarizability_key",
        help="Key of polarizability in training xyz",
        type=str,
        default=DefaultKeys.POLARIZABILITY.value,
    )
    parser.add_argument(
        "--magmom_key",
        help="Key of magnetic moment in training xyz",
        type=str,
        default=DefaultKeys.MAGMOM.value,
    )
    parser.add_argument(
        "--magforces_key",
        help="Key of magnetic forces in training xyz",
        type=str,
        default=DefaultKeys.MAGFORCES.value,
    )
    parser.add_argument(
        "--head_key",
        help="Key of head in training xyz",
        type=str,
        default=DefaultKeys.HEAD.value,
    )
    parser.add_argument(
        "--charges_key",
        help="Key of atomic charges in training xyz",
        type=str,
        default=DefaultKeys.CHARGES.value,
    )
    parser.add_argument(
        "--elec_temp_key",
        help="Key of electronic temperature in training xyz",
        type=str,
        default=DefaultKeys.ELEC_TEMP.value,
    )
    parser.add_argument(
        "--total_spin_key",
        help="Key of total spin in training xyz",
        type=str,
        default=DefaultKeys.TOTAL_SPIN.value,
    )
    parser.add_argument(
        "--total_charge_key",
        help="Key of total charge in training xyz",
        type=str,
        default=DefaultKeys.TOTAL_CHARGE.value,
    )
    parser.add_argument(
        "--embedding_specs",
        help=(
            "Dict of feature‐spec dictionaries. "
            "embedding_specs:\n"
            "  total_spin:\n"
            "    type: categorical\n"
            "    per: graph\n"
            "    num_classes: 101\n"
            "    emb_dim: 64\n"
            "  total_charge:\n"
            "    type: categorical\n"
            "    per: graph\n"
            "    num_classes: 201\n"
            "    emb_dim: 64\n"
            "  temperature:\n"
            "    type: continuous\n"
            "    per: graph\n"
            "    in_dim: 1\n"
            "    emb_dim: 32\n"
        ),
        default=None,
    )
    parser.add_argument(
        "--skip_evaluate_heads",
        help="Comma-separated list of heads to skip during final evaluation",
        type=str,
        default="pt_head",
    )

    # Loss and optimization
    parser.add_argument(
        "--loss",
        help="type of loss",
        default="weighted",
        choices=[
            "ef",
            "weighted",
            "forces_only",
            "virials",
            "stress",
            "dipole",
            "dipole_polar",
            "huber",
            "universal",
            "energy_forces_dipole",
            "l1l2energyforces",
            "defect",
        ],
    )
    parser.add_argument(
        "--forces_weight", help="weight of forces loss", type=float, default=100.0
    )
    parser.add_argument(
        "--swa_forces_weight",
        "--stage_two_forces_weight",
        help="weight of forces loss after starting Stage Two (previously called swa)",
        type=float,
        default=100.0,
        dest="swa_forces_weight",
    )
    parser.add_argument(
        "--magforces_weight",
        help="weight of mag forces loss",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--swa_magforces_weight",
        "--stage_two_magforces_weight",
        help="weight of magforces loss after starting Stage Two (previously called swa)",
        type=float,
        default=100.0,
        dest="swa_magforces_weight",
    )
    parser.add_argument(
        "--energy_weight", help="weight of energy loss", type=float, default=1.0
    )
    parser.add_argument(
        "--swa_energy_weight",
        "--stage_two_energy_weight",
        help="weight of energy loss after starting Stage Two (previously called swa)",
        type=float,
        default=1000.0,
        dest="swa_energy_weight",
    )
    # Charge-aware defect model (MACEDefect / --loss defect)
    parser.add_argument(
        "--delta_energy_weight",
        help="weight of the paired charge-state energy difference loss",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--delta_forces_weight",
        help="weight of the paired charge-state force difference loss",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--total_energy_weight",
        help="weight of the unpaired charged total energy loss (detached base branch)",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--pressure_weight",
        help="weight of the hydrostatic charged-cell pressure loss; leave at zero "
        "until the finite-difference check of the DFT code in use passes",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_u_l2",
        help="L2 penalty on the per-channel carrier energy readouts. Default 0: this was "
        "expected to be what buys localisation, but the divacancy reaches a logit gap of "
        "~11.7 without it. It is also a mean over sites, so it cannot penalise a "
        "bulk-wide level of u, and it updates carrier channels no frame occupies",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_p_l2",
        help="L2 penalty on the polarisation channel of the latent charge",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--defect_qhost_l2",
        help="L2 penalty on the host latent charges",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--eps_inf",
        help="High-frequency dielectric constant of the host (DFPT). Sets the initial "
        "gauge of the screening amplitude and, with --eps_inf_prior_weight, an optional "
        "weak prior on it",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--eps_inf_prior_weight",
        help="Weight of the optional prior on the screening amplitude. Keep it small "
        "enough that genuine data can overrule it, and always report the fitted value",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--carrier_feature_dim",
        help="Width of the invariant features fed to the carrier correction heads",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--counter_embedding_dim",
        help="Width of the carrier counter embedding. It embeds six numbers (four "
        "counters plus derived q and M_s), so it does not need to be wide",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--carrier_mlp_hidden",
        help="Hidden width of the per-channel carrier readouts. These produce one scalar "
        "per channel; the first layer of these eight MLPs dominates the correction "
        "branch's parameter count",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--share_logits_across_spin",
        help="Share the attention logit network between the two spin channels of a "
        "carrier type",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_logit_seed_gamma",
        help="Initial per-channel gamma for the additive novelty logit bias "
        "(logit_i^c += gamma_c * s_hat_i). 0 disables. gamma is TRAINABLE, so a state "
        "for which the localisation prior is wrong can drive it to zero. Unlike the "
        "u-seed this works at step 0 -- s_hat is exact and needs no fit through the "
        "readout. NOTE it is an architecture term, present at inference and part of the "
        "energy, not a training-only initialisation",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_seed_anneal",
        help="Anneal the novelty logit-seed gain to zero during training, per channel, "
        "scheduled on each channel's intrinsic (seed-off) gap rather than on epochs. "
        "The converged model is then bias-free: no inference-time descriptor to carry, "
        "differentiate, or accidentally cache as a constant -- which would silently "
        "break force consistency. Safe only because MLP_l is retained",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_seed_anneal_epochs",
        help="Absolute epoch by which the logit-seed gain reaches zero. Absolute, not a "
        "fraction of max_num_epochs: with early stopping the run length is not known in "
        "advance, and a fractional schedule can let a model converge and stop with the "
        "seed still active -- baking an inference-time descriptor into the shipped model",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--defect_alpha_mode",
        help="How the carrier attention is produced. 'logits': a separate network per "
        "channel (the baseline). 'tied': alpha = softmax(-beta u), derived from the "
        "carrier site energy itself, which removes the additive gauge freedom in the "
        "logits, forbids incoherent bound-but-delocalised states, and makes Delta u a "
        "physical binding energy rather than an arbitrary logit scale (plan stage D)",
        type=str,
        choices=["logits", "tied"],
        default="logits",
    )
    parser.add_argument(
        "--defect_beta",
        help="Inverse energy scale of the tied attention, in 1/eV. A fixed gauge "
        "constant recorded with the model, not a tuning knob. Only used with "
        "--defect_alpha_mode=tied",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--defect_zn_l2",
        help="L2 on the counter-embedding input columns of MLP_u's first layer. Keeps "
        "the carrier site energy close to linear in n, which is the condition under "
        "which a counter that is the sum of two other observed counters over-determines "
        "and so kills the E_base gauge. Not itself a gauge-fixing device",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_size_weight",
        help="Weight on the size-extensivity hinge. alpha = softmax(l) normalises over "
        "every atom, so bulk enters the denominator in proportion to N and attention on "
        "the defect decays as 1/N once N passes k*exp(gap) -- the gap only sets where. "
        "This penalises the energy drift that padding the cell with ideal bulk would "
        "cause. Default 0, i.e. off",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_size_ratio",
        help="Cell-size ratio R the correction must be stable over. Expressed as a ratio "
        "rather than an absolute atom count so it needs no knowledge of production cell "
        "sizes and survives retraining on different cells: the requirement is stability "
        "over R-fold growth beyond whatever it was trained on",
        type=float,
        default=1e4,
    )
    parser.add_argument(
        "--defect_size_tol",
        help="Tolerated drift in eV before the size hinge activates. A hinge rather than "
        "a plain penalty so the term stops pushing once satisfied and does not fight the "
        "energy objective permanently",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--defect_size_warmup_epochs",
        help="Epochs before the size hinge activates, counted on the absolute epoch so a "
        "restart past the warmup resumes with it already on. A backstop only: the term is "
        "naturally silent early, since before u develops defect contrast the drift is ~0",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--defect_size_ema",
        help="EMA decay for the per-channel contrast |c| used to place the size "
        "threshold. The threshold is detached, so smoothing costs nothing and stops it "
        "chasing per-batch noise",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--defect_gauge_weight",
        help="Weight on the level-mode gauge penalty (stage D-opt). The softmax is "
        "shift-invariant, so a uniform offset in u^c moves the energy without moving "
        "alpha -- one unidentified direction per channel. Pins the pooled readout to zero "
        "on pristine cells, where a carrier is a band-to-band transition. Does not fix "
        "extensivity; makes the diluted limit interpretable. Default 0, i.e. off",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_totals_detach_base",
        help="Detach E_base inside L_tot, the pre-A5 behaviour. Off by default: the "
        "correction is intensive (sum_i alpha_i = 1) while E_base is extensive, so a "
        "bulk-wide base error cannot be absorbed by the correction, and detaching it "
        "left total forces at defect geometries supervised by nothing. Ablation only",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--base_lr_factor",
        help="Multiplies the learning rate of the base-branch parameter groups "
        "(embedding, interactions, products, readouts). Below 1 slows the base branch "
        "so it is not a moving target for the correction, which matters once L_tot "
        "trains both branches at defect geometries",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--defect_spectral_head",
        help="Component H: read the carrier energy off the lowest eigenvalue of a learned "
        "short-ranged Hamiltonian instead of a softmax-weighted mean of site energies. "
        "Localisation then becomes bound-state formation -- a threshold -- rather than an "
        "amplitude contest against cell size. Retire the size hinge and seed anneal with it: "
        "a bound state's weight is N-independent by construction",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_spectral_decay",
        help="H1: physical hopping decay with a floor, t = [t_min + softplus(B)] * "
        "exp(-(r - r0)/l) * envelope, with per-species-pair decay lengths. Without it a 10 A "
        "Hamiltonian is nearly a complete graph and every state is broad",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_spectral_first_shell",
        help="H2: build the site energies and hopping prefactor from first-interaction-block "
        "features rather than the concatenation of all blocks. On-site energies are local in "
        "every tight-binding model, and final-layer features carry the whole receptive field, "
        "which lets the head place alpha away from where its own forces act",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_spectral_sigma",
        help="H3: dangling-orbital sigma term. Couples atoms whose coordination gaps face "
        "each other, which is the vacancy pair and essentially nothing else. Geometry only; "
        "silent for substitutionals by construction",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_spectral_r_cut",
        help="Range of the carrier Hamiltonian, in Angstrom. 0 means the trunk's receptive "
        "field (r_max * num_interactions), which is the natural scale: the site energies and "
        "hoppings are functions of features that already aggregate everything within it. "
        "This is NOT the message-passing cutoff -- at r_max = 5.0 the two Pb sharing the hole "
        "(median 5.32 A apart, bridged in bulk by the atom that is now the vacancy) have no "
        "edge at all in 64% of frames, so the correct state is unrepresentable. The graph is "
        "built at this cutoff and the trunk filtered back to r_max",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_counting_head",
        help="Edit 4: the electron-counting head. Four orbitals per atom (s, px, py, pz) "
        "with Slater-Koster angular factors; the carrier energy is the Mermin free-energy "
        "difference between the requested fill and the neutral one, so it is identically "
        "zero at zero counters. REPLACES the spectral head rather than sitting beside it -- "
        "two heads would both write delta_sr and the model would double count. Requires "
        "--defect_spectral_head",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_counting_on_site_range",
        help="gamma, the half-width of the bounded on-site correction, in eV. The audit "
        "found every chlorine pinned at gamma = 1 in 6/6 seeds (pre-tanh -3.60 +- 0.19) "
        "while both cations stayed linear; 3.0 is the scale of the missing-anion Madelung "
        "shift, which is what the bound must be able to express",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--defect_counting_hop_range",
        help="half-width of the bounded hopping correction, as a fraction of V0. 20% "
        "saturated at 0.5, which is real but secondary to the on-site channel",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--defect_counting_t_el",
        help="smearing width for the counting head, in eV. 0.05 matches the label "
        "pipeline (doped's SIGMA default at ISMEAR = 0). The previous 0.025 was k_B * 300 K "
        "and had no connection to the labels",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--defect_counting_smearing",
        help="occupation family for the counting head. 'gaussian' matches the label "
        "pipeline (doped's ISMEAR = 0 default, SIGMA = 0.05 eV); 'fermi' is retained for "
        "sensitivity work. The width comes from --defect_counting_t_el",
        type=str,
        choices=("gaussian", "fermi"),
        default="gaussian",
    )
    parser.add_argument(
        "--defect_protocol",
        help="run the Stage-3 protocol from mace.modules.defect_protocol: Harrison "
        "initialisation, c-shift calibration on the first batch, linear warmup, the "
        "initialisation gate, the head-only trainable mask and the post-step Z projection. "
        "Every Stage-3 result to date came from defect-perovskite/stage_run.py, which "
        "carries these and the trainer did not; 'the joint run comes from config' is empty "
        "until this is on",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_protocol_warmup",
        help="linear LR warmup in epochs. The counting head's correction is eV-scale at "
        "step 0, so the first steps see gradients three orders larger than the converged "
        "ones",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--defect_protocol_head_only",
        help="train only the correction parameters, the carrier head and the Madelung "
        "charges -- the mask the Stage-1..3 screens ran under. OFF by default because the "
        "joint run trains the base too, at a reduced rate via --base_lr_factor",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_protocol_freeze_z",
        help="diagnostic arm: pin the Madelung species charges at their initialisation, so "
        "the run answers whether the LEARNED scale of Z buys anything",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_gap_weight",
        help="weight on the pristine frontier-gap constraint, (gap - E_gap)^2. A SPECTRUM "
        "constraint applied to the stoichiometric cells in each batch, selected by "
        "composition and never by a defect label, so it carries none of the 79-atom "
        "base-extrapolation slope that keeps energy targets out of the head-only stages",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_e_gap",
        help="host band gap in eV that --defect_gap_weight drives the pristine frontier gap "
        "towards. 2.40 for CsPbCl3 at this level of theory",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_gap_composition",
        help="pristine stoichiometry in the model's own species order, comma-separated, "
        "e.g. '3,1,1' for CsPbCl3 with the table sorted (Cl, Cs, Pb). Only the ratio is "
        "used. Required by --defect_gap_weight: without it nothing can tell a defect-free "
        "cell from a defective one without consulting a label",
        type=lambda s: [float(v) for v in s.split(",")] if s else None,
        default=None,
    )
    parser.add_argument(
        "--defect_counting_hop_form",
        help="environment modulation on each hopping integral. 'linear' is "
        "1 + hop_range*tanh(g), what Stage 3 ran, bounded in [0.5, 1.5] and asymmetric in "
        "log space. 'log' is exp(beta*tanh(g)), symmetric and positive by construction, so "
        "widening cannot drive an integral through zero and flip the sign the Harrison "
        "initialisation fixed. Both are exactly 1 at g = 0, so bulk-like bonds are untouched "
        "by the choice. Measured motivation: on the vacancy-flanking Pb-Pb bond the trained "
        "cohort sits AT the linear stop in every d bin",
        type=str,
        choices=("linear", "log"),
        default="linear",
    )
    parser.add_argument(
        "--defect_counting_hop_beta",
        help="beta in the 'log' modulation, so the factor spans [exp(-beta), exp(beta)]. "
        "ln 3 = 1.0986 gives x[1/3, 3]. Ignored by --defect_counting_hop_form linear",
        type=float,
        default=1.0986122886681098,
    )
    # Stage A' spec. Each is a constructor argument of MACEDefect and survives the config
    # round trip (section 5.1); the behaviours land in sections 2.1-2.5.
    parser.add_argument(
        "--defect_counting_decay_learned",
        help="learn the radial decay length per Slater-Koster integral type: "
        "L_b = L0 * exp(beta_L * tanh u_b), four universal scalars shared across hosts, "
        "u_b = 0 at initialisation so the fixed-length head is the starting point",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_counting_decay_beta",
        help="beta_L for the learned decay lengths; ln 2 = 0.6931 keeps every L_b within "
        "[L0/2, 2 L0]",
        type=float,
        default=0.6931471805599453,
    )
    parser.add_argument(
        "--defect_null_reference",
        help="JSON establishing which cell sizes have a NEUTRAL NULL -- "
        '{"nulls": {"<atoms>": {"slope": ..., "ci": [lo, hi]}}} -- a size qualifying when '
        "its interval brackets zero. Charged frames of any other size have their energy "
        "weight zeroed (speed-cycle standing rule 2); their forces are untouched. Required "
        "whenever --defect_charged_energy_share is non-zero",
        type=str,
        default="",
    )
    parser.add_argument(
        "--defect_counting_centre_form",
        help="where the pristine centre is subtracted in the on-site correction "
        "(speed-cycle spec section 2.1). 'argument' is gamma tanh(h(x_i) - h(xbar_s)), "
        "which keeps the channel alive under a drifting h; 'output' is the Stage A' form, "
        "gamma [tanh h(x_i) - tanh h(xbar_s)], kept so that cohort stays scorable",
        type=str,
        choices=("argument", "output"),
        default="argument",
    )
    parser.add_argument(
        "--defect_madelung_site_zeta",
        help="bound in e on the PER-SITE charge deviation, Z_i = Z0[s] + zeta tanh(z(x_i) "
        "- z(xbar_s)) (speed-cycle spec section 2.2). Zero disables the channel, which is "
        "the per-species model exactly. The deviation is centred per graph so the cell's "
        "net charge stays the element baseline's",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_size_grouped_batches",
        help="group training and validation batches by EXACT atom count, so the counting "
        "head can solve a whole batch as one [B, 4n, 4n] eigenproblem (speed-cycle spec "
        "section 1.1). Frames are permuted within their group and the batches are then "
        "permuted across groups; the short tail batch of each group is kept",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_precision_policy",
        help="'uniform' runs the whole model at --default_dtype; 'mixed' runs the trunk at "
        "the process default and the carrier head, Madelung term and long-range branch at "
        "float64 with explicit casts at the boundary",
        type=str,
        choices=("uniform", "mixed"),
        default="uniform",
    )
    parser.add_argument(
        "--defect_neutral_size_upweight_energy",
        help="apply --defect_neutral_size_upweight to the ENERGY loss as well as the force "
        "loss, each channel solved on its own mass (Stage A' spec section 1). The realised "
        "share of both channels is logged every epoch",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_c_shift_per_class",
        help="calibrate c per (charge, size) class over every charged frame at "
        "initialisation, with E_LR's value in the residual (Stage A' spec section 3), "
        "instead of one scalar",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_charged_energy_share",
        help="target share of the CHARGED energy loss carried by the large cells (Stage A' "
        "spec section 3: 0.25-0.5). Applied after the per-frame energy weights, so the "
        "share is realised on the weighted mass. 0 disables",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_base_cache",
        help="cache the frozen base's energy, forces and later-block features per frame "
        "(section 2.5 of the Stage A' spec) and recompute only the first interaction block "
        "each step. Refuses to start unless the base is frozen, the head reads the first "
        "block only, and any long-range branch is detached and frozen",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_base_cache_dir",
        help="where the base cache file is written and looked for (named by the base's "
        "checksum); empty means the checkpoints directory",
        type=str,
        default="",
    )
    parser.add_argument(
        "--defect_on_site_centred",
        help="the on-site correction is the deviation from the pristine environment: "
        "gamma * [tanh h(x_i) - tanh h(xbar_s(i))], with xbar_s the mean first-block "
        "feature of species s over the pristine reference cell",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_protocol_zero_on_site",
        help="start the on-site correction channel at exactly zero output. The measured "
        "channel applies a near-uniform +0.27 eV to every atom -- a global gauge that "
        "contributes no force and is therefore invisible to a forces-only objective. "
        "Zero-init means whatever it carries afterwards was put there by the energy loss",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_neutral_size_upweight",
        help="target share of the NEUTRAL force loss carried by large neutral cells. They "
        "are the base branch's only direct constraint at large d: the rest of the neutral "
        "set stops around 6.0 A and the base extrapolates above it, which is the measured "
        "+0.132 eV/A error in the carrier-free 79-atom residual. 0 disables",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_counting_envelope",
        help="radial envelope on the hopping integrals. 'exp' is exp(-(r - d_ref)/L), what "
        "Stage 3 ran; 'power' is Harrison's own (d_ref/r)^2, which is also the rule V0 is "
        "initialised from. Measured at the vacancy-flanking Pb-Pb bond, 'exp' delivers "
        "t/t_Harrison = 0.14 +- 0.07 with the learned pair modulation pinned at its +50% "
        "bound on the close frames -- the head straining against the envelope and losing",
        type=str,
        choices=("exp", "power"),
        default="exp",
    )
    parser.add_argument(
        "--defect_counting_decay_length",
        help="L in the 'exp' envelope, in Angstrom. Ignored by --defect_counting_envelope "
        "power. Matching Harrison's log-slope at d_ref needs 1.4 and matching it at the hub "
        "bond needs about 2.9, which is why retuning this only moves which separation is "
        "wrong",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--defect_madelung_on_site",
        help="Edit 1: put the host Madelung potential of learnable per-species charges on "
        "the on-site energies, eps_i = eps_local_i - phi_LR_i / eps_inf. Variational, unlike "
        "the retired response channel: an anion vacancy raises phi at the neighbouring "
        "cations, which lowers their electron on-site energy, which is a donor well the "
        "eigenproblem can bind into. phi_LR is the FULL lattice sum -- only the true self "
        "term is excluded, no self-image subtraction",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_madelung_eps_inf",
        help="High-frequency dielectric constant screening phi_LR in the on-site shift. A PER-HOST INPUT with no default: 4.0 is CsPbCl3's number and standing rule 1 forbids a per-host constant in a default, so --defect_madelung_on_site refuses to run without it. The "
        "carrier's own field, which the lattice has not had time to respond to. Distinct "
        "from --eps_inf, which initialises the long-range branch",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_madelung_composition",
        help="Pristine stoichiometry in the model's own species order, comma-separated "
        "(CsPbCl3 with Z-table [17, 55, 82] is '3,1,1'). A property of the training set's "
        "formula unit, not of the defect: Z is projected onto this hyperplane after every "
        "optimiser step, without which the species charges drift as a group",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--defect_madelung_z_init",
        help="Initial per-species charges, comma-separated, in the same order. Nominal "
        "charges ('-1,1,2' for CsPbCl3) rather than zeros: at Z = 0 the Madelung term is "
        "identically absent at epoch 0 and the run has to discover the ionicity of a "
        "rock-salt-like crystal from force residuals",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--defect_base_release_epoch",
        help="Epoch at which the frozen Stage-A base is released to a slow learning rate "
        "(plan T4). 0 disables the two-timescale schedule entirely. Passing seeds settle by "
        "~30, so releasing there adapts the base without letting it re-absorb the site "
        "signal during the epochs that decide localisation",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--defect_base_release_factor",
        help="Base learning rate after release, as a multiple of the head's",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--defect_two_size_upweight",
        help="Target share of the CHARGED force loss carried by charged frames at the "
        "larger cell size (0 disables). Those 17 frames carry the only measurement that "
        "separates a bound carrier from a band state (Test 2, R_DFT = 0.95), and at natural "
        "weighting they are ~3% of the charged force loss. Cell size is a property of the "
        "box, not a defect label",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_spectral_anneal_s0",
        help="Bandwidth anneal: all hoppings are scaled by s0^(1 - e/E_a) for e <= E_a, "
        "then 1. Starting wide and narrowing lets a level separate from the band gradually "
        "rather than having to tunnel out of a converged solution. 0 disables it",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_spectral_anneal_epochs",
        help="E_a for the bandwidth anneal",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--defect_eps_gauge_weight",
        help="T5: weight on (per-frame mean site energy)^2, which pins the uniform "
        "gauge of eps without subtracting the mean. Subtraction removes the mode "
        "exactly but makes lambda cell-size dependent (O(1/N) for a localised well, ~2.6 meV "
        "across the 640-5120 ladder against a <= 1 meV gate). Pair with "
        "--defect_spectral_gauge_penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--defect_spectral_gauge_penalty",
        help="Report the mean site energy instead of subtracting it, so the gauge is "
        "set by --defect_eps_gauge_weight in the loss",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_spectral_states",
        help="Number of lowest eigenpairs kept by the spectral head. Log the truncation "
        "margin (lambda_m - lambda_occupied)/T_s; it must stay well above 1",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--defect_spectral_smearing",
        help="Thermal smearing T_s in eV over those eigenvalues. Regularises eigenvalue "
        "crossings, and lets a gapless cell degrade to a band-edge ensemble rather than "
        "forcing a bound state that is not there",
        type=float,
        default=0.020,
    )
    parser.add_argument(
        "--defect_base_init",
        help="Path to a Stage-A model whose base branch is copied into this model before "
        "training (component S). Pair with --base_lr_factor 0.0 for Stage B: the correction "
        "then trains against a fixed base, and the residual at a charged geometry stands in "
        "for the neutral pair partner the dataset does not contain. Buffers are copied too, "
        "so the energy reference matches the one Stage A was fitted with",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--defect_zero_u_init",
        help="Zero-initialise the carrier energy readout MLP_u. ON by default, for "
        "optimisation conditioning: a random u is structureless noise that the optimiser "
        "must first destroy, arriving back at u ~ 0 with the attention still uniform, "
        "whereas from zero u grows along the gradient and its contrast is "
        "target-aligned from the first step. This is NOT needed for the n = 0 identity, "
        "which is structural via the counter prefactor. Set False to ablate",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--freeze_amplitude",
        help="Freeze the screening amplitude a at 1/sqrt(eps_inf) instead of fitting it "
        "(plan stage E). a is not identifiable from a dipole term alone -- with q = 0 on "
        "every frame there is no monopole for it to scale -- so a free a drifts to "
        "whatever absorbs the electron-hole energy and then reads as a fitted screening "
        "constant while being nothing of the kind",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--defect_seed_gate",
        help="Which readiness measure retires the logit seed. 'gap' uses the raw logit gap "
        "and is the proven schedule (short-range-only localises 4 of 4 seeds under it). "
        "'site' uses the species-blind site structure, which is the right quantity in "
        "principle but currently hands over as a cliff rather than a ramp and cost one "
        "seed its localisation. Both are logged either way",
        type=str,
        choices=["gap", "site"],
        default="gap",
    )
    parser.add_argument(
        "--defect_seed_max_drop",
        help="Largest fractional fall in the seed gain per epoch, 1.0 for no bound. The "
        "site gate measures the right thing but releases the seed as a step once site "
        "structure appears, and a seed whose structure appears late loses its attention in "
        "the two or three epochs that release takes. Bounding the rate turns the same "
        "measure into a hand-over",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--use_polarisation",
        help="Enable the q^pol channel. False ablates it entirely, leaving delta_lr as "
        "carrier^2 + host.carrier. After gating, the polarisation response is contained "
        "inside the receptive field, and a neutral cloud contained inside the receptive "
        "field has a local electrostatic energy that delta_SR can already represent",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--host_charge_detached",
        help="Compute q^host from a DETACHED copy of the shared features, so fitting it "
        "cannot shape the representation the logit and u readouts consume. q^host needs "
        "species to be salient, and E_LR[q^host] is in the loss, so sharing a trunk makes "
        "the attention read off an element ordering instead of a site",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--pol_gate",
        help="Gate q^pol on a smeared UNSIGNED carrier density instead of subtracting a "
        "cell mean. A whole-cell mean gives neutrality, not locality, and leaves every "
        "atom a constant per-species residual whose Madelung energy grows with N. Gating "
        "makes a bulk atom zero because the gate is zero there. Also self-protecting: a "
        "delocalised alpha gives g ~ 1/N, so attention collapse stops being profitable",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--pol_gate_lambda",
        help="Screening length of the polarisation gate, in A. A FIXED gauge constant "
        "recorded with the model, like sigma and beta -- not learned, because an unbounded "
        "lambda recovers the bulk plateau exactly",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--pol_gate_hops",
        help="Propagation steps for the polarisation gate on the message-passing graph. "
        "Two hops at r_max = 5 reaches 10 A",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--host_carrier_coupling",
        help="Keep the cross term between q^host and the carrier cloud in delta_lr. False "
        "repartitions the branch so it carries only what Delta E_SR structurally cannot: "
        "the monopole self-interaction and interactions between separated carriers. The "
        "cross term has the same pooling form as Delta E_SR = sum_c n_c <u>_alpha, so the "
        "two are degenerate; measured on CsPbCl3 it favours the Cs sublattice over the "
        "vacancy shell by 1.25 eV against 0.18 eV of short-range difference",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--carrier_self_isolated",
        help="Define E_LR as the finite-size correction: subtract each carrier channel's "
        "ISOLATED self-energy from the periodic one, leaving only the image interaction. "
        "The in-cell part is self-interaction error -- a single hole has no Hartree "
        "self-repulsion -- and measured at the training cell it pays 0.104 eV to spread "
        "the attention out. Per channel, so cross-channel terms (electron-hole) survive",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--lr_start_epoch",
        help="Hold the long-range branch out of the energy and the loss until this epoch. "
        "Not a ramp on a: the branch is absent entirely, so the model IS the short-range "
        "model until it fires. The short-range model reliably localises within a few "
        "epochs while every long-range run so far has had its attention captured first, so "
        "this asks whether the branch destroys a settled correct answer rather than "
        "whether it can find one",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use_long_range",
        help="Enable the latent-Ewald long-range branch (requires the LES library)",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--virials_weight", help="weight of virials loss", type=float, default=1.0
    )
    parser.add_argument(
        "--swa_virials_weight",
        "--stage_two_virials_weight",
        help="weight of virials loss after starting Stage Two (previously called swa)",
        type=float,
        default=10.0,
        dest="swa_virials_weight",
    )
    parser.add_argument(
        "--stress_weight", help="weight of stress loss", type=float, default=1.0
    )
    parser.add_argument(
        "--swa_stress_weight",
        "--stage_two_stress_weight",
        help="weight of stress loss after starting Stage Two (previously called swa)",
        type=float,
        default=10.0,
        dest="swa_stress_weight",
    )
    parser.add_argument(
        "--dipole_weight", help="weight of dipoles loss", type=float, default=1.0
    )
    parser.add_argument(
        "--swa_dipole_weight",
        "--stage_two_dipole_weight",
        help="weight of dipoles after starting Stage Two (previously called swa)",
        type=float,
        default=1.0,
        dest="swa_dipole_weight",
    )
    parser.add_argument(
        "--swa_polarizability_weight",
        "--stage_two_polarizability_weight",
        help="weight of polarizability after starting Stage Two (previously called swa)",
        type=float,
        default=1.0,
        dest="swa_polarizability_weight",
    )
    parser.add_argument(
        "--polarizability_weight",
        help="weight of polarizability loss",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--config_type_weights",
        help="String of dictionary containing the weights for each config type",
        type=str,
        default='{"Default":1.0}',
    )
    parser.add_argument(
        "--huber_delta",
        help="delta parameter for huber loss",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--optimizer",
        help="Optimizer for parameter optimization",
        type=str,
        default="adam",
        choices=["adam", "adamw", "schedulefree"],
    )
    parser.add_argument(
        "--beta",
        help="Beta parameter for the optimizer",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--beta1_schedulefree",
        help="Beta1 parameter for the ScheduleFree optimizer",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--beta2_schedulefree",
        help="Beta2 parameter for the ScheduleFree optimizer",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--warmup_steps_schedulefree",
        help="Number of linear LR warmup steps for the ScheduleFree optimizer",
        type=int,
        default=0,
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=10)
    parser.add_argument(
        "--valid_batch_size", help="Validation batch size", type=int, default=10
    )
    parser.add_argument(
        "--lr", help="Learning rate of optimizer", type=float, default=0.01
    )
    parser.add_argument(
        "--swa_lr",
        "--stage_two_lr",
        help="Learning rate of optimizer in Stage Two (previously called swa)",
        type=float,
        default=1e-3,
        dest="swa_lr",
    )
    parser.add_argument(
        "--weight_decay", help="weight decay (L2 penalty)", type=float, default=5e-7
    )
    parser.add_argument(
        "--lr_params_factors",
        help="Learning rate factors to multiply on the original lr",
        type=str,
        default='{"embedding_lr_factor": 1.0, "interactions_lr_factor": 1.0, "products_lr_factor": 1.0, "readouts_lr_factor": 1.0}',
    )
    parser.add_argument(
        "--freeze",
        help="Freeze layers from 1 to N. Can be positive or negative, e.g. -1 means the last layer is frozen. 0 or None means all layers are active and is a default setting",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--amsgrad",
        help="use amsgrad variant of optimizer",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--scheduler", help="Type of scheduler", type=str, default="ReduceLROnPlateau"
    )
    parser.add_argument(
        "--lr_factor", help="Learning rate factor", type=float, default=0.8
    )
    parser.add_argument(
        "--scheduler_patience", help="Learning rate factor", type=int, default=50
    )
    parser.add_argument(
        "--lr_scheduler_gamma",
        help="Gamma of learning rate scheduler",
        type=float,
        default=0.9993,
    )
    parser.add_argument(
        "--swa",
        "--stage_two",
        help="use Stage Two loss weight, which decreases the learning rate and increases the energy weight at the end of the training to help converge them",
        action="store_true",
        default=False,
        dest="swa",
    )
    parser.add_argument(
        "--start_swa",
        "--start_stage_two",
        help="Number of epochs before changing to Stage Two loss weights",
        type=int,
        default=None,
        dest="start_swa",
    )
    parser.add_argument(
        "--lbfgs",
        help="Switch to L-BFGS optimizer",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ema",
        help="use Exponential Moving Average",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ema_decay",
        help="Exponential Moving Average decay",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--max_num_epochs", help="Maximum number of epochs", type=int, default=2048
    )
    parser.add_argument(
        "--patience",
        help="Maximum number of consecutive epochs of increasing loss",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--foundation_model",
        help="Path to the foundation model for transfer learning",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--foundation_model_kwargs",
        help="Additional kwargs for the foundation model for transfer learning",
        type=str,
        default="{}",
    )
    parser.add_argument(
        "--foundation_model_readout",
        help="Use readout of foundation model for transfer learning",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--finetune_dipoles_polarizabilities",
        help="Fine-tune an existing AtomicDielectricMACE (MACE-MDP) model on dipoles and polarizabilities only. Requires --foundation_model pointing to the pretrained MDP checkpoint.",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--eval_interval", help="evaluate model every <n> epochs", type=int, default=1
    )
    parser.add_argument(
        "--keep_checkpoints",
        help="keep all checkpoints",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_all_checkpoints",
        help="save all checkpoints",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--restart_latest",
        help="restart optimizer from latest checkpoint",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_cpu",
        help="Save a model to be loaded on cpu",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--clip_grad",
        help="Gradient Clipping Value",
        type=check_float_or_none,
        default=10.0,
    )
    parser.add_argument(
        "--dry_run",
        help="Run all steps upto training to test settings.",
        action="store_true",
        default=False,
    )
    # option for cuequivariance acceleration
    parser.add_argument(
        "--enable_cueq",
        help="Enable cuequivariance acceleration",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--only_cueq",
        help="Only use cuequivariance acceleration",
        type=str2bool,
        default=False,
    )
    # option for openequivariance acceleration
    parser.add_argument(
        "--enable_oeq",
        help="Enable openequivariance acceleration",
        type=str2bool,
        default=False,
    )
    # options for using Weights and Biases for experiment tracking
    # to install see https://wandb.ai
    parser.add_argument(
        "--wandb",
        help="Use Weights and Biases for experiment tracking",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--wandb_dir",
        help="An absolute path to a directory where Weights and Biases metadata will be stored",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--wandb_project",
        help="Weights and Biases project name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--wandb_entity",
        help="Weights and Biases entity name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--wandb_name",
        help="Weights and Biases experiment name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--wandb_log_hypers",
        help="The hyperparameters to log in Weights and Biases",
        nargs="+",
        default=[
            "num_channels",
            "max_L",
            "correlation",
            "lr",
            "swa_lr",
            "weight_decay",
            "batch_size",
            "max_num_epochs",
            "start_swa",
            "energy_weight",
            "forces_weight",
        ],
    )

    # --- magnetic mace specific arguments ---
    parser.add_argument(
        "--num_mag_radial_basis_one_body",
        help="number of radial basis for one body contribution in magnetic mace",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--m_max",
        help=(
            "|m| saturation per element. Either a dict literal mapping atomic "
            'number to m_max (e.g. "{26: 1.8, 28: 1.2}" — only listed elements '
            "are required, others default to 1.0), or a space-separated list of "
            "floats ordered by z_table.zs (legacy)."
        ),
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--max_m_ell",
        help="max_ell for magnetic mace",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_mag_radial_basis",
        help="number of radial basis for magnetic part",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--use_magmom_one_body",
        help="If true, use one body mangetic moment contribution in the model",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--train_one_body_contribution",
        help="If true, include the magmom one-body coefficients in the optimizer "
        "(only relevant when --use_magmom_one_body is set).",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--data_aug_magmom",
        help="Whether to use data augmentation on magnetic moment training. ",
        type=str2bool,
        default=False,
    )

    return parser


def build_preprocess_arg_parser() -> argparse.ArgumentParser:
    try:
        import configargparse

        parser = configargparse.ArgumentParser(
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add(
            "--config",
            type=str,
            is_config_file=True,
            help="config file to aggregate options",
        )
    except ImportError:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument(
        "--train_file",
        help="Training set h5 file",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--valid_file",
        help="Training set xyz file",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--num_process",
        help="The user defined number of processes to use, as well as the number of files created.",
        type=int,
        default=int(os.cpu_count() / 4),
    )
    parser.add_argument(
        "--valid_fraction",
        help="Fraction of training set used for validation",
        type=float,
        default=0.1,
        required=False,
    )
    parser.add_argument(
        "--test_file",
        help="Test set xyz file",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--work_dir",
        help="set directory for all files and folders",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--h5_prefix",
        help="Prefix for h5 files when saving",
        type=str,
        default="",
    )
    parser.add_argument(
        "--r_max", help="distance cutoff (in Ang)", type=float, default=5.0
    )
    parser.add_argument(
        "--config_type_weights",
        help="String of dictionary containing the weights for each config type",
        type=str,
        default='{"Default":1.0}',
    )
    parser.add_argument(
        "--energy_key",
        help="Key of reference energies in training xyz",
        type=str,
        default=DefaultKeys.ENERGY.value,
    )
    parser.add_argument(
        "--forces_key",
        help="Key of reference forces in training xyz",
        type=str,
        default=DefaultKeys.FORCES.value,
    )
    parser.add_argument(
        "--virials_key",
        help="Key of reference virials in training xyz",
        type=str,
        default=DefaultKeys.VIRIALS.value,
    )
    parser.add_argument(
        "--stress_key",
        help="Key of reference stress in training xyz",
        type=str,
        default=DefaultKeys.STRESS.value,
    )
    parser.add_argument(
        "--dipole_key",
        help="Key of reference dipoles in training xyz",
        type=str,
        default=DefaultKeys.DIPOLE.value,
    )
    parser.add_argument(
        "--polarizability_key",
        help="Key of polarizability in training xyz",
        type=str,
        default=DefaultKeys.POLARIZABILITY.value,
    )
    parser.add_argument(
        "--charges_key",
        help="Key of atomic charges in training xyz",
        type=str,
        default=DefaultKeys.CHARGES.value,
    )
    parser.add_argument(
        "--atomic_numbers",
        help="List of atomic numbers",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--compute_statistics",
        help="Compute statistics for the dataset",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--batch_size",
        help="batch size to compute average number of neighbours",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--scaling",
        help="type of scaling to the output",
        type=str,
        default="rms_forces_scaling",
        choices=["std_scaling", "rms_forces_scaling", "no_scaling"],
    )
    parser.add_argument(
        "--E0s",
        help="Dictionary of isolated atom energies",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--shuffle",
        help="Shuffle the training dataset",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--seed",
        help="Random seed for splitting training and validation sets",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--head_key",
        help="Key of head in training xyz",
        type=str,
        default=DefaultKeys.HEAD.value,
    )
    parser.add_argument(
        "--heads",
        help="Dict of heads: containing individual files and E0s",
        type=str,
        default=None,
        required=False,
    )
    return parser


def check_float_or_none(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        if value != "None":
            raise argparse.ArgumentTypeError(
                f"{value} is an invalid value (float or None)"
            ) from None
        return None


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def read_yaml(value: str) -> Dict:
    from pathlib import Path

    import yaml

    if not Path(value).is_file():
        raise argparse.ArgumentTypeError(f"File {value} does not exist.")
    with open(value, "r", encoding="utf-8") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise argparse.ArgumentTypeError(
                f"Error parsing YAML file {value}: {exc}"
            ) from exc
