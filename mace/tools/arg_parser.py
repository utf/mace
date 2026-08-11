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
