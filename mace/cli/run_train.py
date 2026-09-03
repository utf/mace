###########################################################################################
# Training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import ast
import glob
import json
import logging
import os

import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch.distributed
from e3nn.util import jit
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import LBFGS
from torch.utils.data import ConcatDataset
from torch_ema import ExponentialMovingAverage

import mace
from mace import data, tools
from mace.calculators.foundations_models import (
    mace_mp,
    mace_mp_names,
    mace_off,
    mace_omol,
    mace_polar,
    polar_model_names,
)
from mace.cli.convert_cueq_e3nn import run as run_cueq_to_e3nn
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.cli.convert_e3nn_oeq import run as run_e3nn_to_oeq
from mace.cli.convert_oeq_e3nn import run as run_oeq_to_e3nn
from mace.cli.visualise_train import TrainingPlotter
from mace.data import KeySpecification, update_keyspec_from_kwargs
from mace.modules.lora import inject_LoRAs, merge_lora_weights
from mace.tools import torch_geometric
from mace.tools.distributed_tools import init_distributed
from mace.tools.model_script_utils import configure_model
from mace.tools.multihead_tools import (
    HeadConfig,
    apply_pseudolabels_to_pt_head_configs,
    assemble_replay_data,
    dict_head_to_dataclass,
    inherit_magnetic_hyperparameters_from_foundation,
    prepare_default_head,
    prepare_pt_head,
)
from mace.tools.run_train_utils import (
    combine_datasets,
    load_dataset_for_path,
    normalize_file_paths,
)
from mace.tools.scripts_utils import (
    LRScheduler,
    SubsetCollection,
    check_path_ase_read,
    convert_to_json_format,
    dict_to_array,
    extract_config_mace_model,
    get_atomic_energies,
    get_avg_num_neighbors,
    get_config_type_weights,
    get_dataset_from_xyz,
    get_files_with_suffix,
    get_loss_fn,
    get_optimizer,
    get_params_options,
    get_swa,
    print_git_commit,
    remove_pt_head,
    setup_wandb,
)
from mace.tools.tables_utils import create_error_table
from mace.tools.utils import AtomicNumberTable


def main() -> None:
    """
    This script runs the training/fine tuning for mace
    """
    args = tools.build_default_arg_parser().parse_args()
    run(args)



def graph_cutoff(args) -> float:
    """Radius the neighbour graph is built at.

    Normally r_max. With the spectral head this must be the CARRIER Hamiltonian's range,
    which is longer: the two Pb that share the hole sit a median 5.32 A apart and are bridged
    in bulk by the atom that is now the vacancy, so at r_max = 5.0 they have no edge in 64% of
    frames and the correct two-site state cannot be represented at all.

    The trunk is filtered back to r_max in MACEDefect.forward, so it sees exactly the graph it
    always did -- and would even without the filter, since the radial cutoff sends anything
    beyond r_max to zero. The filter only avoids paying for the extra edges.
    """
    if not getattr(args, "defect_spectral_head", False):
        return float(args.r_max)
    explicit = float(getattr(args, "defect_spectral_r_cut", 0.0) or 0.0)
    if explicit > 0:
        return max(float(args.r_max), explicit)
    return float(args.r_max) * int(args.num_interactions)

def run(args) -> None:
    """
    This script runs the training/fine tuning for mace
    """
    tag = tools.get_tag(name=args.name, seed=args.seed)
    args, input_log_messages = tools.check_args(args)

    # default keyspec to update using heads dictionary
    args.key_specification = KeySpecification()
    update_keyspec_from_kwargs(args.key_specification, vars(args))

    # Per-host supercell band edges, used to reference charged-cell energy labels.
    # Per-frame e_cbm_cell / e_vbm_cell keys override this table where present.
    band_edges = None
    if getattr(args, "band_edges_file", None) is not None:
        band_edges = data.load_band_edges(args.band_edges_file)

    if args.device == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            import oneccl_bindings_for_pytorch as oneccl  # pylint: disable=unused-import
        except ImportError as e:
            raise ImportError(
                "Error: Intel extension for PyTorch not found, but XPU device was specified"
            ) from e
    rank, local_rank, world_size = init_distributed(args)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir, rank=rank)
    logging.info("===========VERIFYING SETTINGS===========")
    for message, loglevel in input_log_messages:
        logging.log(level=loglevel, msg=message)

    if args.distributed:
        if args.device == "cuda":
            torch.cuda.set_device(local_rank)
        elif args.device == "xpu":
            torch.xpu.set_device(local_rank)
        logging.info(f"Process group initialized: {torch.distributed.is_initialized()}")
        logging.info(f"Processes: {world_size}")

    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.debug(f"Configuration: {args}")

    tools.set_default_dtype(args.default_dtype)
    device = tools.init_device(args.device)
    commit = print_git_commit()
    model_foundation: Optional[torch.nn.Module] = None
    foundation_model_avg_num_neighbors = 0
    # Filter out None from mace_mp_names to get valid model names
    valid_mace_mp_models = [name for name in mace_mp_names if name is not None]
    args.foundation_model_kwargs = ast.literal_eval(args.foundation_model_kwargs)
    args.foundation_model_kwargs["head"] = args.foundation_head

    # MDP fine-tuning validation
    if args.finetune_dipoles_polarizabilities:
        if args.model != "AtomicDielectricMACE":
            raise ValueError(
                "--finetune_dipoles_polarizabilities only supports "
                "--model AtomicDielectricMACE"
            )
        if args.foundation_model is None:
            raise ValueError(
                "--foundation_model must be provided when using "
                "--finetune_dipoles_polarizabilities"
            )
        if args.loss not in ("weighted", "dipole_polar"):
            raise ValueError(
                "--finetune_dipoles_polarizabilities requires --loss dipole_polar "
                f"(got --loss={args.loss})"
            )
        args.loss = "dipole_polar"
        # multiheads_finetuning defaults to True and would override loss to "universal"
        args.multiheads_finetuning = False
        # AtomicDielectricMACE has no atomic_energies_fn, so E0s="foundation"/"estimated" crash
        if args.E0s is not None and args.E0s.lower() in ("foundation", "estimated"):
            logging.warning(
                f"--E0s={args.E0s} is not supported for AtomicDielectricMACE "
                "(no atomic_energies_fn); falling back to --E0s=average"
            )
            args.E0s = "average"
        logging.info(
            "MDP fine-tuning mode: loss=dipole_polar, multiheads_finetuning disabled"
        )

    if args.foundation_model is not None:
        if args.foundation_model in polar_model_names:
            logging.info(
                f"Using Polar foundation model {args.foundation_model} as initial checkpoint."
            )
            model_foundation = mace_polar(
                model=args.foundation_model,
                device=args.device,
                default_dtype=args.default_dtype,
                return_raw_model=True,
            )
        elif args.foundation_model in valid_mace_mp_models:
            logging.info(
                f"Using foundation model mace {args.foundation_model} as initial checkpoint."
            )
            calc = mace_mp(
                model=args.foundation_model,
                device=args.device,
                default_dtype=args.default_dtype,
                **args.foundation_model_kwargs,
            )
            model_foundation = calc.models[0]

        elif args.foundation_model in ["small_off", "medium_off", "large_off"]:
            model_type = args.foundation_model.split("_")[0]
            logging.info(
                f"Using foundation model mace-off-2023 {model_type} as initial checkpoint. ASL license."
            )
            calc = mace_off(
                model=model_type,
                device=args.device,
                default_dtype=args.default_dtype,
            )
            model_foundation = calc.models[0]
        elif args.foundation_model in ["mace_omol"]:
            logging.info("Using foundation model mace-omol as initial checkpoint.")
            calc = mace_omol(
                device=args.device,
                default_dtype=args.default_dtype,
            )
            model_foundation = calc.models[0]
        else:
            model_foundation = torch.load(
                args.foundation_model, map_location=args.device
            )
            logging.info(
                f"Using foundation model {args.foundation_model} as initial checkpoint."
            )
        args.r_max = model_foundation.r_max.item()
        inherited_magnetic_args = inherit_magnetic_hyperparameters_from_foundation(
            args, model_foundation
        )
        if inherited_magnetic_args:
            logging.info(
                f"Inheriting magnetic hyperparameters from foundation model: {inherited_magnetic_args}"
            )
        if args.finetune_dipoles_polarizabilities:
            foundation_cls = model_foundation.__class__.__name__
            if foundation_cls != "AtomicDielectricMACE":
                raise ValueError(
                    f"--finetune_dipoles_polarizabilities requires an AtomicDielectricMACE "
                    f"checkpoint, but --foundation_model contains a {foundation_cls} model."
                )
        foundation_model_avg_num_neighbors = model_foundation.interactions[
            0
        ].avg_num_neighbors
        if (
            args.foundation_model not in ["small", "medium", "large"]
            and args.pt_train_file is None
        ):
            if args.multiheads_finetuning:
                logging.warning(
                    "Using multiheads finetuning with a foundation model that is not a Materials Project model, need to provied a path to a pretraining file with --pt_train_file."
                )
            args.multiheads_finetuning = False
        if args.multiheads_finetuning:
            assert args.E0s != "average", (
                "average atomic energies cannot be used for multiheads finetuning"
            )
            if not args.force_mh_ft_lr:
                logging.info(
                    "Multihead finetuning mode, setting learning rate to 0.0001 and EMA to True. To use a different learning rate, set --force_mh_ft_lr=True."
                )
                args.lr = 0.0001
                args.ema = True
                args.ema_decay = 0.99999
            logging.info(
                "Using multiheads finetuning mode, setting learning rate to 0.0001 and EMA to True"
            )
        if hasattr(model_foundation, "heads"):
            if len(model_foundation.heads) > 1:
                logging.info(
                    f"Selecting the head {args.foundation_head} as foundation head."
                )
                model_foundation = remove_pt_head(
                    model_foundation, args.foundation_head
                )
    else:
        args.multiheads_finetuning = False

    if args.heads is not None:
        args.heads = ast.literal_eval(args.heads)
        for _, head_dict in args.heads.items():
            # priority is global args < head property_key values < head info_keys+arrays_keys
            head_keyspec = deepcopy(args.key_specification)
            update_keyspec_from_kwargs(head_keyspec, head_dict)
            head_keyspec.update(
                info_keys=head_dict.get("info_keys", {}),
                arrays_keys=head_dict.get("arrays_keys", {}),
            )
            head_dict["key_specification"] = head_keyspec
    else:
        args.heads = prepare_default_head(args)
    if args.multiheads_finetuning:
        pt_keyspec = (
            args.heads["pt_head"]["key_specification"]
            if "pt_head" in args.heads
            else deepcopy(args.key_specification)
        )
        args.heads["pt_head"] = prepare_pt_head(
            args, pt_keyspec, foundation_model_avg_num_neighbors
        )

    logging.info("===========LOADING INPUT DATA===========")
    heads = list(args.heads.keys())
    logging.info(f"Using heads: {heads}")
    logging.info("Using the key specifications to parse data:")
    for name, head_dict in args.heads.items():
        head_keyspec = head_dict["key_specification"]
        logging.info(f"{name}: {head_keyspec}")

    head_configs: List[HeadConfig] = []
    for head, head_args in args.heads.items():
        logging.info(f"=============    Processing head {head}     ===========")
        head_config = dict_head_to_dataclass(head_args, head, args)
        # don't apply user's --atomic_numbers to pt_head, that info needs to come
        # from the actual pt data
        if args.multiheads_finetuning and head_config.head_name == "pt_head":
            head_config.atomic_numbers = None

        # Handle train_file and valid_file - normalize to lists
        if hasattr(head_config, "train_file") and head_config.train_file is not None:
            head_config.train_file = normalize_file_paths(head_config.train_file)
        if hasattr(head_config, "valid_file") and head_config.valid_file is not None:
            head_config.valid_file = normalize_file_paths(head_config.valid_file)
        if hasattr(head_config, "test_file") and head_config.test_file is not None:
            head_config.test_file = normalize_file_paths(head_config.test_file)

        if (
            head_config.statistics_file is not None
            and head_config.head_name != "pt_head"
        ):
            with open(head_config.statistics_file, "r") as f:  # pylint: disable=W1514
                statistics = json.load(f)
            logging.info("Using statistics json file")
            head_config.atomic_numbers = statistics["atomic_numbers"]
            head_config.mean = statistics["mean"]
            head_config.std = statistics["std"]
            head_config.avg_num_neighbors = statistics["avg_num_neighbors"]
            head_config.compute_avg_num_neighbors = False
            if isinstance(statistics["atomic_energies"], str) and statistics[
                "atomic_energies"
            ].endswith(".json"):
                with open(statistics["atomic_energies"], "r", encoding="utf-8") as f:
                    atomic_energies = json.load(f)
                head_config.E0s = atomic_energies
                head_config.atomic_energies_dict = ast.literal_eval(atomic_energies)
            else:
                head_config.E0s = statistics["atomic_energies"]
                head_config.atomic_energies_dict = ast.literal_eval(
                    statistics["atomic_energies"]
                )
        if head_config.train_file in (
            ["mp"],
            ["matpes_pbe"],
            ["matpes_r2scan"],
            ["omat"],
        ):
            assert head_config.head_name == "pt_head", (
                "Only pt_head should use mp as train_file"
            )
            logging.info(
                f"Using filtered Materials Project data for replay ({args.num_samples_pt}, {args.filter_type_pt}, {args.subselect_pt}). "
                "You can also construct a different subset using `fine_tuning_select.py` script."
            )
            collections = assemble_replay_data(
                head_config.train_file[0], args, head_config, tag
            )
            head_config.collections = collections
        elif any(check_path_ase_read(f) for f in head_config.train_file):
            train_files_ase_list = [
                f for f in head_config.train_file if check_path_ase_read(f)
            ]
            valid_files_ase_list = None
            test_files_ase_list = None
            if head_config.valid_file:
                valid_files_ase_list = [
                    f for f in head_config.valid_file if check_path_ase_read(f)
                ]
            if head_config.test_file:
                test_files_ase_list = [
                    f for f in head_config.test_file if check_path_ase_read(f)
                ]
            config_type_weights = get_config_type_weights(
                head_config.config_type_weights
            )
            collections, atomic_energies_dict = get_dataset_from_xyz(
                work_dir=args.work_dir,
                train_path=train_files_ase_list,
                valid_path=valid_files_ase_list,
                valid_fraction=head_config.valid_fraction,
                config_type_weights=config_type_weights,
                test_path=test_files_ase_list,
                seed=args.seed,
                key_specification=head_config.key_specification,
                head_name=head_config.head_name,
                keep_isolated_atoms=head_config.keep_isolated_atoms,
                no_data_ok=(
                    args.pseudolabel_replay
                    and args.multiheads_finetuning
                    and head_config.head_name == "pt_head"
                ),
                prefix=args.name,
                band_edges=band_edges,
            )
            head_config.collections = SubsetCollection(
                train=collections.train,
                valid=collections.valid,
                tests=collections.tests,
            )
            head_config.atomic_energies_dict = atomic_energies_dict
            logging.info(
                f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
                f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}],"
            )
        head_configs.append(head_config)

    if all(
        check_path_ase_read(head_config.train_file[0]) for head_config in head_configs
    ):
        size_collections_train = sum(
            len(head_config.collections.train) for head_config in head_configs
        )
        size_collections_valid = sum(
            len(head_config.collections.valid) for head_config in head_configs
        )
        if size_collections_train < args.batch_size:
            logging.error(
                f"Batch size ({args.batch_size}) is larger than the number of training data ({size_collections_train})"
            )
        if size_collections_valid < args.valid_batch_size:
            logging.warning(
                f"Validation batch size ({args.valid_batch_size}) is larger than the number of validation data ({size_collections_valid})"
            )

    if args.multiheads_finetuning:
        logging.info(
            "==================Using multiheads finetuning mode=================="
        )
        args.loss = "universal"

        all_ase_readable = all(
            all(check_path_ase_read(f) for f in head_config.train_file)
            for head_config in head_configs
        )
        head_config_pt = filter(lambda x: x.head_name == "pt_head", head_configs)
        head_config_pt = next(head_config_pt, None)
        assert head_config_pt is not None, "Pretraining head not found"
        if all_ase_readable:
            ratio_pt_ft = (
                size_collections_train - len(head_config_pt.collections.train)
            ) / len(head_config_pt.collections.train)
            if ratio_pt_ft < args.real_pt_data_ratio_threshold:
                logging.warning(
                    f"Ratio of the number of configurations in the training set and the in the pt_train_file is {ratio_pt_ft}, "
                    f"increasing the number of configurations in the fine-tuning heads by {int(args.real_pt_data_ratio_threshold / ratio_pt_ft)}"
                )
                for head_config in head_configs:
                    if head_config.head_name == "pt_head":
                        continue
                    head_config.collections.train += (
                        head_config.collections.train
                        * int(args.real_pt_data_ratio_threshold / ratio_pt_ft)
                    )
            logging.info(
                f"Total number of configurations in pretraining: train={len(head_config_pt.collections.train)}, valid={len(head_config_pt.collections.valid)}"
            )
        else:
            logging.debug(
                "Using LMDB/HDF5 datasets for pretraining or fine-tuning - skipping ratio check"
            )

    # Atomic number table
    # yapf: disable
    for head_config in head_configs:
        if head_config.atomic_numbers is None:
            assert all(check_path_ase_read(f) for f in head_config.train_file), "Must specify atomic_numbers when using .h5 or .aselmdb train_file input"
            z_table_head = tools.get_atomic_number_table_from_zs(
                z
                for configs in (head_config.collections.train, head_config.collections.valid)
                for config in configs
                for z in config.atomic_numbers
            )
            head_config.atomic_numbers = z_table_head.zs
            head_config.z_table = z_table_head
        else:
            if head_config.statistics_file is None:
                logging.info("Using atomic numbers from command line argument")
            else:
                logging.info("Using atomic numbers from statistics file")
            zs_list = ast.literal_eval(head_config.atomic_numbers)
            assert isinstance(zs_list, list)
            z_table_head = tools.AtomicNumberTable(zs_list)
            head_config.atomic_numbers = zs_list
            head_config.z_table = z_table_head
        # yapf: enable
    all_atomic_numbers = set()
    for head_config in head_configs:
        all_atomic_numbers.update(head_config.atomic_numbers)
    z_table = AtomicNumberTable(sorted(list(all_atomic_numbers)))
    if args.foundation_model_elements and model_foundation:
        z_table = AtomicNumberTable(sorted(model_foundation.atomic_numbers.tolist()))
    logging.info(f"Atomic Numbers used: {z_table.zs}")

    # Atomic energies
    atomic_energies_dict = {}
    for head_config in head_configs:
        if head_config.atomic_energies_dict is None or len(head_config.atomic_energies_dict) == 0:
            assert head_config.E0s is not None, "Atomic energies must be provided"
            if all(check_path_ase_read(f) for f in head_config.train_file) and head_config.E0s.lower() not in ["foundation", "estimated"]:
                atomic_energies_dict[head_config.head_name] = get_atomic_energies(
                    head_config.E0s, head_config.collections.train, head_config.z_table
                )
            elif head_config.E0s.lower() == "foundation":
                assert args.foundation_model is not None
                z_table_foundation = AtomicNumberTable(
                    [int(z) for z in model_foundation.atomic_numbers]
                )
                foundation_atomic_energies = model_foundation.atomic_energies_fn.atomic_energies
                if foundation_atomic_energies.ndim > 1:
                    foundation_atomic_energies = foundation_atomic_energies.squeeze()
                    if foundation_atomic_energies.ndim == 2:
                        foundation_atomic_energies = foundation_atomic_energies[0]
                        logging.info("Foundation model has multiple heads, using the first head as foundation E0s.")
                atomic_energies_dict[head_config.head_name] = {
                    z: foundation_atomic_energies[
                        z_table_foundation.z_to_index(z)
                    ].item()
                    for z in z_table.zs
                }
            elif head_config.E0s.lower() == "estimated":
                assert args.foundation_model is not None, "Foundation model must be provided for E0s estimation"
                assert all(check_path_ase_read(f) for f in head_config.train_file), "E0s estimation requires training data in .xyz format"
                logging.info("Estimating E0s from foundation model predictions on training data")
                z_table_foundation = AtomicNumberTable(
                    [int(z) for z in model_foundation.atomic_numbers]
                )
                foundation_atomic_energies = model_foundation.atomic_energies_fn.atomic_energies
                if foundation_atomic_energies.ndim > 1:
                    foundation_atomic_energies = foundation_atomic_energies.squeeze()
                    if foundation_atomic_energies.ndim == 2:
                        foundation_atomic_energies = foundation_atomic_energies[0]
                        logging.info("Foundation model has multiple heads, using the first head for E0 estimation.")
                foundation_e0s = {
                    z: foundation_atomic_energies[
                        z_table_foundation.z_to_index(z)
                    ].item()
                    for z in z_table_foundation.zs
                }
                atomic_energies_dict[head_config.head_name] = data.estimate_e0s_from_foundation(
                    foundation_model=model_foundation,
                    foundation_e0s=foundation_e0s,
                    collections_train=head_config.collections.train,
                    z_table=head_config.z_table,
                    device=device,
                )
            else:
                atomic_energies_dict[head_config.head_name] = get_atomic_energies(head_config.E0s, None, head_config.z_table)
        else:
            atomic_energies_dict[head_config.head_name] = head_config.atomic_energies_dict

    # Atomic energies for multiheads finetuning
    if args.multiheads_finetuning:
        assert (
            model_foundation is not None
        ), "Model foundation must be provided for multiheads finetuning"
        z_table_foundation = AtomicNumberTable(
            [int(z) for z in model_foundation.atomic_numbers]
        )
        foundation_atomic_energies = model_foundation.atomic_energies_fn.atomic_energies
        if foundation_atomic_energies.ndim > 1:
            foundation_atomic_energies = foundation_atomic_energies.squeeze()
            if foundation_atomic_energies.ndim == 2:
                foundation_atomic_energies = foundation_atomic_energies[0]
                logging.info("Foundation model has multiple heads, using the first head as foundation E0s.")
        atomic_energies_dict["pt_head"] = {
            z: foundation_atomic_energies[
                z_table_foundation.z_to_index(z)
            ].item()
            for z in z_table.zs
        }
    heads = sorted(heads, key=lambda x: -1000 if x == "pt_head" else 0)
    # Padding atomic energies if keeping all elements of the foundation model
    if args.foundation_model_elements and model_foundation:
        atomic_energies_dict_padded = {}
        for head_name, head_energies in atomic_energies_dict.items():
            energy_head_padded = {}
            for z in z_table.zs:
                energy_head_padded[z] = head_energies.get(z, 0.0)
            atomic_energies_dict_padded[head_name] = energy_head_padded
        atomic_energies_dict = atomic_energies_dict_padded

    if args.model == "AtomicDipolesMACE":
        atomic_energies = None
        dipole_only = True
        args.compute_dipole = True
        args.compute_energy = False
        args.compute_forces = False
        args.compute_virials = False
        args.compute_stress = False
        args.compute_polarizability = False
    elif args.model == "AtomicDielectricMACE":
        atomic_energies = None
        dipole_only = False
        args.compute_dipole = True
        args.compute_energy = False
        args.compute_forces = False
        args.compute_virials = False
        args.compute_stress = False
        args.compute_polarizability = True
    else:
        dipole_only = False
        if args.model == "EnergyDipolesMACE":
            args.compute_dipole = True
            args.compute_energy = True
            args.compute_forces = True
            args.compute_virials = False
            args.compute_stress = False
            args.compute_polarizability = False
        elif args.model == "PolarMACE" and args.loss == "energy_forces_dipole":
            args.compute_dipole = True
            args.compute_energy = True
            args.compute_forces = True
            args.compute_virials = False
            args.compute_stress = False
            args.compute_polarizability = False
        else:
            args.compute_energy = True
            args.compute_dipole = False
            args.compute_polarizability = False
        # atomic_energies: np.ndarray = np.array(
        #     [atomic_energies_dict[z] for z in z_table.zs]
        # )
        atomic_energies = dict_to_array(atomic_energies_dict, heads)
        for head_config in head_configs:
            try:
                logging.info(f"Atomic Energies used (z: eV) for head {head_config.head_name}: " + "{" + ", ".join([f"{z}: {atomic_energies_dict[head_config.head_name][z]}" for z in head_config.z_table.zs]) + "}")
            except KeyError as e:
                raise KeyError(f"Atomic number {e} not found in atomic_energies_dict for head {head_config.head_name}, add E0s for this atomic number") from e

    # Load datasets for each head, supporting multiple files per head
    valid_sets = {head: [] for head in heads}
    train_sets = {head: [] for head in heads}

    for head_config in head_configs:
        train_datasets = []

        logging.info(f"Processing datasets for head '{head_config.head_name}'")

        # Apply pseudolabels if this is the pt_head and pseudolabeling is enabled
        if args.pseudolabel_replay and args.multiheads_finetuning and head_config.head_name == "pt_head":
            logging.info("=============    Pseudolabeling for pt_head    ===========")
            if apply_pseudolabels_to_pt_head_configs(
                foundation_model=model_foundation,
                pt_head_config=head_config,
                r_max=graph_cutoff(args),
                device=device,
                batch_size=args.batch_size,
                force_stress=args.pseudolabel_replay_compute_stress,
            ):
                logging.info("Successfully applied pseudolabels to pt_head configurations")
            else:
                logging.warning("Pseudolabeling was not successful, continuing with original configurations")

        ase_files = [f for f in head_config.train_file if check_path_ase_read(f)]
        non_ase_files = [f for f in head_config.train_file if not check_path_ase_read(f)]

        if ase_files:
            dataset = load_dataset_for_path(
            file_path=ase_files,
            # graph_cutoff, NOT r_max. With the spectral head the carrier Hamiltonian is
            # longer-ranged than the trunk, and the trunk is filtered back to r_max inside
            # the model. Building the TRAINING graph at r_max instead left the two
            # vacancy-sharing Pb -- median 5.4 A apart -- with no edge in 0/40 frames, so the
            # two-site state the R2 screen was looking for could not be represented at all.
            # graph_cutoff was previously applied only to the valid-file FALLBACK path, which
            # runs with an explicit valid_file never executes.
            r_max=graph_cutoff(args),
            z_table=z_table,
            head_config=head_config,
            heads=heads,
            collection=head_config.collections.train,
            )
            train_datasets.append(dataset)
            logging.debug(f"Successfully loaded dataset from ASE files: {ase_files}")

        for file in non_ase_files:
            dataset = load_dataset_for_path(
            file_path=file,
            r_max=graph_cutoff(args),
            z_table=z_table,
            head_config=head_config,
            heads=heads,
            )
            train_datasets.append(dataset)
            logging.debug(f"Successfully loaded dataset from non-ASE file: {file}")

        if not train_datasets:
            raise ValueError(f"No valid training datasets found for head {head_config.head_name}")

        train_sets[head_config.head_name] = combine_datasets(train_datasets, head_config.head_name)

        if head_config.valid_file:
            valid_datasets = []

            valid_ase_files = [f for f in head_config.valid_file if check_path_ase_read(f)]
            valid_non_ase_files = [f for f in head_config.valid_file if not check_path_ase_read(f)]

            if valid_ase_files:
                valid_dataset = load_dataset_for_path(
                    file_path=valid_ase_files,
                    r_max=graph_cutoff(args),
                    z_table=z_table,
                    head_config=head_config,
                    heads=heads,
                    collection=head_config.collections.valid,
                )
                valid_datasets.append(valid_dataset)
                logging.debug(f"Successfully loaded validation dataset from ASE files: {valid_ase_files}")
            for valid_file in valid_non_ase_files:
                valid_dataset = load_dataset_for_path(
                file_path=valid_file,
                r_max=graph_cutoff(args),
                z_table=z_table,
                head_config=head_config,
                heads=heads,
            )
                valid_datasets.append(valid_dataset)
                logging.debug(f"Successfully loaded validation dataset from {valid_file}")

            # Combine validation datasets
            if valid_datasets:
                valid_sets[head_config.head_name] = combine_datasets(valid_datasets, f"{head_config.head_name}_valid")
                logging.info(f"Combined validation datasets for {head_config.head_name}")

        # If no valid file is provided but collection exist, use the validation set from the collection
        if head_config.valid_file is None and head_config.collections.valid:
            valid_sets[head_config.head_name] = [
                data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=graph_cutoff(args), heads=heads
                )
                for config in head_config.collections.valid
            ]
        if not valid_sets[head_config.head_name]:
            raise ValueError(f"No valid datasets found for head {head_config.head_name}, please provide a valid_file or a valid_fraction")

        # Create data loader for this head
        if isinstance(train_sets[head_config.head_name], list):
            dataset_size = len(train_sets[head_config.head_name])
        else:
            dataset_size = len(train_sets[head_config.head_name])
        logging.info(f"Head '{head_config.head_name}' training dataset size: {dataset_size}")

        train_loader_head = torch_geometric.dataloader.DataLoader(
            dataset=train_sets[head_config.head_name],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=(not args.lbfgs),
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
        head_config.train_loader = train_loader_head

    # concatenate all the trainsets
    train_set = ConcatDataset([train_sets[head] for head in heads])
    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=(not args.lbfgs),
            seed=args.seed,
        )
        valid_samplers = {}
        for head, valid_set in valid_sets.items():
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_set,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=args.seed,
            )
            valid_samplers[head] = valid_sampler

    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=(train_sampler is None and not args.lbfgs),
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )

    valid_loaders = {heads[i]: None for i in range(len(heads))}
    if not isinstance(valid_sets, dict):
        valid_sets = {"Default": valid_sets}
    for head, valid_set in valid_sets.items():
        valid_loaders[head] = torch_geometric.dataloader.DataLoader(
            dataset=valid_set,
            batch_size=args.valid_batch_size,
            sampler=valid_samplers[head] if args.distributed else None,
            shuffle=False,
            drop_last=False,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )

    loss_fn = get_loss_fn(args, dipole_only, args.compute_dipole)
    args.avg_num_neighbors = get_avg_num_neighbors(head_configs, args, train_loader, device)

    # COUNT NEIGHBOURS AT r_max, NOT AT THE GRAPH CUTOFF. `avg_num_neighbors` divides every
    # message in the trunk, and the trunk's messages only run over edges inside `r_max`. With
    # the carrier head the loader builds its graph at the CARRIER cutoff instead -- 10 A here
    # against r_max 5.0 -- so the count comes back about eight times too large and every
    # message is divided by eight times too much.
    #
    # Measured, not estimated: the Stage-A base was trained without the head and has 14.08; a
    # run with the head on the same data computes 112.5. For a Stage-B run
    # `load_stage_a_base` now repairs this by inheriting the checkpoint's value, but a
    # from-scratch arm has no checkpoint to inherit from -- so without this the staging
    # control would differ from the staged arm in trunk normalisation, which has nothing to do
    # with staging. Rescaling by the edge-count ratio keeps one number and one code path.
    if (getattr(args, "defect_spectral_head", False)
            and float(graph_cutoff(args)) > float(args.r_max)
            and all(h.compute_avg_num_neighbors for h in head_configs)):
        from mace.modules.defect_reach import count_edges_within

        # Over several batches, not one. The from-scratch arm has no checkpoint to inherit a
        # normalisation from, so this ratio IS its trunk normalisation, and one batch of eight
        # frames is a sample of a dataset property. On the first measurement a single batch
        # gave 13.93 against Stage A's 14.08 -- 1.1% out, which is the sampling error and not
        # a disagreement, but it is an avoidable asymmetry between the two arms when the only
        # thing they are meant to differ in is staging.
        wide = narrow = 0
        for k, probe in enumerate(train_loader):
            if k >= 20:
                break
            w, n = count_edges_within(probe, float(args.r_max))
            wide += w
            narrow += n
        if wide > 0:
            ratio = float(narrow) / float(wide)
            rescaled = float(args.avg_num_neighbors) * ratio
            logging.warning(
                "avg_num_neighbors was computed on the %.1f A carrier graph (%.2f); the "
                "trunk only passes messages inside r_max = %.1f A, so it is rescaled by the "
                "edge-count ratio %.4f (%d of %d edges over %d batches) to %.2f. Dividing "
                "every message by the carrier-graph count would be about %.1fx too much.",
                float(graph_cutoff(args)), float(args.avg_num_neighbors),
                float(args.r_max), ratio, narrow, wide, min(k + 1, 20), rescaled,
                1.0 / max(ratio, 1e-9))
            args.avg_num_neighbors = rescaled

    # Model
    model, output_args = configure_model(args, train_loader, atomic_energies, model_foundation, heads, z_table, head_configs)
    model.to(device)

    # Stage B (component S): start from a converged, frozen base so the correction is not
    # racing the base for the same signal. Loading the weights and zeroing their learning
    # rate are separate switches, and only the pair gives Stage B -- warn rather than let a
    # half-configured run look like a staged one.
    if getattr(args, "defect_base_init", None):
        from mace.modules.defect_stage import load_stage_a_base

        if model.__class__.__name__ != "MACEDefect":
            raise RuntimeError("--defect_base_init only applies to MACEDefect")
        load_stage_a_base(model, args.defect_base_init, device=device)
        if float(getattr(args, "base_lr_factor", 1.0)) != 0.0:
            logging.warning(
                "--defect_base_init was given but --base_lr_factor is "
                f"{getattr(args, 'base_lr_factor', 1.0)}, not 0.0, so the loaded base will "
                "drift during training. That is a warm start, not Stage B.")

    # ------------------------------------------------------------------ Stage-3 protocol
    #
    # The five things every Stage-3 result in this programme was produced with, and which
    # the production trainer did not have: Harrison initialisation, c-shift calibration,
    # linear warmup, the initialisation gate, and the head-only trainable mask. They live in
    # mace.modules.defect_protocol and defect-perovskite/stage_run.py calls the SAME
    # functions, so the two drivers cannot drift -- which is the only sense in which "the
    # joint run comes from config" can be true.
    #
    # This half (initialisation and the mask) runs before the optimiser is built, because
    # requires_grad decides which parameter groups exist. The other half (c-shift, the gate,
    # warmup and the post-step projection) needs the data loaders and runs further down.
    protocol_on = bool(getattr(args, "defect_protocol", False))
    if protocol_on:
        from mace.modules import defect_protocol

        if model.__class__.__name__ != "MACEDefect":
            raise RuntimeError("--defect_protocol only applies to MACEDefect")
        if getattr(model, "spectral", None) is None:
            raise RuntimeError(
                "--defect_protocol needs a carrier head; pass --defect_spectral_head and "
                "--defect_counting_head")
        bond_length = float(getattr(args, "defect_protocol_bond_length", 0.0) or 0.0)
        if bond_length <= 0.0:
            raise RuntimeError(
                "--defect_protocol needs --defect_protocol_bond_length. eps0 = 0 leaves "
                "every site degenerate -- the atomic limit, where the bond order vanishes "
                "and nothing can move the hoppings -- so the Harrison initialisation is not "
                "optional, and its scale is a measured property of the host rather than a "
                "constant this code is entitled to guess.")
        harrison_applied = defect_protocol.apply_harrison(
            model, model.atomic_numbers, bond_length)
        logging.info(
            f"Stage-3 protocol: Harrison initialisation at bond length {bond_length} A "
            f"-> {'applied' if harrison_applied else 'NOT APPLIED'}")

        # The on-site correction channel starts at exactly zero output when asked. What it
        # removes is a measured global gauge -- a near-uniform +0.27 eV on every atom, which
        # contributes no force and so is invisible to a forces-only objective. Starting from
        # zero means whatever the joint run's energy loss puts there was put there, not
        # inherited from a random draw and then frozen in by a flat direction.
        if bool(getattr(args, "defect_protocol_zero_on_site", False)):
            if defect_protocol.zero_on_site_correction(model):
                logging.info(
                    "Stage-3 protocol: on-site correction zero-initialised (the +0.27 eV "
                    "uniform gauge removed at step 0)")
            else:
                logging.warning(
                    "Stage-3 protocol: --defect_protocol_zero_on_site was given but no "
                    "on-site correction layer was found; NOTHING was zeroed.")

        freeze_z = bool(getattr(args, "defect_protocol_freeze_z", False))
        if bool(getattr(args, "defect_protocol_head_only", False)):
            frozen = 0
            for name, param in model.named_parameters():
                keep = defect_protocol.trainable_mask(name, freeze_z=freeze_z)
                param.requires_grad_(keep)
                frozen += int(not keep)
            logging.info(
                f"Stage-3 protocol: head-only, {frozen} parameter tensors frozen"
                + (", Z pinned" if freeze_z else ""))
        elif freeze_z:
            for name, param in model.named_parameters():
                if name.startswith("madelung."):
                    param.requires_grad_(False)
            logging.info("Stage-3 protocol: Z pinned at its initialisation")
        # Section 2.2 of the Stage A' spec: the head-only mask above marks the long-range
        # charge MLPs trainable (they are correction parameters); `lr_freeze` re-pins them
        # here, AFTER the mask, or the flag would be silently undone by it.
        if getattr(model, "lr_freeze", False):
            model._apply_long_range_policy()
            logging.info("Long-range branch frozen at its physical initialisation "
                         "(--defect_lr_freeze)")

    if model.__class__.__name__ == "MACEDefect":
        # Ship the band edges that referenced the labels with the model, so inference
        # can undo the referencing with exactly the constants training used rather than
        # with re-derived ones (plan section 7.2).
        registry: Dict[str, Dict[str, float]] = {}
        for head_config in head_configs:
            collections = getattr(head_config, "collections", None)
            if collections is None:
                continue
            registry.update(data.collect_band_edge_registry(collections.train))
        model.band_edge_registry = registry
        # Counter vectors the level-mode gauge probe is evaluated at (stage D-opt). Taken
        # from the training set rather than configured, so the penalty constrains exactly
        # the directions the data actually populates; n = 0 is excluded because its
        # correction is identically zero and constrains nothing.
        #
        # Populated whether or not the penalty is switched on, because stage D-opt asks
        # for mean(u^c) every epoch regardless: it is the free diagnostic that says
        # whether the level mode is drifting, and it is worth nothing if it only exists
        # once you already suspected a problem. `--defect_gauge_weight` gates the loss
        # term alone.
        observed: List[Tuple[int, ...]] = []
        for head_config in head_configs:
            collections = getattr(head_config, "collections", None)
            if collections is None:
                continue
            for config in collections.train:
                counts = config.properties.get("carrier_counts")
                if counts is None:
                    continue
                flat = counts.tolist() if hasattr(counts, "tolist") else counts
                vector = tuple(int(v) for v in flat)
                if sum(vector) > 0 and vector not in observed:
                    observed.append(vector)
        if observed:
            model.register_buffer(
                "gauge_counters",
                torch.as_tensor(observed, dtype=torch.long, device=device),
                persistent=True,
            )
            state = (
                f"penalty at weight {args.defect_gauge_weight}"
                if float(args.defect_gauge_weight) > 0.0
                else "diagnostic only, penalty off"
            )
            logging.info(
                f"Level-mode gauge probe on {len(observed)} counter vector(s) "
                f"{observed} -- {state}"
            )
        elif float(args.defect_gauge_weight) > 0.0:
            logging.warning(
                "--defect_gauge_weight > 0 but no carrier-bearing counters were "
                "found in the training set; the gauge penalty will be inert"
            )
        if registry:
            logging.info(f"Recorded band edges for {len(registry)} (host, size) pairs")
        else:
            logging.warning(
                "No band edges recorded with the model; inference will need them "
                "supplied explicitly to report raw-scale energies"
            )

    if args.lora:
        lora_rank = args.lora_rank
        lora_alpha = args.lora_alpha

        logging.info(
            "Injecting LoRA layers with rank=%s and alpha=%s",
            lora_rank,
            lora_alpha,
        )

        logging.info(
            "Original model has %s trainable parameters.",
            tools.count_parameters(model),
        )

        model = inject_LoRAs(model, rank=lora_rank, alpha=lora_alpha)

        logging.info(
            "Model with LoRA has %s trainable parameters.",
            tools.count_parameters(model),
        )

    logging.info("===========OPTIMIZER INFORMATION===========")
    logging.info(f"Using {args.optimizer.upper()} as parameter optimizer")
    logging.info(f"Batch size: {args.batch_size}")
    if args.ema:
        logging.info(f"Using Exponential Moving Average with decay: {args.ema_decay}")
    logging.info(
        f"Number of gradient updates: {int(args.max_num_epochs*len(train_set)/args.batch_size)}"
    )
    logging.info(f"Learning rate: {args.lr}, weight decay: {args.weight_decay}")
    logging.info(loss_fn)

    # Cueq and OEQ conversion
    if model.__class__.__name__ == "MACEDefect":
        # cuEquivariance is verified against e3nn for the short-range model: energies
        # agree to 9e-14 and forces to 2e-16 in float64 (test_defect_cueq.py). The
        # correction heads are plain dense MLPs, so nothing there is converted; the gain
        # is entirely in the trunk, which is where the time goes.
        if args.enable_oeq:
            raise NotImplementedError(
                "OpenEquivariance conversion of MACEDefect is untested; use "
                "--enable_cueq=True instead"
            )
        # cuEq conversion with the long-range branch was blocked as untested. It is now
        # verified -- see defect-example/verify_cueq_long_range.py, which is the
        # regression test for this and should be re-run if the conversion changes.
        #
        # The block was protecting against something real. The conversion rebuilds the
        # model from `extract_config_mace_model`, so a constructor argument missing from
        # that extractor silently reverts to its default. `freeze_amplitude` was missing
        # and defaults to False, so converting a frozen-amplitude model handed back a
        # *trainable* screening amplitude -- which then reads as a fitted screening
        # constant rather than the input gauge it is. That extractor now captures it (with
        # `high_precision_softmax`, `zero_u_init` and `correction_trunk`, which were also
        # missing but happen to equal their defaults everywhere).
        #
        # Measured on a real long-range model after that fix: amplitude bit-identical and
        # still frozen, E_LR still contributing, energies exact and forces agreeing to
        # 6e-6 eV/A -- float32 summation-order noise.
        if args.enable_cueq and getattr(model, "use_long_range", False):
            logging.info(
                "cuEquivariance with use_long_range=True: verified equivalent to e3nn "
                "for MACEDefect, including the frozen screening amplitude"
            )
        if args.enable_cueq:
            logging.info(
                "cuEquivariance pays off with width: measured 2.4x at 128 channels / "
                "max_L=1, but 0.85x (a slowdown) at 8 channels, where kernel overhead "
                "dominates. Benchmark before using it for small debug runs."
            )
    if args.enable_cueq and args.enable_oeq:
        logging.warning(
            "Both CUEQ and OEQ are enabled, using CUEQ for training. "
            "To use OEQ, disable CUEQ with --disable_cueq."
        )
        args.enable_oeq = False
    if args.enable_cueq and not args.only_cueq:
        logging.info("Converting model to CUEQ for accelerated training")
        assert model.__class__.__name__ in [
            "MACE",
            "ScaleShiftMACE",
            "MACELES",
            "PolarMACE",
            "MagneticScaleShiftMACE",
            "AtomicDielectricMACE",
            "MACEDefect",
        ]
        model = run_e3nn_to_cueq(deepcopy(model), device=device)
    if args.enable_oeq:
        logging.info("Converting model to OEQ for accelerated training")
        assert model.__class__.__name__ in [
            "MACE",
            "ScaleShiftMACE",
            "MACELES",
            "PolarMACE",
            "MagneticScaleShiftMACE",
        ]
        model = run_e3nn_to_oeq(deepcopy(model), device=device)

    # Optimizer
    if (
        hasattr(model, "onebody_magmombasis_coeffs")
        and not args.train_one_body_contribution
    ):
        model.onebody_magmombasis_coeffs.requires_grad_(False)
    param_options = get_params_options(args, model)

    optimizer: torch.optim.Optimizer
    optimizer = get_optimizer(args, param_options)
    logging.info("=== Layer's learning rates ===")
    for name, p in model.named_parameters():
        st = optimizer.state.get(p, {})
        if st:
            logging.info(f"Param: {name}: {list(st.keys())}")

    for i, param_group in enumerate(optimizer.param_groups):
        logging.info(f"Param group {i}: lr = {param_group['lr']}")

    if args.device == "xpu":
        logging.info("Optimzing model and optimzier for XPU")
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    logger = tools.MetricsLogger(
        directory=args.results_dir, tag=tag + "_train"
    )  # pylint: disable=E1123

    lr_scheduler = LRScheduler(optimizer, args)

    swa: Optional[tools.SWAContainer] = None
    swas = [False]
    if args.swa:
        swa, swas = get_swa(args, model, optimizer, swas, dipole_only)

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=args.start_swa,
    )

    start_epoch = 0
    restart_lbfgs = False
    opt_start_epoch = None
    if args.restart_latest:
        try:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=True,
                device=device,
            )
        except Exception:  # pylint: disable=W0703
            try:
                opt_start_epoch = checkpoint_handler.load_latest(
                    state=tools.CheckpointState(model, optimizer, lr_scheduler),
                    swa=False,
                    device=device,
                )
            except Exception: # pylint: disable=W0703
                restart_lbfgs = True
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)

    if args.lbfgs:
        logging.info("Switching optimizer to LBFGS")
        optimizer = LBFGS(model.parameters(),
                          history_size=200,
                          max_iter=20,
                          line_search_fn="strong_wolfe")
        if restart_lbfgs:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
            if opt_start_epoch is not None:
                start_epoch = opt_start_epoch

    if args.wandb:
        setup_wandb(args)
    if args.distributed:
        # device_ids is only valid for single-device CUDA modules; CPU (gloo)
        # requires device_ids=None.
        distributed_model = DDP(
            model, device_ids=[local_rank] if args.device == "cuda" else None
        )
    else:
        distributed_model = None


    train_valid_data_loader = {}
    for head_config in head_configs:
        data_loader_name = "train_" + head_config.head_name
        train_valid_data_loader[data_loader_name] = head_config.train_loader
    for head, valid_loader in valid_loaders.items():
        data_load_name = "valid_" + head
        train_valid_data_loader[data_load_name] = valid_loader

    if args.plot and args.plot_frequency > 0:
        try:
            plotter = TrainingPlotter(
                results_dir=logger.path,
                heads=heads,
                table_type=args.error_table,
                train_valid_data=train_valid_data_loader,
                test_data={},
                output_args=output_args,
                device=device,
                plot_frequency=args.plot_frequency,
                distributed=args.distributed,
                swa_start=swa.start if swa else None,
                plot_interaction_e=args.plot_interaction_e
                )
        except Exception as e:  # pylint: disable=W0718
            logging.debug(f"Creating Plotter failed: {e}")
    else:
        plotter = None

    if args.dry_run:
        logging.info("DRY RUN mode enabled. Stopping now.")
        return

    if args.device == "xpu":
        try:
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
        except ImportError as e:
            logging.error(
                "Intel Extension for PyTorch not found, but XPU device was specified. "
                "Please install it to use XPU device."
            )

    # The novelty descriptor's scale constants are dataset statistics and must be fixed
    # before the first forward: under logit seeding s_hat is an input, so they are part
    # of the model definition and are saved as buffers.
    if model.__class__.__name__ == "MACEDefect" and getattr(model, "logit_seed", False):
        from mace.modules.defect_seed import calibrate_novelty

        calibrate_novelty(model=model, data_loader=train_loader, device=device)

    defect_seed_hook = None
    anneal_seed = (
        model.__class__.__name__ == "MACEDefect"
        and getattr(model, "logit_seed", False)
        and getattr(args, "defect_seed_anneal", False)
    )
    if anneal_seed:
        from mace.modules.defect_seed import anneal_logit_seed

        gamma_init = model.logit_seed_gamma.detach().clone()
        # The schedule owns gamma outright while annealing. Left trainable, the optimizer
        # moves it after each epoch's update and the ratchet then locks in whatever it
        # did -- observed ending at [0.0, -0.0013, 0.0, -0.092], i.e. a live and
        # SIGN-FLIPPED bias that pushes attention away from novel atoms. The whole point
        # of the anneal is that the converged model is bias-free, so gamma cannot also be
        # a free parameter.
        model.logit_seed_gamma.requires_grad_(False)

    if model.__class__.__name__ == "MACEDefect":

        def defect_seed_hook(epoch: int, current_model) -> None:
            # Stage-3 linear warmup, applied to the learning rate BEFORE the epoch's
            # gradient steps. The counting head's correction is eV-scale at step 0, so the
            # first steps see gradients three orders larger than the converged ones and a
            # seed can be thrown somewhere it cannot return from -- observed, not supposed.
            #
            # WHY AN OVERLAY AND NOT A SECOND SCHEDULER. The run's own scheduler is
            # ReduceLROnPlateau or ExponentialLR and both write param_group['lr'] directly,
            # so a second scheduler would have two objects owning one field. Worse, this
            # hook fires BEFORE lr_scheduler.step(): writing an absolute rate here would let
            # ExponentialLR compound its decay on top of the warmup and then have the next
            # epoch's write discard it, quietly costing the run gamma^warm of its schedule.
            #
            # So the warmup is a pure multiplicative overlay that is REMOVED before it is
            # re-applied. The scheduler owns the trajectory throughout; the overlay only
            # scales whatever it currently says. At epoch == warm the factor is 1.0, so the
            # overlay comes off and is never put back.
            warm = int(getattr(args, "defect_protocol_warmup", 0) or 0)
            if protocol_on and warm > 0 and epoch <= warm:
                from mace.modules.defect_protocol import warmup_factor

                previous = getattr(defect_seed_hook, "_warmup_factor", 1.0)
                factor = warmup_factor(epoch, warm)
                for group in optimizer.param_groups:
                    group["lr"] = float(group["lr"]) / previous * factor
                defect_seed_hook._warmup_factor = factor
                if epoch < warm:
                    logging.info(f"Stage-3 warmup: epoch {epoch}, lr x{factor:.3f}")
                if epoch == start_epoch and start_epoch > 0:
                    # Audit only, not a fix. Resuming mid-warmup starts the overlay at 1.0
                    # while the checkpointed rate still carries the old factor, so the
                    # remaining warmup epochs are double-scaled. Logged so a resumed run's
                    # schedule is readable from its own log rather than inferred.
                    logging.warning(
                        f"Stage-3 warmup: RESUMED at epoch {start_epoch} inside the "
                        f"{warm}-epoch warmup. The overlay restarts from 1.0 while the "
                        f"checkpointed lr still carries the previous factor, so the "
                        f"remaining warmup epochs are scaled twice. Audit only.")


            # The **absolute** epoch, taken from the trainer. A counter local to the loss
            # would restart at zero on every resume and silently re-serve the size-hinge
            # warmup -- the same shape of bug as the gamma anneal restarting from scratch.
            loss_fn.current_epoch = int(epoch)
            # The model needs it too: the long-range branch is gated on the same absolute
            # epoch, and it must reach the module that is actually being trained.
            target = getattr(current_model, "module", current_model)
            if hasattr(target, "current_epoch"):
                with torch.no_grad():
                    target.current_epoch.fill_(int(epoch))
            # Gauge visibility. `c_shift` and the mean on-site correction are the two
            # directions a forces-only objective cannot see -- both move the whole spectrum
            # and neither changes a force -- so they are logged every epoch rather than
            # inspected once at the end, when a drift is already baked in.
            if bool(getattr(args, "defect_neutral_size_upweight_energy", False)):
                from mace.data.two_size import realised_shares as _shares

                s = _shares(train_set, population="neutral")
                logging.info("Realised neutral large-cell shares: epoch %d "
                             "realised_share_E %.4f realised_share_F %.4f",
                             epoch, s["energy"], s["forces"])
            if base_cache_rng is not None and getattr(target, "_base_cache", None) is not None:
                from mace.modules import defect_cache as _dc

                drift = _dc.check_drift(target, train_set, device, base_cache_rng)
                if drift:
                    logging.info("Base cache drift guard: epoch %d frame %d (%d atoms) "
                                 "|dE| %.2e eV  max|dF| %.2e eV/A",
                                 epoch, drift["frame"], drift["n_atoms"],
                                 drift["d_energy"], drift["d_forces"])
            if protocol_on:
                head = getattr(target, "spectral", None)
                if head is not None:
                    parts = [f"c_shift {float(head.c_shift):+.4f}"]
                    site = getattr(getattr(head, "h", None), "site", None)
                    if site is not None:
                        last = [m for m in site.modules()
                                if isinstance(m, torch.nn.Linear)]
                        if last:
                            parts.append(f"|W_site| {float(last[-1].weight.abs().mean()):.5f}")
                            if last[-1].bias is not None:
                                parts.append(
                                    f"b_site {float(last[-1].bias.mean()):+.5f}")
                    z = getattr(target, "madelung", None)
                    if z is not None and hasattr(z, "z"):
                        parts.append("Z " + " ".join(f"{v:+.3f}"
                                                     for v in z.z.detach().tolist()))
                    logging.info("Gauge: epoch %d  %s", epoch, "  ".join(parts))

            # Bandwidth anneal. This runs in epoch_hook rather than post_eval_hook because
            # it must be in place BEFORE the epoch's gradient steps, not chosen after them.
            #
            # The flag was declared, passed by the launcher and documented, but nothing ever
            # wrote hop_scale, so it stayed at the 1.0 it is registered with and the anneal
            # arm was identical to the plain arm. Every unit test set hop_scale by hand, so
            # the head's behaviour was covered while the schedule that drives it was not.
            s0 = float(getattr(args, "defect_spectral_anneal_s0", 0.0) or 0.0)
            spectral = getattr(target, "spectral", None)
            if s0 > 0 and spectral is not None and hasattr(spectral, "hop_scale"):
                from mace.modules.defect_spectral import bandwidth_scale

                span = int(getattr(args, "defect_spectral_anneal_epochs", 20))
                scale = bandwidth_scale(epoch, s0, span)
                with torch.no_grad():
                    spectral.hop_scale.fill_(scale)
                if epoch <= span:
                    logging.info(
                        f"Bandwidth anneal: epoch {epoch}, hop_scale {scale:.4f} "
                        f"(s0={s0}, over {span} epochs)")

            if not anneal_seed:
                return
            report = anneal_logit_seed(
                model=current_model,
                data_loader=train_loader,
                device=device,
                epoch=epoch,
                zero_by_epoch=args.defect_seed_anneal_epochs,
                gamma_init=gamma_init,
                gate=args.defect_seed_gate,
                max_drop=args.defect_seed_max_drop,
            )
            if report:
                # gap_site and the raw gap are logged TOGETHER, permanently. The pair is
                # the diagnostic: a raw gap of 24.5 at epoch 1 alongside a gap_site near
                # zero is a species ordering, not a found defect, and reading only the raw
                # gap is what retired the seed into a collapsed attention.
                extra = ""
                if "gap_site" in report:
                    extra = (
                        f", gap_site={report['gap_site']}"
                        f" (seeded {report['gap_site_seeded']})"
                    )
                logging.info(
                    f"Epoch {epoch}: logit-seed anneal gamma={report['gamma']}, "
                    f"intrinsic gap={report['gap_intrinsic']}{extra}"
                )

    # Section 10: record the settings *with the model*, not only in the checkpoint args.
    # The args cover a restart; they do not travel with the artifact someone is handed,
    # and these values define the cell-size range the correction is valid over, which
    # should be readable off the model wherever it is used.
    if model.__class__.__name__ == "MACEDefect":
        model.size_extensivity_settings = {
            "size_ratio": float(args.defect_size_ratio),
            "size_tol": float(args.defect_size_tol),
            "size_weight": float(args.defect_size_weight),
            "size_warmup_epochs": int(args.defect_size_warmup_epochs),
            "gauge_weight": float(args.defect_gauge_weight),
        }

    # The hook fires at the top of each epoch, but `train` evaluates the validation set
    # once before the loop begins. Without this, that first evaluation on a resumed run
    # would report a loss missing the size term while every later one includes it.
    if hasattr(loss_fn, "current_epoch"):
        loss_fn.current_epoch = int(start_epoch)
    _target = getattr(model, "module", model)
    if hasattr(_target, "current_epoch"):
        with torch.no_grad():
            _target.current_epoch.fill_(int(start_epoch))


    # Two-timescale base (plan T4): frozen while attention settles, then released slowly,
    # with a LABEL-FREE rollback guard. Built only when a release epoch is configured, so
    # every other run is untouched.
    base_release = None
    if int(getattr(args, "defect_base_release_epoch", 0)) > 0:
        from mace.modules.defect_release import BaseRelease

        base_release = BaseRelease(
            model, optimizer,
            release_epoch=int(args.defect_base_release_epoch),
            release_factor=float(args.defect_base_release_factor),
        )
        logging.info(
            f"Base release scheduled at epoch {args.defect_base_release_epoch} "
            f"with factor {args.defect_base_release_factor}")

    def _release_hook(epoch, _model, _opt, eval_metrics):
        """Release and guard, on label-free diagnostics only.

        alpha_overlap is computed by the model's own diagnostics against the attention
        recorded at the release epoch; n_eff and the active/null channel ratio come from the
        same per-epoch carrier metrics that are already logged. None of them needs the
        vacancy assignment.
        """
        if base_release is None:
            return
        state = {}
        for key, name in (("defect_n_eff", "n_eff"),
                          ("defect_null_ratio", "null_ratio")):
            value = eval_metrics.get(key)
            if value is not None:
                state[name] = float(value)

        # alpha_overlap: cosine similarity between the current attention over the validation
        # set and the attention at the release epoch. The vector bookkeeping lives here so
        # BaseRelease keeps a float-only interface; the reference is captured on the first
        # call at or after the release epoch, which is the same epoch BaseRelease snapshots
        # the base weights.
        vec = eval_metrics.get("defect_alpha_vec")
        if vec is not None:
            vec = vec.detach().reshape(-1).float()
            ref = getattr(_release_hook, "_alpha_ref", None)
            if ref is not None and ref.numel() == vec.numel():
                denom = float(ref.norm() * vec.norm())
                if denom > 0:
                    state["alpha_overlap"] = float((ref @ vec) / denom)
            if ref is None and epoch >= int(args.defect_base_release_epoch):
                _release_hook._alpha_ref = vec.clone()

        base_release.on_epoch_start(epoch, base_lr=float(args.lr), state=state)
        base_release.on_epoch_end(epoch, state=state)

    # Two-size upweight, applied to the dataset the loss iterates. Placed here, next to the
    # reach assertion, because both are properties of the data the optimiser actually sees --
    # the class of thing this project has repeatedly got wrong by configuring somewhere else.
    realised_shares = {}
    if float(getattr(args, "defect_two_size_upweight", 0.0) or 0.0) > 0:
        from mace.data.two_size import apply_two_size_upweight

        factor, share = apply_two_size_upweight(
            train_set, target_share=float(args.defect_two_size_upweight))
        realised_shares["charged_large"] = share
        logging.info(
            f"Two-size upweight: factor {factor:.2f}, realised charged-force-loss share "
            f"{share:.1%} (target {float(args.defect_two_size_upweight):.0%})")

    # The NEUTRAL large cells get the same treatment, and separately. They are the base
    # branch's only direct constraint at large d -- the rest of the neutral set stops around
    # 6.0 A, and above that the base extrapolates, which is the measured +0.132 eV/A error in
    # the carrier-free 79-atom residual. In the joint run the base is no longer frozen, so
    # this is what lets it LEARN that region instead of leaving the correction to absorb the
    # difference, which is exactly the leakage the adoption rule tests for.
    if float(getattr(args, "defect_neutral_size_upweight", 0.0) or 0.0) > 0:
        from mace.data.two_size import apply_neutral_size_upweight, apply_size_upweight

        if bool(getattr(args, "defect_neutral_size_upweight_energy", False)):
            # Stage A' section 1: the same target in the energy loss AND the force loss.
            both = apply_size_upweight(
                train_set, population="neutral",
                target_share=float(args.defect_neutral_size_upweight),
                channels=("energy", "forces"))
            realised_shares["neutral_large_E"] = both["energy"][1]
            realised_shares["neutral_large_F"] = both["forces"][1]
            logging.info(
                f"Neutral size upweight, both channels: energy factor {both['energy'][0]:.2f} "
                f"-> share {both['energy'][1]:.1%}; forces factor {both['forces'][0]:.2f} "
                f"-> share {both['forces'][1]:.1%} (target "
                f"{float(args.defect_neutral_size_upweight):.0%})")
        else:
            factor, share = apply_neutral_size_upweight(
                train_set, target_share=float(args.defect_neutral_size_upweight))
            realised_shares["neutral_large"] = share
            logging.info(
                f"Neutral two-size upweight: factor {factor:.2f}, realised neutral-force-loss "
                f"share {share:.1%} (target "
                f"{float(args.defect_neutral_size_upweight):.0%})")
    if realised_shares:
        logging.info("Realised large-cell shares: %s", json.dumps(
            {k: round(v, 4) for k, v in realised_shares.items()}, sort_keys=True))

    # ------------------------------------------- Stage-3 protocol, the data-dependent half
    #
    # c-shift, the initialisation gate, warmup and the post-step projection. Everything here
    # runs on the loss's OWN loader, for the same reason the reach assertion below does: a
    # calibration performed on a differently-built batch is a calibration for a run that is
    # not happening.
    # ------------------------------------------------ Stage A' section 2.5: precision, cache
    #
    # The mixed policy keeps the DATA in float64 (labels at float32 lose ~5e-5 eV on a
    # 500 eV cell, which is not the loss's precision) and casts only the trunk down, so the
    # process default must be float64 for it to mean what it says.
    if (model.__class__.__name__ == "MACEDefect"
            and getattr(model, "precision_policy", "uniform") == "mixed"
            and torch.get_default_dtype() != torch.float64):
        raise RuntimeError(
            "--defect_precision_policy mixed needs --default_dtype float64: the policy casts "
            "the trunk to float32 and keeps everything else, data included, at float64")
    base_cache_rng = None
    if bool(getattr(args, "defect_base_cache", False)):
        from mace.modules import defect_cache

        if model.__class__.__name__ != "MACEDefect":
            raise RuntimeError("--defect_base_cache only applies to MACEDefect")
        # Later-block invariant readouts feed only the (detached) long-range charges; they
        # are cached, so they must not train. Freeze them before the cacheability check.
        for i, ro in enumerate(model.defect_feature_readouts):
            if i > 0:
                for p in ro.parameters():
                    p.requires_grad_(False)
        defect_cache.require_cacheable(model)
        n_keys = defect_cache.attach_frame_keys(train_set, z_table=z_table)
        for _head, _vset in valid_sets.items():
            n_keys += defect_cache.attach_frame_keys(_vset, z_table=z_table)
        logging.info("Base cache: frame keys attached to %d frames", n_keys)
        probe_batch = next(iter(train_loader))
        table_before, wall_before = defect_cache.profile_step(
            model, loss_fn, probe_batch, device, output_args)
        logging.info("Profile, full forward (one training step, mean of 3): %.3f s/step\n%s",
                     wall_before, table_before)
        cache_dir = str(getattr(args, "defect_base_cache_dir", "") or args.checkpoints_dir)
        os.makedirs(cache_dir, exist_ok=True)
        digest = defect_cache.base_checksum(model)
        cache_path = os.path.join(cache_dir, f"{args.name}_basecache_{digest.hex()[:12]}.pt")
        cache = defect_cache.build_base_cache(
            model, [train_loader] + list(valid_loaders.values()), device, path=cache_path)
        model.set_base_cache(cache)
        table_after, wall_after = defect_cache.profile_step(
            model, loss_fn, probe_batch, device, output_args)
        logging.info("Profile, cached forward (one training step, mean of 3): %.3f s/step "
                     "(%.2fx)\n%s", wall_after, wall_before / max(wall_after, 1e-9),
                     table_after)
        base_cache_rng = np.random.default_rng(int(args.seed))

    protocol_post_step = None
    if protocol_on:
        from mace.modules import defect_protocol
        from mace.modules.defect_counting import VALENCE

        init_batch = next(iter(train_loader)).to(device)
        # EVERY charged frame the loader holds, in one pass -- not the first batch. `c` is the
        # head's energy zero, set once from data, so a value that depends on which frames the
        # shuffle put first is a run-to-run difference with no physical content. The
        # degenerate case is worse: a first batch that happens to be all neutral skips the
        # calibration entirely, which is what the production smoke hit on a cross-fit fold.
        c, c_n = defect_protocol.calibrate_c_shift_over_loader(model, train_loader, device)
        if c is None:
            # NOT silently zero. Delta_n = 0 on every frame makes the ratio undefined, and a
            # 0.0 written here would be indistinguishable from a calibration that happened.
            logging.warning(
                "Stage-3 protocol: c-shift NOT calibrated -- NO frame in the training set "
                "carries a net carrier, so Delta_n = 0 everywhere and the ratio is undefined. "
                "The head starts at c = 0, which is a choice this run did not make "
                "deliberately; the protocol summary records c_shift_calibrated: false.")
        else:
            with torch.no_grad():
                model.spectral.c_shift.fill_(float(c))
            logging.info(f"Stage-3 protocol: c-shift calibrated to {c:+.4f} eV over "
                         f"{c_n} charged frames (whole training set, shuffle-independent)")

        # The initialisation gate, on a PRISTINE spectrum: bands, not atoms. Reported rather
        # than enforced -- a trip is a statement about the initialisation that the run's
        # record should carry, and stopping here would discard a run for a diagnostic.
        e_gap = float(getattr(args, "defect_e_gap", 0.0) or 0.0)
        composition = getattr(args, "defect_gap_composition", None)
        if e_gap > 0 and composition is not None:
            mask = loss_fn.stoichiometric_mask(init_batch) \
                if hasattr(loss_fn, "stoichiometric_mask") else None
            gate = None
            if mask is not None and bool(mask.any()):
                which = int(torch.nonzero(mask.reshape(-1))[0])
                nodes = init_batch.batch == which
                internals: Dict[str, Any] = {}
                head = model.spectral
                original = head.forward

                def _capture(*a, **k):
                    k["internals"] = internals
                    return original(*a, **k)

                head.forward = _capture
                try:
                    with torch.no_grad():
                        model(init_batch.to_dict(), training=False, compute_force=False)
                finally:
                    head.forward = original
                lam = internals.get("lam")
                if lam is not None:
                    v = lam[which, 0]
                    v = v[v < 500.0]
                    # Doubly-occupied count for THAT graph's own composition, recovered
                    # from the one-hot species through the model's own atomic-number table
                    # rather than from a constant.
                    zs = model.atomic_numbers[
                        init_batch.node_attrs[nodes].argmax(dim=-1)].tolist()
                    n_el = sum(VALENCE[int(z)] for z in zs) / 2.0
                    gate = defect_protocol.initialisation_report(v, n_el, e_gap)
            if gate is None:
                logging.warning(
                    "Stage-3 protocol: initialisation gate UNSCORED -- the first batch "
                    "carries no stoichiometric cell. Do not read that as a pass.")
            else:
                logging.info(
                    f"Stage-3 protocol: init gate edges "
                    f"{gate['edge_spacing_below']:.3f}/{gate['edge_spacing_above']:.3f} eV "
                    f"(need <= {0.5 * e_gap:.2f}), bandwidth {gate['bandwidth']:.2f} eV "
                    f"(need >= {2 * e_gap:.2f}) -> "
                    f"{'PASS' if gate['passed'] else 'TRIP'}")

        protocol_post_step = defect_protocol.post_step
        logging.info("Stage-3 protocol: %s", json.dumps(defect_protocol.protocol_summary(
            stage=3, e_gap=e_gap,
            w_gap=float(getattr(args, "defect_gap_weight", 0.0) or 0.0),
            warmup=int(getattr(args, "defect_protocol_warmup", 5)),
            clip=float(args.clip_grad or 0.0),
            freeze_z=bool(getattr(args, "defect_protocol_freeze_z", False)),
            model=model, c_shift=c, c_shift_n_frames=c_n,
            harrison_applied=harrison_applied), sort_keys=True))

    # Reach assertion on the LOSS'S OWN loader, not a reimplementation of it. graph_cutoff
    # was previously verified only where it was not used, and the resulting 5 A graph -- in
    # which the two vacancy-sharing Pb have no edge -- voided a 20-seed screen. A check that
    # runs on a different code path than the loss is worse than none, because it passes.
    if getattr(args, "defect_spectral_head", False):
        from mace.modules.defect_reach import assert_carrier_reach

        assert_carrier_reach(next(iter(train_loader)), float(args.r_max),
                             graph_cutoff(args))

    tools.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loaders=valid_loaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        save_all_checkpoints=args.save_all_checkpoints,
        output_args=output_args,
        device=device,
        swa=swa,
        ema=ema,
        max_grad_norm=args.clip_grad,
        log_errors=args.error_table,
        log_wandb=args.wandb,
        distributed=args.distributed,
        distributed_model=distributed_model,
        plotter=plotter,
        train_sampler=train_sampler,
        rank=rank,
        data_aug_magmom=args.data_aug_magmom,
        epoch_hook=defect_seed_hook,
        post_eval_hook=_release_hook,
        post_step_hook=protocol_post_step,
    )

    logging.info("")
    logging.info("===========RESULTS===========")

    train_valid_data_loader = {}
    for head_config in head_configs:
        data_loader_name = "train_" + head_config.head_name
        train_valid_data_loader[data_loader_name] = head_config.train_loader
    for head, valid_loader in valid_loaders.items():
        data_load_name = "valid_" + head
        train_valid_data_loader[data_load_name] = valid_loader
    test_sets = {}
    stop_first_test = False
    test_data_loader = {}
    if all(
        head_config.test_file == head_configs[0].test_file
        for head_config in head_configs
    ) and head_configs[0].test_file is not None:
        stop_first_test = True
    if all(
        head_config.test_dir == head_configs[0].test_dir
        for head_config in head_configs
    ) and head_configs[0].test_dir is not None:
        stop_first_test = True
    for head_config in head_configs:
        if all(check_path_ase_read(f) for f in head_config.train_file):
            for name, subset in head_config.collections.tests:
                test_sets[head_config.head_name + "_" + name] = [
                    data.AtomicData.from_config(
                        config, z_table=z_table, cutoff=graph_cutoff(args), heads=heads
                    )
                    for config in subset
                ]
        if head_config.test_dir is not None:
            if not args.multi_processed_test:
                test_files = get_files_with_suffix(head_config.test_dir, "_test.h5")
                for test_file in test_files:
                    name = os.path.splitext(os.path.basename(test_file))[0]
                    test_sets[name] = data.HDF5Dataset(
                        test_file, r_max=graph_cutoff(args), z_table=z_table, heads=heads, head=head_config.head_name
                    )
            else:
                test_folders = glob(head_config.test_dir + "/*")
                for folder in test_folders:
                    name = os.path.splitext(os.path.basename(test_file))[0]
                    test_sets[name] = data.dataset_from_sharded_hdf5(
                        folder, r_max=graph_cutoff(args), z_table=z_table, heads=heads, head=head_config.head_name
                    )
        for test_name, test_set in test_sets.items():
            test_sampler = None
            if args.distributed:
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_set,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,
                    seed=args.seed,
                )
            try:
                drop_last = test_set.drop_last
            except AttributeError as e:  # pylint: disable=W0612
                drop_last = False
            test_loader = torch_geometric.dataloader.DataLoader(
                test_set,
                batch_size=args.valid_batch_size,
                shuffle=(test_sampler is None),
                drop_last=drop_last,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
            test_data_loader[test_name] = test_loader
        if stop_first_test:
            break

    for swa_eval in swas:
        epoch = checkpoint_handler.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler),
            swa=swa_eval,
            device=device,
        )
        model.to(device)
        if args.distributed:
            # re-enable gradients for distributed model for evaluation of stage-two model
            # after param.requires_grad = False was called before evaluating stage-one model
            for param in model.parameters():
                param.requires_grad = True
            distributed_model = DDP(
                model, device_ids=[local_rank] if args.device == "cuda" else None
            )
        model_to_evaluate = model if not args.distributed else distributed_model
        if swa_eval:
            logging.info(f"Loaded Stage two model from epoch {epoch} for evaluation")
        else:
            logging.info(f"Loaded Stage one model from epoch {epoch} for evaluation")

        if rank == 0:
            # Save entire model
            if swa_eval:
                model_path = Path(args.checkpoints_dir) / (tag + "_stagetwo.model")
            else:
                model_path = Path(args.checkpoints_dir) / (tag + ".model")
            logging.info(f"Saving model to {model_path}")
            model_to_save = deepcopy(model)
            if args.lora:
                logging.info("Merging LoRA weights into base model")
                merge_lora_weights(model_to_save)
            if args.enable_cueq and not args.only_cueq:
                logging.info("RUNING CUEQ TO E3NN")
                model_to_save = run_cueq_to_e3nn(deepcopy(model), device=device)
            if args.enable_oeq:
                logging.info("RUNING OEQ TO E3NN")
                model_to_save = run_oeq_to_e3nn(deepcopy(model), device=device)
            if args.save_cpu:
                model_to_save = model_to_save.to("cpu")
            torch.save(model_to_save, model_path)
            extra_files = {
                "commit.txt": commit.encode("utf-8") if commit is not None else b"",
                "config.yaml": json.dumps(
                    convert_to_json_format(extract_config_mace_model(model))
                ),
            }
            os.makedirs(args.model_dir, exist_ok=True)
            if swa_eval:
                torch.save(
                    model_to_save, Path(args.model_dir) / (args.name + "_stagetwo.model")
                )
                try:
                    path_complied = Path(args.model_dir) / (
                        args.name + "_stagetwo_compiled.model"
                    )
                    logging.info(f"Compiling model, saving metadata {path_complied}")
                    model_compiled = jit.compile(deepcopy(model_to_save))
                    torch.jit.save(
                        model_compiled,
                        path_complied,
                        _extra_files=extra_files,
                    )
                except Exception as e:  # pylint: disable=W0718
                    pass
            else:
                torch.save(model_to_save, Path(args.model_dir) / (args.name + ".model"))
                try:
                    path_complied = Path(args.model_dir) / (
                        args.name + "_compiled.model"
                    )
                    logging.info(f"Compiling model, saving metadata to {path_complied}")
                    model_compiled = jit.compile(deepcopy(model_to_save))
                    torch.jit.save(
                        model_compiled,
                        path_complied,
                        _extra_files=extra_files,
                    )
                except Exception as e:  # pylint: disable=W0718
                    pass

        logging.info("Computing metrics for training, validation, and test sets")
        for param in model.parameters():
            param.requires_grad = False
        skip_heads = args.skip_evaluate_heads.split(",") if args.skip_evaluate_heads else []
        if skip_heads:
            logging.info(f"Skipping evaluation for heads: {skip_heads}")
        table_train_valid = create_error_table(
            table_type=args.error_table,
            all_data_loaders=train_valid_data_loader,
            model=model_to_evaluate,
            loss_fn=loss_fn,
            output_args=output_args,
            log_wandb=args.wandb,
            device=device,
            distributed=args.distributed,
            skip_heads=skip_heads,
        )
        logging.info("Error-table on TRAIN and VALID:\n" + str(table_train_valid))

        if test_data_loader:
            table_test = create_error_table(
                table_type=args.error_table,
                all_data_loaders=test_data_loader,
                model=model_to_evaluate,
                loss_fn=loss_fn,
                output_args=output_args,
                log_wandb=args.wandb,
                device=device,
                distributed=args.distributed,
            )
            logging.info("Error-table on TEST:\n" + str(table_test))
        if args.plot:
            try:
                plotter = TrainingPlotter(
                    results_dir=logger.path,
                    heads=heads,
                    table_type=args.error_table,
                    train_valid_data=train_valid_data_loader,
                    test_data=test_data_loader,
                    output_args=output_args,
                    device=device,
                    plot_frequency=args.plot_frequency,
                    distributed=args.distributed,
                    swa_start=swa.start if swa else None
                )
                plotter.plot(epoch, model_to_evaluate, rank)
            except Exception as e:  # pylint: disable=W0718
                logging.debug(f"Plotting failed: {e}")

        if args.distributed:
            torch.distributed.barrier()

    logging.info("Done")
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
