from __future__ import annotations

from enum import Enum


class DefaultKeys(Enum):
    ENERGY = "REF_energy"
    FORCES = "REF_forces"
    STRESS = "REF_stress"
    VIRIALS = "REF_virials"
    DIPOLE = "dipole"
    POLARIZABILITY = "polarizability"
    HEAD = "head"
    CHARGES = "REF_charges"
    TOTAL_CHARGE = "total_charge"
    TOTAL_SPIN = "total_spin"
    ELEC_TEMP = "elec_temp"
    MAGMOM = "REF_magmom"
    MAGFORCES = "REF_magforces"
    # Charge-aware defect models (see charge_aware_defect_mlip_implementation.md)
    CARRIER_COUNTS = "carrier_counts"
    HOST = "host"
    PAIR_ID = "pair_id"
    MULTIPLICITY = "multiplicity"
    M_S_REF_DOUBLED = "m_s_ref_doubled"
    CELL_CHARGE = "cell_charge"
    E_CBM_CELL = "e_cbm_cell"
    E_VBM_CELL = "e_vbm_cell"
    BASE_ENERGY = "REF_base_energy"
    BASE_FORCES = "REF_base_forces"
    BASE_STRESS = "REF_base_stress"

    @staticmethod
    def keydict() -> dict[str, str]:
        key_dict = {}
        for member in DefaultKeys:
            key_name = f"{member.name.lower()}_key"
            key_dict[key_name] = member.value
        return key_dict
