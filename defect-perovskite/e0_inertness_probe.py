"""Prove the correction branch is inert before trusting an E0 base run.

E0 trains "the base branch with all corrections disabled" by feeding it only n = 0 frames.
The energy path really does vanish there -- dE_SR = sum_c n_c (...) is zero when every counter
is zero -- but that is not sufficient on its own. Four regularisers in the defect loss act on
model OUTPUTS rather than on parameters:

    u_l2      -> carrier_readouts
    p_l2      -> polarisation
    qhost_l2  -> latent_charges_host
    zn_l2     -> counter embedding

Each of those is a function of the shared trunk, so with a nonzero weight it pushes gradient
into the trunk even on a frame with no carrier. The size hinge and the seed-anneal schedule do
the same through the logit network. A base trained with them on would not be a geometry-only
base, and E0's premise would be quietly false.

This asserts, on a real n = 0 batch: dE_SR is exactly zero, total energy equals base energy,
and every correction-head parameter receives exactly zero gradient. Anything else is a finding
to chase, not a nuisance to tolerate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD before e3nn is imported)
from e3nn import o3

from mace import data as mace_data
from mace import modules
from mace.data.defects import prepare_defect_configurations
from mace.modules.defect_models import MACEDefect
from mace.tools import AtomicNumberTable, torch_geometric


def build_batch(frames, z_table, cutoff, device):
    keyspec = mace_data.KeySpecification(
        info_keys={"energy": "REF_energy", "carrier_counts": "carrier_counts",
                   "multiplicity": "multiplicity", "host": "host",
                   "m_s_ref_doubled": "m_s_ref_doubled", "cell_charge": "cell_charge",
                   "e_cbm_cell": "e_cbm_cell", "e_vbm_cell": "e_vbm_cell"},
        arrays_keys={"forces": "REF_forces"})
    configs = [mace_data.config_from_atoms(a, key_specification=keyspec) for a in frames]
    prepare_defect_configurations(configs)
    atomic = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=cutoff)
              for c in configs]
    loader = torch_geometric.dataloader.DataLoader(atomic, batch_size=len(atomic),
                                                   shuffle=False)
    return next(iter(loader)).to(device)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=here / "dataset_e0" / "train.xyz")
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from ase.io import read
    frames = read(args.data, f":{args.n_frames}")
    for a in frames:
        c = a.info["carrier_counts"]
        c = [int(x) for x in c.split()] if isinstance(c, str) else np.asarray(c)
        assert not np.asarray(c).any(), "probe needs n = 0 frames"

    torch.manual_seed(0)
    z_table = AtomicNumberTable([17, 55, 82])
    model = MACEDefect(
        r_max=5.0, num_bessel=8, num_polynomial_cutoff=5, max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=3,
        hidden_irreps=o3.Irreps("32x0e + 32x1o"), MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu, atomic_energies=np.zeros((1, 3)),
        avg_num_neighbors=20.0, atomic_numbers=[17, 55, 82], correlation=3,
        atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        carrier_feature_dim=32, counter_embedding_dim=16, carrier_mlp_hidden=32,
        use_long_range=False, zero_u_init=False,
    ).to(args.device)

    batch = build_batch(frames, z_table, 5.0, args.device)
    out = model(batch.to_dict(), training=True, compute_force=True)

    dsr = out["delta_sr_energy"]
    corr = out["correction_energy"]
    gap_e = (out["energy"] - out["base_energy"]).abs().max()
    gap_f = (out["forces"] - out["base_forces"]).abs().max()

    print("=== energy path at n = 0 ===")
    print(f"  max |delta_sr_energy|      {float(dsr.abs().max()):.3e}")
    print(f"  max |correction_energy|    {float(corr.abs().max()):.3e}")
    print(f"  max |energy - base_energy| {float(gap_e):.3e}")
    print(f"  max |forces - base_forces| {float(gap_f):.3e}")

    # Which parameters belong to the correction head rather than the shared trunk?
    correction_prefixes = ("carrier_", "counter_", "latent_charges", "logit",
                           "defect_", "delta_")
    corr_params = {n: p for n, p in model.named_parameters()
                   if any(n.startswith(pre) or f".{pre}" in n for pre in correction_prefixes)}
    print(f"\n=== gradient reach ({len(corr_params)} correction-head tensors) ===")

    # The energy-only objective a base run actually optimises.
    loss = out["energy"].sum() + out["forces"].pow(2).sum()
    model.zero_grad(set_to_none=True)
    loss.backward()

    nonzero = {n: float(p.grad.abs().max()) for n, p in corr_params.items()
               if p.grad is not None and float(p.grad.abs().max()) > 0}
    if nonzero:
        print("  FAIL: correction-head parameters receive gradient from the base objective:")
        for n, v in sorted(nonzero.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {n}: max|grad| = {v:.3e}")
    else:
        print("  PASS: no correction-head parameter receives gradient")

    ok = (float(dsr.abs().max()) == 0.0 and float(gap_e) == 0.0
          and float(gap_f) == 0.0 and not nonzero)
    print(f"\n{'INERT' if ok else 'NOT INERT'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
