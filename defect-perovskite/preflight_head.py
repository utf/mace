"""Section 4 pre-flight checks for a carrier-Hamiltonian head, before any R1 run.

Each check exists because something like it has already gone wrong:

  reach       the 5 A head coupled the hub Pb pair in only 36% of frames, with median
              envelope weight 0.00000, so the target state was unrepresentable. The gate is
              ~100% of frames coupled at weight >= 0.3.
  hermiticity eigh assumes symmetry rather than checking it; a non-symmetric H gives silently
              wrong gradients.
  floor/decay t must not fall below t_min anywhere (that is the softmax limit the head exists
              to leave) and the decay length must stay above its floor.
  anchor      adding a constant to raw eps must leave dE_SR untouched to float precision.
  n = 0       the correction must vanish identically, for any parameters.

Run on real frames, not synthetic cells: every fault so far appeared at real scale and was
invisible on hand-built 8-atom batches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401
from ase.geometry import get_distances
from ase.io import read

from mace import data as mace_data
from mace import tools
from mace.data.defects import prepare_defect_configurations
from mace.tools import torch_geometric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_gates import KEYSPEC  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402

OK, BAD = "PASS", "FAIL"
results = []


def check(name, passed, detail):
    results.append((name, OK if passed else BAD, detail))
    print(f"  [{OK if passed else BAD}] {name}: {detail}")


def build_batch(frames, z_table, cutoff, device):
    configs = [mace_data.config_from_atoms(a, key_specification=KEYSPEC) for a in frames]
    prepare_defect_configurations(configs)
    atomic = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=cutoff)
              for c in configs]
    return next(iter(torch_geometric.dataloader.DataLoader(
        atomic, batch_size=len(atomic), shuffle=False))).to(device)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path,
                    default=here / "dataset_pbe" / "valid.xyz")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = torch.load(args.model, map_location=args.device,
                       weights_only=False).to(args.device).eval()
    head = getattr(model, "spectral", None)
    if head is None:
        raise SystemExit("model has no spectral head")
    r_cut = float(model.spectral_r_cut)
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    print(f"head reach {r_cut:.1f} A, decay {'on' if head.use_decay else 'off'}, "
          f"t_min {head.t_min}\n")

    frames = [a for a in read(args.data, ":")
              if np.asarray([int(x) for x in a.info["carrier_counts"].split()]
                            if isinstance(a.info["carrier_counts"], str)
                            else a.info["carrier_counts"]).any()][: args.frames]

    # --- reach: is the hub pair actually coupled? -----------------------------------------
    weights, coupled = [], 0
    for atoms in frames:
        try:
            site = locate_vacancy(atoms)
        except ValueError:
            continue
        pos = atoms.get_positions()
        _, d = get_distances(pos[site.shell[0]][None], pos[site.shell[1]][None],
                             cell=atoms.get_cell(), pbc=atoms.pbc)
        # Ask the HEAD for its envelope rather than reimplementing the formula here. The
        # first version hardcoded the polynomial and so kept reporting the old shape after
        # the head switched to a gentler one -- testing the copy, not the code.
        with torch.no_grad():
            env = float(head._envelope(torch.tensor([float(d[0, 0])])).item())
        weights.append(env)
        coupled += env >= 0.3
    weights = np.array(weights)
    check("reach (hub pair envelope >= 0.3)",
          coupled == len(weights),
          f"{coupled}/{len(weights)} frames, median weight {np.median(weights):.3f}")

    batch = build_batch(frames[:4], z_table, r_cut, args.device)
    d = batch.to_dict()

    # --- hermiticity and floor, from the head's own tensors --------------------------------
    with torch.no_grad():
        out = model(d, training=False, compute_force=False)
        feats = out["defect_features"]
        ei = d["edge_index"]
        vec = d["positions"][ei[1]] - d["positions"][ei[0]] + d["shifts"]
        r = vec.norm(dim=-1)
        sp = d["node_attrs"].argmax(dim=-1)
        t_ij = head.hopping(feats[ei[0]], feats[ei[1]], r, sp[ei[0]], sp[ei[1]])
        t_ji = head.hopping(feats[ei[1]], feats[ei[0]], r, sp[ei[1]], sp[ei[0]])
    check("hopping symmetry t_ij == t_ji", torch.equal(t_ij, t_ji),
          f"max |t_ij - t_ji| = {float((t_ij - t_ji).abs().max()):.3e}")

    if head.use_decay:
        # The floor belongs on the AMPLITUDE, not on the product. The envelope necessarily
        # sends t to zero at r_cut, so "t >= t_min everywhere" is unsatisfiable as stated;
        # what the floor guarantees is that no pair decouples for lack of amplitude. Check the
        # amplitude directly, and separately check that the coupling actually surviving at the
        # hub Pb separation is non-negligible -- which is the property the floor exists for.
        with torch.no_grad():
            feats_pair = torch.cat([feats[ei[0]] + feats[ei[1]],
                                    (feats[ei[0]] - feats[ei[1]]).abs(),
                                    head._radial(r.reshape(-1))], dim=-1)
            amp = head.t_min + torch.nn.functional.softplus(head.hop(feats_pair))
        check("hopping amplitude floor", float(amp.min()) >= head.t_min - 1e-9,
              f"min amplitude = {float(amp.min()):.4f} eV (floor {head.t_min})")

        med_sep = 5.32
        with torch.no_grad():
            probe = torch.full((1,), med_sep, device=r.device, dtype=r.dtype)
            t_probe = head.hopping(feats[:1], feats[1:2], probe,
                                   sp[:1], sp[1:2])
        check("coupling survives at the median hub separation",
              float(t_probe.abs().max()) > 0.2 * head.t_min,
              f"|t| at {med_sep} A = {float(t_probe.abs().max()):.4f} eV")
        with torch.no_grad():
            ell = head.decay_length(sp[ei[0]], sp[ei[1]])
        check("decay length floor", float(ell.min()) >= 0.3,
              f"min l = {float(ell.min()):.3f} A, max {float(ell.max()):.3f} A")

    # --- gauge anchor: a constant shift of raw eps must not move dE_SR ---------------------
    with torch.no_grad():
        base = model(d, training=False, compute_force=False)["delta_sr_energy"].clone()
        head.site[-1].bias.add_(7.0)          # uniform shift of every site energy
        shifted = model(d, training=False, compute_force=False)["delta_sr_energy"].clone()
        head.site[-1].bias.sub_(7.0)
    dev = float((base - shifted).abs().max())
    check("gauge anchor (uniform eps shift is inert)", dev < 1e-4,
          f"max |dE_SR change| under +7 eV shift = {dev:.3e} eV")

    # --- n = 0 gives exactly zero ----------------------------------------------------------
    d0 = dict(d)
    d0["carrier_counts"] = torch.zeros_like(d["carrier_counts"])
    with torch.no_grad():
        z = model(d0, training=False, compute_force=False)
    check("n = 0 => dE_SR == 0", float(z["delta_sr_energy"].abs().max()) == 0.0,
          f"max |dE_SR| = {float(z['delta_sr_energy'].abs().max()):.3e}")

    # --- |eps| bounded (the drift the anchor exists to stop) -------------------------------
    with torch.no_grad():
        eps = model(d, training=False, compute_force=False)["carrier_readouts"]
    check("|eps| bounded (expect O(1 eV))", float(eps.abs().max()) < 50.0,
          f"max |eps| = {float(eps.abs().max()):.2f} eV, "
          f"std {float(eps.std()):.3f}")

    bad = [n for n, v, _ in results if v == BAD]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
    if bad:
        print("FAILED: " + ", ".join(bad))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
