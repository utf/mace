"""Plan P Phase 3a: shared helpers for base v8 w01 as its own field provider (the base's forces,
energy, Born charges and node dipoles), the neutral production model (base + E_PP, through a
D-SCC checkpoint at zero carriers), and the Gamma Hessian with the acoustic sum rule."""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np, torch
from ase import Atoms
from ase.data import atomic_masses
from mace import data as mace_data, tools
from mace.modules.dscc import data as dd
from mace.tools import torch_geometric

MEV = 1000.0 * 6.582119569e-16 * math.sqrt(1.602176634e-19 / (1.66053907e-27 * 1e-20))   # sqrt(eV/A^2/amu) -> meV
BASE_V8 = "/home/alex/runs/dscc/p1v8/base_v8_w01/base.pt"
BW5_S0 = "/home/alex/runs/dscc/dscc_p2cwbw5_s0/model_avg.pt"


def load_cell(path) -> Atoms:
    doc = json.load(open(Path(path).expanduser()))
    return Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)


class Base:
    """Base v8 w01 alone, float64: forces, energy, node dipoles (zero field), Z* and eps_inf by autograd."""

    def __init__(self, path=BASE_V8, device="cuda"):
        self.dev = device
        self.model = torch.load(Path(path).expanduser(), map_location=device, weights_only=False).double().to(device).eval()
        self.z = tools.AtomicNumberTable([int(x) for x in self.model.atomic_numbers])
        self.r_max = float(self.model.r_max)

    def batch(self, a: Atoms) -> Dict[str, torch.Tensor]:
        a = a.copy(); a.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0, "config_type": "p3a", "source_dir": "p3a"})
        return next(iter(torch_geometric.dataloader.DataLoader(dd.atomic_data([a], self.z, self.r_max), batch_size=1))).to(self.dev).to_dict()

    def forces(self, a: Atoms) -> np.ndarray:
        out = self.model(self.batch(a), training=False, compute_force=True)
        return out["forces"].detach().cpu().numpy()

    def energy(self, a: Atoms) -> float:
        with torch.no_grad():
            return float(self.model(self.batch(a), training=False, compute_force=False)["energy"][0])

    def forces_dipoles(self, a: Atoms) -> Tuple[float, np.ndarray, np.ndarray]:
        """Energy, forces and the node dipoles p [N, 3] (e A) in one pass: the base is a MACEField
        and returns -dE/dE_node attached to the positions."""
        b = self.batch(a)
        leaf = torch.zeros(b["positions"].shape[0], 3, dtype=torch.float64, device=self.dev, requires_grad=True)
        with torch.enable_grad():
            out = self.model(b, training=True, compute_force=True, electric_field_node=leaf, compute_node_dipoles=True)
        return float(out["energy"][0]), out["forces"].detach().cpu().numpy(), out["node_dipoles"].detach().cpu().numpy()

    def dipoles(self, a: Atoms) -> np.ndarray:
        b = self.batch(a)
        leaf = torch.zeros(b["positions"].shape[0], 3, dtype=torch.float64, device=self.dev, requires_grad=True)
        with torch.enable_grad():
            out = self.model(b, training=True, compute_force=False, electric_field_node=leaf, compute_node_dipoles=True)
        return out["node_dipoles"].detach().cpu().numpy()

    def response(self, a: Atoms) -> Tuple[np.ndarray, np.ndarray]:
        """Z*[k, alpha, beta] = d(Omega P_alpha)/dR_k,beta (e) and eps_inf tensor, by autograd (the P1 gate-2 call)."""
        m = self.model
        cfg = mace_data.config_from_atoms(a, key_specification=mace_data.KeySpecification.from_defaults(), head_name=list(m.heads)[-1])
        ad = mace_data.AtomicData.from_config(cfg, z_table=self.z, cutoff=self.r_max, heads=list(m.heads))
        b = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1))).to(self.dev).to_dict()
        out = m(b, training=True, compute_force=False, compute_polarization=True, compute_becs=True, compute_polarizability=True)
        Z = out["becs"].detach().cpu().numpy().reshape(len(a), 3, 3)
        eps = np.eye(3) + out["polarizability"].detach().cpu().numpy()[0]
        return Z, eps


class Neutral:
    """The neutral production model: base + E_PP, read through a D-SCC checkpoint at zero carriers."""

    def __init__(self, path=BW5_S0, device="cuda"):
        self.dev = device
        self.model = torch.load(Path(path).expanduser(), map_location=device, weights_only=False).to(device).eval()
        self.model.base = self.model.base.double()
        assert getattr(self.model, "host_coupling", "") == "polarisation"
        self.z = tools.AtomicNumberTable([int(x) for x in self.model.atomic_numbers])
        self.r_cut = float(self.model.r_cut)

    def batch(self, a: Atoms, counts=(0, 0, 0, 0), charge=0) -> Dict[str, torch.Tensor]:
        a = a.copy(); a.info.update({"carrier_counts": np.array(counts, float), "cell_charge": int(charge), "config_type": "p3a", "source_dir": "p3a"})
        return next(iter(torch_geometric.dataloader.DataLoader(dd.atomic_data([a], self.z, self.r_cut), batch_size=1))).to(self.dev).to_dict()

    def energy(self, a: Atoms) -> Tuple[float, float]:
        """(E, E_PP) at zero carriers."""
        with torch.no_grad():
            out = self.model(self.batch(a), compute_force=False)
        return float(out["energy"][0]), float(out["e_pp"][0])

    def energy_forces(self, a: Atoms) -> Tuple[float, float, np.ndarray]:
        out = self.model(self.batch(a), compute_force=True)
        return float(out["energy"][0]), float(out["e_pp"][0]), out["forces"].detach().cpu().numpy()


def asr(FC: np.ndarray) -> np.ndarray:
    """Acoustic sum rule on a (3N, 3N) force-constant matrix, in place on a copy."""
    FC = 0.5 * (FC + FC.T)
    for a_ in range(3):
        for b_ in range(3):
            blk = FC[a_::3, b_::3]; np.fill_diagonal(blk, np.diag(blk) - blk.sum(1)); FC[a_::3, b_::3] = blk
    return FC


def gamma_hessian(force_fn, atoms: Atoms, h: float = 0.01, log=None) -> np.ndarray:
    """Central-difference Gamma force constants (eV/A^2), ASR applied."""
    N = len(atoms); pos0 = atoms.get_positions(); FC = np.zeros((3 * N, 3 * N))
    for i in range(N):
        for c in range(3):
            p, m = atoms.copy(), atoms.copy()
            pp = pos0.copy(); pp[i, c] += h; p.set_positions(pp)
            pm = pos0.copy(); pm[i, c] -= h; m.set_positions(pm)
            FC[3 * i + c] = -(force_fn(p) - force_fn(m)).reshape(-1) / (2 * h)
        if log and i % 20 == 19:
            log(f"  hessian {i + 1}/{N}")
    return asr(FC)


def gamma_modes(FC: np.ndarray, numbers) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigenvalues w2 (eV/A^2/amu), eigenvectors e (3N, 3N) of the mass-weighted matrix, masses M (amu)."""
    M = np.array([atomic_masses[n] for n in numbers]); Msq = np.repeat(np.sqrt(M), 3)
    w2, e = np.linalg.eigh(FC / np.outer(Msq, Msq))
    return w2, e, M


def tilt_stats(atoms: Atoms) -> Dict[str, float]:
    """Pb-Cl bond lengths and Pb-Cl-Pb angles (the tilt record of p0_tilt_check)."""
    from ase.neighborlist import neighbor_list
    i, j, d, D = neighbor_list("ijdD", atoms, 3.4); z = atoms.get_atomic_numbers()
    pbcl = [(ii, jj, dd, DD) for ii, jj, dd, DD in zip(i, j, d, D) if z[ii] == 82 and z[jj] == 17]
    bonds = np.array([dd for _, _, dd, _ in pbcl]); by = {}
    for ii, jj, dd, DD in pbcl:
        by.setdefault(jj, []).append(-DD)
    ang = []
    for vs in by.values():
        if len(vs) == 2:
            c = np.dot(vs[0], vs[1]) / np.linalg.norm(vs[0]) / np.linalg.norm(vs[1]); ang.append(math.degrees(math.acos(max(-1, min(1, c)))))
    ang = np.array(ang)
    return {"pb_cl_bond_min": float(bonds.min()), "pb_cl_bond_mean": float(bonds.mean()), "pb_cl_bond_max": float(bonds.max()),
            "pb_cl_pb_angle_min": float(ang.min()), "pb_cl_pb_angle_mean": float(ang.mean()), "pb_cl_pb_angle_max": float(ang.max()),
            "tilt_from_180_mean_deg": float(180.0 - ang.mean()), "n_angles": int(len(ang))}
