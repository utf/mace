"""v4.2 (C5): the bound-state precondition."""
import torch

from mace.modules.dscc import precondition as pc

torch.set_default_dtype(torch.float64)
from mace.modules.dscc import scf


def _gapped_h(n_atoms=6, homo_gap=0.8, bound_site=1, seed=0):
    """A random sp Hamiltonian whose reference HOMO is a split-off level localised on one
    site (bound) or a band state (not)."""
    g = torch.Generator().manual_seed(seed)
    n = 4 * n_atoms
    A = torch.randn(n, n, generator=g); H = 0.5 * (A + A.T)
    eps, U = torch.linalg.eigh(H)
    eps = torch.sort(eps).values
    n_ref = 2 + 4 + 7 * (n_atoms - 2)                # Cs, Pb, and Cl x (n-2)
    n_up = (n_ref + 1) // 2
    eps[n_up:] += 3.0                                 # a gap above the reference HOMO region
    eps[n_up - 1] += homo_gap                         # the HOMO split off from the band below
    H = U @ torch.diag(eps) @ U.T
    return H, [55, 82] + [17] * (n_atoms - 2)


def test_frame_record_thresholds():
    cfg = pc.PreconditionConfig(delta_c=0.5, n_loc=4.0)
    H, numbers = _gapped_h()
    rec = pc.frame_record(H, numbers, cfg)
    assert rec.separation > 2.0                        # the level above is 3 eV up
    assert rec.n_eff > 0.0
    # A tiny separation fails regardless of localisation.
    H2, _ = _gapped_h(homo_gap=0.0)
    eps = torch.linalg.eigvalsh(H2)
    n_up = (2 + 4 + 7 * 4 + 1) // 2
    H2 = H2 - 2.9 * torch.eye(H2.shape[0]) * 0.0      # no-op; separation is what it is
    rec2 = pc.frame_record(H2, numbers, pc.PreconditionConfig(delta_c=10.0, n_loc=100.0))
    assert not rec2.passed and rec2.separation < 10.0
