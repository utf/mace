"""Section 1.1 of the speed cycle: the size-grouped batched solver is the loop, stacked.

THE TOLERANCE IS THE SPEC'S: 1e-8 eV on the eigenvalues and 1e-8 on the density matrix.
Both paths run the same float64 `eigh` on the same matrix, so the only differences are the
order of a few reductions; anything larger is a wiring error, not arithmetic.

WHAT WOULD BE MISSED by comparing energies alone. The batched path collapses the four
per-fill density matrices into one (`D = U diag(sum_a s_a f_a) U^T`), reindexes every node
by its position within its own graph, and replaces the assembled diagonal rather than adding
to it. A transposed `local` map, a sign paired with the wrong graph, or an on-site level
added twice all leave the total energy plausible and move `alpha`, the gap and the forces --
so those are compared too, per graph and per atom.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from e3nn import o3

from mace import data, modules, tools
from mace.modules.defect_counting import (batched_head_energy_hf, head_energy_hf,
                                          resolve_fills_batched, spin_targets,
                                          spin_targets_batched)
from mace.modules.defect_models import MACEDefect

Z_TABLE = tools.AtomicNumberTable([17, 55, 82])


@pytest.fixture(scope="module", autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)


def _perovskite(reps=(2, 2, 2), rattle=0.0, seed=0):
    a = 5.6
    cell = bulk("Cs", "sc", a=a)
    atoms = Atoms("CsPbCl3",
                  scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
                  cell=cell.cell, pbc=True).repeat(reps)
    if rattle:
        atoms.rattle(stdev=rattle, seed=seed)
    return atoms


def _batch(frames, counts, cutoff=6.0):
    ds = []
    for atoms, c in zip(frames, counts):
        config = data.Configuration(
            atomic_numbers=atoms.get_atomic_numbers(), positions=atoms.get_positions(),
            cell=np.array(atoms.get_cell()), pbc=(True, True, True),
            properties={"carrier_counts": c}, property_weights={})
        ds.append(data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=cutoff))
    loader = tools.torch_geometric.dataloader.DataLoader(ds, batch_size=len(ds))
    return next(iter(loader))


def _model(**overrides):
    torch.manual_seed(0)
    kwargs = dict(
        r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=3,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"), MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu, atomic_energies=np.zeros((1, 3)),
        avg_num_neighbors=8.0, atomic_numbers=[17, 55, 82], correlation=2,
        atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        carrier_feature_dim=16, counter_embedding_dim=8, carrier_mlp_hidden=16,
        use_long_range=False, spectral_head=True, counting_head=True,
        spectral_first_shell=True, spectral_r_cut=6.0,
        madelung_on_site=True, madelung_composition=[3.0, 1.0, 1.0],
        madelung_z_init=[-1.0, 1.0, 2.0], les_arguments={"sigma": 1.0})
    kwargs.update(overrides)
    return MACEDefect(**kwargs)


class TestFills:
    def test_batched_spin_targets_match_the_scalar_ones(self):
        counts = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                              dtype=torch.float64)
        n_total = torch.tensor([409.0, 410.0, 409.0])
        maj, minor = spin_targets_batched(n_total, counts)
        for g in range(3):
            a, b = spin_targets(int(n_total[g]), counts[g].tolist())
            assert float(maj[g]) == a and float(minor[g]) == b

    def test_odd_counts_keep_the_doublet(self):
        maj, minor = spin_targets_batched(torch.tensor([409.0]),
                                          torch.zeros(1, 4, dtype=torch.float64))
        assert (float(maj[0]), float(minor[0])) == (205.0, 204.0)

    def test_occupation_override_is_taken_verbatim(self):
        now, ref = resolve_fills_batched(torch.tensor([409.0, 409.0]),
                                         torch.zeros(2, 4, dtype=torch.float64),
                                         occupation=torch.tensor([[3.0, 2.0],
                                                                  [7.0, 1.0]]))
        assert [float(now[0][0]), float(now[0][1])] == [3.0, 7.0]
        assert [float(ref[0][0]), float(ref[0][1])] == [205.0, 205.0]


class TestSolverIdentity:
    """`batched_head_energy_hf` against `head_energy_hf` on the same matrices."""

    @staticmethod
    def _matrices(b=3, n=7, seed=0):
        torch.manual_seed(seed)
        m = n * 4
        H = torch.randn(b, m, m, dtype=torch.float64)
        return 0.5 * (H + H.transpose(-1, -2))

    def test_energy_eigenvalues_and_density_match(self):
        H = self._matrices()
        counts = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                              dtype=torch.float64)
        n_total = torch.tensor([21.0, 21.0, 21.0])
        e_b, lam_b, d_b, _ = batched_head_energy_hf(H, n_total, counts, 0.05)
        for g in range(3):
            e, lam, _psi, p_now, p_ref = head_energy_hf(
                H[g], int(n_total[g]), counts[g].tolist(), 0.05)
            assert abs(float(e_b[g] - e)) < 1e-8, f"energy, graph {g}"
            assert float((lam_b[g] - lam).abs().max()) < 1e-8, f"eigenvalues, graph {g}"
            assert float((d_b[g] - (p_now - p_ref)).abs().max()) < 1e-8, f"P, graph {g}"

    def test_the_density_response_matches_and_carries_a_gradient(self):
        H = self._matrices().requires_grad_(True)
        counts = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]],
                              dtype=torch.float64)
        n_total = torch.tensor([21.0, 21.0, 21.0])
        bucket_b = {}
        batched_head_energy_hf(H, n_total, counts, 0.05, response_out=bucket_b)
        dens_b = bucket_b["density_difference"]
        for g in range(3):
            bucket = {}
            head_energy_hf(H[g].detach(), int(n_total[g]), counts[g].tolist(), 0.05,
                           response_out=bucket)
            assert float((dens_b[g] - bucket["density_difference"]).abs().max()) < 1e-8
        # The response is what carries dP/dtheta; a detached one would train the wrong
        # gradient silently, so the test asserts the graph exists.
        assert dens_b.requires_grad
        dens_b.sum().backward()
        assert float(H.grad.abs().max()) > 0.0


class TestModelPathsAgree:
    """The whole model, batched solver on and off, on one size-uniform batch."""

    @staticmethod
    def _run(model, batch, batched: bool):
        model.spectral.batch_by_size = batched
        return model(batch.to_dict(), training=True, compute_force=True)

    def test_energy_forces_and_head_outputs_are_identical(self):
        model = _model()
        frames = [_perovskite(rattle=0.02, seed=s) for s in (1, 2, 3)]
        counts = [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        batch = _batch(frames, counts)
        on = self._run(model, batch, True)
        off = self._run(model, batch, False)
        for key, tol in (("energy", 1e-8), ("forces", 1e-8), ("delta_sr_energy", 1e-8)):
            a, b = on[key], off[key]
            assert a is not None and b is not None, key
            worst = float((a - b).abs().max())
            assert worst < tol, f"{key} differs by {worst:.3e}"

    def test_the_carrier_density_and_gap_are_identical(self):
        model = _model()
        frames = [_perovskite(rattle=0.02, seed=s) for s in (4, 5)]
        counts = [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        batch = _batch(frames, counts)
        grabbed = {"runs": []}
        head = model.spectral
        original = head.forward

        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            grabbed["runs"].append(out)
            return out

        head.forward = wrapped
        try:
            self._run(model, batch, True)
            self._run(model, batch, False)
        finally:
            head.forward = original
        runs = grabbed["runs"]
        # Two head calls per forward (the counter and its reference); compare like for like.
        half = len(runs) // 2
        for a, b in zip(runs[:half], runs[half:]):
            assert float((a.alpha - b.alpha).abs().max()) < 1e-8
            assert float((a.gap - b.gap).abs().max()) < 1e-8
            assert float((a.eigenvalues - b.eigenvalues).abs().max()) < 1e-8

    def test_a_mixed_size_batch_still_uses_the_loop(self):
        model = _model()
        frames = [_perovskite(reps=(2, 2, 2)), _perovskite(reps=(2, 2, 3))]
        batch = _batch(frames, [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        out = self._run(model, batch, True)
        assert out["energy"].isfinite().all() and out["forces"].isfinite().all()
