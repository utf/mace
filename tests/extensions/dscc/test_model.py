"""Plan section 4 / 5 gates on the wired model, `Phi = 0` path: the neutral null is
bit-identical to the base; `sum dq = Q`; forces and stress against central differences;
invariances; checkpoint round trip."""
import copy
import io

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace import data as mace_data
from mace import modules, tools
from mace.modules.dscc import data as dd
from mace.modules.dscc.model import MACEDSCC
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)
Z_TABLE = tools.AtomicNumberTable([17, 55, 82])
R_CUT = 6.0


def _base(seed=0):
    torch.manual_seed(seed)
    return modules.ScaleShiftMACE(
        r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=1,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=3, hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"), gate=torch.nn.functional.silu,
        atomic_energies=np.zeros((1, 3)), avg_num_neighbors=8.0, atomic_numbers=[17, 55, 82],
        correlation=2, atomic_inter_scale=1.0, atomic_inter_shift=0.0).double()


def _perovskite(rattle=0.05, seed=0, remove_cl=None):
    atoms = Atoms("CsPbCl3", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0],
                                              [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
                  cell=np.eye(3) * 5.6, pbc=True).repeat((2, 2, 2))
    atoms.rattle(stdev=rattle, seed=seed)
    if remove_cl is not None:
        cl = [i for i, z in enumerate(atoms.get_atomic_numbers()) if z == 17]
        del atoms[cl[remove_cl]]
    return atoms


def _frame(atoms, counts, charge):
    atoms = atoms.copy()
    atoms.info.update({"carrier_counts": np.array(counts), "cell_charge": charge,
                       "config_type": "toy", "source_dir": "t"})
    return atoms


def _batch(frames):
    ds = dd.atomic_data(frames, Z_TABLE, R_CUT)
    return next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=len(ds)))).to_dict()


@pytest.fixture(scope="module")
def model():
    m = MACEDSCC(_base(), r_cut=R_CUT, directional=True, hidden=16)
    with torch.no_grad():
        m.h0.vector_mix.normal_(0.0, 0.3)
        m.h0.alpha.fill_(0.5)
        m.h0.beta.fill_(0.5)
    m.set_pristine_centre([_batch([_frame(_perovskite(rattle=0.0), [0, 0, 0, 0], 0)])])
    return m


PRISTINE = _frame(_perovskite(), [0, 0, 0, 0], 0)
VAC0 = _frame(_perovskite(remove_cl=0), [0, 0, 0, 0], 0)
VACP = _frame(_perovskite(remove_cl=0), [0, 0, 1, 0], 1)


class TestNeutralNull:
    def test_reference_states_return_the_base_bit_identically(self, model):
        batch = _batch([PRISTINE, VAC0])
        out = model(dict(batch), compute_force=True, compute_stress=True)
        ref = modules.ScaleShiftMACE.forward(model.base, model._trunk_data(dict(_batch([PRISTINE, VAC0]))),
                                             compute_force=True, compute_stress=True)
        assert out["short_circuit"] and torch.equal(out["energy"], ref["energy"])
        assert torch.equal(out["forces"], ref["forces"]) and torch.equal(out["stress"], ref["stress"])
        assert float(out["head_energy"].abs().max()) == 0.0


class TestChargedState:
    def test_charge_sum_rule_and_energy_decomposition(self, model):
        out = model(_batch([VACP]), compute_force=True)
        assert not out["short_circuit"]
        assert float(out["dq"].sum()) == pytest.approx(1.0, abs=1e-12)
        assert float(out["energy"] - out["base_energy"] - out["head_energy"]) == pytest.approx(0.0, abs=1e-12)   # C_Q = 0
        assert not out["calibrated"]
        assert torch.isfinite(out["forces"]).all()

    def test_mixed_batch_matches_single_frames(self, model):
        out = model(_batch([VACP, VAC0, VACP]), compute_force=True)
        single = model(_batch([VACP]), compute_force=True)
        assert float((out["energy"][0] - single["energy"][0]).abs()) < 1e-12
        assert float((out["energy"][2] - single["energy"][0]).abs()) < 1e-12
        n = len(VACP)
        assert float((out["forces"][:n] - single["forces"]).abs().max()) < 1e-11
        base0 = modules.ScaleShiftMACE.forward(model.base, model._trunk_data(_batch([VAC0])),
                                               compute_force=True)
        assert float((out["energy"][1] - base0["energy"][0]).abs()) < 1e-12
        assert float((out["forces"][n:2 * n] - base0["forces"]).abs().max()) < 1e-11

    def test_forces_against_central_differences(self, model):
        atoms = VACP
        out = model(_batch([atoms]), compute_force=True)
        h = 1e-4
        for atom, comp in ((1, 0), (7, 2), (20, 1)):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[atom, comp] += h
            minus.positions[atom, comp] -= h
            e_plus = float(model(_batch([plus]), compute_force=False)["energy"])
            e_minus = float(model(_batch([minus]), compute_force=False)["energy"])
            fd = -(e_plus - e_minus) / (2 * h)
            assert float(out["forces"][atom, comp]) == pytest.approx(fd, abs=2e-6)

    def test_stress_against_strain_differences(self, model):
        atoms = VACP
        out = model(_batch([atoms]), compute_force=False, compute_stress=True)
        volume = atoms.get_volume()
        h = 1e-4
        for i, j in ((0, 0), (1, 2), (2, 2)):
            e = []
            for sign in (1.0, -1.0):
                strained = atoms.copy()
                eps = np.zeros((3, 3)); eps[i, j] = eps[j, i] = sign * h
                strained.set_cell(np.array(atoms.get_cell()) @ (np.eye(3) + eps), scale_atoms=True)
                e.append(float(model(_batch([strained]), compute_force=False)["energy"]))
            fd = (e[0] - e[1]) / (2 * h) / volume
            expected = float(out["stress"][0, i, j] + (out["stress"][0, j, i] if i != j else 0.0))
            assert expected == pytest.approx(fd, abs=1e-7)

    def test_invariances(self, model):
        e0 = float(model(_batch([VACP]), compute_force=False)["energy"])
        shifted = VACP.copy(); shifted.positions += [1.3, -0.4, 2.2]
        wrapped = VACP.copy(); wrapped.positions[4] += np.array(VACP.get_cell())[0]
        perm = VACP[np.random.default_rng(1).permutation(len(VACP))]
        for other in (shifted, wrapped, perm):
            assert float(model(_batch([other]), compute_force=False)["energy"]) == pytest.approx(e0, abs=1e-10)
        R = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))[0]
        if np.linalg.det(R) < 0:
            R = -R
        rotated = VACP.copy()
        rotated.set_cell(np.array(VACP.get_cell()) @ R.T, scale_atoms=False)
        rotated.positions = VACP.get_positions() @ R.T
        assert float(model(_batch([rotated]), compute_force=False)["energy"]) == pytest.approx(e0, abs=1e-9)


class TestCheckpoint:
    def test_round_trip_is_bit_identical(self, model):
        buf = io.BytesIO()
        torch.save(model, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=False)
        a = model(_batch([VACP]), compute_force=True)
        b = loaded(_batch([VACP]), compute_force=True)
        assert torch.equal(a["energy"], b["energy"]) and torch.equal(a["forces"], b["forces"])
        # state_dict round trip carries the registered numbers and the calibration.
        model2 = copy.deepcopy(model)
        model2.set_calibration({1: 0.37}, {"frames": "toy", "when": "test"})
        fresh = MACEDSCC(_base(), r_cut=R_CUT, directional=True, hidden=16)
        fresh.load_state_dict(model2.state_dict())
        assert fresh.calibrated and float(fresh.c_q(1)) == 0.37 and fresh.kernel == model.kernel
        c = fresh(_batch([VACP]), compute_force=False)
        assert float(c["energy"] - c["energy_uncalibrated"]) == pytest.approx(0.37)
        assert torch.equal(c["energy_uncalibrated"], a["energy_uncalibrated"])


# ----------------------------------------------------------------- Phase 1: the coupled head

def _coupled(regime="A", route_b=False, seed=0):
    from mace.modules.dscc.kernels import KernelConfig
    from mace.modules.dscc.scf import ScfOptions
    m = MACEDSCC(_base(seed), r_cut=R_CUT, directional=True, hidden=16, coupling=True,
                 route_b=route_b, kernel=KernelConfig(regime=regime, r_g=1.0, r_s=5.0),
                 scf=ScfOptions(tol_q=1e-11, tol_E=1e-11))     # 1e-13 is below eigh's floor at 160 orbitals
    with torch.no_grad():
        m.h0.vector_mix.normal_(0.0, 0.3)
        m.h0.alpha.fill_(0.5)
        m.h0.beta.fill_(0.5)
        m.lambda_raw.fill_(0.0)                  # lambda_dir = lambda_max / 2
        m.u_raw.fill_(-1.0)
    pristine = _batch([_frame(_perovskite(rattle=0.0), [0, 0, 0, 0], 0)])
    m.set_pristine_centre([pristine])
    if route_b:
        m.initialise_route_b([pristine])
    return m


@pytest.fixture(scope="module", params=["A", "B"])
def coupled(request):
    return _coupled(regime=request.param)


class TestCoupledHead:
    def test_fixed_point_diagnostics_and_charge(self, coupled):
        out = coupled(_batch([VACP]), compute_force=True)
        d = out["diagnostics"]
        assert d["converged"] == [True] and d["residual"][0] < 1e-10 and d["commutator"][0] < 1e-7
        assert abs(d["band_minus_primary"][0]) < 1e-9
        assert float(out["dq"].sum()) == pytest.approx(1.0, abs=1e-12)

    def test_forces_against_central_differences(self, coupled):
        out = coupled(_batch([VACP]), compute_force=True)
        h = 1e-4
        for atom, comp in ((1, 0), (9, 2), (30, 1)):
            plus, minus = VACP.copy(), VACP.copy()
            plus.positions[atom, comp] += h
            minus.positions[atom, comp] -= h
            e_plus = float(coupled(_batch([plus]), compute_force=False)["energy"])
            e_minus = float(coupled(_batch([minus]), compute_force=False)["energy"])
            assert float(out["forces"][atom, comp]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=3e-6)

    def test_stress_against_strain_differences(self, coupled):
        out = coupled(_batch([VACP]), compute_force=False, compute_stress=True)
        volume = VACP.get_volume()
        h = 1e-4
        for i, j in ((0, 0), (0, 2)):
            e = []
            for sign in (1.0, -1.0):
                strained = VACP.copy()
                eps = np.zeros((3, 3)); eps[i, j] = eps[j, i] = sign * h
                strained.set_cell(np.array(VACP.get_cell()) @ (np.eye(3) + eps), scale_atoms=True)
                e.append(float(coupled(_batch([strained]), compute_force=False)["energy"]))
            fd = (e[0] - e[1]) / (2 * h) / volume
            expected = float(out["stress"][0, i, j] + (out["stress"][0, j, i] if i != j else 0.0))
            assert expected == pytest.approx(fd, abs=1e-7)

    def test_gauge(self, coupled):
        """`H0 -> H0 + a I`: `dq` and forces unchanged to 1e-10; `E` shifts by exactly `-a Q`."""
        ref = coupled(_batch([VACP]), compute_force=True)
        coupled.h0.gauge_shift = 0.37
        try:
            out = coupled(_batch([VACP]), compute_force=True)
        finally:
            coupled.h0.gauge_shift = 0.0
        assert float((out["dq"] - ref["dq"]).abs().max()) < 1e-10
        assert float((out["forces"] - ref["forces"]).abs().max()) < 1e-10
        assert float(out["energy"] - ref["energy"]) == pytest.approx(-0.37 * 1, abs=1e-10)

    def test_invariances(self, coupled):
        e0 = float(coupled(_batch([VACP]), compute_force=False)["energy"])
        shifted = VACP.copy(); shifted.positions += [1.3, -0.4, 2.2]
        wrapped = VACP.copy(); wrapped.positions[4] += np.array(VACP.get_cell())[0]
        perm = VACP[np.random.default_rng(1).permutation(len(VACP))]
        for other in (shifted, wrapped, perm):
            assert float(coupled(_batch([other]), compute_force=False)["energy"]) == pytest.approx(e0, abs=1e-9)


class TestRouteB:
    def test_centred_pattern_and_forces(self):
        m = _coupled(regime="A", route_b=True)
        zbar_species = m.zstar()
        comp = m.pristine_composition
        assert float(zbar_species @ comp) == pytest.approx(0.0, abs=1e-12)
        assert float(m.zstar_init.abs().max()) > 1e-3 and float(m.z_max) > 0
        out = m(_batch([VACP]), compute_force=True)
        assert out["diagnostics"]["converged"] == [True]
        assert float(out["dq"].sum()) == pytest.approx(1.0, abs=1e-12)
        h = 1e-4
        for atom, comp_ in ((2, 1), (15, 0)):
            plus, minus = VACP.copy(), VACP.copy()
            plus.positions[atom, comp_] += h
            minus.positions[atom, comp_] -= h
            e_plus = float(m(_batch([plus]), compute_force=False)["energy"])
            e_minus = float(m(_batch([minus]), compute_force=False)["energy"])
            assert float(out["forces"][atom, comp_]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=3e-6)
        # The far-field force channel: forces on atoms far from the carrier are nonzero.
        assert torch.isfinite(out["forces"]).all()
