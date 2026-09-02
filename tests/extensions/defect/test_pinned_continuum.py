"""A2: the pinned continuum, restated for Edit 1.

V3's anti-escape guarantee was a bit-identity: an atom further than `r_max` from the vacancy
has a block-1 descriptor IDENTICAL to its pristine counterpart, so the far block of H is
pinned to the host and the continuum cannot slide. That is what makes `Delta_bind` mean
something -- V1 met the band-edge inequality by moving the host origin instead of splitting a
level off, and reached +11.9 eV against a 0.5 eV target.

Edit 1 breaks the bit-identity ON PURPOSE. `phi_LR` is long-ranged, so the on-site energy of a
far atom is no longer equal to its pristine value; it differs by exactly the Madelung
potential of the missing charge. Without restating the test, the guarantee silently
disappears -- the test would simply be deleted as "no longer true" and nothing would replace
it.

The restated form has two clauses, and both are required:

1. the LEARNED parts -- on-site and hopping -- are still bit-identical to pristine beyond
   `r_max`;
2. the TOTAL on-site difference equals the analytic `-Delta phi_LR / eps_inf` to float
   precision.

Together they say the far block moved by electrostatics and by nothing else.

THE GEOMETRY MUST BE UNRELAXED. On a relaxed frame the far atoms have physically moved, so
their block-1 environments differ from pristine and clause 1 fails for a reason that has
nothing to do with the head. The test builds a vacancy cell as pristine-minus-one-atom, which
is the only geometry on which the bit-identity is even well posed.
"""

import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from e3nn import o3

from mace import data, modules, tools
from mace.data.defects import prepare_defect_configurations
from mace.modules.defect_madelung import self_potential_of, site_potential
from mace.modules.defect_models import MACEDefect
from mace.modules.latent_ewald import LatentEwald
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)

Z_TABLE = tools.AtomicNumberTable([17, 55, 82])          # Cl, Cs, Pb
CUTOFF = 4.0
EPS_INF = 4.0
# Pristine stoichiometry in Z_TABLE order: Cl 3, Cs 1, Pb 1.
COMPOSITION = [3.0, 1.0, 1.0]


def pristine_cell(a=5.6, reps=3):
    """Cubic CsPbCl3, ideal positions. No rattle: clause 1 is only well posed here."""
    motif = [("Pb", [0.0, 0.0, 0.0]), ("Cs", [0.5, 0.5, 0.5]),
             ("Cl", [0.5, 0.0, 0.0]), ("Cl", [0.0, 0.5, 0.0]), ("Cl", [0.0, 0.0, 0.5])]
    pos, sym = [], []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                for s, f in motif:
                    pos.append((np.array([i, j, k], float) + np.array(f)) * a)
                    sym.append(s)
    atoms = Atoms(symbols=sym, positions=np.array(pos), cell=np.eye(3) * (a * reps),
                  pbc=True)
    # The SAME counters as the vacancy cell. The parent head conditions eps on
    # counter_emb, so a neutral reference would differ from the charged one for a reason
    # that has nothing to do with descriptor locality -- and clause 1 is only about
    # locality. Physically odd, deliberately: this is a reference, not a configuration.
    atoms.info["charge"] = 1.0
    atoms.info["spin_multiplicity"] = 1
    atoms.arrays["forces"] = np.zeros((len(atoms), 3))
    atoms.info["energy"] = 0.0
    return atoms


def with_vacancy(atoms, charge=1.0):
    """Pristine minus one Cl, geometry otherwise untouched."""
    z = atoms.get_atomic_numbers()
    victim = int(np.where(z == 17)[0][len(np.where(z == 17)[0]) // 2])
    keep = [i for i in range(len(atoms)) if i != victim]
    out = atoms[keep]
    out.info["charge"] = charge
    out.info["spin_multiplicity"] = 1
    out.arrays["forces"] = np.zeros((len(out), 3))
    out.info["energy"] = 0.0
    return out, atoms.get_positions()[victim]


def build_model(madelung=True, seed=0, gauge_penalty=True):
    """`gauge_penalty=True` means eps is NOT mean-subtracted (see the head: the flag selects
    penalising the mean over removing it). A2's two clauses are exact statements about eps,
    so they are checked on the ungauged head; the gauged case is checked separately, where
    the same statement holds up to one per-cell constant."""
    torch.manual_seed(seed)
    return MACEDefect(
        spectral_gauge_penalty=gauge_penalty,
        r_max=CUTOFF, num_bessel=6, num_polynomial_cutoff=5, max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=len(Z_TABLE),
        hidden_irreps=o3.Irreps("16x0e + 16x1o"), MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([1.0, 2.0, 3.0]), avg_num_neighbors=6,
        atomic_numbers=Z_TABLE.zs, correlation=2,
        atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        carrier_feature_dim=8, counter_embedding_dim=8, carrier_mlp_hidden=16,
        use_long_range=False, spectral_head=True, spectral_first_shell=True,
        spectral_r_cut=8.0,
        madelung_on_site=madelung, madelung_eps_inf=EPS_INF,
        madelung_composition=COMPOSITION,
        madelung_z_init=[-1.0, 1.0, 2.0] if madelung else None,
    )


def batch_of(atoms_list, cutoff=8.0):
    configs = prepare_defect_configurations(
        [data.config_from_atoms(a) for a in atoms_list])
    dataset = [data.AtomicData.from_config(c, z_table=Z_TABLE, cutoff=cutoff)
               for c in configs]
    loader = torch_geometric.dataloader.DataLoader(dataset, batch_size=len(dataset),
                                                   shuffle=False)
    return next(iter(loader))


def eps_of(model, atoms, key="eps"):
    """The head's site energies, per atom. `key='eps_raw'` is the LEARNED term alone."""
    grabbed = {}
    head = model.spectral
    original = head.forward
    head.forward = lambda *a, **k: original(*a, **dict(k, internals=grabbed))
    try:
        with torch.no_grad():
            model(batch_of([atoms]).to_dict(), training=False, compute_force=False)
    finally:
        head.forward = original
    return grabbed[key].detach().clone(), grabbed


@pytest.fixture(scope="module")
def geometry():
    pristine = pristine_cell()
    vac, site = with_vacancy(pristine)
    # Far atoms: beyond r_max of the removed site, minimum image. Their block-1 descriptors
    # cannot know the vacancy exists.
    cell = np.diag(np.array(pristine.get_cell()))
    delta = vac.get_positions() - site
    delta -= np.round(delta / cell) * cell
    far = np.linalg.norm(delta, axis=1) > CUTOFF + 1e-9
    assert far.sum() > 10, "the test cell is too small to have a far region"
    return pristine, vac, site, far


class TestPinnedContinuum:
    def test_the_learned_parts_are_still_bit_identical_beyond_r_max(self, geometry):
        """Clause 1. The LEARNED on-site term -- eps_raw, before Edit 1 and before the gauge
        -- must still be bit-identical beyond r_max, with the Madelung term ON. That is the
        version that says the descriptor is local; comparing the gauged `eps` would compare
        two per-cell means as well."""
        pristine, vac, _, far = geometry

        model = build_model(madelung=True)
        eps_vac, _ = eps_of(model, vac, key="eps_raw")
        eps_pri, _ = eps_of(model, pristine, key="eps_raw")

        # Align the vacancy cell's atoms to their pristine counterparts. The vacancy cell is
        # the pristine list with one entry removed, so indices shift by one after the victim.
        z = pristine.get_atomic_numbers()
        victim = int(np.where(z == 17)[0][len(np.where(z == 17)[0]) // 2])
        keep = [i for i in range(len(pristine)) if i != victim]
        mapped = np.array(keep)

        diff = (eps_vac[far] - eps_pri[mapped[far]]).abs().max()
        assert float(diff) < 1e-10, (
            f"the learned on-site term moved by {float(diff):.3e} eV beyond r_max with the "
            "Madelung term OFF -- the block-1 descriptor is not local and the anti-escape "
            "guarantee is gone")

    def test_the_total_shift_is_exactly_the_madelung_difference(self, geometry):
        """Clause 2. With the term on, the far block moves by electrostatics and by nothing
        else -- checked against the analytic potential, not against the model."""
        pristine, vac, _, far = geometry
        model = build_model(madelung=True)
        eps_vac, _ = eps_of(model, vac)
        eps_pri, _ = eps_of(model, pristine)

        z = pristine.get_atomic_numbers()
        victim = int(np.where(z == 17)[0][len(np.where(z == 17)[0]) // 2])
        keep = [i for i in range(len(pristine)) if i != victim]
        mapped = np.array(keep)

        expected = self._expected_shift(model, pristine, vac, far, mapped)

        # eps is [n_nodes, C] and the Madelung shift is channel-independent, so any channel
        # carries it; the difference is taken on the same channel in both cells.
        got = (eps_vac[far, 0] - eps_pri[mapped[far], 0])
        assert torch.allclose(got, expected, atol=1e-9), (
            "the far-block on-site difference is not the analytic Madelung difference. "
            f"max discrepancy {float((got - expected).abs().max()):.3e} eV")

    def test_the_madelung_term_is_not_trivially_zero(self, geometry):
        """Guard: the two clauses above are both satisfied by a term that does nothing."""
        pristine, vac, _, far = geometry
        on = build_model(madelung=True)
        off = build_model(madelung=False)
        # Same seed, so the learned parts are identical and the difference is the term.
        assert float((eps_of(on, vac)[0] - eps_of(off, vac)[0]).abs().max()) > 1e-3

    def test_under_the_difference_gauge_the_residual_is_one_per_cell_constant(self,
                                                                             geometry):
        """The Stage-1/2 reading, stated exactly rather than waved at.

        With the mean-subtracting gauge on, clause 2 cannot hold as an equality: the gauge
        removes the uniform part of the Madelung shift along with everything else uniform.
        What it CAN hold as is `(eps_vac - eps_pri)[far] - expected[far] = const`, one
        constant per cell pair. If that residual varied from atom to atom, something other
        than a gauge would be moving the far block."""
        pristine, vac, _, far = geometry
        model = build_model(madelung=True, gauge_penalty=False)
        eps_vac, _ = eps_of(model, vac)
        eps_pri, _ = eps_of(model, pristine)

        z = pristine.get_atomic_numbers()
        victim = int(np.where(z == 17)[0][len(np.where(z == 17)[0]) // 2])
        mapped = np.array([i for i in range(len(pristine)) if i != victim])

        expected = self._expected_shift(model, pristine, vac, far, mapped)
        residual = (eps_vac[far, 0] - eps_pri[mapped[far], 0] - expected).numpy()
        assert np.ptp(residual) < 1e-9, (
            f"the far-block residual varies by {np.ptp(residual):.3e} eV across atoms; "
            "under the difference gauge it must be one constant")

    @staticmethod
    def _expected_shift(model, pristine, vac, far, mapped):
        ewald = LatentEwald(None).double()
        zv = torch.as_tensor([float(model.madelung.z[Z_TABLE.z_to_index(int(n))])
                              for n in vac.get_atomic_numbers()])
        zp = torch.as_tensor([float(model.madelung.z[Z_TABLE.z_to_index(int(n))])
                              for n in pristine.get_atomic_numbers()])

        def phi(atoms, charges):
            c = torch.as_tensor(np.array(atoms.get_cell())).reshape(1, 3, 3)
            b = torch.zeros(len(atoms), dtype=torch.long)
            return site_potential(ewald, charges,
                                  torch.as_tensor(atoms.get_positions()), c, b,
                                  self_potential=self_potential_of(ewald, c)).detach()

        return -(phi(vac, zv)[far] - phi(pristine, zp)[mapped[far]]) / EPS_INF


class TestConfigRoundTrip:
    def test_madelung_survives_the_round_trip(self):
        """The cuEq conversion rebuilds the model from its extracted config at the end of
        every run. A term dropped there is a term the saved model does not have."""
        from mace.tools.scripts_utils import extract_config_mace_model

        model = build_model(madelung=True)
        config = extract_config_mace_model(model)
        assert config["madelung_on_site"] is True
        assert config["madelung_eps_inf"] == pytest.approx(EPS_INF)
        assert config["madelung_composition"] == pytest.approx(COMPOSITION)
        rebuilt = MACEDefect(**config)
        assert rebuilt.madelung is not None

    def test_the_response_channel_is_gone(self):
        """Edit 2. Both terms are the carrier's electrostatics; both present is double
        counting, so the old flag must not merely default to False -- it must not exist."""
        import inspect

        assert "response_channel" not in inspect.signature(MACEDefect.__init__).parameters
        with pytest.raises(ImportError):
            import mace.modules.defect_response  # noqa: F401
