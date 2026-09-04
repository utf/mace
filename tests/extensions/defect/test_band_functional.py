"""Plan v8 sections 3 and 7.1: the band functional's identity, and the mock policy's contract.

`F_band(H, N)` is defined by `dF_band/dH_ab = P_ba` at fixed N per spin -- the statement
that makes Hellmann-Feynman forces and the density response exact. Under Gaussian smearing
`F = sum f_k eps_k - (sigma / 2 sqrt(pi)) sum exp(-x_k^2)`, under Fermi-Dirac it is the
Mermin free energy. Both are the production fills, both are tested, both spins, by central
finite differences in H against the density matrix the fill itself builds. Perturbing
`H_ab` and `H_ba` together (H is symmetric) compares to `P_ab + P_ba`.

The mock policy is held to the same matrix-level contract on a known occupation vector,
and to nothing else.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from mace.modules import defect_counting as dc
from mace.modules import defect_state as ds
from tests.extensions.defect import mock_policy

torch.set_default_dtype(torch.float64)


def synthetic_h(seed: int, n: int = 12) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(n, n, generator=g, dtype=torch.float64)
    m = 0.5 * (m + m.T)
    # A gap in the middle, so a frontier that sits inside it AND one that does not are
    # both reachable by the choice of N.
    lam, U = torch.linalg.eigh(m)
    lam = lam + torch.where(torch.arange(n) >= n // 2, 3.0, 0.0)
    return (U * lam) @ U.T


def band_free_energy(H, n_maj, n_min, t_el, family):
    """`F_band(H, N)` summed over both spins, from the production fill of THIS H."""
    previous = dc.use_smearing(family, t_el)
    try:
        lam = torch.linalg.eigvalsh(H)
        return (dc.free_energy(lam, float(n_maj), t_el)
                + dc.free_energy(lam, float(n_min), t_el))
    finally:
        dc.use_smearing(*previous)


def density(H, n_maj, n_min, t_el, family):
    previous = dc.use_smearing(family, t_el)
    try:
        lam, U = torch.linalg.eigh(H)
        f = dc.fermi_fill(lam, float(n_maj), t_el) + dc.fermi_fill(lam, float(n_min), t_el)
        return (U * f) @ U.T
    finally:
        dc.use_smearing(*previous)


def fd_matrix(fn, H, h=1e-5):
    """Central differences of `fn(H)` in every (a, b) with a <= b, perturbing H_ab and H_ba
    together for a != b. Returns the symmetric matrix G with G_ab = dF/dH_ab + dF/dH_ba."""
    n = H.shape[0]
    G = torch.zeros_like(H)
    for a in range(n):
        for b in range(a, n):
            E = torch.zeros_like(H)
            E[a, b] = 1.0
            E[b, a] = 1.0
            G[a, b] = G[b, a] = (fn(H + h * E) - fn(H - h * E)) / (2 * h)
    return G


@pytest.fixture
def tight_mu(monkeypatch):
    """The bisection's 1e-10 tolerance on `mu` is a finite-difference FLOOR, not a model
    error: at a partially occupied frontier `dN/dmu ~ 1/T`, so a 1e-10 error in `mu` is a
    ~1e-10 eV error in F, which divided by 2h = 2e-5 is a 5e-6 error in the derivative --
    exactly the `epsilon / h` term of section 7.2, and exactly what the first run of this
    test measured on the Fermi-Dirac electron case (4.3e-6). Tightening the solver for the
    measurement, rather than loosening the assertion, keeps the identity a strict one."""
    import functools

    monkeypatch.setattr(dc, "find_mu", functools.partial(dc.find_mu, tol=1e-14))


@pytest.mark.parametrize("family", ["gaussian", "fermi"])
@pytest.mark.parametrize("fills", [(6, 6), (7, 6), (5, 6), (6, 5)],
                         ids=["reference", "electron", "hole_maj", "hole_min"])
def test_dF_dH_is_the_density_matrix_for_the_production_fills(family, fills, tight_mu):
    """Both families, both spins, the frontier in the gap and at a band edge."""
    H = synthetic_h(0)
    t_el = 0.05
    n_maj, n_min = fills
    fn = lambda M: float(band_free_energy(M, n_maj, n_min, t_el, family))  # noqa: E731
    G = fd_matrix(fn, H)
    P = density(H, n_maj, n_min, t_el, family)
    expect = P + P.T - torch.diag(torch.diagonal(P))     # P_ab + P_ba, once on the diagonal
    err = float((G - expect).abs().max())
    assert err < 1e-7, f"{family} {fills}: max |FD - (P_ab + P_ba)| = {err:.2e}"


def test_the_identity_fails_when_the_entropy_term_is_dropped():
    """The counterfactual: `sum f_k eps_k` alone with SMEARED occupations is not the band
    functional, and the finite difference then disagrees with P wherever a level is
    fractionally occupied. This is what makes the test above non-vacuous."""
    H = synthetic_h(3)
    t_el = 0.5                                              # wide, so fractions exist
    n = 6.5

    def wrong(M):
        previous = dc.use_smearing("gaussian", t_el)
        try:
            lam = torch.linalg.eigvalsh(M)
            return float((dc.fermi_fill(lam, n, t_el) * lam).sum())
        finally:
            dc.use_smearing(*previous)

    G = fd_matrix(wrong, H)
    P = density(H, n, 0.0, t_el, "gaussian")
    expect = P + P.T - torch.diag(torch.diagonal(P))
    assert float((G - expect).abs().max()) > 1e-3


class TestMockPolicy:
    """The mock's contract and NOTHING else: no physical state, no force, no stress."""

    @pytest.fixture(autouse=True)
    def installed(self):
        mock_policy.install()

    def test_it_is_dispatchable_and_not_configurable(self):
        assert isinstance(ds.dispatch(mock_policy.MOCK_KEY), mock_policy.MockKnownOccupation)
        from tests.extensions.defect.test_neutral_reference_skip import _model

        with pytest.raises(ValueError, match="not a production policy"):
            _model(occupation_policy=mock_policy.MOCK_KEY)

    def test_schema_and_count_consistency(self):
        n = 12
        f_maj = [1.0] * 6 + [0.0] * 6
        f_min = [1.0] * 5 + [0.5, 0.5] + [0.0] * 5
        # 12 electrons split 6/6 at the reference; this state has delta_n = (0, 0).
        good = mock_policy.mock_state(f_maj, f_min, q_formal=0, delta_n=(0, 0))
        mock_policy.check_counts(good, ds.reference_state(), n, 12)
        bad_charge = mock_policy.mock_state(f_maj, f_min, q_formal=1, delta_n=(0, 0))
        with pytest.raises(ValueError, match="Q_formal - Q_ref"):
            mock_policy.check_counts(bad_charge, ds.reference_state(), n, 12)
        bad_trace = mock_policy.mock_state([1.0] * 7 + [0.0] * 5, f_min, q_formal=0,
                                           delta_n=(0, 0))
        with pytest.raises(ValueError, match="Tr\\[P_sigma"):
            mock_policy.check_counts(bad_trace, ds.reference_state(), n, 12)
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            mock_policy.check_counts(
                mock_policy.mock_state([2.0] + [0.0] * 11, f_min, 0, (0, 0)),
                ds.reference_state(), n, 12)

    def test_equality_to_s_ref_is_on_the_physical_key_not_the_charge(self):
        """Same Q_formal, same delta_n, different policy: NOT the reference, and the head
        correction under the mock is not zero there. Charge alone does not identify it."""
        H = synthetic_h(1)
        f_maj = [1.0] * 5 + [0.0, 1.0] + [0.0] * 5           # a hole below, an electron above
        f_min = [1.0] * 6 + [0.0] * 6
        s = mock_policy.mock_state(f_maj, f_min, q_formal=0, delta_n=(0, 0))
        assert s.q_formal == ds.reference_state().q_formal
        assert s.delta_n == ds.reference_state().delta_n
        assert not s.is_reference(ds.reference_state())
        e, *_ = ds.dispatch(mock_policy.MOCK_KEY).solve(H, 12, s)
        assert float(e) > 0.0, "an excitation costs energy; the correction is not null"

    def test_cache_keys_separate_the_mock_from_the_count_fill(self):
        s_mock = mock_policy.mock_state([1.0] * 6 + [0.0] * 6, [1.0] * 6 + [0.0] * 6, 0, (0, 0))
        assert s_mock.key_digest() != ds.reference_state().key_digest()
        # and two mocks with different known occupations are different keys
        other = mock_policy.mock_state([1.0] * 5 + [0.0, 1.0] + [0.0] * 5,
                                       [1.0] * 6 + [0.0] * 6, 0, (0, 0))
        assert other.key_digest() != s_mock.key_digest()

    def test_the_matrix_level_contract_dF_dH_equals_P(self):
        """On a known occupation vector, by finite differences, both spins."""
        H = synthetic_h(2)
        f_maj = [1.0] * 5 + [0.3, 0.7] + [0.0] * 5
        f_min = [1.0] * 4 + [0.5, 0.5, 0.5, 0.5] + [0.0] * 4
        s = mock_policy.mock_state(f_maj, f_min, q_formal=0, delta_n=(0, 0))
        policy = ds.dispatch(mock_policy.MOCK_KEY)

        def energy(M):
            e, *_ = policy.solve(M, 12, s)
            return float(e)

        G = fd_matrix(energy, H)
        _, _, _, p_now, p_ref = policy.solve(H, 12, s)
        P = p_now - p_ref
        expect = P + P.T - torch.diag(torch.diagonal(P))
        assert float((G - expect).abs().max()) < 1e-7

    def test_it_refuses_a_count_fill_state_and_has_no_batched_path(self):
        policy = ds.dispatch(mock_policy.MOCK_KEY)
        with pytest.raises(ValueError, match="only a mock state"):
            policy.solve(synthetic_h(0), 12, ds.reference_state())
        with pytest.raises(NotImplementedError):
            policy.solve_batched(None, None, None, None)

    def test_no_production_module_imports_the_mock(self):
        root = Path(__file__).resolve().parents[3] / "mace"
        offenders = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module]
                if any("mock_policy" in m or m.startswith("tests") for m in mods):
                    offenders.append(str(path.relative_to(root)))
                    break
        assert not offenders, offenders
