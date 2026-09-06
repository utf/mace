"""Addendum sections 5.2 and 6.1: the image-active density and the boundary functional,
as pure functions (the forward wiring is `test_stage4_boundary.py`)."""

from __future__ import annotations

import math

import pytest
import torch

from mace.modules import defect_image as di
from mace.modules.defect_density import GaussianDensity
from mace.modules.defect_lift import build_lift
from mace.modules.defect_madelung import COULOMB_CONSTANT, self_potential_of
from mace.modules.latent_ewald import LatentEwald

torch.set_default_dtype(torch.float64)

ALPHA_M_SC = 2.8372974          # simple-cubic Madelung constant, jellium
SIGMA = 1.0


def cell(a: float) -> torch.Tensor:
    return torch.eye(3, dtype=torch.float64) * a


def density(charges, centres, a: float, sigma: float = SIGMA) -> GaussianDensity:
    return GaussianDensity(torch.tensor(charges, dtype=torch.float64),
                           torch.tensor(centres, dtype=torch.float64), sigma, cell(a))


def record_for(centres, a: float):
    """A lift branch anchored on the given (compact) support."""
    scaled = torch.tensor(centres, dtype=torch.float64) / a
    return build_lift(-torch.ones(len(centres), dtype=torch.float64), scaled)


#: k_max sigma = 2 pi: the reciprocal sum is converged to 1e-6 (see TestConvergence).
DL = 1.0


@pytest.fixture(scope="module")
def ewald():
    return LatentEwald({"sigma": SIGMA, "dl": DL}).double()


def _isolated(ewald, charges, centres):
    return ewald.isolated_energy(charges, centres, torch.zeros(charges.numel(),
                                                               dtype=torch.long), 1).reshape(())


# ------------------------------------------------------------------ the kernel


class TestIsolatedKernel:
    def test_the_direct_sum_is_the_les_isolated_evaluator(self, ewald):
        """Same smearing convention as the periodic half, or K_img is not an image term."""
        torch.manual_seed(0)
        q = torch.randn(7)
        r = torch.rand(7, 3) * 6.0
        c = di.coulomb_constant_of(ewald)
        mine = di.isolated_self_energy(q, r, SIGMA, c)
        les = _isolated(ewald, q, r)
        assert float(mine) == pytest.approx(float(les), abs=1e-9)
        # With the tabulated constant instead, the two differ at 4e-7 relative -- the
        # reason the evaluator's own constant is read rather than assumed.
        assert abs(float(di.isolated_self_energy(q, r, SIGMA)) - float(les)) > 1e-6

    def test_the_bilinear_form_is_the_les_cross_term(self, ewald):
        torch.manual_seed(1)
        qa, ra = torch.randn(4), torch.rand(4, 3) * 5.0
        qb, rb = torch.randn(5), torch.rand(5, 3) * 5.0 + 1.0
        cross = (_isolated(ewald, torch.cat([qa, qb]), torch.cat([ra, rb]))
                 - _isolated(ewald, qa, ra) - _isolated(ewald, qb, rb))
        assert float(di.isolated_bilinear(qa, ra, qb, rb, SIGMA,
                                          coulomb=di.coulomb_constant_of(ewald))
                     ) == pytest.approx(float(cross), abs=1e-9)

    def test_coincident_primitives_are_finite_with_a_finite_derivative(self):
        """The reason the isolated half is a direct sum and not LES's real-space fallback."""
        ra = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
        rb = torch.tensor([[1.0, 1.0, 1.0]])
        value = di.isolated_bilinear(torch.ones(1), ra, torch.ones(1), rb, SIGMA)
        expected = COULOMB_CONSTANT * 2.0 / (SIGMA * math.sqrt(2.0) * math.sqrt(math.pi))
        assert float(value) == pytest.approx(expected, rel=1e-12)
        (grad,) = torch.autograd.grad(value, ra)
        assert torch.isfinite(grad).all() and float(grad.abs().max()) == pytest.approx(0.0)

    def test_the_series_and_the_far_branch_agree_at_the_switch(self):
        r2 = torch.tensor([di._NEAR ** 2 * (1 - 1e-9), di._NEAR ** 2 * (1 + 1e-9)])
        k = di.isolated_pair_kernel(r2, SIGMA)
        assert float(k[0]) == pytest.approx(float(k[1]), rel=1e-12)

    def test_the_kernel_is_the_smeared_coulomb_law_at_range(self):
        r = torch.tensor([3.0, 5.0, 7.0])
        k = di.isolated_pair_kernel(r * r, SIGMA)
        for ri, ki in zip(r.tolist(), k.tolist()):
            assert ki == pytest.approx(COULOMB_CONSTANT * math.erf(ri / (SIGMA * math.sqrt(2)))
                                       / ri, rel=1e-12)

    def test_the_long_range_filter_splits_the_kernel_exactly(self):
        torch.manual_seed(2)
        qa, ra = torch.randn(6), torch.rand(6, 3) * 8.0
        qb, rb = torch.randn(6), torch.rand(6, 3) * 8.0
        full = di.isolated_bilinear(qa, ra, qb, rb, SIGMA)
        lr = di.isolated_bilinear(qa, ra, qb, rb, SIGMA, r_split=4.0)
        d = ra[:, None, :] - rb[None, :, :]
        r = d.norm(dim=-1)
        sr = qa @ (di.isolated_pair_kernel(r * r, SIGMA)
                   * (1.0 - di.long_range_envelope(r, 4.0))) @ qb
        assert float(lr + sr) == pytest.approx(float(full), abs=1e-10)
        # The filter is C^2 at both ends: zero with zero slope at r = 0, one beyond r_split.
        env = di.long_range_envelope(torch.tensor([0.0, 1e-3, 4.0, 9.0]), 4.0)
        assert float(env[0]) == 0.0 and float(env[1]) < 1e-15
        assert float(env[2]) == 1.0 and float(env[3]) == 1.0

    def test_gradients_check(self):
        torch.manual_seed(3)
        qa, qb = torch.randn(3), torch.randn(4)
        ra = (torch.rand(3, 3) * 4.0).requires_grad_(True)
        rb = torch.rand(4, 3) * 4.0

        def f(x):
            return di.isolated_bilinear(qa, x, qb, rb, SIGMA, r_split=2.5)

        assert torch.autograd.gradcheck(f, (ra,), eps=1e-6, atol=1e-6)


# ------------------------------------------------------------------ the image energy


class TestImageEnergy:
    @pytest.mark.parametrize("a", [8.0, 12.0, 16.0])
    def test_a_lone_charge_pays_the_makov_payne_image_energy(self, ewald, a):
        """`Phi_img^PBC = (E_PBC - E_inf) / eps` -- with E the energy `1/2 q A q`, so no
        further half. A single Gaussian has no isolated energy, so the whole of it is the
        self-image term `1/2 A_ii q^2`, which is Makov-Payne's `-alpha_M q^2 / (2 L)`."""
        eps = 4.0
        rho = density([1.0], [[a / 2] * 3], a)
        rec = record_for([[a / 2] * 3], a)
        phi = di.phi_img_pbc(rho, rec, ewald, eps)
        a_ii = float(self_potential_of(ewald, cell(a))[0])
        assert float(phi) == pytest.approx(0.5 * a_ii / eps, rel=1e-10)
        assert float(phi) == pytest.approx(-ALPHA_M_SC * di.coulomb_constant_of(ewald)
                                           / (2.0 * a) / eps, rel=1e-5)
        assert float(phi) < 0.0

    def test_it_is_a_pure_image_term_for_a_neutral_pair(self, ewald):
        """Two opposite charges: the isolated energy is their direct interaction and the
        periodic one adds only the dipole's images, `(2 pi / 3) p^2 / V` in jellium."""
        values = []
        for a in (10.0, 20.0, 40.0):
            c = [[a / 2, a / 2, a / 2], [a / 2 + 1.5, a / 2, a / 2]]
            rho = density([1.0, -1.0], c, a)
            values.append(abs(float(di.phi_img_pbc(rho, record_for(c, a), ewald, 1.0))))
        assert values[0] > values[1] > values[2]
        dipole = 2.0 * math.pi / 3.0 * 1.5 ** 2 * di.coulomb_constant_of(ewald) / 40.0 ** 3
        assert values[2] == pytest.approx(dipole, rel=0.05)

    def test_the_isolated_half_acts_on_the_lift_not_on_the_wrapped_density(self, ewald):
        """A compact pair straddling a cell face: on wrapped positions the two primitives
        are a cell apart and the 'isolated' energy is wrong by their whole interaction; on
        the lift they are 1 A apart, and the image energy matches the pair placed mid-cell."""
        a = 12.0
        mid = [[5.5, 6.0, 6.0], [6.5, 6.0, 6.0]]
        straddle = [[11.5, 6.0, 6.0], [0.5, 6.0, 6.0]]
        q = [1.0, -1.0]
        phi_mid = di.phi_img_pbc(density(q, mid, a), record_for(mid, a), ewald, 1.0)
        phi_face = di.phi_img_pbc(density(q, straddle, a), record_for(straddle, a), ewald, 1.0)
        assert float(phi_face) == pytest.approx(float(phi_mid), abs=1e-8)
        wrapped = (ewald.energy(torch.tensor(q), torch.tensor(straddle), cell(a).reshape(1, 3, 3),
                                torch.zeros(2, dtype=torch.long))
                   - _isolated(ewald, torch.tensor(q), torch.tensor(straddle)))
        assert abs(float(wrapped) - float(phi_mid)) > 1.0

    def test_translating_a_compact_object_through_every_face_is_invariant(self, ewald):
        """Section 11.1: continuous motion and, at a handoff, a common lattice translation
        -- the isolated energy is invariant to the numerical floor."""
        a = 12.0
        base = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.4, -0.3], [-0.8, 0.9, 0.5]])
        q = [1.0, -0.6, -0.4]
        values = []
        for axis in range(3):
            for t in torch.linspace(0.0, a, 13):
                shift = torch.zeros(3)
                shift[axis] = float(t)
                c = ((base + 6.0 + shift) % a).tolist()
                values.append(float(di.phi_img_pbc(density(q, c, a), record_for(c, a),
                                                   ewald, 1.0)))
        assert max(values) - min(values) < 1e-8


# ------------------------------------------------------------------ the static-frontier term


class TestStaticFrontier:
    def test_with_no_filter_it_is_the_isolated_cross_term(self, ewald):
        a = 14.0
        s = density([1.0, -0.5], [[7.0, 7.0, 7.0], [9.0, 7.0, 7.0]], a)
        f = density([-1.0, 0.5], [[7.5, 7.5, 7.0], [6.0, 8.0, 7.0]], a)
        rec = record_for([[7.0, 7.0, 7.0]], a)
        phi = di.phi_sf_lr(s, f, rec, SIGMA, r_split=1e-6, eps_inf=2.0,
                           coulomb=di.coulomb_constant_of(ewald))
        cross = (_isolated(ewald, torch.cat([s.charges, f.charges]),
                           torch.cat([s.centres, f.centres]))
                 - _isolated(ewald, s.charges, s.centres) - _isolated(ewald, f.charges, f.centres))
        assert float(phi) == pytest.approx(float(cross) / 2.0, abs=1e-9)

    def test_it_vanishes_when_everything_is_inside_the_split(self):
        a = 14.0
        s = density([1.0], [[7.0, 7.0, 7.0]], a)
        f = density([-1.0], [[7.5, 7.5, 7.0]], a)
        rec = record_for([[7.0, 7.0, 7.0]], a)
        # The filter is a C^2 polynomial, O((r / r_split)^6) inside: small, not exactly 0.
        assert abs(float(di.phi_sf_lr(s, f, rec, SIGMA, r_split=50.0, eps_inf=1.0))) < 1e-8

    def test_it_is_boundary_common_and_reported_under_both(self, ewald):
        a = 12.0
        s = density([1.0], [[6.0, 6.0, 6.0]], a)
        ch = di.Channel("e0", torch.tensor([0.7, 0.3]), 1, -1.0, torch.tensor(1.0))
        pos = torch.tensor([[6.5, 6.0, 6.0], [5.5, 6.0, 6.0]])
        dens = di.image_active_density(s, [ch], pos, q_core=1)
        rec = record_for([[6.0, 6.0, 6.0]], a)
        per = di.boundary_functional(dens, rec, ewald, r_split=3.0, eps_inf=4.0,
                                     boundary="periodic")
        iso = di.boundary_functional(dens, rec, ewald, r_split=3.0, eps_inf=4.0,
                                     boundary="isolated")
        assert float(per["phi_sf"]) == float(iso["phi_sf"])
        assert float(iso["phi_img"]) == 0.0 and float(per["phi_img"]) != 0.0
        assert float(per["phi_img_pbc"]) == float(iso["phi_img_pbc"]) == float(per["phi_img"])
        assert float(per["phi"]) == pytest.approx(float(per["phi_sf"]) + float(per["phi_img"]))
        assert float(iso["phi"]) == float(iso["phi_sf"])
        with pytest.raises(ValueError, match="boundary"):
            di.boundary_functional(dens, rec, ewald, r_split=3.0, eps_inf=4.0, boundary="pbc")


# ------------------------------------------------------------------ the densities


class TestImageDensity:
    @staticmethod
    def _make(w_e, w_h, n_e=1, n_h=1):
        a = 10.0
        s = density([1.25, -0.25], [[5.0, 5.0, 5.0], [7.0, 5.0, 5.0]], a)   # int = Q_core
        pos = torch.tensor([[5.0, 5.0, 5.0], [7.0, 5.0, 5.0], [5.0, 7.0, 5.0]])
        chans = [di.Channel("e0", torch.tensor([0.5, 0.3, 0.2]), n_e, -1.0, torch.tensor(w_e)),
                 di.Channel("h1", torch.tensor([0.1, 0.1, 0.8]), n_h, 1.0, torch.tensor(w_h)),
                 di.Channel("h0", torch.tensor([1.0, 0.0, 0.0]), 0, 1.0, torch.tensor(0.0))]
        return di.image_active_density(s, chans, pos, q_core=1)

    def test_every_integral_is_its_monopole(self):
        d = self._make(0.6, 1.0, n_e=2, n_h=1)
        assert float(d.frontier.integral()) == pytest.approx(d.q_f) and d.q_f == -1
        assert float(d.delta.integral()) == pytest.approx(d.q_formal) and d.q_formal == 0
        assert float(d.img.integral()) == pytest.approx(float(d.q_img))
        # q_img = Q_core - sum n_e w_e + sum n_h w_h
        assert float(d.q_img) == pytest.approx(1.0 - 2 * 0.6 + 1 * 1.0)
        assert "h0" not in d.weights          # a zero count is absent, not a zero weight

    def test_q_img_is_q_formal_only_on_the_compact_plateau(self):
        assert float(self._make(1.0, 1.0).q_img) == self._make(1.0, 1.0).q_formal
        assert self._make(1.0, 1.0).compact
        d = self._make(1.0, 0.999)
        assert not d.compact and float(d.q_img) != d.q_formal

    def test_rho_img_is_rho_delta_at_w_one(self):
        d = self._make(1.0, 1.0)
        assert torch.allclose(d.img.charges, d.delta.charges)
        d = self._make(0.5, 1.0)
        assert not torch.allclose(d.img.charges, d.delta.charges)

    def test_a_mismatched_channel_is_refused(self):
        s = density([1.0], [[5.0, 5.0, 5.0]], 10.0)
        ch = di.Channel("e0", torch.tensor([1.0, 0.0]), 1, -1.0, torch.tensor(1.0))
        with pytest.raises(ValueError, match="site densities"):
            di.image_active_density(s, [ch], torch.zeros(3, 3), q_core=1)


class TestConvergence:
    def test_the_periodic_half_must_be_converged_or_the_image_term_inherits_its_truncation(self):
        """LES truncates the reciprocal sum at `k_max = 2 pi / dl`. At its default `dl = 2`
        with `sigma = 1` the Gaussian weight there is still 7e-3, and because the isolated
        half is exact the whole remainder lands in `Phi_img`: 0.5 % on a lone charge and
        2.4 % on a 64-site lattice. At `k_max sigma = 2 pi` it is 1e-6. The unified regime
        therefore ties `dl` to `r_res`; the legacy evaluator keeps LES's default."""
        a = 12.0
        rho = density([1.0], [[a / 2] * 3], a)
        rec = record_for([[a / 2] * 3], a)
        mp = -ALPHA_M_SC * (90.4756 / (2.0 * math.pi)) / (2.0 * a)
        loose = float(di.phi_img_pbc(rho, rec, LatentEwald({"sigma": SIGMA, "dl": 2.0}).double(),
                                     1.0))
        tight = float(di.phi_img_pbc(rho, rec, LatentEwald({"sigma": SIGMA, "dl": 1.0}).double(),
                                     1.0))
        assert abs(loose / mp - 1.0) > 3e-3
        assert abs(tight / mp - 1.0) < 1e-5

    def test_the_registered_rule_converges_the_evaluator(self):
        for sigma in (0.7, 1.0, 1.5):
            dl = di.converged_dl(sigma)
            assert 2.0 * math.pi * sigma / dl >= 2.0 * math.pi - 1e-12
            assert dl <= 2.0
        assert di.converged_dl(1.0, 0.5) == 0.5      # never loosened past the config


class TestRefusals:
    def test_eps_inf_and_the_width_are_checked(self, ewald):
        a = 10.0
        rho = density([1.0], [[5.0] * 3], a)
        rec = record_for([[5.0] * 3], a)
        with pytest.raises(ValueError, match="eps_inf"):
            di.phi_img_pbc(rho, rec, ewald, 0.0)
        with pytest.raises(ValueError, match="eps_inf"):
            di.phi_sf_lr(rho, rho, rec, SIGMA, 2.0, -1.0)
        other = density([1.0], [[5.0] * 3], a, sigma=0.7)
        with pytest.raises(ValueError, match="smearing convention"):
            di.phi_img_pbc(other, rec, ewald, 1.0)
