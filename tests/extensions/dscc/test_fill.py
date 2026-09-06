"""Plan section 4 gate: `dF_band/dH_ab = P_ba` to 1e-10; the Gaussian fill identities."""
import math

import pytest
import torch

from mace.modules.dscc import fill as fl

torch.set_default_dtype(torch.float64)


def _random_h(n: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    return scale * 0.5 * (a + a.T)


class TestFill:
    def test_occupations_sum_to_n_and_the_entropy_is_the_gaussian_one(self):
        H = _random_h(12, 0)
        res = fl.fill(H, 5.0)
        assert float(res.f.sum()) == pytest.approx(5.0, abs=1e-10)
        x = (res.eps - res.mu) / fl.SIGMA_S
        assert float(res.entropy) == pytest.approx(
            float(-fl.SIGMA_S * torch.exp(-x * x).sum() / (2 * math.sqrt(math.pi))), abs=1e-14)
        assert float(res.F_band) == pytest.approx(float(fl.band_free_energy_value(res.eps, 5.0)),
                                                  abs=1e-12)
        assert float(torch.trace(res.P)) == pytest.approx(5.0, abs=1e-10)
        assert torch.allclose(res.P, res.P.T, atol=1e-14)

    def test_df_band_dh_is_the_density_matrix(self):
        """Central differences of `F_band(H + h E_ab)` against `P_ba` on a spectrum with
        fractional levels (small scale: levels within a few widths of `mu`)."""
        H = _random_h(10, 1, scale=0.05)
        N = 4.0
        res = fl.fill(H, N)
        assert 0.0 < float(res.f[3]) < 1.0 or 0.0 < float(res.f[4]) < 1.0   # fractional
        h = 1e-5
        for a, b in ((0, 0), (2, 5), (3, 4), (7, 1)):
            plus, minus = H.clone(), H.clone()
            plus[a, b] += h; plus[b, a] += h if a != b else 0.0
            minus[a, b] -= h; minus[b, a] -= h if a != b else 0.0
            fd = (float(fl.band_free_energy_value(torch.linalg.eigvalsh(plus), N))
                  - float(fl.band_free_energy_value(torch.linalg.eigvalsh(minus), N))) / (2 * h)
            expected = float(res.P[b, a] + (res.P[a, b] if a != b else 0.0))
            assert fd == pytest.approx(expected, abs=1e-8)
        # And autograd returns exactly P (the envelope construction).
        Hg = H.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(fl.fill(Hg, N).F_band, Hg)
        assert float((grad - res.P).abs().max()) < 1e-12

    def test_density_response_gradient_is_the_divided_difference(self):
        """`d Tr(P A)/dH` from the reused Function against finite differences."""
        H = _random_h(8, 2, scale=0.05)
        A = _random_h(8, 3)
        N = 3.0
        Hg = H.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad((fl.fill(Hg, N).P * A).sum(), Hg)
        h = 1e-5
        for a, b in ((1, 1), (0, 4), (2, 6)):
            plus, minus = H.clone(), H.clone()
            plus[a, b] += h; minus[a, b] -= h
            if a != b:
                plus[b, a] += h; minus[b, a] -= h
            fd = (float((fl.fill(plus, N).P * A).sum()) - float((fl.fill(minus, N).P * A).sum())) / (2 * h)
            expected = float(grad[a, b] + (grad[b, a] if a != b else 0.0))
            assert fd == pytest.approx(expected, abs=1e-6, rel=1e-6)

    def test_batched_fill_matches_per_frame(self):
        Hs = torch.stack([_random_h(9, 4), _random_h(9, 5)])
        res = fl.fill(Hs, torch.tensor([3.0, 5.0]))
        for i, n in enumerate((3.0, 5.0)):
            single = fl.fill(Hs[i], n)
            assert torch.allclose(res.P[i], single.P, atol=1e-13)
            assert float(res.F_band[i]) == pytest.approx(float(single.F_band), abs=1e-12)

    def test_non_gaussian_setting_and_float32_are_refused(self):
        from mace.modules.dscc import legacy as cnt
        with pytest.raises(TypeError):
            fl.fill(_random_h(4, 6).float(), 2.0)
        previous = cnt.use_smearing("fermi", 0.05)
        try:
            with pytest.raises(RuntimeError):
                fl.fill(_random_h(4, 6), 2.0)
        finally:
            cnt.use_smearing(*previous) if isinstance(previous, tuple) else cnt.use_smearing("gaussian", 0.05)


def test_newton_chemical_potential_matches_bisection_to_the_floor():
    from mace.modules import defect_counting as cnt
    for seed, n, scale in ((0, 5.0, 1.0), (1, 4.0, 0.05), (2, 7.0, 3.0)):
        H = _random_h(12, seed, scale=scale)
        eps = torch.linalg.eigvalsh(H)
        mu = fl.chemical_potential(eps, n, fl.SIGMA_S)
        ref = cnt.find_mu(eps, n, fl.SIGMA_S, "gaussian", tol=1e-13)
        assert float(fl.occupations(eps, mu, fl.SIGMA_S).sum()) == pytest.approx(n, abs=1e-12)
        # mu is not unique inside a gap (any value beyond the smearing tails gives the same
        # occupations): compare the occupations, which are what the fill uses.
        assert torch.allclose(fl.occupations(eps, mu, fl.SIGMA_S), fl.occupations(eps, ref, fl.SIGMA_S), atol=1e-12)
    # batched, per-frame counts
    Hs = torch.stack([_random_h(9, 4), _random_h(9, 5)]); eps = torch.linalg.eigvalsh(Hs)
    mu = fl.chemical_potential(eps, torch.tensor([3.0, 5.0]), fl.SIGMA_S)
    assert torch.allclose(fl.occupations(eps, mu, fl.SIGMA_S).sum(-1), torch.tensor([3.0, 5.0]), atol=1e-12)
