"""The frozen-pristine spectral gauge (addendum section 3.1, gates in section 11.1)."""

import pytest
import torch

from mace.modules.defect_gauge import (
    DEFAULT_GAP_FLOOR,
    GaugeError,
    GaugeRecord,
    apply_gauge,
    assert_orthonormal,
    gauge_scalar,
    shift_edges,
    valence_trace,
)


def _pristine(n: int = 12, seed: int = 0) -> torch.Tensor:
    """A symmetric toy Hamiltonian with a clean gap after the fourth level."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=generator, dtype=torch.float64)
    h = 0.5 * (a + a.T)
    lam, vec = torch.linalg.eigh(h)
    # Open a 2 eV gap above rank 4 so the occupied manifold is unambiguous.
    lam = lam.clone()
    lam[4:] += 2.0
    return (vec * lam).matmul(vec.T)


class TestGaugeScalar:
    def test_rank_normalised_mean_of_the_occupied_manifold(self):
        h = _pristine()
        lam = torch.linalg.eigvalsh(h)
        mu, record = gauge_scalar(lam, (4, 4), pristine_key="toy")
        # Both spins share the spectrum, so the gauge is the mean occupied eigenvalue.
        assert mu.item() == pytest.approx(float(lam[:4].mean()), abs=1e-12)
        assert record.n_valence == 8
        assert record.ranks == (4, 4)

    def test_unequal_spin_ranks_weight_by_rank(self):
        h = _pristine()
        lam = torch.linalg.eigvalsh(h)
        mu, record = gauge_scalar(lam, (4, 3), pristine_key="toy")
        expected = float((lam[:4].sum() + lam[:3].sum()) / 7.0)
        assert mu.item() == pytest.approx(expected, abs=1e-12)
        assert record.n_valence == 7

    def test_extensive_trace_is_not_used(self):
        """Rank normalisation: doubling the cell must not double the gauge.

        An unnormalised occupied trace is extensive, so a gauge built from it would depend
        on the tiling of the reference cell. Two identical blocks are the exact analogue.
        """
        h = _pristine()
        lam = torch.linalg.eigvalsh(h)
        doubled = torch.cat([lam, lam])
        mu_one, _ = gauge_scalar(lam, (4, 4), pristine_key="toy")
        mu_two, _ = gauge_scalar(doubled, (8, 8), pristine_key="toy_2x")
        assert mu_two.item() == pytest.approx(mu_one.item(), abs=1e-12)


class TestGaugeInvariance:
    """Section 11.1: `H -> H + aI` shifts mu_g by exactly `a` and changes nothing else."""

    @pytest.mark.parametrize("a", [-3.0, -0.25, 0.25, 7.5])
    def test_common_shift_moves_mu_g_by_exactly_a(self, a):
        h = _pristine()
        lam = torch.linalg.eigvalsh(h)
        mu, _ = gauge_scalar(lam, (4, 4), pristine_key="toy")
        mu_shifted, _ = gauge_scalar(lam + a, (4, 4), pristine_key="toy")
        assert (mu_shifted - mu).item() == pytest.approx(a, abs=1e-12)

    @pytest.mark.parametrize("a", [-3.0, 0.25, 7.5])
    def test_gauge_fixed_hamiltonian_is_unchanged(self, a):
        h = _pristine()
        lam = torch.linalg.eigvalsh(h)
        identity = torch.eye(h.shape[-1], dtype=h.dtype)

        mu, _ = gauge_scalar(lam, (4, 4), pristine_key="toy")
        gauged = apply_gauge(h, mu)

        raw_shifted = h + a * identity
        mu_shifted, _ = gauge_scalar(torch.linalg.eigvalsh(raw_shifted), (4, 4),
                                     pristine_key="toy")
        gauged_shifted = apply_gauge(raw_shifted, mu_shifted)

        assert torch.allclose(gauged, gauged_shifted, atol=1e-12, rtol=0.0)

    @pytest.mark.parametrize("a", [-3.0, 0.25])
    def test_aligned_edges_shift_with_the_hamiltonian(self, a):
        """An edge left ungauged would move relative to the spectrum by mu_g."""
        h = _pristine()
        lam = torch.linalg.eigvalsh(h)
        identity = torch.eye(h.shape[-1], dtype=h.dtype)

        mu, _ = gauge_scalar(lam, (4, 4), pristine_key="toy")
        edge = float(lam[3])
        gauged_edge = shift_edges(edge, mu)
        gauged_spectrum = torch.linalg.eigvalsh(apply_gauge(h, mu))

        mu_s, _ = gauge_scalar(lam + a, (4, 4), pristine_key="toy")
        gauged_edge_s = shift_edges(edge + a, mu_s)
        gauged_spectrum_s = torch.linalg.eigvalsh(apply_gauge(h + a * identity, mu_s))

        assert gauged_edge_s == pytest.approx(gauged_edge, abs=1e-12)
        assert torch.allclose(gauged_spectrum, gauged_spectrum_s, atol=1e-12, rtol=0.0)

    def test_shift_edges_handles_containers(self):
        assert shift_edges([1.0, 2.0], 0.5) == [0.5, 1.5]
        assert shift_edges({"vbm": 1.0, "tag": "x"}, 0.5) == {"vbm": 0.5, "tag": "x"}


class TestGaugeGuards:
    def test_lost_separating_gap_is_an_error(self):
        """No gap means no well-defined occupied subspace, so the record is void."""
        lam = torch.linspace(0.0, 1.0, 12, dtype=torch.float64)
        with pytest.raises(GaugeError, match="separating gap"):
            gauge_scalar(lam, (4, 4), pristine_key="toy", gap_floor=0.5)

    def test_rank_at_the_top_of_the_spectrum_is_refused(self):
        lam = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
        with pytest.raises(GaugeError, match="no state above"):
            gauge_scalar(lam, (6,), pristine_key="toy")

    def test_nonorthogonal_representation_is_refused(self):
        overlap = torch.eye(4, dtype=torch.float64)
        assert_orthonormal(overlap)          # identity is fine
        overlap[0, 1] = overlap[1, 0] = 0.3
        with pytest.raises(GaugeError, match="orthonormal"):
            assert_orthonormal(overlap)

    def test_absent_overlap_is_treated_as_orthonormal(self):
        assert_orthonormal(None)

    def test_bad_rank_is_refused(self):
        lam = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
        with pytest.raises(GaugeError, match="outside"):
            valence_trace(lam, (0,))


class TestGaugeRecord:
    def test_round_trip_and_fingerprint(self):
        lam = torch.linalg.eigvalsh(_pristine())
        _, record = gauge_scalar(lam, (4, 4), pristine_key="toy")
        again = GaugeRecord.from_dict(record.to_dict())
        assert again == record
        assert again.fingerprint == record.fingerprint

    def test_fingerprint_separates_different_gauges(self):
        lam = torch.linalg.eigvalsh(_pristine())
        _, a = gauge_scalar(lam, (4, 4), pristine_key="toy")
        _, b = gauge_scalar(lam + 1.0, (4, 4), pristine_key="toy")
        assert a.fingerprint != b.fingerprint

    def test_gap_floor_default_is_recorded(self):
        lam = torch.linalg.eigvalsh(_pristine())
        _, record = gauge_scalar(lam, (4, 4), pristine_key="toy")
        assert record.gap_floor == DEFAULT_GAP_FLOOR
        assert min(record.gaps) >= DEFAULT_GAP_FLOOR


class TestGaugeGradient:
    def test_mu_g_is_differentiated_through_the_spectrum(self):
        """mu_g depends on theta and must carry gradient; it has no coordinate dependence."""
        h = _pristine().clone().requires_grad_(True)
        lam = torch.linalg.eigvalsh(h)
        mu, _ = gauge_scalar(lam, (4, 4), pristine_key="toy")
        mu.backward()
        assert h.grad is not None
        assert torch.isfinite(h.grad).all()
        # d mu_g / dH = P_V / n_valence, so the gradient trace is sum_sigma M_sigma / n = 1.
        assert float(h.grad.diagonal().sum()) == pytest.approx(1.0, abs=1e-10)
