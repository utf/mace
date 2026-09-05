"""Plan v8 Stage 1.3: the SR/LR diagnostic switch on the Madelung on-site term.

`V_full = V_SR(r_split) + V_LR(r_split)` (section 7.3) with `V_SR` the smeared Coulomb
potential of the ions within `r_split`, summed over the head's neighbour list (images
included) and switched off smoothly at `r_split`. Pinned: `V_SR` against a brute-force sum
over lattice images on a small cell; "off" is an exact zero; the three modes give three
different Hamiltonians and the band-term force still passes the finite-difference harness
under "long_range" (the switch is smooth, the edge lengths carry the graph); the flag
survives the config round trip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mace.modules import defect_fd as fd
from mace.modules import defect_madelung as dm
from mace.modules.defect_context import ForwardContext
from mace.modules.latent_ewald import LatentEwald
from tests.extensions.defect.test_neutral_reference_skip import (_batch, _gapped,
                                                                  _perovskite)

torch.set_default_dtype(torch.float64)


def _brute_force_sr(charges, positions, cell, r_split, sigma, p=6, n_images=2):
    n = positions.shape[0]
    out = torch.zeros(n, dtype=positions.dtype)
    shifts = [torch.tensor(s, dtype=positions.dtype) @ cell
              for s in np.array(np.meshgrid(*[range(-n_images, n_images + 1)] * 3)).T.reshape(-1, 3)]
    for i in range(n):
        for j in range(n):
            for shift in shifts:
                r = float(torch.linalg.norm(positions[j] + shift - positions[i]))
                if r < 1e-9 or r >= r_split:
                    continue
                x = r / r_split
                env = (1 - (p + 1) * (p + 2) / 2 * x ** p + p * (p + 2) * x ** (p + 1)
                       - p * (p + 1) / 2 * x ** (p + 2))
                out[i] += (dm.COULOMB_CONSTANT * float(charges[j])
                           * math.erf(r / (sigma * math.sqrt(2))) / r * env)
    return out


class TestShortRangePotential:
    def test_matches_a_brute_force_image_sum_on_a_small_cell(self):
        atoms = _perovskite(reps=(1, 1, 1), rattle=0.05, seed=3)
        cell = torch.tensor(np.array(atoms.get_cell()))
        pos = torch.tensor(atoms.get_positions())
        charges = torch.tensor([-1.0, 1.0, 2.0, -1.0, -1.0])[
            torch.tensor([{17: 0, 55: 1, 82: 2}[z] for z in atoms.get_atomic_numbers()])]
        charges = torch.tensor([{17: -1.0, 55: 1.0, 82: 2.0}[z]
                                for z in atoms.get_atomic_numbers()])
        r_split, sigma = 5.0, 1.0
        # the neighbour list with images, built by hand
        src, dst, lengths = [], [], []
        n = pos.shape[0]
        for i in range(n):
            for j in range(n):
                for a in range(-2, 3):
                    for b in range(-2, 3):
                        for c in range(-2, 3):
                            shift = torch.tensor([a, b, c], dtype=pos.dtype) @ cell
                            r = float(torch.linalg.norm(pos[j] + shift - pos[i]))
                            if 1e-9 < r < r_split + 0.5:
                                src.append(j); dst.append(i); lengths.append(r)
        edge_index = torch.tensor([src, dst])
        got = dm.short_range_potential(charges, edge_index, torch.tensor(lengths), r_split,
                                       sigma, n)
        want = _brute_force_sr(charges, pos, cell, r_split, sigma)
        assert torch.allclose(got, want, atol=1e-10)
        assert float(got.abs().max()) > 1.0, "the short-range potential is not small"

    def test_the_switch_is_smooth_at_r_split(self):
        r_split = 5.0
        charges = torch.tensor([1.0, 1.0])
        for r in (4.999, 4.9999):
            edge_index = torch.tensor([[0], [1]])
            v = dm.short_range_potential(charges, edge_index, torch.tensor([r]), r_split, 1.0, 2)
            assert abs(float(v[1])) < 1e-6 * (r_split - r) ** 2 * 1e6


class TestTheModes:
    @pytest.fixture(scope="class")
    def model(self):
        frames = [_perovskite(seed=1)]
        return _gapped(frames)

    def test_the_three_modes_are_three_hamiltonians_and_off_is_zero(self, model):
        batch = _batch([_perovskite(seed=1)], [[0.0, 0.0, 1.0, 0.0]])
        shifts = {}
        energies = {}
        original = model.madelung.on_site_shift

        def grab(*a, **k):
            out = original(*a, **k)
            shifts[model.madelung_range] = out.detach().clone()
            return out

        model.madelung.on_site_shift = grab
        try:
            for mode in dm.MADELUNG_RANGES:
                model.madelung_range = mode
                with torch.no_grad():
                    out = model(batch.to_dict(), training=False, compute_force=False)
                energies[mode] = float(out["delta_sr_energy"][0])
        finally:
            model.madelung.on_site_shift = original
            model.madelung_range = "full"
        assert torch.equal(shifts["off"], torch.zeros_like(shifts["off"]))
        assert float(shifts["full"].abs().max()) > 0.1
        # V_LR = V_full - V_SR, and V_SR is not small at r_split = 4 A on this cell
        assert float((shifts["full"] - shifts["long_range"]).abs().max()) > 0.1
        assert len({round(v, 6) for v in energies.values()}) == 3

    def test_long_range_equals_full_minus_the_short_range_potential(self, model):
        """The identity V_full = V_SR + V_LR, read off the module with the head's own edges."""
        batch = _batch([_perovskite(seed=2)], [[0.0, 0.0, 0.0, 0.0]])
        d = batch.to_dict()
        captured = {}
        original = model.madelung.on_site_shift

        def grab(*a, **k):
            captured.update(k)
            captured["args"] = a
            return original(*a, **k)

        model.madelung.on_site_shift = grab
        model.madelung_range = "long_range"
        try:
            with torch.no_grad():
                model(d, training=False, compute_force=False)
        finally:
            model.madelung.on_site_shift = original
            model.madelung_range = "full"
        ewald, species, positions, cell, b = captured["args"]
        kw = dict(feats=captured["feats"], centre=captured["centre"],
                  num_graphs=captured["num_graphs"])
        with torch.no_grad():
            full = model.madelung.on_site_shift(ewald, species, positions, cell, b,
                                                eps_inf=model.madelung_eps_inf, **kw)
            lr = model.madelung.on_site_shift(
                ewald, species, positions, cell, b, eps_inf=model.madelung_eps_inf,
                madelung_range="long_range", edge_index=captured["edge_index"],
                edge_lengths=captured["edge_lengths"], r_split=captured["r_split"], **kw)
            charges = model.madelung.charges(species, captured["feats"], captured["centre"],
                                             b, captured["num_graphs"])
            sr = dm.short_range_potential(charges, captured["edge_index"],
                                          captured["edge_lengths"], captured["r_split"],
                                          ewald.sigma, charges.shape[0])
        assert torch.allclose(full - lr, -sr / model.madelung_eps_inf, atol=1e-10)
        assert captured["r_split"] == pytest.approx(model.functional["r_split"])

    def test_the_band_force_passes_the_harness_under_long_range(self, model):
        ctx = ForwardContext.production(model)
        batch = _batch([_perovskite(seed=1)], [[0.0, 0.0, 1.0, 0.0]], cutoff=ctx.cutoff)
        data = ctx.forward_dict(batch)
        model.madelung_range = "long_range"
        try:
            reports = {r.term: r for r in fd.force_check(
                model, data, components=[(0, 0), (7, 2), (21, 1)], tol=1e-5,
                terms=["band", "frontier"])}
        finally:
            model.madelung_range = "full"
        assert reports["band"].status == "pass", reports["band"].fit
        assert reports["frontier"].status == "pass", reports["frontier"].fit

    def test_the_flag_survives_the_round_trip_and_refuses_nonsense(self, model):
        from mace.modules.defect_models import MACEDefect
        from mace.tools.scripts_utils import extract_config_mace_model

        model.madelung_range = "long_range"
        try:
            config = extract_config_mace_model(model)
        finally:
            model.madelung_range = "full"
        assert config["madelung_range"] == "long_range"
        rebuilt = MACEDefect(**config)
        assert rebuilt.madelung_range == "long_range"
        with pytest.raises(ValueError, match="madelung_range"):
            MACEDefect(**{**config, "madelung_range": "half"})

    def test_a_short_head_cutoff_is_refused_under_long_range(self, model):
        r_cut = model.spectral_r_cut
        model.spectral_r_cut = 2.0
        model.madelung_range = "long_range"
        batch = _batch([_perovskite(seed=1)], [[0.0, 0.0, 0.0, 0.0]])
        try:
            with pytest.raises(ValueError, match="too short"):
                model(batch.to_dict(), training=False, compute_force=False)
        finally:
            model.spectral_r_cut = r_cut
            model.madelung_range = "full"
