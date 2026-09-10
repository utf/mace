"""v5 amendment A1: reference-free environment dependence.

Four things the amendment asks to be testable, plus the static check it names explicitly.
The neutral null and the gauge-shift invariance it also lists are unchanged and live in
`test_gates.py`; they are re-run, not rewritten.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from mace.modules.dscc import graph as gr
from mace.modules.dscc.hamiltonian import SATURATED, H0, delta_from_baselines
from mace.modules.dscc.legacy import ORBITALS_PER_ATOM, sk_block

from .test_hamiltonian import ZS, _inputs, _perovskite

RUNTIME = Path(__file__).resolve().parents[3] / "mace" / "modules" / "dscc"


def _model(beta_b=0.0, beta_a=0.0, seed=1):
    torch.manual_seed(seed)
    m = H0(ZS, feature_dim=8, n_vectors=6, r_cut=6.0, hidden=16, readout_hidden=16,
           beta_b=beta_b, beta_a=beta_a)
    with torch.no_grad():
        m.vector_mix.normal_(0.0, 0.5)
        m.alpha.fill_(0.7)
        m.beta.fill_(-0.4)
        # The readouts are initialised near zero on purpose (the species-default start);
        # push them off zero so the tests exercise a live correction.
        for r in (m.g_read, m.f_read):
            for lin in [x for x in r.modules() if isinstance(x, torch.nn.Linear)]:
                lin.weight.normal_(0.0, 0.5)
                lin.bias.normal_(0.0, 0.5)
    return m


class TestStaticCheck:
    def test_no_runtime_module_reads_pristine_feature_statistics(self):
        """A1's static check, verbatim: the symbol is absent from the H0 assembly path.

        On IDENTIFIERS, not on text. Grepping the source would fail on the amendment's own
        record of what was removed, and passing that grep would mean deleting the history
        rather than the code. The parse walks every name, attribute, argument and keyword of
        every runtime module and asserts that nothing named `centre` survives, and that the
        only `pristine` symbols left are the two A1 keeps.

        The exemptions are named rather than implied. `q0_pristine` and `pristine_atoms` are
        a CHARGE reference and an atom count: Route B''s `R_eff` diagnostic and the size
        classes. Neither is a feature statistic and neither is read by `H0`. `pristine_gap`
        is the host gap regulariser, which A1 keeps as one of exactly two host-specific
        items. `vacancy_centre` is a geometric midpoint in real space and has nothing to do
        with feature means."""
        import ast
        import re

        centre_ok = {"vacancy_centre", "vacancy_centre_full"}
        pristine_ok = {"q0_pristine", "pristine_atoms", "pristine_gap",
                       "set_pristine_reference", "pristine_indices", "pristine",
                       "static_pristine_cell"}
        offenders = []
        for path in sorted(RUNTIME.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Name):
                    names = [node.id]
                elif isinstance(node, ast.Attribute):
                    names = [node.attr]
                elif isinstance(node, ast.arg):
                    names = [node.arg]
                elif isinstance(node, ast.keyword) and node.arg:
                    names = [node.arg]
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    names = [node.name]
                for name in names:
                    if re.search(r"(^|_)centre(_|$)", name) and name not in centre_ok:
                        offenders.append(f"{path.name}: {name}")
                    if "pristine" in name and name not in pristine_ok:
                        offenders.append(f"{path.name}: {name}")
        assert not offenders, ("pristine feature statistics back in the runtime path: "
                               + ", ".join(sorted(set(offenders))))

    def test_h0_has_no_centre_state_and_needs_no_setup_call(self):
        m = H0(ZS, feature_dim=8, n_vectors=6, hidden=16)
        assert not hasattr(m, "centre") and not hasattr(m, "centre_set")
        assert not hasattr(m, "set_centre")
        assert "centre" not in dict(m.state_dict())
        # It builds a Hamiltonian straight out of the constructor: no reference to set.
        atoms = _perovskite(rattle=0.05)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        H = m(scalars, vectors, species, ei, gr.edge_vectors(positions, cell, ei, S))
        assert torch.isfinite(H).all()


class TestZeroReadoutLimit:
    def test_zeroing_every_readout_gives_the_species_default_hamiltonian(self):
        """A1: "with all readouts forced to zero the model equals the species-default `H0` to
        1e-12". The reference is BUILT HERE from `eps0` and `v0 * radial` -- comparing one H0
        against another H0 would only prove the code agrees with itself."""
        m = _model(beta_b=0.5, beta_a=0.5)
        m.zero_readouts()
        atoms = _perovskite(rattle=0.05)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        ev = gr.edge_vectors(positions, cell, ei, S)
        H = m(scalars, vectors, species, ei, ev).to(torch.float64)

        n = int(species.shape[0])
        sk = m.sk
        src, dst = ei[0], ei[1]
        r = ev.to(torch.float64).norm(dim=-1).clamp_min(1e-9)
        # tanh(0) = 0, so the modulation is exactly 1 and the integrals are v0 * radial.
        radial = sk.radial(r, species[src], species[dst])
        if radial.dim() == 1:
            radial = radial.unsqueeze(-1)
        v = sk.v0(species[src], species[dst]) * radial
        ref = torch.zeros(4 * n, 4 * n, dtype=torch.float64)
        blocks = sk_block(ev.to(torch.float64) / r.unsqueeze(-1), v)
        for e in range(int(src.shape[0])):
            i, j = int(src[e]) * ORBITALS_PER_ATOM, int(dst[e]) * ORBITALS_PER_ATOM
            ref[i:i + 4, j:j + 4] += blocks[e]
        ref = 0.5 * (ref + ref.T)
        levels = sk.eps0.detach().to(torch.float64)[species]
        diag = torch.cat([levels[:, :1], levels[:, 1:].expand(-1, 3)], dim=-1).reshape(-1)
        ref = ref - torch.diag(torch.diagonal(ref)) + torch.diag(diag)
        # a_Z, b_Z survive (they are parameters, not readouts); the environment FACTORS are 1.
        a_i, b_i = m.site_coefficients(scalars.to(torch.float64), species)
        assert float((a_i - m.coefficients()[0][species]).abs().max()) < 1e-12
        assert float((b_i - m.coefficients()[1][species]).abs().max()) < 1e-12
        dirblock = m.directional_block(scalars, vectors.to(torch.float64), species, ei, ev)
        assert float((H - (ref + dirblock)).abs().max()) < 1e-12


class TestEnvironmentFactors:
    def test_beta_zero_is_the_species_coefficient_and_removes_the_readout_from_the_graph(self):
        """W4's `beta = 0` variant must be the species-level model exactly, not the species
        model times a learned one -- otherwise "off" leaves a gradient path open."""
        atoms = _perovskite(rattle=0.05)
        off, on = _model(beta_b=0.0), _model(beta_b=0.5)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, off.r_cut)
        _, b_off = off.site_coefficients(scalars.to(torch.float64), species)
        assert float((b_off - off.coefficients()[1][species]).abs().max()) == 0.0
        s_req = scalars.clone().to(torch.float64).requires_grad_(True)
        _, b_on = on.site_coefficients(s_req, species)
        assert torch.autograd.grad(b_on.sum(), s_req, allow_unused=True)[0] is not None
        s_req2 = scalars.clone().to(torch.float64).requires_grad_(True)
        _, b_off2 = off.site_coefficients(s_req2, species)
        assert torch.autograd.grad(b_off2.sum(), s_req2, allow_unused=True)[0] is None

    def test_the_factor_stays_inside_its_bound(self):
        atoms = _perovskite(rattle=0.05)
        m = _model(beta_b=0.5, beta_a=0.5)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        a_i, b_i = m.site_coefficients(scalars.to(torch.float64), species)
        a_z, b_z = m.coefficients()
        for got, ref in ((a_i, a_z[species]), (b_i, b_z[species])):
            ratio = got / ref.to(got.dtype)
            assert float(ratio.min()) > 0.5 - 1e-9 and float(ratio.max()) < 1.5 + 1e-9

    def test_batched_matches_per_graph_with_the_environment_factors_on(self):
        """The two assembly paths must stay matrix-identical: A1 adds a term to both."""
        m = _model(beta_b=0.5, beta_a=0.5)
        atoms = [_perovskite(rattle=0.05, seed=s) for s in (1, 2, 3)]
        singles, feats = [], []
        for a in atoms:
            f = _inputs(a, m.r_cut, seed=int(a.get_positions()[0, 0] * 1000) % 97)
            species, scalars, vectors, positions, cell, ei, S = f
            singles.append(m(scalars, vectors, species, ei, gr.edge_vectors(positions, cell, ei, S)))
            feats.append(f)
        n = len(atoms[0])
        species = torch.cat([f[0] for f in feats])
        scalars = torch.cat([f[1] for f in feats])
        vectors = torch.cat([f[2] for f in feats])
        ei = torch.cat([f[5] + g * n for g, f in enumerate(feats)], dim=1)
        ev = torch.cat([gr.edge_vectors(f[3], f[4], f[5], f[6]) for f in feats])
        batch = torch.arange(len(atoms)).repeat_interleave(n)
        H = m.batched(scalars, vectors, species, ei, ev, batch, len(atoms), n)
        for g, single in enumerate(singles):
            assert float((H[g] - single).abs().max()) < 1e-10


class TestSaturationAndL2:
    def test_saturation_report_has_every_term_and_species(self):
        m = _model(beta_b=0.5, beta_a=0.5)
        atoms = _perovskite(rattle=0.05)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        sites = torch.zeros(int(species.shape[0]), dtype=torch.bool)
        sites[:2] = True
        rep = m.saturation(scalars, species, ei, sites)
        for term in ("on_site", "rank2", "rank1", "hop"):
            assert 0.0 <= rep[term] <= 1.0
            assert set(rep[term + "_by_species"]) <= set(ZS)
        assert len(rep["site_on_site"]) == 4 and len(rep["site_rank2"]) == 2
        # The reported fraction is the fraction above the registered threshold.
        t = torch.tanh(m.sk.site(torch.cat([scalars.to(torch.float64), m.sk.elem(species)], dim=-1)))
        assert rep["on_site"] == pytest.approx(float((t.abs() > SATURATED).to(torch.float64).mean()))

    def test_the_l2_pools_over_every_h0_of_a_step_and_is_differentiable(self):
        m = _model(beta_b=0.5)
        atoms = _perovskite(rattle=0.05)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        ev = gr.edge_vectors(positions, cell, ei, S)
        m.sk.collect_regularisation(True)
        m(scalars, vectors, species, ei, ev)
        one = {k: float(v) for k, v in m.sk.regularisation().items()}
        assert set(one) == {"site", "hop", "rank2"}
        assert all(0.0 <= v <= 1.0 for v in one.values())
        m(scalars, vectors, species, ei, ev)                       # a second H0 in the step
        two = m.sk.regularisation()
        assert float(two["site"]) == pytest.approx(one["site"], rel=1e-12)
        assert torch.autograd.grad(sum(two.values()), m.sk.site[-1].bias, retain_graph=True)[0].abs().sum() > 0
        m.sk.reset_regularisation()
        assert m.sk.regularisation() == {}


class TestDelta:
    def test_delta_is_the_registered_fraction_of_the_baseline_spread(self):
        m = H0(ZS, feature_dim=8, n_vectors=6, hidden=16)
        eps0 = m.sk.eps0.detach()
        expect = 0.40 * float(eps0.reshape(-1).std(unbiased=False))
        assert float(m.sk.delta.min()) == pytest.approx(expect, rel=1e-6)
        assert float(m.sk.delta.max()) == pytest.approx(expect, rel=1e-6)
        assert expect == pytest.approx(3.079, abs=2e-3)     # the registered CsPbCl3 value
        # A buffer, not a parameter: a registered bound is not something the fit may widen.
        assert "sk.delta" in dict(m.named_buffers())
        assert all(not p.requires_grad or p is not m.sk.delta for p in m.parameters())

    def test_the_on_site_correction_is_bounded_by_delta(self):
        m = _model()
        atoms = _perovskite(rattle=0.05)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        levels = m.sk.on_site(scalars.to(torch.float64), species, None)
        dev = (levels - m.sk.eps0.detach().to(torch.float64)[species]).abs()
        assert float(dev.max()) < float(m.sk.delta.max()) + 1e-9

    def test_delta_from_baselines_is_host_free(self):
        """The rule reads the element table through `eps0`; scaling the baselines scales
        `Delta` with them, and nothing about this crystal enters."""
        eps0 = torch.tensor([[-24.63, -11.74], [-3.36, -1.80], [-15.19, -8.04]])
        d1 = delta_from_baselines(eps0, 0.40)
        d2 = delta_from_baselines(2.0 * eps0, 0.40)
        assert float(d2.max()) == pytest.approx(2.0 * float(d1.max()), rel=1e-6)
        assert float(np.std([-24.63, -11.74, -3.36, -1.80, -15.19, -8.04])) == pytest.approx(
            float(d1.max()) / 0.40, rel=1e-5)
