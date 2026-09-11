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
                       "set_pristine_reference", "pristine_indices", "pristine_batches",
                       "pristine", "static_pristine_cell"}
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


class TestPreA1Checkpoints:
    def test_a_pre_a1_pickle_is_refused_rather_than_scored_in_the_wrong_form(self, tmp_path):
        """The failure this prevents is SILENT: unpickling restores `_buffers` whatever
        `__init__` would have made, so a centred checkpoint would load and then run the
        uncentred `on_site` on centred weights, with no error anywhere."""
        import io
        import pickle

        m = H0(ZS, feature_dim=8, n_vectors=6, hidden=16)
        state = m.__getstate__() if hasattr(m, "__getstate__") else dict(m.__dict__)
        buf = io.BytesIO()
        torch.save(m, buf)
        buf.seek(0)
        ok = torch.load(buf, weights_only=False)                 # a post-A1 pickle loads
        assert not hasattr(ok, "centre")
        # The same model as it would have been pickled before A1.
        m.register_buffer("centre", torch.zeros(len(ZS), 8))
        buf = io.BytesIO()
        torch.save(m, buf)
        buf.seek(0)
        with pytest.raises(RuntimeError, match="pre-a1"):
            torch.load(buf, weights_only=False)


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

    def test_the_coefficient_and_its_environment_shift_stay_bounded(self):
        """`b_i = b_max tanh(beta_Z + beta_b tanh g)`: the outer tanh bounds the coefficient
        by `b_max`, the inner one bounds the environment's shift of the pre-activation by
        `beta_b`."""
        atoms = _perovskite(rattle=0.05)
        m = _model(beta_b=0.5, beta_a=0.5)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        a_i, b_i = m.site_coefficients(scalars.to(torch.float64), species)
        assert float(a_i.abs().max()) < m.a_max and float(b_i.abs().max()) < m.b_max
        shift = m.env_shift(m.g_read, m.beta_b, scalars.to(torch.float64), species, "rank2")
        assert float(shift.abs().max()) <= m.beta_b + 1e-12

    def test_the_readout_gradient_does_not_depend_on_the_species_coefficient(self):
        """The defect this form exists to remove. Under `b_Z (1 + beta tanh g)` the readout
        gradient is `beta * b_Z * sech^2`, and `b_Z` is initialised at zero -- the readout
        starts with exactly no signal and is still ~7x starved after 60 epochs. Inside the
        tanh the gradient is `b_max * beta * sech^2(.) * sech^2(g)`, independent of `b_Z`."""
        atoms = _perovskite(rattle=0.05)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, _model().r_cut)
        grads = []
        for b_z_value in (0.0, 0.4):          # beta_Z at its initialisation, and grown
            m = _model(beta_b=0.5)
            with torch.no_grad():
                m.beta.fill_(b_z_value)
            last = [x for x in m.g_read.modules() if isinstance(x, torch.nn.Linear)][-1]
            _, b_i = m.site_coefficients(scalars.to(torch.float64), species)
            g = torch.autograd.grad(b_i.sum(), last.bias, allow_unused=True)[0]
            grads.append(0.0 if g is None else float(g.abs().max()))
        assert grads[0] > 0.0                                   # live at beta_Z = 0
        assert abs(grads[0] - grads[1]) / max(grads) < 0.5      # not proportional to b_Z

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


class TestA11Standardisation:
    """v5 amendment A1.1: per-species input standardisation for the reference-free readouts."""

    def test_standardise_whitens_per_species_and_is_frozen(self):
        torch.manual_seed(0)
        m = H0(ZS, feature_dim=8, n_vectors=6, hidden=16)
        species = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2])
        # A species-mean far larger than the deviation around it: base v2's actual regime
        # (||mu|| / median ||x - mu|| = 29-46, per-channel |mu|/sd = 19-28).
        x = torch.randn(8, 8) * 0.3
        x = x + torch.tensor([10.0, -9.0, 11.0])[species].unsqueeze(-1)
        mean = torch.stack([x[species == z].mean(0) for z in range(3)])
        sd = torch.stack([x[species == z].std(0, unbiased=False) for z in range(3)])
        m.sk.set_feature_stats(mean, sd)
        h = m.sk.standardise(x, species)
        for z in range(3):
            assert float(h[species == z].mean().abs()) < 1e-9
            assert float(h[species == z].std(unbiased=False).mean()) == pytest.approx(1.0, abs=1e-6)
        # A buffer, not a parameter: frozen after computation, and it travels with the model.
        assert "sk.feat_mean" in dict(m.named_buffers()) and "sk.feat_sd" in dict(m.named_buffers())
        # By identity, not by shape: `elem`'s embedding weight happens to share the shape.
        assert all(p is not m.sk.feat_mean and p is not m.sk.feat_sd for p in m.parameters())
        assert not m.sk.feat_mean.requires_grad and not m.sk.feat_sd.requires_grad

    def test_dead_channels_are_zeroed_not_amplified(self):
        """A channel the base does not vary carries no information. Dividing by its (near
        zero) sd amplifies numerical noise without bound -- the first implementation floored
        the sd instead of zeroing the channel, and a symmetry-perfect cell then drove `H0` to
        nonsense and the SCF to non-convergence. The test that caught it is
        `test_sparse.py::test_sparse_scf_and_frontier_forces_match_the_dense_model`."""
        m = H0(ZS, feature_dim=8, n_vectors=6, hidden=16)
        sd = torch.ones(3, 8)
        sd[:, :3] = 1e-12                    # channels the base never varies
        dead = m.sk.set_feature_stats(torch.zeros(3, 8), sd)
        assert all(v == 3 for v in dead.values())
        x = torch.full((3, 8), 1e-11)
        h = m.sk.standardise(x, torch.tensor([0, 1, 2]))
        assert float(h[:, :3].abs().max()) == 0.0          # zeroed, not 1e-11/1e-12 = 10
        assert torch.isfinite(h).all()

    def test_a_species_with_no_variation_at_all_is_finite_and_stable(self):
        """Every channel dead -- an unrattled cell, all atoms of a species equivalent. The
        readouts then contribute their bias and nothing else: degenerate, but finite."""
        m = H0(ZS, feature_dim=8, n_vectors=6, hidden=16)
        dead = m.sk.set_feature_stats(torch.randn(3, 8), torch.zeros(3, 8))
        assert all(v == 8 for v in dead.values())
        h = m.sk.standardise(torch.randn(5, 8) * 1e3, torch.tensor([0, 1, 2, 2, 0]))
        assert float(h.abs().max()) == 0.0
        atoms = _perovskite(rattle=0.05)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        m.sk.set_feature_stats(torch.zeros(3, 8), torch.zeros(3, 8))
        H = m(scalars, vectors, species, ei, gr.edge_vectors(positions, cell, ei, S))
        assert torch.isfinite(H).all()

    def test_identity_until_set_so_a_fresh_module_still_builds(self):
        m = H0(ZS, feature_dim=8, n_vectors=6, hidden=16)
        x = torch.randn(4, 8)
        sp = torch.tensor([0, 1, 2, 2])
        assert float((m.sk.standardise(x, sp) - x).abs().max()) == 0.0

    def test_the_conditioning_argument_is_what_the_fix_addresses(self):
        """With a large species mean, the same readout weights give a pre-activation dominated
        by the constant part on raw features and by the environment on standardised ones. This
        is the mechanism A1.1 records, stated as a test rather than as prose."""
        torch.manual_seed(0)
        n_ch = 64
        w = torch.randn(n_ch) / n_ch ** 0.5
        mu = torch.full((n_ch,), 3.0)
        dev = torch.randn(200, n_ch) * 0.1
        raw = dev + mu
        const_part = float((w @ mu).abs())
        env_part = float((dev @ w).abs().mean())
        assert const_part > 10 * env_part                      # raw: the constant dominates
        std = (raw - raw.mean(0)) / raw.std(0, unbiased=False)
        assert float((std.mean(0) @ w).abs()) < 1e-5           # standardised: no constant left


class TestSaturationBarrier:
    """The guard rail added after seed 0 of the first A1.1 arm diverged into saturation."""

    def _pre(self, m, value):
        """Force every readout pre-activation to `value` by zeroing the weights and setting
        the final bias, so the barrier and L2 can be read at a known operating point."""
        with torch.no_grad():
            for mod in (m.sk.site, m.sk.hop, m.g_read, m.f_read):
                last = [x for x in mod.modules() if isinstance(x, torch.nn.Linear)][-1]
                last.weight.zero_()
                last.bias.fill_(value)

    def test_an_l2_on_the_tanh_output_cannot_rescue_a_saturated_unit(self):
        """The reason the barrier is on the pre-activation. `d/dpre tanh(pre)^2` is
        `2 tanh sech^2`, which vanishes as `|pre|` grows -- the restoring force disappears
        exactly where it is needed. `d/dpre relu(|pre| - knee)^2` grows instead."""
        for pre_value, expect_l2_grad in ((0.5, 0.5), (6.0, 1e-4)):
            pre = torch.tensor([pre_value], dtype=torch.float64, requires_grad=True)
            g_l2 = torch.autograd.grad((torch.tanh(pre) ** 2).sum(), pre)[0].abs().item()
            pre2 = torch.tensor([pre_value], dtype=torch.float64, requires_grad=True)
            g_bar = torch.autograd.grad(((pre2.abs() - 2.0).clamp_min(0.0) ** 2).sum(), pre2)[0].abs().item()
            if pre_value < 2.0:
                assert g_bar == 0.0                       # silent in normal operation
            else:
                assert g_l2 < 1e-4 and g_bar > 5.0        # L2 dead, barrier strong

    def test_the_barrier_is_zero_in_normal_operation_and_bites_at_saturation(self):
        atoms = _perovskite(rattle=0.05)
        m = _model(beta_b=0.5)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        ev = gr.edge_vectors(positions, cell, ei, S)
        m.sk.collect_regularisation(True)
        self._pre(m, 0.05)                                 # a healthy run sits here
        m(scalars, vectors, species, ei, ev)
        assert max(float(v) for v in m.sk.barrier().values()) == 0.0
        m.sk.reset_regularisation()
        self._pre(m, 4.0)                                  # where seed 0 ended up
        m(scalars, vectors, species, ei, ev)
        bar = m.sk.barrier()
        assert min(float(v) for v in bar.values()) == pytest.approx(4.0, rel=1e-6)
        # 1e-3 * 4 per term against a force loss of ~1e-5: a hard stop, as intended.
        assert 1e-3 * sum(float(v) for v in bar.values()) > 100 * 1e-5

    def test_the_l2_still_reads_the_tanh_and_both_come_from_one_store(self):
        atoms = _perovskite(rattle=0.05)
        m = _model()
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        m.sk.collect_regularisation(True)
        self._pre(m, 1.0)
        m(scalars, vectors, species, ei, gr.edge_vectors(positions, cell, ei, S))
        reg, bar = m.sk.regularisation(), m.sk.barrier()
        assert set(reg) == set(bar)
        expect = float(torch.tanh(torch.tensor(1.0)) ** 2)
        assert float(reg["site"]) == pytest.approx(expect, rel=1e-6)
        assert float(bar["site"]) == 0.0


class TestRank1Deadlock:
    """`vector_mix` and `alpha` were both zero-initialised, and each one's gradient is
    proportional to the other, so the rank-1 s-p block could never leave zero. Measured in
    the converged W3 heads: |vector_mix| and |alpha| exactly 0.0 after 60 epochs."""

    def _grads(self, zero_vector_mix):
        torch.manual_seed(0)
        m = H0(ZS, feature_dim=8, n_vectors=6, r_cut=6.0, hidden=16)
        if zero_vector_mix:
            with torch.no_grad():
                m.vector_mix.zero_()                      # the pre-fix initialisation
        atoms = _perovskite(rattle=0.05)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        H = m(scalars, vectors, species, ei, gr.edge_vectors(positions, cell, ei, S))
        # A GENERIC cotangent. `(H**2).sum()` would be the wrong probe: its gradient is `2H`,
        # which is identically zero at the s-p entries exactly because the block starts
        # absent, so it reports no gradient whether or not the deadlock is fixed. A real
        # loss reaches `H` through the eigendecomposition and has no such null.
        torch.manual_seed(7)
        loss = (H * torch.randn_like(H)).sum()
        g_alpha, g_mix = torch.autograd.grad(loss, [m.alpha, m.vector_mix], allow_unused=True)
        return (0.0 if g_alpha is None else float(g_alpha.abs().max()),
                0.0 if g_mix is None else float(g_mix.abs().max()))

    def test_zero_vector_mix_is_a_saddle_the_optimiser_cannot_leave(self):
        g_alpha, g_mix = self._grads(zero_vector_mix=True)
        assert g_alpha == 0.0 and g_mix == 0.0

    def test_the_fixed_initialisation_gives_alpha_a_gradient(self):
        g_alpha, _ = self._grads(zero_vector_mix=False)
        assert g_alpha > 0.0

    def test_the_block_still_starts_exactly_absent(self):
        """`alpha` stays at zero, so `a_Z = 0` and the s-p block is identically zero at
        initialisation -- the neutral null and the Phase-1 gates are untouched."""
        torch.manual_seed(0)
        m = H0(ZS, feature_dim=8, n_vectors=6, r_cut=6.0, hidden=16)
        assert float(m.alpha.abs().max()) == 0.0
        assert float(m.vector_mix.abs().max()) > 0.0
        atoms = _perovskite(rattle=0.05)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        blocks = m.directional_site_blocks(scalars, vectors.to(torch.float64), species, ei,
                                           gr.edge_vectors(positions, cell, ei, S))
        assert float(blocks[:, 0, 1:].abs().max()) == 0.0      # s-p block exactly absent
        assert float(blocks[:, 1:, 1:].abs().max()) == 0.0     # b_Z = 0 at init too
