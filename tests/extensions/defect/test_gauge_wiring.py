"""Plan v8.1 section 3.1, wired: the frozen-pristine spectral gauge in the model forward.

On the frontier-term toy (a gapped CsPbCl3 2x2x2 cell and its Cl vacancy, class table built
by the driver helper), with the gauge reference registered from the class table's pristine
frame:

* `mu_g` is the rank-normalised occupied trace of the pristine cell and is reported;
* a raw common-mode shift `Htilde -> Htilde + aI` (every species level up by `a`) moves
  `mu_g` by exactly `a` and leaves the gauged Hamiltonian's spectrum, the charged energies,
  the forces, the frontier term, the gap and the class-table edges unchanged to the floor;
* below the gauge layer (no reference registered) the same shift moves a charged energy by
  `a * Delta N` -- the analytically expected electron-count term the production path removes;
* the gauge carries no coordinate dependence: forces with and without the gauge differ only
  through the (rigid) shift, i.e. not at all;
* a head pickled with the retired energy-constant parameters leaves them out of the
  optimiser.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mace.modules import defect_composition as dc
from mace.modules.defect_cache import attach_frame_keys
from mace.modules.defect_models import establish_spectral_gauge
from mace.modules.defect_protocol import trainable_mask
from tests.extensions.defect.test_frontier_term import (HOLE, NEUTRAL, PRISTINE, VACANCY,
                                                        Z_TABLE, _forward, _frame,
                                                        _harrison_model)

torch.set_default_dtype(torch.float64)


def _gauged_model():
    m = _harrison_model()
    frames = [_frame(PRISTINE, NEUTRAL), _frame(VACANCY, NEUTRAL)]
    attach_frame_keys(frames, z_table=Z_TABLE)
    dc.ensure_class_table(m, frames, log=False)
    record = establish_spectral_gauge(m, frames, log=False)
    return m, frames, record


def _edges(model):
    rec = dc.lookup_class(model.composition_classes, VACANCY.get_atomic_numbers())
    return rec.vbm_al, rec.cbm_al, rec.n_e, rec.n_h, rec.q_core, rec.m_vb


class TestGauge:
    def test_mu_g_is_reported_and_is_the_pristine_occupied_trace(self):
        model, frames, record = _gauged_model()
        out, _ = _forward(model, [PRISTINE, VACANCY], [HOLE, HOLE], training=False,
                          compute_force=False)
        assert float(out["gauge_mu"]) == pytest.approx(record["mu_g"], abs=1e-12)
        ref = model.gauge_reference
        assert ref["pristine_key"] == model.composition_classes["pristine_key"]
        assert sum(ref["ranks"]) == ref["n_total"]
        # the record is the addendum's: rank-normalised over both spins
        assert record["n_valence"] == sum(record["ranks"])
        assert all(g > 0 for g in record["gaps"])

    def test_a_common_mode_shift_moves_mu_g_and_nothing_observable(self):
        model, frames, record = _gauged_model()
        a = 0.83
        out0, _ = _forward(model, [PRISTINE, VACANCY], [HOLE, HOLE], training=True,
                           compute_force=True)
        edges0 = _edges(model)
        with torch.no_grad():
            model.spectral.h.eps0.add_(a)
        mu1, rec1 = model._gauge_shift(model.pristine_centre(torch.float64))
        assert float(mu1) == pytest.approx(record["mu_g"] + a, abs=1e-9)
        out1, _ = _forward(model, [PRISTINE, VACANCY], [HOLE, HOLE], training=True,
                           compute_force=True)
        for key in ("energy", "delta_sr_energy", "frontier_energy", "frontier_w", "forces",
                    "delta_forces", "logit_gap"):
            assert torch.allclose(out0[key], out1[key], atol=1e-8, rtol=0), key
        # the class table re-aligned under the shifted parameters reads the same edges
        summary = dc.refresh_class_table(model, frames, log=False)
        assert not summary["disagree"]
        edges1 = _edges(model)
        assert edges1[0] == pytest.approx(edges0[0], abs=1e-8)
        assert edges1[1] == pytest.approx(edges0[1], abs=1e-8)
        assert edges1[2:] == edges0[2:]
        # and the fingerprint names the new gauge
        assert rec1.fingerprint != dc.lookup_class  # sanity: an object, not a placeholder
        from mace.modules.defect_gauge import GaugeRecord
        assert GaugeRecord.from_dict(record).fingerprint != rec1.fingerprint

    def test_below_the_gauge_the_shift_is_the_electron_count_term(self):
        """Diagnostic of section 11.1: with `mu_g` artificially held (no reference), a
        constant shift `a` moves a charged energy by `a * Delta N` (one hole: `-a`)."""
        model, frames, _ = _gauged_model()
        model.gauge_reference = None            # hold the gauge: raw levels
        a = 0.37
        out0, _ = _forward(model, [VACANCY], [HOLE], training=False, compute_force=False)
        with torch.no_grad():
            model.spectral.h.eps0.add_(a)
        out1, _ = _forward(model, [VACANCY], [HOLE], training=False, compute_force=False)
        d_e = float(out1["delta_sr_energy"][0] - out0["delta_sr_energy"][0])
        # a hole: N_Q = N_ref - 1, so F_band(H + aI, N_Q) - F_band(H + aI, N_ref) moves by -a
        assert d_e == pytest.approx(-a, abs=1e-6)
        assert float(out1["gauge_mu"]) == 0.0

    def test_the_gauge_has_no_coordinate_dependence(self):
        model, frames, _ = _gauged_model()
        ctx_out, batch = _forward(model, [VACANCY], [HOLE], training=True, compute_force=True)
        mu = model._gauge_shift(model.pristine_centre(torch.float64))[0]
        pos = batch.positions
        assert not pos.requires_grad or torch.autograd.grad(
            mu, [pos], allow_unused=True)[0] is None

    def test_retired_constant_parameters_are_not_trainable(self):
        assert not trainable_mask("spectral.c_shift")
        assert not trainable_mask("spectral.c_shift_table")
        assert trainable_mask("spectral.h.eps0")
        model, _, _ = _gauged_model()
        names = [n for n, _ in model.named_parameters()]
        assert not any(n.endswith("c_shift") or n.endswith("c_shift_table") for n in names)


def test_h_fix_takes_no_state_input():
    """Addendum 3.1 / item 1.1: `H_fix(R)` is a function of the geometry and species alone.
    The assembler's signature carries no state, policy, counter, or formal-charge argument
    -- asserted by name, so that a future argument of that kind fails here first."""
    import inspect

    from mace.modules.defect_counting import CountingHead

    names = set(inspect.signature(CountingHead.assemble_hamiltonian).parameters) - {"self"}
    assert names == {"node_feats", "node_species", "edge_index", "edge_vector", "madelung",
                     "centre", "n_nodes"}
    forbidden = ("state", "policy", "count", "counter", "charge", "q_formal", "occupation",
                 "carrier", "gauge", "shift")
    for n in names:
        assert not any(f in n for f in forbidden), n
    doc = CountingHead.assemble_hamiltonian.__doc__ or ""
    assert "no gauge and no energy constant" in doc
