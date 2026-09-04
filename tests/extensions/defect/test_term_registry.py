"""Plan v8 sections 2.5 and 2.6: the term registry describes the forward that exists, and
the two kernels are one entry point."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mace.modules import defect_terms as dt
from mace.modules.defect_models import MACEDefect
from mace.modules.latent_ewald import LatentEwald
from tests.extensions.defect.test_neutral_reference_skip import (_batch, _model,
                                                                  _perovskite)
from tests.unit.test_base_cache_precision import _model as _model_no_lr

torch.set_default_dtype(torch.float64)


class TestRegistry:
    def test_every_registered_term_is_reported_by_the_forward_and_sums_to_the_energy(self):
        model = _model()
        batch = _batch([_perovskite(seed=1), _perovskite(seed=2)],
                       [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        with torch.no_grad():
            out = model(batch.to_dict(), training=False, compute_force=False)
        energies = dt.term_energies(model, out)
        names = [t.name for t in model.terms()]
        assert names == ["base", "lr_host", "band", "lr_carrier"]
        total = sum(energies[n] for n in names)
        assert torch.allclose(total, energies["assembled"], atol=1e-10)

    def test_a_model_without_the_long_range_branch_registers_two_terms(self):
        model = _model_no_lr()
        assert [t.name for t in model.terms()] == ["base", "band"]

    def test_the_band_term_is_the_one_with_a_potential_and_lr_carrier_has_none(self):
        """The v6 fact the harness must report: E_LR of the carrier depends on P and puts
        nothing into H. Stage 5 changes this entry; nothing else may."""
        model = _model()
        by_name = {t.name: t for t in model.terms()}
        assert by_name["band"].depends_on_P and by_name["band"].potential == "band"
        assert by_name["lr_carrier"].depends_on_P and by_name["lr_carrier"].potential == "absent"
        assert not by_name["base"].depends_on_P and by_name["base"].potential == "none"

    def test_gauge_dependence_is_a_column(self):
        model = _model()
        by_name = {t.name: t for t in model.terms()}
        assert not by_name["base"].gauge_dependent
        assert by_name["lr_host"].gauge_dependent and by_name["lr_carrier"].gauge_dependent
        assert by_name["band"].gauge_dependent, "the Madelung shift inside H is periodic"

    def test_the_registry_refuses_a_forward_that_does_not_report_a_term(self):
        model = _model_no_lr()
        with pytest.raises(KeyError, match="registry and the forward disagree"):
            dt.term_energies(model, {"energy": torch.zeros(1)})


class TestTwoKernels:
    @staticmethod
    def charges_and_geometry():
        rng = np.random.default_rng(0)
        pos = torch.tensor(rng.uniform(0, 8, size=(6, 3)))
        q = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.25, -0.25])
        cell = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3) * 9.0
        batch = torch.zeros(6, dtype=torch.long)
        return q, pos, cell, batch

    def test_the_entry_point_selects_the_two_evaluators_by_name(self):
        ewald = LatentEwald({"sigma": 1.0})
        q, pos, cell, batch = self.charges_and_geometry()
        per = dt.evaluate_kernel(ewald, "periodic", q, pos, cell, batch, 1)
        iso = dt.evaluate_kernel(ewald, "isolated", q, pos, cell, batch, 1)
        assert torch.equal(per, ewald.energy(q, pos, cell, batch))
        assert torch.equal(iso, ewald.isolated_energy(q, pos, batch, 1))
        assert torch.equal(ewald.evaluate(dt.Kernel.PBC, q, pos, cell, batch, 1), per)
        assert float((per - iso).abs()) > 0.0, "the image term is not zero on this cell"

    def test_an_unknown_kernel_is_refused(self):
        ewald = LatentEwald({"sigma": 1.0})
        q, pos, cell, batch = self.charges_and_geometry()
        with pytest.raises(ValueError):
            dt.evaluate_kernel(ewald, "tin_foil", q, pos, cell, batch, 1)

    def test_the_gauge_is_config_and_the_isolated_gauge_is_the_dilute_call(self):
        """`gauge="isolated"` on the model is the same forward as `dilute=True` on the
        periodic one: one functional, the kernel chosen by a flag."""
        from mace.tools.scripts_utils import extract_config_mace_model

        periodic = _model()
        assert periodic.gauge == "periodic"
        config = extract_config_mace_model(periodic)
        assert config["gauge"] == "periodic"
        config["gauge"] = "isolated"
        isolated = MACEDefect(**config)
        isolated.load_state_dict(periodic.state_dict())
        batch = _batch([_perovskite(seed=4)], [[0.0, 0.0, 1.0, 0.0]])
        with torch.no_grad():
            a = periodic(batch.to_dict(), training=False, compute_force=False, dilute=True)
            b = isolated(batch.to_dict(), training=False, compute_force=False)
            c = periodic(batch.to_dict(), training=False, compute_force=False)
        assert torch.equal(a["energy"], b["energy"])
        assert float((a["energy"] - c["energy"]).abs()) > 0.0
        with pytest.raises(ValueError, match="gauge"):
            _model(gauge="tin_foil")
