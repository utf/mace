"""Sections 2.1 and 2.2 of the speed cycle's spec: the corrected centred forms.

    2.1   corr_i = gamma * tanh( h(x_i) - h(xbar_{s(i)}) )
    2.2   Z_i    = Z0[s(i)] + zeta * tanh( z(x_i) - z(xbar_{s(i)}) )

WHAT CHANGED AND WHY IT IS NOT COSMETIC. The Stage A' form centred the OUTPUT,
`gamma [tanh h(x_i) - tanh h(xbar_s)]`. That is also zero on the pristine cell and also
invariant to a constant shift of `h` -- and the invariance is the problem: nothing in the
loss opposes `h` drifting, and once every atom of a species is past |tanh| = 0.98 the
difference of two saturated tanhs is identically zero and so is its gradient. Two of six
Stage B seeds ran chlorine into that state and three ran caesium into it, which is what
F10's 0-of-6 measured. Moving the subtraction inside the tanh keeps both properties and
makes saturation a property of the DEVIATION, which the data bounds.

The tests below are written so that they fail on the old form: the pristine identity holds
for both, so it cannot be the discriminator; what separates them is that the gradient
survives a large common shift of `h`.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from e3nn import o3

from mace import data, modules, tools
from mace.modules.defect_counting import CountingHead
from mace.modules.defect_madelung import MadelungOnSite
from mace.modules.defect_models import MACEDefect
from mace.modules.latent_ewald import LatentEwald

Z_TABLE = tools.AtomicNumberTable([17, 55, 82])


@pytest.fixture(scope="module", autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)


def _head(form="argument", gamma=3.0):
    torch.manual_seed(0)
    return CountingHead(num_elements=3, feature_dim=8, atomic_numbers=[17, 55, 82],
                        on_site_range=gamma, centre_form=form)


def _inputs(n=12, dim=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(n, dim, generator=g)
    species = torch.arange(n) % 3
    centre = torch.randn(3, dim, generator=g)
    return feats, species, centre


class TestOnSiteForm:
    def test_the_correction_is_zero_at_the_species_centre(self):
        """The pristine identity, and it holds for BOTH forms -- which is why it cannot be
        the test that tells them apart."""
        for form in ("argument", "output"):
            h = _head(form)
            _, _, centre = _inputs()
            species = torch.arange(3)
            feats = centre[species]
            levels = h.h.on_site(feats, species, centre=centre)
            base = h.h.eps0[species]
            assert float((levels - base).abs().max()) < 1e-12, form

    def test_the_argument_form_is_the_tanh_of_the_difference(self):
        h = _head("argument")
        feats, species, centre = _inputs()
        levels = h.h.on_site(feats, species, centre=centre)
        e = h.h.elem(species)
        pre = h.h.site(torch.cat([feats, e], dim=-1))
        pre_c = h.h.site(torch.cat([centre[species], e], dim=-1))
        want = h.h.eps0[species] + h.h.on_site_range * torch.tanh(pre - pre_c)
        assert torch.allclose(levels, want, atol=1e-12)

    def test_the_output_form_is_still_reachable_for_the_stage_b_cohort(self):
        h = _head("output")
        feats, species, centre = _inputs()
        levels = h.h.on_site(feats, species, centre=centre)
        e = h.h.elem(species)
        pre = h.h.site(torch.cat([feats, e], dim=-1))
        pre_c = h.h.site(torch.cat([centre[species], e], dim=-1))
        want = (h.h.eps0[species]
                + h.h.on_site_range * (torch.tanh(pre) - torch.tanh(pre_c)))
        assert torch.allclose(levels, want, atol=1e-12)

    def test_a_model_pickled_before_the_form_existed_runs_the_output_form(self):
        h = _head("output")
        del h.h.centre_form
        feats, species, centre = _inputs()
        levels = h.h.on_site(feats, species, centre=centre)
        e = h.h.elem(species)
        pre = h.h.site(torch.cat([feats, e], dim=-1))
        pre_c = h.h.site(torch.cat([centre[species], e], dim=-1))
        want = (h.h.eps0[species]
                + h.h.on_site_range * (torch.tanh(pre) - torch.tanh(pre_c)))
        assert torch.allclose(levels, want, atol=1e-12)

    @pytest.mark.parametrize("form,alive", [("argument", True), ("output", False)])
    def test_a_saturated_h_kills_the_output_form_and_not_the_argument_form(self, form,
                                                                          alive):
        """THE MEASUREMENT F10 FAILED ON. Push `h` far past saturation with a constant --
        exactly the drift the centring leaves unopposed -- and ask whether the correction
        still responds to the features. Under the output form both tanhs are pinned at 1 and
        the gradient is dead; under the argument form the deviation is untouched, because a
        constant cancels inside the tanh."""
        h = _head(form)
        feats, species, centre = _inputs()
        with torch.no_grad():
            last = [m for m in h.h.site.modules() if isinstance(m, torch.nn.Linear)][-1]
            last.bias.add_(40.0)          # every pre-activation past |tanh| = 1 - 1e-30
        feats = feats.clone().requires_grad_(True)
        levels = h.h.on_site(feats, species, centre=centre)
        grad = torch.autograd.grad(levels.sum(), feats, allow_unused=True)[0]
        size = 0.0 if grad is None else float(grad.abs().max())
        if alive:
            assert size > 1e-6, f"{form}: channel dead, |dcorr/dx| = {size:.3e}"
        else:
            assert size < 1e-12, f"{form}: expected a dead channel, got {size:.3e}"

    def test_the_argument_form_is_invariant_to_a_constant_shift_of_h(self):
        """The property the centring exists for, kept by the corrected form."""
        h = _head("argument")
        feats, species, centre = _inputs()
        before = h.h.on_site(feats, species, centre=centre)
        with torch.no_grad():
            last = [m for m in h.h.site.modules() if isinstance(m, torch.nn.Linear)][-1]
            last.bias.add_(0.7)
        after = h.h.on_site(feats, species, centre=centre)
        assert torch.allclose(before, after, atol=1e-12)


class TestPerSiteCharges:
    @staticmethod
    def _madelung(zeta=1.0, dim=8):
        torch.manual_seed(0)
        return MadelungOnSite(num_elements=3, composition=[3.0, 1.0, 1.0],
                              z_init=[-1.0, 1.0, 2.0], site_zeta=zeta, feature_dim=dim)

    def test_zeta_zero_is_the_per_species_model_exactly(self):
        m = self._madelung(zeta=0.0)
        feats, species, centre = _inputs()
        assert m.deviation(species, feats, centre) is None
        assert torch.allclose(m.charges(species, feats, centre), m.z[species])

    def test_on_a_pristine_cell_every_charge_is_the_element_baseline(self):
        m = self._madelung()
        _, _, centre = _inputs()
        species = torch.arange(3)
        feats = centre[species]
        batch = torch.zeros(3, dtype=torch.long)
        q = m.charges(species, feats, centre, batch, 1)
        assert float((q - m.z[species]).abs().max()) < 1e-12

    def test_the_deviation_is_bounded_by_zeta(self):
        for zeta in (0.5, 1.0):
            m = self._madelung(zeta=zeta)
            feats, species, centre = _inputs(n=60, seed=4)
            with torch.no_grad():
                last = [x for x in m.site.modules()
                        if isinstance(x, torch.nn.Linear)][-1]
                last.weight.mul_(200.0)
            dev = m.deviation(species, feats, centre)
            # Centring per graph is not applied here, so the raw bound is exact.
            assert float(dev.abs().max()) <= zeta + 1e-12

    def test_the_deviation_is_centred_per_graph_so_the_cell_charge_is_the_baseline(self):
        """NEUTRALITY. Without the per-graph centring `sum_i dZ_i` is a learnable per-frame
        net charge, which moves the Ewald G = 0 constant frame by frame and breaks the c
        table's per-(charge, size) premise."""
        m = self._madelung()
        feats, species, centre = _inputs(n=24, seed=5)
        batch = torch.cat([torch.zeros(12, dtype=torch.long),
                           torch.ones(12, dtype=torch.long)])
        with torch.no_grad():
            last = [x for x in m.site.modules() if isinstance(x, torch.nn.Linear)][-1]
            last.weight.mul_(50.0)
            last.bias.add_(0.3)
        dev = m.deviation(species, feats, centre, batch, 2)
        assert float(dev.abs().max()) > 1e-3, "the channel produced nothing to centre"
        for g in (0, 1):
            assert abs(float(dev[batch == g].sum())) < 1e-12

    def test_a_scorer_without_features_gets_the_baseline_rather_than_a_guess(self):
        m = self._madelung()
        species = torch.arange(3)
        assert torch.allclose(m.charges(species), m.z[species])

    def test_the_channel_needs_a_feature_width(self):
        with pytest.raises(ValueError, match="feature width"):
            MadelungOnSite(num_elements=3, composition=[3.0, 1.0, 1.0], site_zeta=1.0)


def _perovskite(reps=(2, 2, 2)):
    a = 5.6
    cell = bulk("Cs", "sc", a=a)
    return Atoms("CsPbCl3",
                 scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5],
                                   [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
                 cell=cell.cell, pbc=True).repeat(reps)


def _model(**overrides):
    torch.manual_seed(0)
    kwargs = dict(
        r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=3,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"), MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu, atomic_energies=np.zeros((1, 3)),
        avg_num_neighbors=8.0, atomic_numbers=[17, 55, 82], correlation=2,
        atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        carrier_feature_dim=16, counter_embedding_dim=8, carrier_mlp_hidden=16,
        use_long_range=False, spectral_head=True, counting_head=True,
        spectral_first_shell=True, spectral_r_cut=6.0,
        madelung_on_site=True, madelung_composition=[3.0, 1.0, 1.0],
        madelung_z_init=[-1.0, 1.0, 2.0], les_arguments={"sigma": 1.0},
        on_site_centred=True)
    kwargs.update(overrides)
    return MACEDefect(**kwargs)


def _batch(frames, counts, cutoff=6.0):
    ds = []
    for atoms, c in zip(frames, counts):
        config = data.Configuration(
            atomic_numbers=atoms.get_atomic_numbers(), positions=atoms.get_positions(),
            cell=np.array(atoms.get_cell()), pbc=(True, True, True),
            properties={"carrier_counts": c}, property_weights={})
        ds.append(data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=cutoff))
    loader = tools.torch_geometric.dataloader.DataLoader(ds, batch_size=len(ds))
    return next(iter(loader))


class TestEndToEnd:
    def test_both_channels_run_and_the_site_charges_move_the_energy(self):
        pristine = [_perovskite()]
        loader = tools.torch_geometric.dataloader.DataLoader(
            [_batch(pristine, [[0.0] * 4])], batch_size=None)
        off = _model(madelung_site_zeta=0.0)
        on = _model(madelung_site_zeta=1.0)
        for model in (off, on):
            model.collect_pristine_centre([_batch(pristine, [[0.0] * 4])])
        assert bool(off.pristine_centre_set) and bool(on.pristine_centre_set)
        # delta_L comes off the same pass and must be a real number.
        assert float(on.pristine_level_spacing) > 0.0
        batch = _batch([_perovskite()], [[0.0, 0.0, 1.0, 0.0]])
        a = off(batch.to_dict(), training=True, compute_force=True)
        b = on(_batch([_perovskite()], [[0.0, 0.0, 1.0, 0.0]]).to_dict(),
               training=True, compute_force=True)
        assert a["forces"].isfinite().all() and b["forces"].isfinite().all()
        # The site channel has an effect: the two models differ on a defect-free cell too,
        # because the deviation is a function of the environment rather than of the carrier.
        assert float((a["delta_sr_energy"] - b["delta_sr_energy"]).abs().max()) > 0.0
        del loader

    def test_the_correction_vanishes_on_the_cell_it_was_centred_on(self):
        """The pristine identity end to end -- per SPECIES, which is what the centre is.
        On a real host with more than one Wyckoff site per species this is NOT zero atom by
        atom, and the residual is a measurement rather than a failure; the cubic toy cell
        here has one environment per species, so it is zero."""
        pristine = _perovskite()
        model = _model()
        model.collect_pristine_centre([_batch([pristine], [[0.0] * 4])])
        centre = model.pristine_centre(torch.get_default_dtype())
        out = model(_batch([pristine], [[0.0] * 4]).to_dict(), training=False,
                    compute_force=False)
        feats = model.defect_feature_readouts[0](out["trunk_block0"])[
            :, : model.spectral_feature_dim]
        species = _batch([pristine], [[0.0] * 4]).node_attrs.argmax(dim=-1)
        levels = model.spectral.h.on_site(feats, species, centre=centre)
        base = model.spectral.h.eps0[species]
        worst = float((levels - base).abs().max())
        print(f"\\npristine-cell residual of the centred correction: {worst:.3e} eV")
        assert worst < 1e-8
