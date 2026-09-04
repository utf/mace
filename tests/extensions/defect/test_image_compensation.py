"""The one-shot image-compensation term and its identities.

Section 2.3 of the Stage A' spec, as corrected by section 2.5 of the speed cycle's:

  * zero on any neutral cell: no carrier, no density, no potential;
  * on a pristine cell with a DELOCALISED carrier (uniform q_c = -1/N) the PERIODIC part
    `A_per q_c` is constant across atoms. The A' spec asserted that of the difference,
    which measurement refuted (variance 6.1e-2 eV^2 on a 40-atom cell): the isolated
    evaluator's potential of a finite cluster is not constant, and the corrected identity
    is on the periodic part alone. Both are measured here, and the difference's spread is
    reported rather than asserted;
  * with the flag on, the model runs, the extras carry the term and the bound switch, and
    the term is exactly zero on a neutral frame while non-zero on a charged one;
  * the switch itself: `s = sigmoid((depth/delta_L - 2)/0.5)`, 1 on a split-off level and 0
    in the continuum, and the term scales with it.
"""

import numpy as np
import pytest
import torch
from ase.build import bulk
from e3nn import o3

from mace import data, modules, tools
from mace.modules.defect_image import image_potential
from mace.modules.defect_models import MACEDefect
from mace.modules.latent_ewald import LatentEwald

Z_TABLE = tools.AtomicNumberTable([17, 55, 82])


def _perovskite(reps=(2, 2, 2)):
    """Cubic CsPbCl3 at a = 5.6 A, tiled; 5 atoms per formula unit."""
    a = 5.6
    cell = bulk("Cs", "sc", a=a)
    from ase import Atoms
    atoms = Atoms("CsPbCl3",
                  scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
                  cell=cell.cell, pbc=True)
    return atoms.repeat(reps)


def _batch(atoms, counts, cutoff=6.0):
    config = data.Configuration(
        atomic_numbers=atoms.get_atomic_numbers(), positions=atoms.get_positions(),
        cell=np.array(atoms.get_cell()), pbc=(True, True, True),
        properties={"carrier_counts": counts}, property_weights={})
    d = data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=cutoff)
    return next(iter(tools.torch_geometric.dataloader.DataLoader([d], batch_size=1)))


@pytest.fixture(scope="module", autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)


def test_zero_charge_gives_zero_potential():
    atoms = _perovskite()
    ewald = LatentEwald({"sigma": 1.0})
    n = len(atoms)
    positions = torch.tensor(atoms.get_positions())
    cell = torch.tensor(np.array(atoms.get_cell())).reshape(1, 3, 3)
    batch = torch.zeros(n, dtype=torch.long)
    phi = image_potential(ewald, torch.zeros(n), positions, cell, batch, 1)
    assert float(phi.abs().max()) == 0.0


def _site_potentials(atoms):
    from mace.modules.defect_madelung import site_potential

    ewald = LatentEwald({"sigma": 1.0})
    n = len(atoms)
    positions = torch.tensor(atoms.get_positions())
    cell = torch.tensor(np.array(atoms.get_cell())).reshape(1, 3, 3)
    batch = torch.zeros(n, dtype=torch.long)
    q = -torch.ones(n) / n
    return (site_potential(ewald, q, positions, cell, batch),
            image_potential(ewald, q, positions, cell, batch, n and 1))


def test_equal_charges_on_a_bravais_lattice_feel_a_constant_periodic_potential():
    """WHERE THE IDENTITY IS AN IDENTITY. On a lattice with ONE site per primitive cell,
    equal charges are a periodic array of identical environments, so `A_per q` is the same
    at every site. This is the kernel's own property, and it is what the spec's constancy
    clause is really about."""
    lattice = bulk("Cs", "sc", a=5.6).repeat((3, 3, 3))
    periodic, _ = _site_potentials(lattice)
    spread = float(periodic.max() - periodic.min())
    print(f"\nperiodic potential, {len(lattice)}-site simple-cubic lattice: "
          f"mean {float(periodic.mean()):+.6f} eV, spread {spread:.3e} eV")
    assert spread < 1e-8, f"A_per q is not constant on a Bravais lattice: {spread:.3e} eV"


def test_on_the_perovskite_neither_part_is_constant_and_the_spread_is_reported():
    """AND WHERE IT IS NOT. The A' spec asserted the DIFFERENCE was constant for a uniform
    carrier; measurement refuted that (variance 6.1e-2 eV^2). The speed cycle's spec moved
    the claim to the periodic part -- and that is refuted too, for a reason that is about
    the crystal rather than the code: CsPbCl3 is not a Bravais lattice, so EQUAL CHARGES ON
    THE ATOMS ARE NOT A UNIFORM CHARGE DENSITY. The structure factor of five inequivalent
    sublattices is non-zero away from k = 0, and the potential varies by construction.

    So neither form of the identity holds on this host. Both spreads are measured and
    reported, and the term is not tuned to either -- the adoption test is the tiling drift,
    which is a statement about size dependence and does not rest on this."""
    atoms = _perovskite()
    periodic, phi = _site_potentials(atoms)
    print(f"\nperovskite, {len(atoms)} atoms, equal charges q_i = -1/N:")
    print(f"  periodic   mean {float(periodic.mean()):+.4f} eV, "
          f"spread {float(periodic.max() - periodic.min()):.3e} eV")
    print(f"  difference mean {float(phi.mean()):+.4f} eV, "
          f"variance {float(phi.var()):.3e} eV^2, range "
          f"[{float(phi.min()):+.4f}, {float(phi.max()):+.4f}]")
    assert np.isfinite(float(phi.var()))


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
        image_compensation=True)
    kwargs.update(overrides)
    return MACEDefect(**kwargs)


def _with_spacing(model, value=0.05):
    """The bound switch needs `delta_L`; the trainer records it on the pristine pass."""
    model.set_pristine_spacing(value)
    return model


def test_the_bound_switch_is_one_when_bound_and_zero_when_resonant():
    """`s = sigmoid((depth/delta_L - 2)/0.5)`. The threshold and width are the spec's, and
    the condition `depth/delta_L > 2` is the one the dilution scorer has used since Stage 3
    -- made differentiable so it multiplies the term instead of selecting frames."""
    model = _with_spacing(_model(), 0.05)
    depth = torch.tensor([0.0, 0.05, 0.10, 0.20, 1.0])
    s = model.bound_switch(depth)
    assert float(s[0]) < 0.02                    # deep in the continuum
    assert abs(float(s[2]) - 0.5) < 1e-6         # exactly at depth = 2 delta_L
    assert float(s[4]) > 0.99                    # far above the threshold
    assert bool((s.diff() > 0).all())            # monotone in depth


def test_a_missing_spacing_is_refused_rather_than_defaulted():
    model = _model()
    with pytest.raises(RuntimeError, match="needs delta_L"):
        model.bound_switch(torch.tensor([0.3]))


def test_the_term_is_zero_on_a_neutral_frame_and_present_on_a_charged_one():
    model = _with_spacing(_model())
    atoms = _perovskite()
    neutral = model(_batch(atoms, [0.0] * 4).to_dict(), compute_force=True)
    assert neutral["energy"].isfinite().all()
    charged = model(_batch(atoms, [1.0, 0.0, 0.0, 0.0]).to_dict(), compute_force=True)
    assert charged["energy"].isfinite().all() and charged["forces"].isfinite().all()
    # The term's effect on the energy: the same charged frame with the flag off differs.
    off = _with_spacing(_model(image_compensation=False))
    off.load_state_dict({k: v for k, v in model.state_dict().items()}, strict=False)
    charged_off = off(_batch(atoms, [1.0, 0.0, 0.0, 0.0]).to_dict(), compute_force=False)
    assert float((charged["delta_sr_energy"] - charged_off["delta_sr_energy"]).abs()) > 0.0
    neutral_off = off(_batch(atoms, [0.0] * 4).to_dict(), compute_force=False)
    assert torch.allclose(neutral["energy"], neutral_off["energy"])


def test_the_pristine_pass_that_records_delta_L_does_not_also_consume_it():
    """The regression this file was missing, and arm B died of.

    Every other test here calls `_with_spacing` first, which sets `delta_L` by hand. The
    TRAINER cannot: it calls `collect_pristine_centre`, and that method's own forward is
    where `delta_L` is measured. With the image term on, that forward reached the bound
    switch, which raises when `delta_L` is unset -- so `collect_pristine_centre` raised
    inside itself and arm B failed on every seed within a minute of launch, while arm A
    (term off) ran to completion.

    The fix is a guard on `_collecting_centre`, and the reason it costs nothing is physical:
    the collection pass is a stoichiometric, carrier-free cell, where `alpha` is zero and
    the compensation is the image potential of a carrier that is not there.
    """
    model = _model(image_compensation=True)
    assert not bool(model.pristine_spacing_set), "the fixture must start without delta_L"
    pristine = _perovskite()
    n = model.collect_pristine_centre([_batch(pristine, [[0.0] * 4])])
    assert n > 0
    assert bool(model.pristine_spacing_set), "the pass must RECORD delta_L"
    assert float(model.pristine_level_spacing) > 0.0
    # And the term works immediately afterwards, on the same model, with no manual setup.
    charged = model(_batch(pristine, [[1.0, 0.0, 0.0, 0.0]]).to_dict(), compute_force=False)
    assert charged["energy"].isfinite().all()


def test_the_guard_is_scoped_to_the_collection_and_released_afterwards():
    """`_collecting_centre` must not latch. If it did, the image term would be silently off
    for the whole run -- arm B would train, report no error, and be arm A."""
    model = _with_spacing(_model(image_compensation=True))
    atoms = _perovskite()
    assert not getattr(model, "_collecting_centre", False)
    model.collect_pristine_centre([_batch(atoms, [[0.0] * 4])])
    assert not getattr(model, "_collecting_centre", False)
    out = model(_batch(atoms, [[1.0, 0.0, 0.0, 0.0]]).to_dict(), compute_force=False)
    comp = out.get("image_compensation")
    assert comp is not None, "the term must be live again after the collection"
    assert float(comp.abs().max()) > 0.0, "a charged frame must carry a non-zero term"
