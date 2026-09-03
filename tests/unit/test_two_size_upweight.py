"""The two-size upweight must hit its target share and stay label-free.

The 17 charged 159-atom frames carry the only measurement that distinguishes a bound carrier
from a band state (Test 2: R_DFT = 0.95). At natural weighting they are ~1.6% of the charged
force loss. The share must be computed in weight x atoms, not frame count: a 159-atom frame
already contributes twice the terms, and counting frames would overshoot 2x.
"""

import torch

from mace.data.two_size import apply_two_size_upweight


class FakeData:
    def __init__(self, n, charged, weight=1.0):
        self.positions = torch.zeros(n, 3)
        self.carrier_counts = torch.tensor([0.0, 0.0, 1.0, 0.0]) if charged \
            else torch.zeros(4)
        self.forces_weight = torch.tensor(weight)


def make(n_small=1030, n_large=17):   # the real ratio: 1030 vs 17
    return ([FakeData(79, True) for _ in range(n_small)]
            + [FakeData(159, True) for _ in range(n_large)]
            + [FakeData(79, False) for _ in range(50)])


def test_realised_share_hits_the_target():
    """At the real 1030:17 ratio the large frames are ~3% of the charged force loss, so
    reaching 25% needs a factor near 10. (With only 100 small frames they would already be
    at 25% and the correct factor is BELOW 1 -- the target is a share, not an increase.)"""
    ds = make()
    factor, share = apply_two_size_upweight(ds, target_share=0.25)
    assert abs(share - 0.25) < 1e-6, f"share {share}"
    assert 5.0 < factor < 20.0, f"factor {factor}"


def test_share_is_weighted_by_atom_count_not_frame_count():
    """A 159-atom frame contributes twice the force terms; frame-counting would overshoot."""
    ds = make(n_small=100, n_large=100)   # equal frames, unequal atoms
    _, share = apply_two_size_upweight(ds, target_share=0.5)
    assert abs(share - 0.5) < 1e-6


def test_neutral_frames_are_untouched():
    ds = make()
    before = [float(d.forces_weight) for d in ds if float(d.carrier_counts.abs().sum()) == 0]
    apply_two_size_upweight(ds, target_share=0.25)
    after = [float(d.forces_weight) for d in ds if float(d.carrier_counts.abs().sum()) == 0]
    assert before == after


def test_small_charged_frames_are_untouched():
    ds = make()
    apply_two_size_upweight(ds, target_share=0.25)
    for d in ds:
        if d.positions.shape[0] == 79 and float(d.carrier_counts.abs().sum()) > 0:
            assert float(d.forces_weight) == 1.0


def test_no_large_frames_is_a_warning_not_a_crash():
    ds = [FakeData(79, True) for _ in range(10)]
    factor, share = apply_two_size_upweight(ds, target_share=0.25)
    assert factor == 1.0 and share == 0.0


def test_idempotent_target_when_applied_to_an_already_weighted_set():
    """Re-applying must re-solve to the same share, not compound the factor."""
    ds = make()
    apply_two_size_upweight(ds, target_share=0.25)
    _, share = apply_two_size_upweight(ds, target_share=0.25)
    assert abs(share - 0.25) < 1e-6


# ---------------------------------------------------------------- Stage A' section 1
# The same target share in the ENERGY loss and the FORCE loss, each solved on its own mass,
# with the frame-level `weight` (config_type_weights) counted in both.

from mace.data.two_size import apply_size_upweight, realised_shares  # noqa: E402


class FakeData2(FakeData):
    """Carries every weight column a real AtomicData has: the generic pair the totals
    terms read (charged frames) and the base pair the base terms read (neutral frames)."""
    def __init__(self, n, charged, weight=1.0, frame_weight=1.0):
        super().__init__(n, charged, weight)
        self.energy_weight = torch.tensor(weight)
        self.base_energy_weight = torch.tensor(weight)
        self.base_forces_weight = torch.tensor(weight)
        self.weight = torch.tensor(frame_weight)


def make2(n_small=1057, n_large=15, n_pristine=544):
    return ([FakeData2(79, False) for _ in range(n_small)]
            + [FakeData2(159, False) for _ in range(n_large)]
            + [FakeData2(80, False, frame_weight=5.0) for _ in range(n_pristine)]
            + [FakeData2(79, True) for _ in range(30)])


def test_both_channels_reach_the_same_target_on_their_own_mass():
    ds = make2()
    res = apply_size_upweight(ds, population="neutral", target_share=0.25)
    assert abs(res["energy"][1] - 0.25) < 1e-6
    assert abs(res["forces"][1] - 0.25) < 1e-6
    # Atom count enters the force mass and not the energy mass, so the factors differ.
    assert res["energy"][0] != res["forces"][0]
    after = realised_shares(ds, population="neutral")
    assert abs(after["energy"] - 0.25) < 1e-6 and abs(after["forces"] - 0.25) < 1e-6


def test_the_frame_weight_is_part_of_the_mass():
    """Pristine frames carry weight 5.0; a share that ignored it would be a share of a
    different loss from the one being minimised."""
    a = make2(n_pristine=544)
    b = make2(n_pristine=0)
    fa = apply_size_upweight(a, population="neutral", target_share=0.25)["forces"][0]
    fb = apply_size_upweight(b, population="neutral", target_share=0.25)["forces"][0]
    assert fa > fb


def test_charged_frames_are_untouched_by_the_neutral_upweight():
    ds = make2()
    apply_size_upweight(ds, population="neutral", target_share=0.25)
    for d in ds:
        if float(d.carrier_counts.abs().sum()) > 0:
            assert float(d.energy_weight) == 1.0 and float(d.forces_weight) == 1.0


def test_the_neutral_upweight_reaches_the_loss_the_base_terms_read():
    """The columns differ by population, and the first neutral upweight scaled one no term
    reads. This test builds real AtomicData through the defect pipeline and checks the
    DefectLoss VALUE moves -- outcome, not the function's own arithmetic."""
    import numpy as np

    from mace import data as mace_data
    from mace import tools
    from mace.data.defects import prepare_defect_configurations
    from mace.data.two_size import weight_column
    from mace.modules.loss import DefectLoss

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    z = tools.AtomicNumberTable([17, 55, 82])
    configs = []
    for n in (12, 24, 12):
        numbers = np.array([17] * (n // 2) + [55] * (n // 4) + [82] * (n // 4))
        configs.append(mace_data.Configuration(
            atomic_numbers=numbers, positions=rng.uniform(0, 8, size=(n, 3)),
            cell=np.eye(3) * 9.0, pbc=(True, True, True),
            properties={"energy": float(rng.normal()), "forces": rng.normal(size=(n, 3)),
                        "carrier_counts": [0.0, 0.0, 0.0, 0.0]},
            property_weights={"energy": 1.0, "forces": 1.0}))
    prepare_defect_configurations(configs)
    ds = [mace_data.AtomicData.from_config(c, z_table=z, cutoff=4.0) for c in configs]
    assert weight_column("neutral", "forces") == "base_forces_weight"
    big = ds[1]
    assert float(big.base_forces_weight) == 1.0 and float(big.base_energy_weight) == 1.0
    apply_size_upweight(ds, population="neutral", target_share=0.75, size_threshold=20)
    assert float(big.base_forces_weight) != 1.0, "the base force column did not move"
    assert float(big.base_energy_weight) != 1.0, "the base energy column did not move"

    loader = tools.torch_geometric.dataloader.DataLoader(ds, batch_size=3)
    batch = next(iter(loader))
    # A fake prediction with a fixed error, so the loss is a pure function of the weights.
    pred = dict(base_energy=batch.base_energy + 0.1, base_forces=batch.base_forces + 0.1,
                delta_energy=batch.delta_energy, delta_forces=batch.delta_forces,
                energy=batch.energy, forces=batch.forces, correction_energy=torch.zeros(3),
                counter_input_l2=torch.zeros(()), stress=None, virials=None)
    loss_fn = DefectLoss(energy_weight=1.0, forces_weight=1.0, total_energy_weight=0.0,
                         delta_energy_weight=0.0, delta_forces_weight=0.0)
    after = float(loss_fn(batch, pred))
    with torch.no_grad():
        batch.base_forces_weight[1] = 1.0
        batch.base_energy_weight[1] = 1.0
    before = float(loss_fn(batch, pred))
    assert after != before, "the upweighted column does not change the loss value"
