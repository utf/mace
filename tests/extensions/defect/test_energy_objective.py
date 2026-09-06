"""v8.1 addendum section 8 / 11.1: the Stage 1-4 charged-energy objective."""
import numpy as np
import pytest
import torch

from mace.modules.defect_objective import (
    ObjectiveStage, PROVENANCE_PAIRED, PROVENANCE_UNPAIRED, StrataTable, Stratum,
    WithinStratumPairSampler, assert_no_split_groups, manifest, profile_charge_constant,
    profile_intercepts, sampled_pair_term, shape_loss_centred, shape_loss_pairs,
    stratum_key,
)

torch.set_default_dtype(torch.float64)


def _toy(seed=0, n=40, n_strata=3):
    g = torch.Generator().manual_seed(seed)
    xi = torch.randn(n, generator=g) * 0.7
    sid = torch.randint(0, n_strata, (n,), generator=g)
    w = torch.rand(n, generator=g) + 0.1
    W = torch.tensor([1.0, 0.5, 2.0])
    return xi, sid, w, W


def test_centred_and_pair_forms_agree_in_value_and_gradient():
    xi, sid, w, W = _toy()
    xa = xi.clone().requires_grad_(True)
    xb = xi.clone().requires_grad_(True)
    la = shape_loss_centred(xa, sid, w, W)
    lb = shape_loss_pairs(xb, sid, w, W)
    assert torch.allclose(la, lb, atol=1e-12, rtol=0)
    la.backward()
    lb.backward()
    assert torch.allclose(xa.grad, xb.grad, atol=1e-12, rtol=0)


def test_invariant_to_a_constant_shift_within_a_stratum():
    xi, sid, w, W = _toy()
    base = shape_loss_centred(xi, sid, w, W)
    shifted = xi + torch.tensor([3.0, -1.5, 0.25])[sid]     # a different constant per stratum
    assert torch.allclose(shape_loss_centred(shifted, sid, w, W), base, atol=1e-12, rtol=0)


def test_total_cell_units_and_stratum_weight_independent_of_count():
    # two strata of very different sizes, the same within-stratum variance and W: equal terms
    g = torch.Generator().manual_seed(1)
    small = torch.randn(4, generator=g)
    big = torch.randn(400, generator=g)
    small = (small - small.mean()) / small.std(unbiased=False)
    big = (big - big.mean()) / big.std(unbiased=False)
    xi = torch.cat([small, big])
    sid = torch.cat([torch.zeros(4, dtype=torch.long), torch.ones(400, dtype=torch.long)])
    w = torch.ones_like(xi)
    W = torch.tensor([1.0, 1.0])
    l0 = shape_loss_centred(xi, sid, w, W)
    assert torch.allclose(l0, torch.tensor(2.0), atol=1e-10)     # variance 1 each, W = 1
    # the atom count never enters: a 79- and a 159-atom stratum with the same residuals
    # give the same term (there is simply no N_at in the formula)


def test_uninformative_stratum_contributes_nothing():
    xi = torch.tensor([0.3, 1.0, 1.5, 7.0])
    sid = torch.tensor([0, 0, 0, 1])          # stratum 1 has one member
    w = torch.ones(4)
    W = torch.tensor([1.0, 1.0])
    only0 = shape_loss_centred(xi[:3], sid[:3], w[:3], W)
    assert torch.allclose(shape_loss_centred(xi, sid, w, W), only0)


def test_profiler_recovers_injected_offset_to_the_floor():
    xi, sid, w, W = _toy()
    xi_np, sid_np, w_np = xi.numpy(), sid.numpy(), w.numpy()
    c0 = profile_intercepts(xi_np, sid_np, w_np)
    c1 = profile_intercepts(xi_np + 0.7, sid_np, w_np)
    for g in c0:
        assert abs((c1[g] - c0[g]) + 0.7) < 1e-12
        m = sid_np == g
        assert abs((w_np[m] * (xi_np[m] + c0[g])).sum() / w_np[m].sum()) < 1e-12


def test_charge_constant_profile_is_one_per_charge_and_order_independent():
    rng = np.random.default_rng(0)
    xi = rng.normal(size=60) + 2.0
    q = np.where(np.arange(60) < 50, 1, 1)
    sid = np.where(np.arange(60) < 50, 0, 1)      # 79-atom stratum and 159-atom stratum
    w = np.where(sid == 0, 1.0 / 50, 1.0 / 10)
    W = {0: 1.0, 1: 1.0}
    out = profile_charge_constant(xi, q, sid, W, w, q_ref=0)
    assert set(out) == {1}
    assert abs(out[1]["optimality"]) < 1e-12
    perm = rng.permutation(60)
    out2 = profile_charge_constant(xi[perm], q[perm], sid[perm], W, w[perm], q_ref=0)
    assert abs(out2[1]["C"] - out[1]["C"]) < 1e-12
    # the reference charge is pinned at zero
    q_mixed = np.where(np.arange(60) < 30, 0, 1)
    out3 = profile_charge_constant(xi, q_mixed, sid, W, w, q_ref=0)
    assert out3[0]["C"] == 0.0


def test_sampled_pair_term_is_unbiased_for_the_shape_loss():
    torch.manual_seed(0)
    xi, sid, w, W = _toy(n=30, n_strata=2)
    W = torch.tensor([1.0, 3.0])
    strata = {}
    for g in (0, 1):
        m = (sid == g).nonzero().flatten().tolist()
        ww = w[sid == g]
        strata[f"s{g}"] = Stratum(key=f"s{g}", index=g, weight=float(W[g]), members=m,
                                  member_weights=(ww / ww.sum()).tolist())
    table = StrataTable(strata=strata, host="toy", pristine_atoms=1, cell_convention="pbc")
    sampler = WithinStratumPairSampler(n_items=30, batch_size=4, table=table,
                                       n_pair_slots=1,
                                       generator=torch.Generator().manual_seed(3))
    acc, n = 0.0, 0
    for _ in range(20000):
        i, j = sampler.draw_pair()
        acc += 0.5 * float((xi[i] - xi[j]) ** 2)
        n += 1
    exact = float(shape_loss_centred(xi, sid, w, W)) / float(W.sum())
    assert abs(acc / n - exact) < 0.03 * max(exact, 1e-3) + 2e-3


def test_pair_sampler_layout_pairs_share_a_stratum_and_both_sizes_are_drawn():
    strata = {
        "small": Stratum(key="small", index=0, weight=1.0, members=list(range(0, 20)),
                         member_weights=[1 / 20] * 20),
        "large": Stratum(key="large", index=1, weight=1.0, members=list(range(20, 24)),
                         member_weights=[1 / 4] * 4),
    }
    table = StrataTable(strata=strata, host="toy", pristine_atoms=1, cell_convention="pbc")
    sampler = WithinStratumPairSampler(n_items=24, batch_size=4, table=table, n_pair_slots=2,
                                       generator=torch.Generator().manual_seed(0))
    seen_large = seen_small = 0
    for batch in sampler:
        assert len(batch) == 4 + 4
        for a, b in zip(batch[4::2], batch[5::2]):
            assert (a < 20) == (b < 20)              # both members of one stratum
            seen_large += a >= 20
            seen_small += a < 20
    assert seen_large > 0 and seen_small > 0


def test_groups_never_split_and_stage_exclusivity():
    class D:  # noqa: D401 - a stand-in AtomicData
        def __init__(self, pid):
            self.pair_id = pid
    assert_no_split_groups({"train": [D("a"), D("b")], "valid": [D("c")]})
    with pytest.raises(ValueError):
        assert_no_split_groups({"train": [D("a")], "valid": [D("a")]})
    assert ObjectiveStage.NUISANCE != ObjectiveStage.PRODUCTION
    m = manifest(StrataTable(strata={}, host="h", pristine_atoms=80, cell_convention="pbc"),
                 ObjectiveStage.NUISANCE, 1.0, 100.0, 2, {"floor": 1e-9})
    assert m["stage"] == "nuisance_intercepts" and "C_Q" not in json_dumps(m)


def json_dumps(m):
    import json
    return json.dumps(m)


def test_stratum_key_fields():
    k = stratum_key(PROVENANCE_UNPAIRED, "CsPbCl3", 1, "17x47,55x16,82x16", "pbc", 1)
    assert k == "unpaired|CsPbCl3|Q+1|17x47,55x16,82x16|pbc|1x"
    assert stratum_key(PROVENANCE_PAIRED, "h", -1, "c", "pbc", 2).startswith("paired|")


# ------------------------------------------------------------------ the loss wiring


def _toy_dataset(n_small=12, n_large=4, seed=0):
    """AtomicData frames: charged/neutral, two sizes, with energies and forces."""
    from mace import data
    from mace.tools import AtomicNumberTable

    rng = np.random.default_rng(seed)
    z_table = AtomicNumberTable([17, 55, 82])
    ds = []
    for k in range(n_small + n_large):
        n = 4 if k < n_small else 8
        charged = k % 3 != 2
        counts = [0.0, 0.0, 1.0, 0.0] if charged else [0.0] * 4
        numbers = [17, 55, 82, 17] if n == 4 else [17, 55, 82, 17, 17, 55, 82, 17]
        config = data.Configuration(
            atomic_numbers=np.array(numbers), positions=rng.normal(size=(n, 3)) * 3,
            cell=np.eye(3) * 12.0, pbc=(True, True, True),
            properties={"carrier_counts": counts, "energy": float(rng.normal()),
                        "forces": rng.normal(size=(n, 3))},
            property_weights={"energy": 1.0, "forces": 1.0})
        ds.append(data.AtomicData.from_config(config, z_table=z_table, cutoff=5.0))
    return ds, z_table


def test_strata_are_stamped_and_the_pair_batches_score():
    from mace.modules.defect_objective import WithinStratumPairSampler, assign_strata
    from mace.modules.loss import DefectLoss
    from mace.tools import torch_geometric

    ds, z_table = _toy_dataset()
    table = assign_strata(ds, z_table, host="toy", pristine_atoms=4, log=False)
    assert len(table.strata) == 2 and all(s.informative for s in table.strata.values())
    assert all(int(d.stratum_id) == -1 for d in ds if int(d.carrier_counts.sum()) == 0)
    sampler = WithinStratumPairSampler(len(ds), batch_size=4, table=table, n_pair_slots=2,
                                       generator=torch.Generator().manual_seed(1))
    loader = torch_geometric.dataloader.DataLoader(ds, batch_sampler=sampler)
    loss_fn = DefectLoss(energy_weight=0.0, forces_weight=0.0, delta_energy_weight=0.0,
                         delta_forces_weight=0.0, total_energy_weight=0.0,
                         energy_shape_weight=2.0, energy_pair_slots=2)
    batch = next(iter(loader))
    assert int(batch.num_graphs) == 8
    # a prediction off by a per-stratum constant: the constant never enters the term
    e_pred = batch.energy + torch.tensor([5.0, -3.0])[batch.stratum_id.reshape(-1).clamp_min(0)]
    pred = {"energy": e_pred, "delta_energy": torch.zeros_like(e_pred)}
    assert float(loss_fn.energy_shape(batch, pred)) == pytest.approx(0.0, abs=1e-12)
    noise = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.3, -0.1, 0.2, 0.6])
    pred = {"energy": e_pred + noise, "delta_energy": torch.zeros_like(e_pred)}
    expected = 2.0 * 0.5 * ((0.3 + 0.1) ** 2 + (0.2 - 0.6) ** 2) / 2
    assert float(loss_fn.energy_shape(batch, pred)) == pytest.approx(expected, abs=1e-12)
    # a batch not built by the sampler is refused
    plain = torch_geometric.dataloader.DataLoader(ds, batch_size=8, shuffle=False)
    with pytest.raises(ValueError):
        loss_fn.energy_shape(next(iter(plain)), {"energy": torch.zeros(8),
                                                  "delta_energy": torch.zeros(8)})


def test_the_shape_term_refuses_a_second_charged_energy_path():
    from mace.modules.loss import DefectLoss

    with pytest.raises(ValueError):
        DefectLoss(total_energy_weight=0.25, delta_energy_weight=0.0,
                   energy_shape_weight=1.0, energy_pair_slots=1)
    with pytest.raises(ValueError):
        DefectLoss(total_energy_weight=0.0, delta_energy_weight=10.0,
                   energy_shape_weight=1.0, energy_pair_slots=1)
    DefectLoss(total_energy_weight=0.0, delta_energy_weight=0.0, energy_shape_weight=1.0,
               energy_pair_slots=1)
