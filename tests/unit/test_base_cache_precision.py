"""Section 2.5 of the Stage A' spec: trunk float32, head float64, base outputs cached.

Three claims, each a test:

  * the mixed policy leaves the HEAD's outputs where the all-float64 model puts them
    (1e-6 eV, 1e-6 eV/A on a fixed frame) and the total energy within 1e-3 eV -- F20's
    identity half; the speed half is measured on the production model, not here;
  * a cached forward returns the same base energy, base forces and head outputs as the
    full forward, to float32 tolerance, and the model refuses to cache a configuration
    under which something cached would be training;
  * the drift guard trips when the base moves.
"""

import copy

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.modules import defect_cache
from mace.modules.defect_models import MACEDefect
from mace.modules.defect_protocol import trainable_mask

Z_TABLE = tools.AtomicNumberTable([17, 55, 82])


def _model(**overrides) -> MACEDefect:
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
        madelung_z_init=[-1.0, 1.0, 2.0], les_arguments={"sigma": 1.0})
    kwargs.update(overrides)
    model = MACEDefect(**kwargs)
    # Randomise the correction heads off their zero initialisation so the head outputs are
    # not trivially identical between two models.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if trainable_mask(name):
                p.add_(0.05 * torch.randn_like(p))
    return model


def _frames(n: int, natoms: int = 12, seed: int = 0, charged: bool = True):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        positions = rng.uniform(0, 7, size=(natoms, 3))
        numbers = np.array([17] * (natoms // 2) + [55] * (natoms // 4) + [82] * (natoms // 4))
        counts = [1.0, 0.0, 0.0, 0.0] if (charged and k % 2 == 0) else [0.0] * 4
        config = data.Configuration(
            atomic_numbers=numbers, positions=positions, cell=np.eye(3) * 9.0,
            pbc=(True, True, True),
            properties={"energy": float(rng.normal()), "forces": rng.normal(size=(natoms, 3)),
                        "carrier_counts": counts},
            property_weights={"energy": 1.0, "forces": 1.0})
        out.append(data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=6.0))
    return out


def _batch(frames):
    loader = tools.torch_geometric.dataloader.DataLoader(frames, batch_size=len(frames))
    return next(iter(loader))


def _freeze_base(model):
    for name, p in model.named_parameters():
        p.requires_grad_(trainable_mask(name))
    for i, ro in enumerate(model.defect_feature_readouts):
        if i > 0:
            for p in ro.parameters():
                p.requires_grad_(False)


def _head_outputs(out):
    return dict(delta_sr=out["delta_sr_energy"], alpha=out["carrier_alpha"],
                eps=out["carrier_readouts"], delta_forces=out["delta_forces"],
                correction_forces=out["forces"] - out["base_forces"])


@pytest.fixture(scope="module")
def frames():
    torch.set_default_dtype(torch.float64)
    return _frames(4)


def test_the_mixed_policy_casts_the_trunk_down_and_the_head_up(frames):
    model = _model(precision_policy="mixed")
    assert next(model.interactions.parameters()).dtype == torch.float32
    assert next(model.readouts.parameters()).dtype == torch.float32
    assert next(model.spectral.parameters()).dtype == torch.float64
    assert next(model.defect_feature_readouts.parameters()).dtype == torch.float64
    assert model.madelung.z.dtype == torch.float64
    out = model(_batch(frames).to_dict(), training=True, compute_force=True)
    assert out["energy"].dtype == torch.float64
    assert out["forces"].dtype == torch.float64
    assert out["delta_sr_energy"].dtype == torch.float64


def test_f20_head_outputs_match_the_all_float64_model(frames):
    """Same weights, one model uniform float64, the other mixed. The head sees float32-
    rounded trunk features under mixed, so this is a measurement of how much that costs."""
    uniform = _model(precision_policy="uniform")
    mixed = copy.deepcopy(uniform)
    mixed.precision_policy = "mixed"
    mixed._apply_precision_policy()
    batch = _batch(frames)
    out_u = uniform(batch.to_dict(), training=True, compute_force=True)
    out_m = mixed(_batch(frames).to_dict(), training=True, compute_force=True)
    hu, hm = _head_outputs(out_u), _head_outputs(out_m)
    for key in ("delta_sr", "eps"):
        diff = float((hu[key] - hm[key]).abs().max())
        assert diff < 1e-6, f"{key}: {diff:.2e} eV"
    for key in ("delta_forces", "correction_forces"):
        diff = float((hu[key] - hm[key]).abs().max())
        assert diff < 1e-6, f"{key}: {diff:.2e} eV/A"
    assert float((out_u["energy"] - out_m["energy"]).abs().max()) < 1e-3


def test_the_cache_reproduces_the_full_forward(frames, tmp_path):
    model = _model(precision_policy="mixed")
    _freeze_base(model)
    defect_cache.attach_frame_keys(frames, z_table=Z_TABLE)
    loader = tools.torch_geometric.dataloader.DataLoader(frames, batch_size=2)
    full = model(_batch(frames).to_dict(), training=True, compute_force=True)
    cache = defect_cache.build_base_cache(model, [loader], device="cpu",
                                         path=tmp_path / "cache.pt")
    assert len(cache) == len(frames)
    assert not torch.equal(model.base_cache_checksum, torch.zeros(32, dtype=torch.uint8))
    model.set_base_cache(cache)
    fast = model(_batch(frames).to_dict(), training=True, compute_force=True)
    assert float((full["base_energy"] - fast["base_energy"]).abs().max()) < 1e-5
    assert float((full["base_forces"] - fast["base_forces"]).abs().max()) < 1e-5
    assert float((full["energy"] - fast["energy"]).abs().max()) < 1e-5
    assert float((full["forces"] - fast["forces"]).abs().max()) < 1e-5
    hf, hc = _head_outputs(full), _head_outputs(fast)
    for key in hf:
        assert float((hf[key] - hc[key]).abs().max()) < 1e-6, key
    # The parameter gradient of the head is what training consumes; it must agree too.
    params = [p for p in model.parameters() if p.requires_grad]
    g_full = torch.autograd.grad(full["forces"].square().sum() + full["energy"].sum(),
                                 params, allow_unused=True)
    g_fast = torch.autograd.grad(fast["forces"].square().sum() + fast["energy"].sum(),
                                 params, allow_unused=True)
    for a, b in zip(g_full, g_fast):
        if a is None and b is None:
            continue
        assert a is not None and b is not None
        assert float((a - b).abs().max()) < 1e-5 * max(1.0, float(a.abs().max()))
    # A second build from disk returns the same entries without recomputing.
    again = defect_cache.build_base_cache(model, [loader], device="cpu",
                                          path=tmp_path / "cache.pt")
    assert again.checksum == cache.checksum and len(again) == len(cache)


def test_a_trainable_base_is_refused(frames):
    model = _model(precision_policy="mixed")
    with pytest.raises(RuntimeError, match="trainable"):
        defect_cache.require_cacheable(model)
    model.spectral_first_shell = False
    _freeze_base(model)
    with pytest.raises(RuntimeError, match="first_shell"):
        defect_cache.require_cacheable(model)


def test_the_drift_guard_trips_when_the_base_moves(frames, tmp_path):
    model = _model(precision_policy="mixed")
    _freeze_base(model)
    defect_cache.attach_frame_keys(frames, z_table=Z_TABLE)
    loader = tools.torch_geometric.dataloader.DataLoader(frames, batch_size=2)
    model.set_base_cache(defect_cache.build_base_cache(model, [loader], device="cpu"))
    rng = np.random.default_rng(0)
    result = defect_cache.check_drift(model, frames, "cpu", rng)
    assert result["d_energy"] < 1e-5 and result["d_forces"] < 1e-5
    with torch.no_grad():
        for p in model.interactions[1].parameters():
            p.add_(1e-2)
            break
    with pytest.raises(RuntimeError, match="DRIFT"):
        for _ in range(len(frames)):
            defect_cache.check_drift(model, frames, "cpu", rng)


def test_frame_keys_are_content_hashes():
    numbers = np.array([17, 55, 82])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cell = np.eye(3) * 8.0
    k = defect_cache.frame_key(numbers, positions, cell)
    assert defect_cache.frame_key(numbers, positions + 1e-8, cell) == k
    assert defect_cache.frame_key(numbers, positions + 1e-3, cell) != k
    assert defect_cache.frame_key(numbers[::-1], positions, cell) != k


def test_a_float32_batch_through_a_mixed_model_keeps_the_trunk_in_the_forces(frames):
    """The scorers build float32 batches. The head-boundary cast then makes a NEW positions
    tensor; the force derivative must still be taken with respect to the data's positions,
    or the base forces lose the trunk entirely (measured: 0.30 against 0.010 eV/A RMS on a
    neutral frame before the fix)."""
    uniform = _model(precision_policy="uniform")
    mixed = copy.deepcopy(uniform)
    mixed.precision_policy = "mixed"
    mixed._apply_precision_policy()
    torch.set_default_dtype(torch.float32)
    try:
        f32 = _frames(2, seed=3, charged=False)
        batch = _batch(f32)
        out_u = uniform.float()(batch.to_dict(), training=False, compute_force=True)
        out_m = mixed(_batch(f32).to_dict(), training=False, compute_force=True)
    finally:
        torch.set_default_dtype(torch.float64)
    d = float((out_u["base_forces"].double() - out_m["base_forces"].double()).abs().max())
    assert d < 1e-4, f"base forces differ by {d:.3e} eV/A between uniform-f32 and mixed"
    assert float(out_m["base_forces"].abs().max()) > 1e-3
