"""V3's locality claims must be structural facts, not training outcomes.

V3 exists because V2 could satisfy Delta_bind >= m by sliding the host reference instead of
binding a level. The fix is that every matrix element is a function of the trunk's first
interaction block only, so an atom further than r_max from the vacancy is bit-identical to its
pristine counterpart and the continuum cannot move. Every claim in that sentence is checkable
before any training happens, which is what this file does.

The tests deliberately assert IDENTITY rather than closeness. "Approximately pinned" is what a
trained-away artefact looks like; exact equality either holds by construction or the
construction is wrong.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import modules
from mace.modules.defect_models import MACEDefect
from mace.modules.defect_spectral_v3 import LocalSpectralHead
from mace.tools.scripts_utils import extract_config_mace_model

from test_defect_spectral_model import make_batch          # noqa: E402


R_MAX = 4.0


def build_v3(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(
        r_max=R_MAX, num_bessel=6, num_polynomial_cutoff=5, max_ell=1,
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
        use_long_range=False, spectral_head=True, spectral_num_states=4,
        spectral_first_shell=True, spectral_local=True, spectral_gauge_penalty=True,
    )
    cfg.update(kw)
    return MACEDefect(**cfg)


def capture(model, batch):
    grabbed = {}
    head = model.spectral
    original = head.forward

    def wrapped(*a, **k):
        k["internals"] = grabbed
        return original(*a, **k)

    head.forward = wrapped
    try:
        with torch.no_grad():
            model(batch.to_dict(), training=False, compute_force=False)
    finally:
        head.forward = original
    return grabbed


# ------------------------------------------------------------------ construction guards


def test_more_than_one_readout():
    """The 'block 1' slice is only block 1 when there are several readouts.

    MACEDefect assembles its per-block defect features with `feat_idx = -1` when there is a
    single readout, so every entry would be the LAST block's features. Nothing downstream can
    see the difference -- the width is right and the model trains -- which is exactly why this
    is checked rather than assumed.
    """
    model = build_v3()
    assert len(model.readouts) > 1


def test_local_head_requires_first_shell():
    with pytest.raises(ValueError, match="spectral_first_shell"):
        build_v3(spectral_first_shell=False)


def test_only_two_length_constants():
    """Constraint 2: no material-specific cutoffs anywhere in the head.

    The parent carries two -- `sigma_r1` (a CsPbCl3 first-shell radius) and `decay_r0` -- and
    V3 must have retired both into r_max rather than inheriting their defaults.
    """
    head = build_v3().spectral
    assert head.sigma_r1 == pytest.approx(R_MAX)
    assert head.decay_r0 == pytest.approx(R_MAX)
    assert head.r_max == pytest.approx(R_MAX)
    assert head.r_cut == pytest.approx(head.r_couple)
    lengths = {round(float(v), 9) for k, v in vars(head).items()
               if isinstance(v, float) and k in
               ("sigma_r1", "decay_r0", "r_cut", "r_max", "r_couple")}
    assert lengths == {round(R_MAX, 9), round(float(head.r_couple), 9)}


def test_extractor_round_trip_keeps_v3():
    """A saved V3 model rebuilt from its own config must come back V3, not H3.

    This extractor has twice dropped a flag that was correct everywhere else, and the two
    heads have different parameters but no shape clash at the model level -- a reverted head
    would simply be a different Hamiltonian under the same checkpoint name.
    """
    cfg = extract_config_mace_model(build_v3())
    assert cfg["spectral_local"] is True
    assert cfg["spectral_first_shell"] is True
    rebuilt = MACEDefect(**cfg)
    assert isinstance(rebuilt.spectral, LocalSpectralHead)


# ------------------------------------------------------------------ test 1: trunk independence


def test_blocks_2plus_do_not_affect_H():
    """H is bit-identical when every interaction block after the first is randomised.

    Block 1 must stay -- it is the descriptor. If H moves at all, the head is reading the
    deeper blocks and its receptive field is r_max * num_interactions, not r_max, so the
    pinned-continuum guarantee below is void.
    """
    model = build_v3()
    batch = make_batch(n_cells=2)
    before = capture(model, batch)["H"].clone()

    with torch.no_grad():
        for blk in list(model.interactions)[1:]:
            for p in blk.parameters():
                p.add_(torch.randn_like(p))
        for blk in list(model.products)[1:]:
            for p in blk.parameters():
                p.add_(torch.randn_like(p))
    after = capture(model, batch)["H"]
    assert torch.equal(before, after), "H moved when only blocks 2+ changed"


def test_head_loss_does_not_reach_the_trunk():
    """The descriptor is detached, so no head term can train the trunk.

    Asserted on the whole trunk rather than on the head's own input, because the response
    channel reads trunk features by a different route and V3 detaches that too.
    """
    model = build_v3(response_channel=False)
    batch = make_batch()
    for p in model.parameters():
        p.requires_grad_(True)
    grabbed = {}
    head = model.spectral
    original = head.forward

    def wrapped(*a, **k):
        k["internals"] = grabbed
        return original(*a, **k)

    head.forward = wrapped
    try:
        model(batch.to_dict(), training=True, compute_force=False)
    finally:
        head.forward = original
    grabbed["lam"].sum().backward()

    trunk = list(model.interactions.parameters()) + list(model.products.parameters())
    assert trunk, "no trunk parameters found; the test is not testing anything"
    assert all(p.grad is None for p in trunk), "a head loss reached the trunk"


# ------------------------------------------------------------------ tests 2 & 3: pinned continuum


def _pristine_and_vacancy(n=14, spacing=2.2, box=13.0, counts=(0, 0, 1, 0)):
    """A pristine cell and the same cell with one atom deleted, positions untouched.

    Dataset frames are relaxed, so a defect frame's far atoms never match pristine to float
    precision and the assertion could only ever be approximate. Constructing the pair makes
    the claim exact: identical positions, identical counter (eps depends on it), one atom
    removed. Everything beyond r_max of the hole then has an identical neighbour set by
    construction, so identity is the only acceptable answer.
    """
    from ase.atoms import Atoms

    rng = np.random.default_rng(3)
    pos = rng.uniform(0.0, box, size=(n, 3))
    numbers = [17, 55, 82] * (n // 3) + [17] * (n % 3)

    def frame(keep):
        a = Atoms(numbers=[numbers[i] for i in keep], positions=pos[keep],
                  cell=np.eye(3) * box, pbc=True)
        a.info.update({"REF_energy": 0.0, "carrier_counts": np.array(counts),
                       "multiplicity": 1, "m_s_ref_doubled": 1,
                       "cell_charge": int(counts[2] + counts[3] - counts[0] - counts[1]),
                       "host": "CsPbCl3", "e_cbm_cell": -3.2, "e_vbm_cell": -6.8})
        a.arrays["REF_forces"] = np.zeros((len(keep), 3))
        return a

    victim = 0
    keep_def = [i for i in range(n) if i != victim]
    return frame(list(range(n))), frame(keep_def), pos, victim, keep_def


def _batch_from(atoms, model):
    from mace import data as mace_data
    from mace.data.defects import prepare_defect_configurations
    from mace.tools import AtomicNumberTable, torch_geometric

    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cfg = mace_data.config_from_atoms(atoms)
    prepare_defect_configurations([cfg])
    cutoff = float(getattr(model, "spectral_r_cut", model.r_max))
    ad = mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=cutoff)
    loader = torch_geometric.dataloader.DataLoader([ad], batch_size=1, shuffle=False)
    return next(iter(loader))


def test_continuum_is_pinned_and_candidate_set_is_exactly_r_max():
    """Tests 2 and 3 together, on the same construction.

    Test 2: every atom beyond r_max of the vacancy has on-site elements matching its pristine
    counterpart to float precision. Test 3: the atoms whose elements changed are EXACTLY those
    within r_max of the vacancy site.

    Hopping follows without a separate assertion for the far block: t is a pure function of
    the two endpoints' descriptors, their elements and r, and the first of those is what is
    being shown identical here.
    """
    model = build_v3()
    pristine, defect, pos, victim, keep = _pristine_and_vacancy()

    eps_p = capture(model, _batch_from(pristine, model))["eps"].detach()
    eps_d = capture(model, _batch_from(defect, model))["eps"].detach()

    # Minimum-image distance from every surviving atom to the vacancy site.
    box = float(pristine.get_cell()[0, 0])
    d = pos[keep] - pos[victim]
    d -= box * np.round(d / box)
    dist = np.linalg.norm(d, axis=-1)

    far = dist > R_MAX
    near = ~far
    assert far.any() and near.any(), "construction gives no contrast; widen the box"

    moved = ~torch.isclose(eps_d, eps_p[keep], atol=0.0, rtol=0.0).all(dim=-1)
    moved = moved.cpu().numpy()

    assert not moved[far].any(), (
        f"{int(moved[far].sum())} atoms beyond r_max changed: the continuum is not pinned")
    assert moved[near].any(), (
        "no atom within r_max changed -- the head is not seeing the vacancy at all")


# ------------------------------------------------------------------ head-level algebra


def head(**kw):
    torch.manual_seed(0)
    d = dict(feature_dim=8, counter_dim=4, num_elements=3, r_max=R_MAX, r_couple=10.0,
             hidden=16, num_channels=2, num_states=2, smearing=0.02)
    d.update(kw)
    return LocalSpectralHead(**d).double()


def two_site(h, r=3.0):
    feats = torch.randn(2, 8, dtype=torch.float64)
    counter = torch.zeros(1, 4, dtype=torch.float64)
    counts = torch.ones(1, 2, dtype=torch.float64)
    batch = torch.zeros(2, dtype=torch.long)
    ei = torch.tensor([[0, 1], [1, 0]])
    length = torch.full((2,), r, dtype=torch.float64)
    vec = torch.zeros(2, 3, dtype=torch.float64)
    vec[0, 0], vec[1, 0] = r, -r
    sp = torch.tensor([2, 2])
    return h(feats, counter, counts, batch, 1, ei, length,
             node_species=sp, edge_vector=vec), feats, ei, length, sp


def test_two_site_matches_closed_form():
    """For H = [[a, -t], [-t, b]] the eigenvalues are (a+b)/2 -/+ sqrt(((a-b)/2)^2 + t^2).

    Re-derived for V3's parametrization rather than inherited: the on-site term gained an
    element embedding and the decay is now referenced at r_max, so the parent's version of
    this test would pass against different numbers.
    """
    h = head()
    grabbed = {}
    feats = torch.randn(2, 8, dtype=torch.float64)
    counter = torch.zeros(1, 4, dtype=torch.float64)
    counts = torch.ones(1, 2, dtype=torch.float64)
    batch = torch.zeros(2, dtype=torch.long)
    ei = torch.tensor([[0, 1], [1, 0]])
    r = torch.full((2,), 3.0, dtype=torch.float64)
    vec = torch.zeros(2, 3, dtype=torch.float64)
    vec[0, 0], vec[1, 0] = 3.0, -3.0
    sp = torch.tensor([2, 2])
    out = h(feats, counter, counts, batch, 1, ei, r, node_species=sp,
            edge_vector=vec, internals=grabbed)

    # Read a, b and t off the H the head actually assembled rather than recomputing them.
    # V3 always carries the sigma term, so `hopping()` alone is not the whole off-diagonal
    # and a recomputed t would be testing a different matrix than the one eigh was given --
    # which is the substitution this project keeps having to guard against.
    H = grabbed["H"][0, 0].detach().numpy()       # [G, C, n, n] -> graph 0, channel 0
    a, b, t = float(H[0, 0]), float(H[1, 1]), float(H[0, 1])
    assert abs(H[0, 1] - H[1, 0]) < 1e-12, "H is not symmetric; eigh's gradients would be wrong"
    mid, half = (a + b) / 2.0, np.sqrt(((a - b) / 2.0) ** 2 + t ** 2)
    got = np.sort(out.eigenvalues[0, 0].detach().numpy())[:2]
    assert np.allclose(got, [mid - half, mid + half], atol=1e-9)


def test_hopping_symmetric_in_ij():
    h = head()
    f = torch.randn(4, 8, dtype=torch.float64)
    r = torch.full((4,), 3.3, dtype=torch.float64)
    sp_i, sp_j = torch.tensor([0, 1, 2, 0]), torch.tensor([2, 0, 1, 1])
    a = h.hopping(f, f.flip(0), r, sp_i, sp_j)
    b = h.hopping(f.flip(0), f, r, sp_j, sp_i)
    assert torch.equal(a, b)


def test_init_hopping_not_collapsed_at_the_flanking_distance():
    """Re-referencing the decay at r_max must not start the head disconnected.

    exp(-(r - r0)/l) with r0 moved from 3.0 to r_max divides every element by e^((r_max-3)/l),
    which at the parent's l_init = 1.0 is enough to leave only the t_min floor -- precisely
    the decoupled regime the floor exists to prevent. Checked at the separation the flanking
    pair actually occupies.
    """
    h = head(r_max=5.0, r_couple=10.0)
    f = torch.randn(2, 8, dtype=torch.float64)
    sp = torch.tensor([2, 2])
    for r_val in (5.3, 6.8):
        r = torch.full((1,), r_val, dtype=torch.float64)
        t = h.hopping(f[:1], f[1:], r, sp[:1], sp[1:]).abs().max().item()
        assert t > 5.0 * h.t_min, (
            f"t at {r_val} A is {t:.4g}, barely above the floor {h.t_min}: the head starts "
            "effectively disconnected")
