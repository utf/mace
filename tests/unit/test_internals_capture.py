"""The D1 internals capture must be inert unless asked for.

D1 needs the head's assembled H and its eigenvectors to split the axial force into on-site
and hopping parts. Those are contracted against the H the head ACTUALLY built -- for H3 that
includes the sigma term -- rather than a formula re-derived in the analysis script, which
would silently drop whatever the head does that the formula forgets.

The risk in exposing them is that a diagnostic path changes the production one. These tests
pin that it cannot: same outputs bit-for-bit with and without the capture, and nothing stored
on the module (a stashed tensor is not deepcopy-safe and would break the cuEq conversion --
that has already happened once with eps_mean).
"""

import copy

import torch

from mace.modules.defect_spectral import SpectralCarrierHead


def make_head(**kw):
    torch.manual_seed(0)
    return SpectralCarrierHead(feature_dim=8, counter_dim=4, hidden=16, num_channels=4,
                               num_states=4, radial_dim=6, r_cut=5.0, num_elements=3, **kw)


def make_inputs(n_nodes=6):
    torch.manual_seed(1)
    node_feats = torch.randn(n_nodes, 8)
    counter_emb = torch.randn(1, 4)
    counts = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    batch = torch.zeros(n_nodes, dtype=torch.long)
    src = torch.arange(n_nodes).repeat_interleave(2)
    dst = (src + torch.tensor([1, 2]).repeat(n_nodes)) % n_nodes
    edge_index = torch.stack([src, dst])
    edge_length = torch.full((edge_index.shape[1],), 3.0)
    species = torch.tensor([0, 1, 2] * (n_nodes // 3))
    return dict(node_feats=node_feats, counter_emb=counter_emb, counts=counts, batch=batch,
                num_graphs=1, edge_index=edge_index, edge_length=edge_length,
                node_species=species)


def test_outputs_are_identical_with_and_without_capture():
    head = make_head().eval()
    args = make_inputs()
    with torch.no_grad():
        plain = head(**args)
        grabbed = {}
        withcap = head(**args, internals=grabbed)
    for a, b, name in zip(plain, withcap, plain._fields):
        assert torch.equal(a, b), f"{name} changed when internals were captured"


def test_capture_fills_what_d1_needs():
    head = make_head().eval()
    grabbed = {}
    with torch.no_grad():
        head(**make_inputs(), internals=grabbed)
    for key in ("H", "psi", "lam", "w", "eps", "batch", "local"):
        assert key in grabbed, f"internals missing {key}"
    # H must be the assembled, symmetrised operator the solver saw.
    H = grabbed["H"]
    assert H.shape[0] == 1 and H.shape[1] == 4, f"unexpected H shape {tuple(H.shape)}"
    assert torch.allclose(H, H.transpose(-1, -2), atol=1e-10), "captured H is not symmetric"


def test_nothing_is_stored_on_the_module():
    """A tensor stashed as an attribute breaks deepcopy, which the cuEq conversion does."""
    head = make_head().eval()
    before = set(head.__dict__)
    grabbed = {}
    with torch.no_grad():
        head(**make_inputs(), internals=grabbed)
    assert set(head.__dict__) == before, "forward added an attribute to the module"
    copy.deepcopy(head)  # must not raise


def test_capture_keeps_the_graph_for_autograd():
    """D1 differentiates through eps and H, so they must not arrive detached."""
    head = make_head()
    args = make_inputs()
    args["node_feats"] = args["node_feats"].requires_grad_(True)
    grabbed = {}
    head(**args, internals=grabbed)
    assert grabbed["eps"].requires_grad, "eps was captured detached"
    assert grabbed["H"].requires_grad, "H was captured detached"
