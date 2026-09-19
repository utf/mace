"""Plan P Phase 2b §3: the Brillouin-zone form of the gap regulariser.
`H0.bloch` at Gamma reproduces the real-space H0 bit-exact (so the Gamma-only mesh reproduces the
existing regulariser); H(k) is Hermitian with real bands; the penalty is zero on a Gamma-only mesh,
finite and differentiable on the 3x3x3 mesh; the mesh is Gamma-centred with Gamma first."""
import importlib.util, os
import torch

from tests.extensions.dscc.test_model import _batch, _frame, _perovskite

_spec = importlib.util.spec_from_file_location("_pol_helpers", os.path.join(os.path.dirname(__file__), "test_polarisation_head.py"))
_pol = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_pol)
torch.set_default_dtype(torch.float64)


def _pristine_batch():
    return _batch([_frame(_perovskite(rattle=0.03, seed=5), [0, 0, 0, 0], 0)])


def _pieces(m, b):
    scalars, vectors = m.features(m.first_block(m._trunk_data(dict(b))))
    species = b["node_attrs"].argmax(-1)
    pos, cell = b["positions"], b["cell"].view(-1, 3, 3)
    src, dst = b["edge_index"][0], b["edge_index"][1]
    shifts = torch.einsum("ei,eij->ej", b["unit_shifts"], cell[b["batch"][src]])
    ev = pos[dst] - pos[src] + shifts
    return scalars, vectors, species, ev


def test_mesh_is_gamma_centred_gamma_first():
    m = _pol._model(coupling="mulliken")
    b = _pristine_batch()
    k = m.bz_mesh(b["cell"], 3)
    assert k.shape == (27, 3) and torch.allclose(k[0], torch.zeros(3))
    # closed under inversion
    for i in range(27):
        assert any(torch.allclose(-k[i], k[j], atol=1e-12) for j in range(27))


def test_bloch_at_gamma_is_h0_and_bands_are_real_hermitian():
    m = _pol._model(coupling="mulliken")
    b = _pristine_batch()
    scalars, vectors, species, ev = _pieces(m, b)
    n = int(species.shape[0])
    H0 = m.h0.batched(scalars, vectors, species, b["edge_index"], ev, b["batch"], 1, n)[0]
    k = m.bz_mesh(b["cell"], 3)
    Hk = m.h0.bloch(scalars, vectors, species, b["edge_index"], ev, k)
    assert Hk.shape == (27, 4 * n, 4 * n)
    assert torch.allclose(Hk[0].real, H0, atol=1e-12) and float(Hk[0].imag.abs().max()) < 1e-12
    for i in (1, 13, 26):
        assert float((Hk[i] - Hk[i].conj().transpose(0, 1)).abs().max()) < 1e-12
    eps = torch.linalg.eigvalsh(Hk)
    assert torch.isfinite(eps).all()
    # bands at -k equal bands at k (time reversal)
    for i in range(27):
        j = next(jj for jj in range(27) if torch.allclose(-k[i], k[jj], atol=1e-12))
        assert torch.allclose(eps[i], eps[j], atol=1e-9)


def test_penalty_zero_on_gamma_only_and_differentiable_on_mesh():
    m = _pol._model(coupling="mulliken")
    b = _pristine_batch()
    feats = tuple(t.detach() for t in m.features(m.first_block(m._trunk_data(dict(b)))))
    pen0, ro0 = m.bz_penalty([b], [feats], [m.bz_mesh(b["cell"], 1)])
    assert float(pen0) == 0.0
    pen, ro = m.bz_penalty([b], [feats], [m.bz_mesh(b["cell"], 3)])
    assert torch.isfinite(pen) and pen.requires_grad
    pen.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.h0.parameters() if p.requires_grad)
    assert set(ro) == {"bz_cbm_min_dev_eV", "bz_vbm_max_dev_eV", "gap_gamma_eV"}
    # consistency with the existing Gamma regulariser: the Gamma gap of the bands equals pristine_gap
    gap = m.pristine_gap(b, feats)
    assert abs(float(gap[0]) - ro["gap_gamma_eV"]) < 1e-9
