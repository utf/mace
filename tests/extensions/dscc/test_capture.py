"""Edge-bucket padding (`mace.modules.dscc.capture`): the prerequisite for CUDA-graph capture.

The whole scheme rests on one claim -- that a dummy edge beyond every cutoff contributes
EXACTLY zero, not approximately zero -- so these gates ask for bit-identity and not a
tolerance. If a future base or head grows a term that does not multiply by the cutoff, this
is where it shows up.
"""
import pytest
import torch

from mace.modules.dscc.capture import GRANULARITY, dummy_unit_shift, edge_bucket, pad_batch, pad_edges
from tests.extensions.dscc.test_model import _batch
from tests.extensions.dscc.test_screening import _o5
from tests.extensions.dscc.test_w6 import VACP, _o3, _w6

torch.set_default_dtype(torch.float64)


class TestBucketRule:
    def test_capacity_has_headroom_and_granularity(self):
        for n in (1, 100, 808, 9122):
            cap = edge_bucket(n)
            assert cap >= n and cap % GRANULARITY == 0

    def test_hysteresis_keeps_a_bucket_that_still_fits(self):
        cap = edge_bucket(9122)
        assert edge_bucket(9100, cap) == cap          # a relaxation drifting by a few edges
        assert edge_bucket(9122 + GRANULARITY, cap) > cap    # genuine growth re-buckets
        assert edge_bucket(100, cap) < cap                   # a genuine shrink does too

    def test_the_dummy_shift_clears_the_cutoff(self):
        for cell in (torch.eye(3) * 5.6, torch.eye(3) * 14.3,
                     torch.tensor([[5.6, 0.0, 0.0], [0.2, 22.0, 0.0], [0.1, 0.3, 30.0]])):
            n = dummy_unit_shift(cell, 10.0)
            assert float((n.to(cell.dtype) @ cell).norm()) > 10.0

    def test_selection_keeps_order_and_marks_the_live_edges(self):
        ei = torch.stack([torch.arange(6), torch.arange(6).flip(0)])
        us = torch.zeros(6, 3, dtype=torch.long)
        keep = torch.tensor([True, False, True, True, False, False])
        out_ei, _, live = pad_edges(ei, us, 8, torch.tensor([9, 9, 9]), keep)
        assert out_ei.shape == (2, 8) and int(live.sum()) == 3
        assert out_ei[0, :3].tolist() == [0, 2, 3]          # stable, in the original order


class TestPaddingIsExact:
    @pytest.mark.parametrize("kind", ["o3", "w6"])
    def test_padding_leaves_every_output_unchanged(self, kind):
        """Bit-identical, both heads. `w6` is the one that matters for O1: its hopping MLP is
        deduplicated by unordered node pair, and every dummy edge is the pair (0, 0), which
        adds one row to the unique set and must not reach a real pair's input."""
        model = _o3() if kind == "o3" else _w6()
        batch = _batch([VACP])
        n = int(batch["edge_index"].shape[1])
        ref = model(dict(batch), compute_force=True)
        padded = model(pad_batch(dict(batch), edge_bucket(n), model.r_cut), compute_force=True)
        assert float((ref["energy"] - padded["energy"]).abs().max()) == 0.0
        assert float((ref["forces"] - padded["forces"]).abs().max()) == 0.0
        assert float((ref["dq"] - padded["dq"]).abs().max()) == 0.0

    def test_a_capacity_far_above_the_count_is_still_exact(self):
        model = _o3()
        batch = _batch([VACP])
        ref = model(dict(batch), compute_force=True)
        padded = model(pad_batch(dict(batch), 4 * GRANULARITY, model.r_cut), compute_force=True)
        assert float((ref["forces"] - padded["forces"]).abs().max()) == 0.0

    @pytest.mark.parametrize("kw", [{"linear": True}, {"mlp": True}])
    def test_the_crystal_field_contributes_nothing_from_a_dummy_edge(self, kw):
        """O5's damped Coulomb is a THIRD consumer of the edge list, and its claim to the
        capture path is that it is two-body like the others. A dummy edge carries a real
        species charge into `V_0`, so `erf(r/a)/r . f_cut` and both its derivatives have to be
        zero past the cutoff -- the clamp alone is not enough, which is how the `T''` sign
        error at the clamp was found.

        WHAT IS ASKED IS COUNT-INDEPENDENCE, NOT BIT-IDENTITY. `V` and `E` are bit-identical;
        the traceless Hessian and (through it) the autograd forces move in the last bit,
        because adding dummy edges repartitions the REAL edges across the scatter's reduction
        and the head takes autograd forces here (the analytic route refuses O5). A leak would
        grow with the number of dummies. This runs 216, 1240 and 3288 of them and requires the
        discrepancy not to move at all, which separates reassociation from contamination in a
        way bit-identity cannot.
        """
        model = _o5(seed=3, **kw)
        with torch.no_grad():                     # a zero-init shift would pad exactly for free
            model.h0.sk.polarity.add_(0.05)
            if kw.get("linear"):
                model.h0.cf_scale.normal_(0.0, 0.05)
            else:
                model.h0.cf_site[-1].weight.normal_(0.0, 0.05)
        batch = _batch([VACP])
        n = int(batch["edge_index"].shape[1])
        sp = batch["node_attrs"].argmax(dim=-1)
        V0, E0, _ = model.h0.cfield(sp, batch["edge_index"], model.head_edge_vectors(batch), len(sp))
        ref = model(dict(batch), compute_force=True)
        seen = set()
        for cap in (edge_bucket(n), 4 * GRANULARITY, 8 * GRANULARITY):
            padded = pad_batch(dict(batch), cap, model.r_cut)
            spp = padded["node_attrs"].argmax(dim=-1)
            V, E, _ = model.h0.cfield(spp, padded["edge_index"],
                                      model.head_edge_vectors(padded), len(spp))
            assert float((V - V0).abs().max()) == 0.0        # the potential is exact
            assert float((E - E0).abs().max()) == 0.0        # so is the field
            out = model(padded, compute_force=True)
            assert float((ref["energy"] - out["energy"]).abs().max()) == 0.0
            assert float((ref["dq"] - out["dq"]).abs().max()) == 0.0
            seen.add(float((ref["forces"] - out["forces"]).abs().max()))
        assert len(seen) == 1, f"force discrepancy moves with the dummy count: {sorted(seen)}"
        assert max(seen) < 1e-15, f"{max(seen)} is too large for reassociation"

    def test_a_multi_graph_batch_is_refused(self):
        with pytest.raises(ValueError):
            pad_batch(dict(_batch([VACP, VACP])), 2048, 10.0)


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")


@cuda_only
class TestGraphedStep:
    """Three CUDA graphs around a CPU segment, against the eager model.

    The tolerance is 1e-9 and not the base's ~1e-6 nondeterminism floor: a captured graph
    replays ONE recorded kernel order, so it is more reproducible than the eager path, and
    every quantity here is compared against a single eager evaluation of the same frame.
    """

    @pytest.fixture(scope="class")
    def step(self):
        from mace.modules.dscc.capture import GraphedStep
        model = _o3().cuda().eval()
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _batch([VACP]).items()}
        return model, data, GraphedStep(model, data)

    def test_it_reproduces_the_eager_model(self, step):
        model, data, graphed = step
        ref = model(dict(data), compute_force=True)
        out = graphed(data)
        assert float((ref["energy"] - out["energy"]).abs().max()) < 1e-9
        assert float((ref["head_energy"] - out["head_energy"]).abs().max()) < 1e-9
        assert float((ref["dq"] - out["dq"]).abs().max()) < 1e-9
        assert float((ref["forces"] - out["forces"]).abs().max()) < 1e-9
        assert float(ref["forces"].abs().max()) > 1e-3          # not a comparison of zeros

    def test_a_moved_frame_replays_through_the_same_graphs(self, step):
        """The edge list's VALUES change from step to step; only its length is fixed. If the
        graphs had baked in the first frame's neighbours this is what would catch it."""
        model, _, graphed = step
        moved = VACP.copy()
        moved.positions[3, 1] += 0.07
        moved.positions[10, 0] -= 0.05
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _batch([moved]).items()}
        ref = model(dict(data), compute_force=True)
        out = graphed(data)
        assert float((ref["energy"] - out["energy"]).abs().max()) < 1e-9
        assert float((ref["forces"] - out["forces"]).abs().max()) < 1e-9

    def test_the_base_bucket_holds_its_live_edges(self, step):
        _, data, graphed = step
        out = graphed(data)
        assert 0 < int(out["base_edges_live"]) <= graphed.base_capacity

    def test_a_frame_beyond_the_bucket_is_refused(self, step):
        model, data, graphed = step
        big = dict(data)
        big["edge_index"] = torch.cat([data["edge_index"]] * 8, dim=1)
        assert not graphed.fits(big)
        with pytest.raises(RuntimeError):
            graphed(big)

    @pytest.mark.parametrize("kw", [{"linear": True}, {"mlp": True}])
    def test_an_o5_head_now_captures_through_the_vjp(self, kw):
        """O5 was REFUSED, because the captured tail was the closed-form gradient and the
        closed form has no `dV/dR`. Before that refusal existed it captured and replayed
        19 meV/A wrong. It now captures correctly: the tail dispatches to
        `h0_position_gradient_vjp`, which differentiates what the assembly actually computed
        instead of a transcription of it."""
        from mace.modules.dscc.capture import GraphedStep
        model = _o5(seed=4, **kw).cuda().eval()
        with torch.no_grad():
            model.h0.sk.polarity.add_(0.05)
            if kw.get("linear"):
                model.h0.cf_scale.normal_(0.0, 0.05)
            else:
                model.h0.cf_site[-1].weight.normal_(0.0, 0.05)
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _batch([VACP]).items()}
        graphed = GraphedStep(model, data)
        assert graphed.fits(data)
        ref = model(dict(data), compute_force=True)
        out = graphed(data)
        assert float(ref["forces"].abs().max()) > 1e-3
        assert float((ref["energy"] - out["energy"]).abs().max()) < 1e-9
        assert float((ref["dq"] - out["dq"]).abs().max()) < 1e-9
        # The number that was 19 meV/A wrong.
        assert float((ref["forces"] - out["forces"]).abs().max()) < 1e-9

    def test_a_block_zero_reading_head_captures(self):
        """THE PRODUCTION ROUTE. W6 reads the base's block 0, so its position derivative runs
        through the base's graph and no closed form is available -- which is why capture
        refused it and why the decoupled lineage was pursued at all. It does not need one: at
        fixed cotangent the gradient is one backward, and a backward captures.

        This is the gate that says the shipped model can be captured."""
        from mace.modules.dscc.capture import GraphedStep
        model = _w6(0).cuda().eval()
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _batch([VACP]).items()}
        graphed = GraphedStep(model, data)
        assert graphed.fits(data)
        ref = model(dict(data), compute_force=True)
        out = graphed(data)
        assert float(ref["forces"].abs().max()) > 1e-3
        assert float((ref["energy"] - out["energy"]).abs().max()) < 1e-9
        assert float((ref["dq"] - out["dq"]).abs().max()) < 1e-9
        assert float((ref["forces"] - out["forces"]).abs().max()) < 1e-9

    def test_a_moved_block_zero_frame_replays(self):
        """A replay on new positions, not a re-capture: the block-0 forward and its backward
        are inside the graph, so a moved frame must go through them."""
        from mace.modules.dscc.capture import GraphedStep
        model = _w6(0).cuda().eval()
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _batch([VACP]).items()}
        graphed = GraphedStep(model, data)
        moved = VACP.copy()
        moved.positions[3, 1] += 0.04
        moved.positions[11, 0] -= 0.03
        d2 = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _batch([moved]).items()}
        ref = model(dict(d2), compute_force=True)
        out = graphed(d2)
        assert float((ref["forces"] - out["forces"]).abs().max()) < 1e-9
        assert float((ref["energy"] - out["energy"]).abs().max()) < 1e-9

    def test_the_two_refusals_are_separate_and_each_is_honest(self):
        """`analytic_refusal` says what the CLOSED FORM cannot differentiate; `capture_refusal`
        says what cannot be CAPTURED. They are different questions and the second is a shorter
        list -- capture has the VJP. What must never happen is a head that neither refuses and
        whose gradient silently omits a term, which is the 19 meV/A bug."""
        from mace.modules.dscc.analytic import analytic_refusal, capture_refusal
        assert analytic_refusal(_o3()) is None and capture_refusal(_o3()) is None
        for kw in ({"linear": True}, {"mlp": True}):
            m = _o5(**kw)
            assert analytic_refusal(m) is not None, "the kernel has no dV/dR"
            assert capture_refusal(m) is None, "but the VJP serves it"
            out = m(_batch([VACP]), compute_force=True)      # autograd route, not the kernel
            assert float(out["forces"].abs().max()) > 1e-3
        w = _w6(0)
        assert analytic_refusal(w) is not None and capture_refusal(w) is None

    def test_a_genuinely_COUPLED_head_is_still_refused(self):
        """This gate used to refuse `_w6` for "not decoupled". That is no longer the reason
        anything is refused: a block-0-reading head captures through the VJP, and W6 is the
        production route. What capture still cannot take is a head that SOLVES -- the SCF is a
        host iteration with a value-dependent trip count, so there is no fixed kernel sequence
        to record."""
        from mace.modules.dscc.capture import GraphedStep
        from mace.modules.dscc.analytic import capture_refusal
        model = _w6(0, scf_free=False).cuda().eval()
        assert capture_refusal(model) is not None
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _batch([VACP]).items()}
        with pytest.raises(ValueError):
            GraphedStep(model, data)

    def test_releasing_a_capture_restores_the_model(self):
        """Capture turns the O1 pair deduplication off for the life of the object. If it were
        never turned back on, every model that had ever been captured would stay 2-3x slower on
        its eager path for the rest of the process -- silently, since the values agree."""
        from mace.modules.dscc.capture import GraphedStep
        model = _w6(0).cuda().eval()
        assert model.h0.sk.dedupe_pairs is not False
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _batch([VACP]).items()}
        g = GraphedStep(model, data)
        assert model.h0.sk.dedupe_pairs is False, "capture must switch it off"
        g.release()
        assert model.h0.sk.dedupe_pairs is True, "release must switch it back"
        g.release()                                   # idempotent
        with pytest.raises(RuntimeError, match="released"):
            g(data)

    def test_the_pair_dedupe_switch_changes_no_value(self):
        """Capture turns off the O1 pair deduplication because `torch.unique` is a
        value-dependent shape. The per-edge path must be the SAME function -- not close, the
        same -- or every captured W6 force is quietly wrong by the difference."""
        model = _w6(0)
        batch = _batch([VACP])
        model.h0.sk.dedupe_pairs = True
        a = model(dict(batch), compute_force=True)
        model.h0.sk.dedupe_pairs = False
        b = model(dict(batch), compute_force=True)
        model.h0.sk.dedupe_pairs = True
        assert float(a["energy"].abs().max()) > 1e-6
        # The energy and the charges are BIT-IDENTICAL: same function, same inputs.
        assert float((a["energy"] - b["energy"]).abs().max()) == 0.0
        assert float((a["dq"] - b["dq"]).abs().max()) == 0.0
        # The forces move in the last bit and only there. The hop MLP's GEMM has a different
        # number of rows in the two paths (unique pairs against all edges), so its reduction is
        # tiled differently; that is reassociation, not a different function. 3.5e-17 against
        # forces of order 0.5 eV/A is ~1e-16 relative, one ULP of float64.
        df = float((a["forces"] - b["forces"]).abs().max())
        assert df < 1e-15, f"{df} is too large to be reassociation"

    def test_a_different_cell_is_refused(self, step):
        """The reciprocal plan, the Madelung constant and the padding translation are cell
        constants held on the capture. Replaying another cell through it would be wrong by
        millielectronvolts and raise nothing -- so it raises."""
        model, data, graphed = step
        other = dict(data)
        other["cell"] = data["cell"] * 1.01
        assert not graphed.fits(other)
        with pytest.raises(RuntimeError):
            graphed(other)

    def test_a_different_charge_state_is_refused(self, step):
        model, data, graphed = step
        other = dict(data)
        other["carrier_counts"] = data["carrier_counts"].clone()
        other["carrier_counts"][0] += 1
        assert not graphed.fits(other)
