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

    def test_a_coupled_head_is_refused(self):
        from mace.modules.dscc.capture import GraphedStep
        model = _w6().cuda().eval()               # route B', not decoupled
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _batch([VACP]).items()}
        with pytest.raises(ValueError):
            GraphedStep(model, data)

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
