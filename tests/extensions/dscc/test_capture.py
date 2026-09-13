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
