"""Section 1.1: batches whose frames all have the same atom count.

WHY IT EXISTS. The counting head's `[B, 4n, 4n]` solver needs one `n` per batch, and a
shuffled loader over this dataset gives it one about one batch in six -- 2300-odd 79-atom
frames, 544 at 80, 34 at 159/160, drawn at random. Without grouping the batched path is
almost never taken and section 1.1 does not reach training at all.

THE GROUPING IS BY EXACT ATOM COUNT, not by a size class. 79 and 80 are different `n` even
though they are the same "small cell", and a batch mixing them cannot be stacked.

WHAT IS AND IS NOT RANDOM. Within a group the frames are permuted every epoch, and then the
BATCHES ARE PERMUTED ACROSS GROUPS -- so a step is as likely to be a large-cell step early in
the epoch as late. Grouping without that second shuffle would put every 159-atom frame in a
run of consecutive steps at a fixed point in the epoch, which is a systematic change to the
optimiser's trajectory dressed up as a speed fix.

THE TAIL BATCH IS KEPT. `drop_last` on a grouped sampler drops up to `batch_size - 1` frames
from EVERY group, and the 159-atom group has 34 frames: dropping two of them per epoch would
discard 6% of the population the two-size upweight exists to protect. A short batch is still
size-uniform, so the batched solver takes it.

The realised-share bookkeeping is untouched: it is an aggregate over the epoch's frames on
the weight column the loss term reads, and which batch a frame landed in does not enter it.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

import torch

__all__ = ["frame_sizes", "SizeGroupedBatchSampler"]


def frame_sizes(dataset) -> List[int]:
    """Atom count per dataset index.

    Reads `num_nodes` when the item has it and falls back to the position count, so it works
    for `AtomicData`, for a `ConcatDataset` of lists of them, and for anything else that
    carries positions. A dataset item that has neither raises here rather than silently
    grouping everything into one bucket.
    """
    sizes: List[int] = []
    for item in dataset:
        n = getattr(item, "num_nodes", None)
        if n is None:
            positions = getattr(item, "positions", None)
            if positions is None:
                raise ValueError(
                    "the size-grouped sampler cannot read an atom count off a dataset item "
                    f"of type {type(item).__name__}; it needs `num_nodes` or `positions`")
            n = int(positions.shape[0])
        sizes.append(int(n))
    return sizes


class SizeGroupedBatchSampler(torch.utils.data.Sampler):
    """Yields lists of dataset indices, every list of one atom count."""

    def __init__(self, sizes: Sequence[int], batch_size: int, shuffle: bool = True,
                 drop_last: bool = False,
                 generator: Optional[torch.Generator] = None) -> None:
        # `torch.utils.data.Sampler.__init__` takes no argument in this torch; calling it
        # with the historical `data_source` positional raises. Skipped rather than guessed.
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.generator = generator
        groups: Dict[int, List[int]] = {}
        for i, n in enumerate(sizes):
            groups.setdefault(int(n), []).append(i)
        # Sorted so the epoch's batch count and the log line are reproducible.
        self.groups = {n: groups[n] for n in sorted(groups)}
        self._length = sum(self._n_batches(len(v)) for v in self.groups.values())

    def _n_batches(self, n_items: int) -> int:
        if self.drop_last:
            return n_items // self.batch_size
        return (n_items + self.batch_size - 1) // self.batch_size

    def describe(self) -> str:
        parts = ", ".join(f"{n} atoms x{len(v)}" for n, v in self.groups.items())
        return (f"{len(self.groups)} size groups ({parts}) -> {self._length} batches "
                f"of at most {self.batch_size}")

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[List[int]]:
        batches: List[List[int]] = []
        for indices in self.groups.values():
            order = list(indices)
            if self.shuffle:
                perm = torch.randperm(len(order), generator=self.generator).tolist()
                order = [order[p] for p in perm]
            limit = self._n_batches(len(order)) * self.batch_size
            for start in range(0, min(limit, len(order)), self.batch_size):
                batches.append(order[start: start + self.batch_size])
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[p] for p in perm]
        return iter(batches)
