"""Head-homogeneous batching for multi-head training.

Every batch holds samples from a single head, so a batch's per-config property
weights are uniform and the training step can skip the field derivatives (Born
charges, polarizability) for heads whose frames carry no response labels.
"""
from typing import Iterator, List, Sequence

import torch
from torch.utils.data import ConcatDataset, Sampler


def concat_group_ids(dataset: ConcatDataset) -> List[int]:
    """Group id (= position in the concatenation) of every sample of a ConcatDataset."""
    ids: List[int] = []
    for g, ds in enumerate(dataset.datasets):
        ids.extend([g] * len(ds))
    return ids


class HeadGroupedBatchSampler(Sampler[List[int]]):
    """Yield index batches that never mix groups; reshuffled on every iteration."""

    def __init__(
        self,
        group_ids: Sequence[int],
        batch_size: int,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.group_ids = list(group_ids)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.groups = {}
        for idx, g in enumerate(self.group_ids):
            self.groups.setdefault(g, []).append(idx)

    def _batches(self) -> List[List[int]]:
        batches: List[List[int]] = []
        for members in self.groups.values():
            perm = torch.randperm(len(members), generator=self.generator).tolist()
            order = [members[i] for i in perm]
            for start in range(0, len(order), self.batch_size):
                chunk = order[start : start + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                batches.append(chunk)
        perm = torch.randperm(len(batches), generator=self.generator).tolist()
        return [batches[i] for i in perm]

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        n = 0
        for members in self.groups.values():
            q, r = divmod(len(members), self.batch_size)
            n += q + (0 if (r == 0 or self.drop_last) else 1)
        return n
