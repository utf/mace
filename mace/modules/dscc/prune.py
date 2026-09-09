"""v5 W1.2: prune a foundation ScaleShiftMACE to the elements the head will see.

The fine-tuned base keeps every element present in its fine-tuning and replay data (dozens);
the D-SCC head's per-species tables (covalent radii, `U_max`, valence) are deliberately host-free
and raise on elements they do not cover, and its one-hot width follows the base's table. The
element-indexed parts of a ScaleShiftMACE are the node embedding (`Linear(n_el x 0e -> k x 0e)`),
the interaction blocks' `source_embedding` / `target_embedding` (same shape, the non-linear
blocks), the atomic energies `[heads, n_el]` and the `atomic_numbers` buffer; the symmetric
contractions of the MH-1 / MPA-0 family carry an element axis of length 1 (agnostic) and the
radial embedding and pair repulsion index full-periodic-table tables by Z. Pruning slices those
rows; every prediction on a frame made of the kept elements is unchanged (tested to 1e-12)."""
from __future__ import annotations

import copy
from typing import Sequence

import torch
from e3nn import o3


def _prune_linear_in(linear: o3.Linear, keep_idx: torch.Tensor, n_el: int) -> o3.Linear:
    """A new `Linear(len(keep) x 0e -> out)` reproducing a `Linear(n_el x 0e -> out)` on the
    kept one-hot rows. e3nn's path normalisation scales a Linear's output by `1/sqrt(mul_in)`
    (path_normalization "element"), so the kept rows are rescaled by `sqrt(n_keep / n_el)`."""
    irreps_in = o3.Irreps(linear.irreps_in)
    assert len(irreps_in) == 1 and irreps_in[0].mul == n_el and irreps_in[0].ir.l == 0, irreps_in
    assert getattr(linear, "path_normalization", "element") == "element", linear.path_normalization
    w = linear.weight.detach().reshape(n_el, -1)                    # one instruction: path_shape (mul_in, mul_out)
    new = o3.Linear(o3.Irreps(f"{len(keep_idx)}x0e"), linear.irreps_out, internal_weights=True, shared_weights=True)
    with torch.no_grad():
        new.weight.copy_((w[keep_idx] * (len(keep_idx) / n_el) ** 0.5).reshape(-1).to(new.weight.dtype))
    return new


def prune_elements(model: torch.nn.Module, keep: Sequence[int]) -> torch.nn.Module:
    """Return a deep copy of `model` whose element table is `keep` (atomic numbers, any order;
    stored sorted as MACE does). Raises if an element of `keep` is not in the model."""
    model = copy.deepcopy(model)
    zs = [int(z) for z in model.atomic_numbers.tolist()]
    keep_sorted = sorted(int(z) for z in keep)
    missing = [z for z in keep_sorted if z not in zs]
    if missing:
        raise ValueError(f"elements {missing} are not in the model's table {zs}")
    keep_idx = torch.tensor([zs.index(z) for z in keep_sorted], dtype=torch.long)
    n_el = len(zs)
    model.node_embedding.linear = _prune_linear_in(model.node_embedding.linear, keep_idx, n_el)
    for block in model.interactions:
        for name in ("source_embedding", "target_embedding"):
            lin = getattr(block, name, None)
            if isinstance(lin, o3.Linear) and o3.Irreps(lin.irreps_in)[0].mul == n_el:
                setattr(block, name, _prune_linear_in(lin, keep_idx, n_el))
        # Blocks whose skip connection is a tensor product with the one-hot (the older
        # RealAgnosticResidual family) are not handled here: refuse rather than mis-slice.
        skip = getattr(block, "skip_tp", None)
        if skip is not None and not isinstance(skip, o3.Linear):
            raise NotImplementedError(f"skip_tp of type {type(skip).__name__} carries per-element weights; pruning not implemented")
        block.node_attrs_irreps = o3.Irreps(f"{len(keep_idx)}x0e") if hasattr(block, "node_attrs_irreps") else None
    for prod in model.products:
        sc = prod.symmetric_contractions
        for c in sc.contractions:
            for p in c.parameters():
                if p.shape[0] not in (1,):
                    raise NotImplementedError("per-element symmetric-contraction weights; pruning not implemented")
    ae = model.atomic_energies_fn.atomic_energies.detach()
    model.atomic_energies_fn.atomic_energies = torch.nn.Parameter(ae[..., keep_idx].clone(), requires_grad=ae.requires_grad) \
        if isinstance(model.atomic_energies_fn.atomic_energies, torch.nn.Parameter) else ae[..., keep_idx].clone()
    if not isinstance(model.atomic_energies_fn.atomic_energies, torch.nn.Parameter):
        model.atomic_energies_fn.register_buffer("atomic_energies", ae[..., keep_idx].clone())
    model.atomic_numbers = torch.tensor(keep_sorted, dtype=torch.int64)
    return model
