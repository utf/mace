"""Section 2.5 of the Stage A' spec: the frozen base's outputs are computed once.

WHY A CACHE IS EXACT HERE AND NOT MERELY CONVENIENT. Stage B trains the carrier head against a
base that does not move: `--base_lr_factor 0.0`, and `assert_base_frozen` checks it after
every step. A function of a frozen network and a fixed geometry is a constant, so `E_base`
and `F_base` per frame can be computed once and looked up. What CANNOT be cached is anything
the head differentiates through: the head consumes the first interaction block's invariant
features and needs their position gradient for forces, so that block is recomputed every
step (in float32, section 2.5) and everything after it is read from the cache.

WHAT IS CACHED, per frame, keyed by a content hash of the geometry:

  * `energy`      the trunk's base energy, `e0 + inter_e`, WITHOUT the long-range host term
                  (that term is recomputed each step from the cached host charges so the
                  epoch gate `lr_start_epoch` and its position gradient keep working);
  * `forces`      the trunk's base forces, `-d(inter_e)/dR`;
  * `feats_rest`  the invariant readouts of every block after the first, which only the
                  long-range charge MLPs consume -- and under `lr_detach_density` they carry
                  no gradient, so a cached value is exact.

WHAT MAKES A CACHED RUN REFUSE TO START. The head must read the first block only
(`spectral_first_shell`), the base must be frozen, and with the long-range branch on both
`lr_detach_density` and `lr_freeze` must hold; otherwise something the cache treats as a
constant would be training, and the run would be training against stale values with no
error anywhere. Every one of these is checked by `require_cacheable`.

THE CHECKSUM. SHA-256 over the base branch's tensors, the trunk normalisation constants and
the frozen long-range parameters, written into `model.base_cache_checksum` so a checkpoint
records which base its head was trained against, and used to name and validate the cache
file on disk. A cache whose checksum does not match the model is discarded, not trusted.

THE DRIFT GUARD. Once per epoch a random training frame is run through the FULL forward
with the cache disabled and compared against the cached values. A base that has moved --
an optimiser group that escaped the freeze, a second optimiser, weight decay without a
gradient -- shows up here as a number, and the run stops rather than continue against a
target that has drifted. This is the same fault class `assert_base_frozen` covers, tested on
the outputs instead of the weights.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["frame_key", "attach_frame_keys", "base_checksum", "BaseOutputCache",
           "require_cacheable", "build_base_cache", "check_drift", "profile_step"]

CACHE_FORMAT = 1


# ----------------------------------------------------------------------------- frame keys

def frame_key(atomic_numbers, positions, cell) -> int:
    """A content hash of the geometry, as a signed 64-bit integer.

    Positions are rounded to 1e-6 A and the cell to 1e-6 A before hashing, so a frame
    re-read from a file with a different float formatting still keys to the same entry; two
    frames that differ by more than that are different frames. Label-free by construction:
    only numbers, positions and cell enter.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(np.asarray(atomic_numbers, dtype=np.int64).tobytes())
    h.update(np.round(np.asarray(positions, dtype=np.float64), 6).tobytes())
    h.update(np.round(np.asarray(cell, dtype=np.float64), 6).reshape(-1).tobytes())
    return int(np.frombuffer(h.digest(), dtype=np.int64)[0])


def attach_frame_keys(dataset: Sequence, z_table=None) -> int:
    """Set `frame_key` on every AtomicData in `dataset`; returns how many were set.

    The key is a one-element long tensor so torch_geometric collates it to `[n_graphs]`,
    and the model finds it in `data["frame_key"]`.
    """
    n = 0
    for d in dataset:
        numbers = d.node_attrs.argmax(dim=-1).cpu().numpy()
        if z_table is not None:
            numbers = np.array([z_table.zs[int(i)] for i in numbers], dtype=np.int64)
        key = frame_key(numbers, d.positions.detach().cpu().numpy(),
                        d.cell.detach().cpu().numpy())
        d.frame_key = torch.tensor([key], dtype=torch.long)
        n += 1
    return n


# ------------------------------------------------------------------------------- checksum

def base_checksum(model) -> bytes:
    """SHA-256 over everything the cache treats as constant."""
    from mace.modules.defect_stage import _base_state

    h = hashlib.sha256()
    for name, tensor in sorted(_base_state(model).items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().to(torch.float64).contiguous().numpy().tobytes()
                 if tensor.is_floating_point()
                 else tensor.detach().cpu().contiguous().numpy().tobytes())
    for block in getattr(model, "interactions", []):
        h.update(repr(float(block.avg_num_neighbors)).encode())
    # The long-range branch, when frozen, is a constant too; the cached host charges are
    # its output. Hash its parameters so a cache built against one set is not read by
    # a model carrying another.
    charges = getattr(model, "latent_charges", None)
    if charges is not None:
        for name, p in sorted(charges.named_parameters()):
            h.update(name.encode())
            h.update(p.detach().cpu().to(torch.float64).contiguous().numpy().tobytes())
    # The later-block invariant readouts feed only the long-range charges; cached, so fixed.
    readouts = getattr(model, "defect_feature_readouts", None)
    if readouts is not None:
        for i, ro in enumerate(readouts):
            if i == 0:
                continue
            for name, p in sorted(ro.named_parameters()):
                h.update(f"{i}.{name}".encode())
                h.update(p.detach().cpu().to(torch.float64).contiguous().numpy().tobytes())
    h.update(str(CACHE_FORMAT).encode())
    return h.digest()


def checksum_to_buffer(digest: bytes) -> torch.Tensor:
    return torch.tensor(list(digest), dtype=torch.uint8)


# ---------------------------------------------------------------------------------- cache

class BaseOutputCache:
    """Per-frame base outputs, looked up by `frame_key`, resident on one device."""

    def __init__(self, checksum: bytes, device="cpu"):
        self.checksum = bytes(checksum)
        self.device = torch.device(device)
        self.entries: Dict[int, Dict[str, torch.Tensor]] = {}

    # -- storage -----------------------------------------------------------------------
    def put(self, key: int, energy: torch.Tensor, forces: torch.Tensor,
            feats_rest: Optional[torch.Tensor], q_host: Optional[torch.Tensor]) -> None:
        self.entries[int(key)] = dict(
            energy=energy.detach().to(self.device, torch.float64).reshape(()),
            forces=forces.detach().to(self.device, torch.float64),
            feats_rest=(None if feats_rest is None
                        else feats_rest.detach().to(self.device, torch.float64)),
            q_host=(None if q_host is None
                    else q_host.detach().to(self.device, torch.float64)),
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, key) -> bool:
        return int(key) in self.entries

    def to(self, device) -> "BaseOutputCache":
        self.device = torch.device(device)
        for e in self.entries.values():
            for k, v in e.items():
                if v is not None:
                    e[k] = v.to(self.device)
        return self

    # -- lookup ------------------------------------------------------------------------
    def lookup(self, keys: torch.Tensor):
        """`(energy [n_graphs], forces [n_atoms, 3], feats_rest [n_atoms, d] | None,
        q_host [n_atoms] | None)` for the graphs in `keys`, in batch order.

        A missing key raises. A cache that silently fell back to a full forward for some
        frames would make the run's cost depend on the shuffle and hide a keying bug.
        """
        energies, forces, feats, qh = [], [], [], []
        for k in keys.tolist():
            e = self.entries.get(int(k))
            if e is None:
                raise KeyError(
                    f"frame_key {k} is not in the base cache ({len(self.entries)} entries); "
                    "the cache was built on a different dataset, or the keys were not "
                    "attached to this loader's frames")
            energies.append(e["energy"])
            forces.append(e["forces"])
            feats.append(e["feats_rest"])
            qh.append(e["q_host"])
        energy = torch.stack(energies)
        force = torch.cat(forces, dim=0)
        feats_rest = None if feats[0] is None else torch.cat(feats, dim=0)
        q_host = None if qh[0] is None else torch.cat(qh, dim=0)
        return energy, force, feats_rest, q_host

    # -- persistence -------------------------------------------------------------------
    def save(self, path) -> None:
        payload = dict(format=CACHE_FORMAT, checksum=self.checksum,
                       entries={k: {kk: (None if vv is None else vv.cpu())
                                    for kk, vv in v.items()}
                                for k, v in self.entries.items()})
        torch.save(payload, path)

    @classmethod
    def load(cls, path, checksum: bytes, device="cpu") -> Optional["BaseOutputCache"]:
        """The cache at `path`, or None when it does not match `checksum`."""
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            return None
        if payload.get("format") != CACHE_FORMAT or payload.get("checksum") != bytes(checksum):
            logging.warning("Base cache at %s does not match this model's checksum; "
                            "it will be rebuilt", path)
            return None
        cache = cls(checksum, device=device)
        for k, v in payload["entries"].items():
            cache.entries[int(k)] = {kk: (None if vv is None else vv.to(cache.device))
                                     for kk, vv in v.items()}
        return cache


def require_cacheable(model) -> None:
    """Refuse a configuration under which something cached would be training."""
    problems = []
    if not getattr(model, "spectral_head", False):
        problems.append("the carrier head is the attention head, which reads every block")
    if not getattr(model, "spectral_first_shell", False):
        problems.append("spectral_first_shell is off: the head reads every block's features, "
                        "and only the first block is recomputed")
    base = [p for n, p in model.named_parameters()
            if n.startswith(("interactions.", "products.", "readouts.", "node_embedding.",
                             "radial_embedding."))]
    if any(p.requires_grad for p in base):
        problems.append("the base branch has trainable parameters; cache only a frozen base "
                        "(--base_lr_factor 0.0 and the head-only mask)")
    if getattr(model, "use_long_range", False):
        if not getattr(model, "lr_detach_density", False):
            problems.append("use_long_range without lr_detach_density: the charge MLPs "
                            "would need gradients through features that are cached")
        if not getattr(model, "lr_freeze", False):
            problems.append("use_long_range without lr_freeze: the host charges are cached "
                            "as constants but their MLP would be training")
    readouts = getattr(model, "defect_feature_readouts", None)
    if readouts is not None:
        for i, ro in enumerate(readouts):
            if i > 0 and any(p.requires_grad for p in ro.parameters()):
                problems.append(f"defect_feature_readouts[{i}] is trainable; its output is "
                                "cached (freeze it: only the long-range charges read it, "
                                "and they are detached)")
    if problems:
        raise RuntimeError("the base cache cannot be used with this configuration:\n  - "
                           + "\n  - ".join(problems))


# ---------------------------------------------------------------------------------- build

def _forward_full(model, batch, device, compute_force: bool):
    """The uncached forward, with the long-range branch OFF, in eval mode."""
    was_training = model.training
    cache = getattr(model, "_base_cache", None)
    lr = bool(getattr(model, "use_long_range", False))
    model.eval()
    model._base_cache = None
    model.use_long_range = False
    try:
        data = batch.to(device).to_dict()
        with torch.enable_grad():
            out = model(data, training=False, compute_force=compute_force)
    finally:
        model.use_long_range = lr
        model._base_cache = cache
        model.train(was_training)
    return out


def _host_charges(model, out, batch, device):
    """`q_host` for every atom of `batch`, from the (frozen) host-charge MLP, or None."""
    charges = getattr(model, "latent_charges", None)
    if charges is None or not getattr(model, "use_long_range", False):
        return None
    feats = out["defect_features"]
    with torch.no_grad():
        n_graphs = int(batch.ptr.numel() - 1)
        host_input = feats.to(next(charges.parameters()).dtype)
        host_raw = charges.host_charge(host_input).squeeze(-1)
        from mace.tools.scatter import scatter_mean
        host_mean = scatter_mean(host_raw, batch.batch.to(device), dim=0, dim_size=n_graphs)
        return host_raw - host_mean[batch.batch.to(device)]


def build_base_cache(model, loaders: Iterable, device, path=None) -> BaseOutputCache:
    """One full forward per batch of every loader, then never again."""
    require_cacheable(model)
    digest = base_checksum(model)
    if path is not None:
        cached = BaseOutputCache.load(path, digest, device=device)
        if cached is not None:
            logging.info("Base cache: loaded %d frames from %s", len(cached), path)
            with torch.no_grad():
                model.base_cache_checksum.copy_(checksum_to_buffer(digest))
            return cached
    cache = BaseOutputCache(digest, device=device)
    cfd = int(getattr(model, "carrier_feature_dim", 0))
    t0 = time.time()
    n_batches = 0
    for loader in loaders:
        for batch in loader:
            if not hasattr(batch, "frame_key"):
                raise RuntimeError("the loader's frames carry no frame_key; call "
                                   "attach_frame_keys on the dataset first")
            out = _forward_full(model, batch, device, compute_force=True)
            q_host = _host_charges(model, out, batch, device)
            feats = out["defect_features"].detach()
            feats_rest = feats[:, cfd:] if feats.shape[1] > cfd else None
            ptr = batch.ptr.tolist()
            for g, key in enumerate(batch.frame_key.tolist()):
                lo, hi = ptr[g], ptr[g + 1]
                cache.put(key, out["base_energy"][g], out["base_forces"][lo:hi],
                          None if feats_rest is None else feats_rest[lo:hi],
                          None if q_host is None else q_host[lo:hi])
            n_batches += 1
    with torch.no_grad():
        model.base_cache_checksum.copy_(checksum_to_buffer(digest))
    logging.info("Base cache: %d frames from %d batches in %.1f s (checksum %s)",
                 len(cache), n_batches, time.time() - t0, digest.hex()[:12])
    if path is not None:
        cache.save(path)
        logging.info("Base cache: written to %s", path)
    return cache


# ---------------------------------------------------------------------------- drift guard

def check_drift(model, dataset: Sequence, device, rng: np.random.Generator,
                tol_energy: float = 1e-4, tol_forces: float = 1e-4) -> Dict[str, float]:
    """One random frame through the full forward against its cached values.

    Returns the discrepancies and raises when either exceeds its tolerance. The tolerances
    are float32 tolerances -- the trunk runs in float32 under the mixed policy, so the fresh
    forward and the cached one differ by float32 non-associativity at most.
    """
    from mace.tools import torch_geometric

    cache = getattr(model, "_base_cache", None)
    if cache is None or len(dataset) == 0:
        return {}
    idx = int(rng.integers(len(dataset)))
    d = dataset[idx]
    batch = next(iter(torch_geometric.dataloader.DataLoader([d], batch_size=1)))
    out = _forward_full(model, batch, device, compute_force=True)
    energy, forces, _, _ = cache.lookup(batch.frame_key)
    d_e = float((out["base_energy"].detach().to(torch.float64) - energy).abs().max())
    d_f = float((out["base_forces"].detach().to(torch.float64) - forces).abs().max())
    result = dict(frame=idx, key=int(batch.frame_key[0]), d_energy=d_e, d_forces=d_f,
                  n_atoms=int(d.positions.shape[0]))
    if d_e > tol_energy or d_f > tol_forces:
        raise RuntimeError(
            f"base cache DRIFT on frame {idx}: |dE| = {d_e:.3e} eV (tol {tol_energy:.1e}), "
            f"max|dF| = {d_f:.3e} eV/A (tol {tol_forces:.1e}). The base is not the one the "
            "cache was built from -- something cached is training.")
    return result


# ------------------------------------------------------------------------------ profiler

def profile_step(model, loss_fn, batch, device, output_args: Dict[str, bool],
                 steps: int = 3, row_limit: int = 15) -> Tuple[str, float]:
    """`torch.profiler` over `steps` training steps (forward + backward), and the mean
    wall time per step. The table is what goes in the run log, per section 2.5."""
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.device(device).type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    model.train()
    batch = batch.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    def one_step():
        out = model(batch.to_dict(), training=True, compute_force=True,
                    compute_virials=bool(output_args.get("virials", False)),
                    compute_stress=bool(output_args.get("stress", False)))
        loss = loss_fn(pred=out, ref=batch)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        del grads

    one_step()                                   # warm-up, outside the profile
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(steps):
            one_step()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
    wall = (time.time() - t0) / max(steps, 1)
    sort_key = ("cuda_time_total" if torch.device(device).type == "cuda"
                else "cpu_time_total")
    try:
        table = prof.key_averages().table(sort_by=sort_key, row_limit=row_limit)
    except Exception as err:  # pragma: no cover - profiler table formatting is best-effort
        table = f"(profiler table unavailable: {err})"
    return table, wall
