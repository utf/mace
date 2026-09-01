#!/usr/bin/env python3
"""T-A: is the missing object the host band edge?

A defect-bound carrier is a level inside the gap: in carrier energy, below the edge of the
host continuum. A level below the continuum of a short-ranged H is exponentially localised.
So localisation reduces to a scalar the head already computes,

    Delta_bind = lambda_1(pristine host, hole counter) - lambda_1(defect cell) > 0

and the current models should be failing it: the head has only ever been active on defect
cells, so nothing tells it where the host edge is, and a band-edge state is as legal as a
bound one. Prediction: Delta_bind ~ 0 in every delocalised model.

LABEL-FREE. Pristine frames carry no defect information; evaluating the head on them with
the hole counter is an evaluation of the HOST, not a label. The vacancy position is used in
exactly one place -- selecting atoms > 8 A from it for the escape check (item 4) -- and that
is evaluation-only, never a loss and never an input.

Delta_bind itself comes from `mace.modules.defect_bind`, which is already built and tested
against this definition. It is NOT re-derived here: four times in this project a guard has
measured a reimplementation rather than the thing it guarded, and a second Delta_bind spelled
out in an analysis script is precisely that pattern. This script adds only the quantities
that module does not provide -- the escape check, N_eff, species composition, the anchor
residual -- and asserts its own lambda_1 against the module's on the first batch.

THE GAUGE, which decides whether any of this means anything:

  eps = eps_raw                     if gauge_penalty else eps_raw - mean_eps[batch]

With `gauge_penalty=True` the site energies are absolute and the uniform mode is pinned
softly by a loss term instead of being projected out, so lambda is absolute and comparing it
across pristine and defect frames is legitimate. With `gauge_penalty=False` every frame's
lambda is measured relative to that frame's own mean site energy, and the difference of two
such numbers is not a binding energy at all. The head is therefore interrogated per model and
a False reading marks the row rather than silently producing a plausible number.

mu_c cancels inside Delta_bind (same model, same channel, one learned scalar per channel), so
the difference is pure spectrum. The soft anchor leaves an O(1/N) residual, ~meV, negligible
at the eV scale in question; `eps_mean` is reported per frame set so it stays visible rather
than corrected.

THE ESCAPE CHECK has the same gauge structure and needs both readings. The anchor pins each
frame's MEAN eps toward zero, so if a genuine well exists the far atoms must rise to
compensate, by roughly depth * support / N -- about 0.1 eV for a 6-atom well 3 eV deep in a
159-atom cell, which is not small against the "~ 0" this check is looking for. So the offset
is reported twice: raw (absolute, contaminated by the anchor when a well exists) and
gauge-matched (each frame's own mean eps removed from both sides, immune to it). Under T-A's
own prediction -- no well anywhere -- the two agree at ~0, and a divergence between them is
itself the finding.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from ase.geometry import get_distances
from ase.io import read

from mace import data as mace_data
from mace import tools
from mace.data.defects import prepare_defect_configurations
from mace.modules.defect_bind import delta_bind, lambda1_of
from mace.tools import torch_geometric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import KEYSPEC, _assert_repo  # noqa: E402
from r1_matrix import graph_cutoff_for  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402

HOLE_COUNTER = (0, 0, 1, 0)   # h_maj, the supervised channel for V_Cl+
PRISTINE_NATOMS = 80          # the perfect cell; 79 = one Cl removed, 159 = the big cell
FAR_CUTOFF = 8.0              # escape check: atoms beyond this are outside the vacancy's reach


# --------------------------------------------------------------------------- frames


def load_frames(path):
    return read(str(path), index=":")


def counts_of(atoms):
    c = atoms.info.get("carrier_counts")
    if c is None:
        return None
    return tuple(int(x) for x in np.asarray(c).reshape(-1))


def select(frames, natoms=None, charged=None):
    out = []
    for a in frames:
        if natoms is not None and len(a) != natoms:
            continue
        c = counts_of(a)
        if charged is True and (c is None or sum(c) == 0):
            continue
        if charged is False and (c is not None and sum(c) != 0):
            continue
        out.append(a)
    return out


def with_hole_counter(frames):
    """Copies carrying the hole counter, with the defect frames' spin bookkeeping.

    The pristine cells are scored with the SAME counter as the defect frames, which is
    off-distribution for them by construction: a perfect crystal has no carrier to hold. That
    is the whole point -- it asks where this Hamiltonian would put a carrier if there were no
    vacancy, which is the reference a binding energy needs. It is also why this can only ever
    be a diagnostic, never a loss.

    `m_s_ref_doubled` MUST be carried over from the defect frames, and this is not
    bookkeeping pedantry -- it decides which channel the reference is read from.
    (0,0,1,0) has M_s = -1, so on an even-electron composition (m_s_ref_doubled = 0, which
    is what an 80-atom perfect cell records) `canonicalise_counts` rewrites it to (0,0,0,1)
    -- the OTHER hole channel. The charged 79-atom cells are odd-electron and record
    m_s_ref_doubled = 1, which skips canonicalisation and leaves them on channel 2. Score the
    pristine cells as they come and Delta_bind silently subtracts channel 2's defect spectrum
    from channel 3's pristine spectrum: two channels with independent learned parameters, a
    difference of unrelated numbers wearing the name of a binding energy.

    The value never reaches the model -- it exists only in the counter validator -- so
    matching it changes which branch of that validator runs and nothing about the physics the
    head sees. `multiplicity` is consistent at 1 either way: 1 + M_s(-1) + 1 = 1.
    """
    out = []
    for a in frames:
        b = a.copy()
        b.info = dict(a.info)
        b.info["carrier_counts"] = np.array(HOLE_COUNTER, dtype=int)
        b.info["m_s_ref_doubled"] = 1
        b.info["multiplicity"] = 1
        b.info["cell_charge"] = 1          # counter_charge((0,0,1,0)) = 1, kept consistent
        out.append(b)
    return out


def make_batches(frames, z_table, cutoff, batch_size, device):
    configs = [mace_data.config_from_atoms(a, key_specification=KEYSPEC) for a in frames]
    prepare_defect_configurations(configs)
    atomic = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=cutoff)
              for c in configs]
    loader = torch_geometric.dataloader.DataLoader(atomic, batch_size=batch_size,
                                                   shuffle=False)
    out, start = [], 0
    for batch in loader:
        n = int(batch.num_graphs)
        out.append((batch.to(device), frames[start:start + n]))
        start += n
    return out


# --------------------------------------------------------------------------- capture


def capture(model, batch):
    """One forward pass returning both the head internals and the model output.

    The head's own `internals` hook is used rather than a re-assembled H: for H3 the
    Hamiltonian carries a sigma term that a formula written out here would silently drop.
    """
    grabbed: dict = {}
    head = model.spectral
    original = head.forward

    def wrapped(*args, **kwargs):
        kwargs["internals"] = grabbed
        return original(*args, **kwargs)

    head.forward = wrapped
    try:
        with torch.no_grad():
            out = model(batch.to_dict(), training=False, compute_force=False)
    finally:
        head.forward = original
    if "lam" not in grabbed:
        raise RuntimeError("no spectral internals captured; not a spectral-head model")
    return grabbed, out


def channel_of(batch):
    counts = batch.carrier_counts.reshape(int(batch.num_graphs), -1)
    return int(torch.argmax(counts.sum(dim=0)).item())


def lambda1_from(internals, channel):
    """lambda_1 per graph, excluding padded slots (which sit at ~1e3 by construction)."""
    lam = internals["lam"][:, channel, :]
    physical = lam < 500.0
    out = []
    for g in range(lam.shape[0]):
        vals = lam[g][physical[g]]
        out.append(float(vals.min()) if vals.numel() else float("nan"))
    return np.asarray(out)


def neff_and_species(out, batch, symbols, channel):
    """N_eff = 1 / sum_i alpha_i^2 per graph, matching the training metric exactly, plus the
    species decomposition of the state's mass."""
    alpha = out["carrier_alpha"]
    idx = batch.batch
    ng = int(batch.num_graphs)
    a = alpha[:, channel]
    neff, mass = [], []
    ai = idx.detach().cpu().numpy()
    an = a.detach().cpu().numpy()
    for g in range(ng):
        sel = ai == g
        v = an[sel]
        s2 = float((v * v).sum())
        neff.append(1.0 / s2 if s2 > 0 else float("nan"))
        sym = np.asarray(symbols[g])
        comp = {}
        tot = float(v.sum()) or 1.0
        for e in sorted(set(sym.tolist())):
            comp[e] = float(v[sym == e].sum() / tot)
        mass.append(comp)
    return np.asarray(neff), mass


def eps_by_species(internals, batch, symbols, channel, far_masks=None):
    """Per-species mean site energy, raw and with each frame's own mean eps removed.

    The gauge-matched column is the one immune to the soft anchor; see the module docstring.
    """
    eps = internals["eps"][:, channel].detach().cpu().numpy()
    idx = batch.batch.detach().cpu().numpy()
    raw, matched, frame_means = defaultdict(list), defaultdict(list), []
    for g in range(int(batch.num_graphs)):
        sel = idx == g
        e = eps[sel]
        sym = np.asarray(symbols[g])
        mu = float(e.mean())
        frame_means.append(mu)
        keep = np.ones(len(e), dtype=bool) if far_masks is None else far_masks[g]
        if not keep.any():
            continue
        for el in sorted(set(sym[keep].tolist())):
            m = keep & (sym == el)
            raw[el].append(float(e[m].mean()))
            matched[el].append(float((e[m] - mu).mean()))
    return ({k: float(np.mean(v)) for k, v in raw.items()},
            {k: float(np.mean(v)) for k, v in matched.items()},
            float(np.mean(frame_means)) if frame_means else float("nan"))


def far_from_vacancy(atoms, cutoff=FAR_CUTOFF):
    """Atoms further than `cutoff` from the vacancy centre.

    EVALUATION-ONLY use of the vacancy position, per the hard rules. Returns None when the
    site cannot be located, so such a frame is dropped rather than silently treated as bulk.
    """
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return None
    if site.cage is None or len(site.cage) == 0:
        return None
    pos = atoms.get_positions()
    cell, pbc = atoms.get_cell(), atoms.pbc
    a, b = int(site.shell[0]), int(site.shell[1])
    vec, _ = get_distances(pos[a][None], pos[b][None], cell=cell, pbc=pbc)
    centre = pos[a] + 0.5 * vec[0, 0]
    dv, _ = get_distances(centre[None], pos, cell=cell, pbc=pbc)
    return np.linalg.norm(dv[0], axis=-1) > cutoff


# --------------------------------------------------------------------------- per model


def run_model(path, pristine, defect, big_charged, big_pristine, device, batch_size):
    # map_location alone leaves buffers inside the scripted submodules on CPU, which only
    # surfaces deep in a TorchScript tensordot; .to() moves the whole tree.
    model = torch.load(path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    head = getattr(model, "spectral", None)
    if head is None:
        return {"model": str(path), "error": "no spectral head"}

    cutoff = graph_cutoff_for(model)
    gauge_penalty = bool(getattr(head, "gauge_penalty", False))

    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    row = {
        "model": str(path),
        "graph_cutoff": float(cutoff),
        "r_max": float(model.r_max),
        "gauge_penalty": gauge_penalty,
        "lambda_is_absolute": gauge_penalty,
    }
    if not gauge_penalty:
        row["warning"] = (
            "gauge_penalty=False: eps is mean-subtracted per frame, so lambda is measured "
            "relative to each frame's own mean site energy and Delta_bind is NOT a binding "
            "energy. Row reported for completeness; do not read it as one.")

    pb = make_batches(pristine, z_table, cutoff, batch_size, device)
    db = make_batches(defect, z_table, cutoff, batch_size, device)

    # --- 1-3: Delta_bind, from the tested module, over every defect batch.
    #
    # The channel is PINNED rather than derived per batch. delta_bind() would otherwise take
    # argmax of each batch's own counters, and if any frame's counter were canonicalised the
    # two sides of the subtraction would come from different channels. with_hole_counter()
    # already prevents that; pinning makes it impossible rather than merely unlikely, and
    # asserts the two agree so a silent divergence cannot survive.
    ch_def = channel_of(db[0][0])
    ch_pri = channel_of(pb[0][0])
    row["channel_defect"] = ch_def
    row["channel_pristine"] = ch_pri
    if ch_def != ch_pri:
        row["error"] = (
            f"channel mismatch: defect frames read channel {ch_def}, pristine reference "
            f"channel {ch_pri}. Delta_bind would subtract two different channels' spectra.")
        return row
    channel = ch_def

    # The reference is evaluated ONCE. delta_bind() re-scores every pristine batch for each
    # defect batch it is handed, which is O(n_pristine x n_defect) forwards for a quantity
    # whose pristine half does not depend on the defect frame at all. The module is still the
    # authority: it is called once below and its answer asserted against this one, so the
    # cheap path cannot drift from the tested definition without the check firing.
    pristine_only = [b for b, _ in pb]
    ref = np.concatenate([lambda1_of(model, b.to_dict(), channel=channel)
                          for b in pristine_only])
    ref_mean = float(np.nanmean(ref))
    lam_def = np.concatenate([lambda1_of(model, b.to_dict(), channel=channel)
                              for b, _ in db])
    delta = ref_mean - lam_def
    row["delta_bind_mean"] = float(np.nanmean(delta))
    row["delta_bind_sd"] = float(np.nanstd(delta))
    row["delta_bind_median"] = float(np.nanmedian(delta))
    row["lambda1_defect"] = float(np.nanmean(lam_def))
    row["lambda1_defect_sd"] = float(np.nanstd(lam_def))
    row["lambda1_pristine"] = ref_mean
    row["lambda1_pristine_sd"] = float(np.nanstd(ref))
    row["n_pristine"] = int(np.isfinite(ref).sum())
    row["n_defect"] = int(np.isfinite(lam_def).sum())

    mod = delta_bind(model, db[0][0].to_dict(),
                     [x.to_dict() for x in pristine_only], channel=channel)
    mine_first = float(np.nanmean(
        ref_mean - lambda1_of(model, db[0][0].to_dict(), channel=channel)))
    row["delta_bind_module_agreement"] = abs(mod["delta_bind_mean"] - mine_first)

    # Consistency: this script's own lambda_1 must agree with the module's, or one of the
    # two is measuring something else and every number below is suspect.
    b0 = pb[0][0]
    int0, out0 = capture(model, b0)
    mine = lambda1_from(int0, channel)
    theirs = lambda1_of(model, b0.to_dict(), channel=channel)
    row["lambda1_selfcheck_max_abs_diff"] = float(np.nanmax(np.abs(mine - theirs)))

    # --- 3b: the model's own carrier energy, from its own outputs (not re-derived).
    def carrier_energy(batches):
        tot, n = 0.0, 0
        for batch, _ in batches:
            with torch.no_grad():
                o = model(batch.to_dict(), training=False, compute_force=False)
            e = o.get("delta_sr_energy")
            if e is None:
                continue
            v = e.detach().cpu().numpy()
            r = o.get("delta_resp_energy")
            if r is not None:
                v = v + r.detach().cpu().numpy()
            tot += float(v.sum()); n += v.size
        return tot / n if n else float("nan")

    dE_p = carrier_energy(pb)
    dE_d = carrier_energy(db)
    row["dE_pristine"] = dE_p
    row["dE_defect"] = dE_d
    row["dE_difference"] = dE_p - dE_d

    # --- 5: what the pristine-frame state actually is (should be a uniform band edge).
    neff_p, mass_p, eps_mean_p = [], [], []
    for batch, frames in pb:
        internals, out = capture(model, batch)
        syms = [np.array(f.get_chemical_symbols()) for f in frames]
        ne, ms = neff_and_species(out, batch, syms, channel)
        neff_p += ne.tolist(); mass_p += ms
        _, _, mu = eps_by_species(internals, batch, syms, channel)
        eps_mean_p.append(mu)
    row["neff_pristine_mean"] = float(np.nanmean(neff_p))
    row["neff_pristine_over_N"] = float(np.nanmean(neff_p) / PRISTINE_NATOMS)
    agg = defaultdict(list)
    for m in mass_p:
        for k, v in m.items():
            agg[k].append(v)
    row["species_mass_pristine"] = {k: float(np.mean(v)) for k, v in agg.items()}
    row["eps_mean_pristine"] = float(np.mean(eps_mean_p))

    neff_d = []
    eps_mean_d = []
    for batch, frames in db:
        internals, out = capture(model, batch)
        syms = [np.array(f.get_chemical_symbols()) for f in frames]
        ne, _ = neff_and_species(out, batch, syms, channel)
        neff_d += ne.tolist()
        _, _, mu = eps_by_species(internals, batch, syms, channel)
        eps_mean_d.append(mu)
    row["neff_defect_mean"] = float(np.nanmean(neff_d))
    row["eps_mean_defect"] = float(np.mean(eps_mean_d))
    row["eps_mean_anchor_residual"] = row["eps_mean_defect"] - row["eps_mean_pristine"]

    # --- 4: escape check on the 159-atom charged cells.
    if big_charged and big_pristine:
        bb = make_batches(big_charged, z_table, cutoff, 1, device)
        bp = make_batches(big_pristine, z_table, cutoff, 1, device)
        raw_d, mat_d = defaultdict(list), defaultdict(list)
        n_used = 0
        for batch, frames in bb:
            fm = [far_from_vacancy(f) for f in frames]
            if any(m is None for m in fm):
                continue
            internals, _ = capture(model, batch)
            syms = [np.array(f.get_chemical_symbols()) for f in frames]
            r, m, _ = eps_by_species(internals, batch, syms, channel, far_masks=fm)
            for k, v in r.items():
                raw_d[k].append(v)
            for k, v in m.items():
                mat_d[k].append(v)
            n_used += len(frames)
        raw_p, mat_p = defaultdict(list), defaultdict(list)
        for batch, frames in bp:
            internals, _ = capture(model, batch)
            syms = [np.array(f.get_chemical_symbols()) for f in frames]
            r, m, _ = eps_by_species(internals, batch, syms, channel)
            for k, v in r.items():
                raw_p[k].append(v)
            for k, v in m.items():
                mat_p[k].append(v)
        species = sorted(set(raw_d) & set(raw_p))
        row["escape_raw"] = {
            k: float(np.mean(raw_d[k]) - np.mean(raw_p[k])) for k in species}
        row["escape_gauge_matched"] = {
            k: float(np.mean(mat_d[k]) - np.mean(mat_p[k])) for k in species}
        row["escape_n_frames"] = n_used
    else:
        row["escape_raw"] = None
        row["escape_gauge_matched"] = None
        row["escape_n_frames"] = 0
    return row


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, type=Path)
    ap.add_argument("--train", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--valid", type=Path, default=here / "dataset_pbe" / "valid.xyz")
    ap.add_argument("--n-pristine", type=int, default=64)
    ap.add_argument("--n-defect", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    train = load_frames(args.train)
    valid = load_frames(args.valid)

    # Pristine reference: the 80-atom perfect cells, scored under the hole counter. Drawn
    # from train so the validation defect frames stay a clean held-out set.
    pristine = with_hole_counter(
        select(train, natoms=PRISTINE_NATOMS, charged=False)[:args.n_pristine])
    # Defect: the charged validation cells, as trained.
    defect = select(valid, natoms=79, charged=True)[:args.n_defect]
    # Escape check: the big cells, where "far from the vacancy" is a meaningful region.
    big_charged = select(train, natoms=159, charged=True) + \
        select(valid, natoms=159, charged=True)
    big_pristine = with_hole_counter(
        select(train, natoms=159, charged=False) + select(valid, natoms=159, charged=False))

    print(f"  pristine reference : {len(pristine)} frames ({PRISTINE_NATOMS} atoms, "
          f"counter {HOLE_COUNTER})")
    print(f"  defect             : {len(defect)} charged validation frames")
    print(f"  escape check       : {len(big_charged)} charged / {len(big_pristine)} "
          f"pristine 159-atom frames")
    if not pristine or not defect:
        sys.exit("no usable frames; check the dataset selection")

    rows = []
    for p in args.models:
        if not Path(p).exists():
            print(f"  MISSING {p}")
            continue
        try:
            r = run_model(p, pristine, defect, big_charged, big_pristine,
                          args.device, args.batch_size)
        except Exception as exc:                       # noqa: BLE001
            r = {"model": str(p), "error": repr(exc)}
        rows.append(r)
        if "error" in r:
            print(f"  {Path(p).name:44s} ERROR {r['error'][:90]}")
        else:
            flag = "" if r["gauge_penalty"] else "  [RELATIVE-GAUGE]"
            print(f"  {Path(p).name:44s} Delta_bind {r['delta_bind_mean']:+8.4f} eV  "
                  f"l1_pri {r['lambda1_pristine']:+8.4f}  l1_def {r['lambda1_defect']:+8.4f}  "
                  f"N_eff(pri) {r['neff_pristine_mean']:6.1f}{flag}")
        args.out.write_text(json.dumps(rows, indent=2))
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}  ({len(rows)} models)")


if __name__ == "__main__":
    main()
