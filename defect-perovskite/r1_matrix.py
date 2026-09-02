"""R1 on the corrected graph: does the head PREFER the hub, or is the loss indifferent?

DIAGNOSTIC ONLY. Uses the vacancy assignment to build masks; production configs refuse
`clamp_mask` and `probe_loss_mask`.

The previous R1 is void. It ran on a 5 A neighbour graph in which the two vacancy-sharing Pb
had no edge in 0 of 40 charged frames, so the `hub2` mask was two UNCOUPLED atoms: no bonding
state, no dt/dd force, and "cage beats hub by 28%" never tested the bonding hypothesis at all.

What is still open is a PREFERENCE question. On the void graph we established only that a
delocalised solution EXISTS which fits the axial residual (r = 0.995). Whether a bound
solution fits the same forces BETTER is untested, and it decides the branch: hub-preferred
means the remaining failure is optimisation; flat means the loss cannot tell bound from
delocalised and only a constraint will.

Masks (equal-cardinality controls throughout):
    hub2    the two vacancy-sharing Pb
    lig2    the two most nearly trans ligands of the cage      (2-atom control, right species)
    rand2   two random atoms, 3 independent draws              (2-atom null)
    cage10  the ligand cage
    nbhd12  hub2 | cage10
    noPb12  the 12 nearest non-Pb atoms, hub excluded          (12-atom control, no Pb)
    free    unclamped

Loss arms: `full` scores every atom; `nbhd` scores only nbhd12, i.e. only the
physics-carrying atoms. For `free` the arm is not a no-op -- under `full` the unclamped head
also absorbs base error on the ~67 atoms the carrier barely touches, under `nbhd` it is judged
only where the carrier lives -- so `free` crosses with it and both are run.

Primary metric:
    axial_red = 1 - RMSE_axial(head) / RMSE_axial(base-only)
with axial the component along the unit vector joining the two hub Pb, RMSE_axial(head) over
(F_total - F_DFT), RMSE_axial(base-only) over (F_base - F_DFT), pooled over both hub Pb and
all held-out charged frames, both against the SAME frozen Stage-A base the head trained
against. So base-only = 0, a perfect axial fit = 1, and negative means the head made the hub
worse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from ase.geometry import get_distances
from ase.io import read

from mace import data as mace_data
from mace import tools
from mace.data.defects import prepare_defect_configurations
from mace.modules.defect_stage import is_correction_param
from mace.tools import torch_geometric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import KEYSPEC, _assert_repo  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402

TWO_ATOM = ("hub2", "lig2", "rand2_0", "rand2_1", "rand2_2")
# 5 seeds only where the verdict lives. The rule compares hub2 against lig2 "by more than
# the seed spread", and a spread estimated from three points is the weakest link in the
# reading. rand2 stays at 3 draws x 3 seeds = nine samples of the 2-atom null.
FIVE_SEED = ("hub2", "lig2")
BIG = ("cage10", "nbhd12", "noPb12")
ALL_MASKS = TWO_ATOM + BIG + ("free",)


def graph_cutoff_for(model, override=None):
    if override:
        return float(override)
    return max(float(model.r_max), float(getattr(model, "spectral_r_cut", 0.0) or 0.0))


def masks_for(atoms, rng):
    """Per-atom boolean masks, plus the hub axis used for every axial projection."""
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return None
    if site.cage is None or len(site.cage) == 0:
        return None
    n = len(atoms)
    pos = atoms.get_positions()
    cell, pbc = atoms.get_cell(), atoms.pbc
    a, b = int(site.shell[0]), int(site.shell[1])

    vec, dist = get_distances(pos[a][None], pos[b][None], cell=cell, pbc=pbc)
    axis = vec[0, 0] / max(float(dist[0, 0]), 1e-12)

    hub = np.zeros(n, dtype=bool)
    hub[[a, b]] = True
    cage_idx = np.asarray(site.cage, dtype=int)
    cage = np.zeros(n, dtype=bool)
    cage[cage_idx] = True

    # lig2: a genuine TRANS pair through the vacancy -- two ligands on opposite sides of the
    # vacancy centre, at a separation comparable to the hub pair.
    #
    # An earlier version took the most widely separated cage pair, which lands at a median
    # 10.5 A: beyond the hopping range, coupled in only 38% of frames, |H_ij| = 0. That made
    # the control two ISOLATED atoms and turned hub2-vs-lig2 into coupled-vs-uncoupled -- the
    # very defect that voided the original R1, reintroduced in the control instead of the
    # treatment. The control must differ from hub2 in WHICH sites, not in whether the pair is
    # a pair at all.
    #
    # DISTANCE-MATCHED to the hub pair. Antipodality about the vacancy was tried and gives a
    # median 7.7 A -- 88% coupled but at |H_ij| = 0.007 eV against the hub's 0.097, i.e. a
    # thirteenfold weaker pair. The cage spans BOTH Pb, so any trans pair across it is far.
    #
    # Matching the separation matches the coupling regime, which is precisely what has to be
    # held fixed: the control must differ from hub2 in WHICH SITES, not in how strongly the
    # two sites talk to each other. Otherwise the comparison measures coupling again.
    d_target = float(dist[0, 0])
    best, pair = float("inf"), (int(cage_idx[0]), int(cage_idx[min(1, len(cage_idx) - 1)]))
    for i in range(len(cage_idx)):
        for j in range(i + 1, len(cage_idx)):
            _, d_ij = get_distances(pos[cage_idx[i]][None], pos[cage_idx[j]][None],
                                    cell=cell, pbc=pbc)
            score = abs(float(d_ij[0, 0]) - d_target)
            if score < best:
                best, pair = score, (int(cage_idx[i]), int(cage_idx[j]))
    lig2 = np.zeros(n, dtype=bool)
    lig2[list(pair)] = True

    sym = np.array(atoms.get_chemical_symbols())
    is_pb = sym == "Pb"

    # noPb12: the 12 non-Pb atoms nearest the vacancy centre. Equal cardinality to nbhd12,
    # same region of space, but no Pb -- so it isolates "does it need Pb amplitude".
    centre = pos[a] + 0.5 * (vec[0, 0])
    dv, _ = get_distances(centre[None], pos, cell=cell, pbc=pbc)
    dist_to_v = np.linalg.norm(dv[0], axis=-1)
    cand = np.where(~is_pb)[0]
    order = cand[np.argsort(dist_to_v[cand])][:12]
    noPb12 = np.zeros(n, dtype=bool)
    noPb12[order] = True

    # Shell-resolved masks for the fine structure. A cage-centred state also produces an
    # axial hub force: each hub Pb sits inside five ligand charges instead of six, so the net
    # field on it is axial by the same missing-neighbour asymmetry that made the hopping
    # fingerprint non-specific. Hub- and cage-centred states therefore both reproduce the
    # COARSE footprint, and what separates them is finer -- the relative Cl-versus-Cs force
    # magnitudes and the d(Pb-Pb) dependence.
    out = dict(hub2=hub, lig2=lig2, cage10=cage, nbhd12=hub | cage, noPb12=noPb12,
               free=np.ones(n, dtype=bool), is_pb=is_pb,
               lig_shell=cage, cs_shell=(sym == "Cs"),
               d_hub=float(dist[0, 0]),
               hub_idx=(a, b), axis=axis)
    for k in range(3):
        r = np.zeros(n, dtype=bool)
        r[rng.choice(n, size=2, replace=False)] = True
        out[f"rand2_{k}"] = r
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


def stacked(frame_masks, which, device):
    return torch.tensor(np.concatenate([m[which] for m in frame_masks]),
                        dtype=torch.bool, device=device)


def coupling_fraction(model, batches, frame_masks, device, which='hub2'):
    """The void-R1 failure mode, stated as a test: is the hub pair actually coupled?

    Restricted to the hub2 mask the Hamiltonian is 2x2; if its off-diagonal is at the floor
    the mask is two isolated atoms and the whole comparison is meaningless -- which is exactly
    what the previous R1 measured without noticing.
    """
    coupled = total = 0
    head = model.spectral
    original = head.forward
    grabbed: dict = {}

    def wrapped(*args, **kwargs):
        kwargs["internals"] = grabbed
        return original(*args, **kwargs)

    head.forward = wrapped
    try:
        with torch.no_grad():
            for batch, frames in batches[:4]:
                fm = [frame_masks[id(a)] for a in frames]
                d = batch.to_dict()
                d["_clamp_mask"] = stacked(fm, which, device)
                model(d, training=False, compute_force=False)
                H, local = grabbed["H"], grabbed["local"]
                idx = batch.batch.detach().cpu().numpy()
                for k, atoms in enumerate(frames):
                    sel = np.where(idx == k)[0]
                    if which == "hub2":
                        a, b = fm[k]["hub_idx"]
                    else:
                        a, b = (int(x) for x in np.where(fm[k][which])[0][:2])
                    sa, sb = int(local[sel[a]]), int(local[sel[b]])
                    total += 1
                    if abs(float(H[k, 2, sa, sb])) > 1e-4:
                        coupled += 1
    finally:
        head.forward = original
    frac = coupled / max(total, 1)
    if which == "hub2" and frac < 0.99:
        raise RuntimeError(
            f"HUB COUPLING FAILURE: the hub pair is coupled in only {frac:.1%} of frames. "
            "The hub2 mask is two isolated atoms and this matrix would repeat the void R1.")
    return frac


def axial_stats(err_head, err_base, fm, sel):
    """Axial components of both error fields on the two hub Pb, plus the D1 paired pair."""
    a, b = fm["hub_idx"]
    axis = fm["axis"]
    eh, eb = err_head[sel], err_base[sel]
    head_ax = [float(np.dot(eh[a], axis)), float(np.dot(eh[b], axis))]
    base_ax = [float(np.dot(eb[a], axis)), float(np.dot(eb[b], axis))]
    # D1 convention for the paired statistics: 0.5*(F_b - F_a).axis
    target = 0.5 * (float(np.dot(-eb[b], axis)) - float(np.dot(-eb[a], axis)))
    corr = 0.5 * (float(np.dot(eh[b] - eb[b], axis)) - float(np.dot(eh[a] - eb[a], axis)))
    return head_ax, base_ax, target, corr


def evaluate(model, batches, frame_masks, device, clamp=None, ctx=None):
    """Head/base axial statistics over `batches`.

    `ctx` is a ForwardContext; when given it decides the clamp and every other forward-pass
    property, so this path cannot disagree with training. `clamp` is the pre-context
    argument, kept for the archived scripts.
    """
    model.eval()
    head_ax, base_ax, tgt, cor, pb_frac = [], [], [], [], []
    d_list = []
    shell_sq = {"lig": 0.0, "cs": 0.0}
    shell_n = {"lig": 0, "cs": 0}
    sq_all = cnt_all = 0.0
    sq_nb = cnt_nb = 0.0
    if True:
        for batch, frames in batches:
            fm = [frame_masks[id(a)] for a in frames]
            # C1 measures a CLAMPED head; evaluating it unclamped would report the metrics of
            # a different model than the one trained -- N_eff came back 35 on a two-atom clamp.
            if ctx is not None:
                d = ctx.forward_dict(batch, frames, requires_grad=True)
            else:
                d = batch.to_dict()
                d["positions"] = batch.positions.detach().clone().requires_grad_(True)
                d["_clamp_mask"] = None if clamp is None else stacked(fm, clamp, device)
            with torch.enable_grad():
                out = model(d, training=False, compute_force=True)
            err_h = (out["forces"] - batch.forces).detach().cpu().numpy()
            err_b = (out["base_forces"] - batch.forces).detach().cpu().numpy()
            alpha = out["carrier_alpha"].detach().cpu().numpy()
            idx = batch.batch.detach().cpu().numpy()
            for k in range(len(frames)):
                sel = np.where(idx == k)[0]
                h, bs, t, c = axial_stats(err_h, err_b, fm[k], sel)
                head_ax += h
                base_ax += bs
                tgt.append(t)
                cor.append(c)
                d_list.append(fm[k]["d_hub"])
                _e = err_h[sel]
                for _name, _key in (("lig", "lig_shell"), ("cs", "cs_shell")):
                    _m = fm[k][_key]
                    if _m.any():
                        shell_sq[_name] += float((_e[_m] ** 2).sum())
                        shell_n[_name] += _e[_m].size
                e = err_h[sel]
                sq_all += float((e ** 2).sum())
                cnt_all += e.size
                nb = fm[k]["nbhd12"]
                sq_nb += float((e[nb] ** 2).sum())
                cnt_nb += e[nb].size
                act = alpha[sel][:, 2]
                if act.sum() > 0:
                    pb_frac.append(float(act[fm[k]["is_pb"]].sum() / act.sum()))

    head_ax, base_ax = np.array(head_ax), np.array(base_ax)
    tgt, cor = np.array(tgt), np.array(cor)
    rmse_h = float(np.sqrt((head_ax ** 2).mean()))
    rmse_b = float(np.sqrt((base_ax ** 2).mean()))
    return dict(
        axial_red=float(1.0 - rmse_h / max(rmse_b, 1e-30)),
        rmse_axial_head=rmse_h, rmse_axial_base=rmse_b,
        pearson=float(np.corrcoef(tgt, cor)[0, 1]) if len(tgt) > 2 else float("nan"),
        ratio=float(np.abs(cor).mean() / max(np.abs(tgt).mean(), 1e-30)),
        rmse_all=float(np.sqrt(sq_all / max(cnt_all, 1))) * 1000.0,
        rmse_nbhd=float(np.sqrt(sq_nb / max(cnt_nb, 1))) * 1000.0,
        pb_fraction=float(np.mean(pb_frac)) if pb_frac else float("nan"),
        rmse_lig_shell=float(np.sqrt(shell_sq["lig"] / max(shell_n["lig"], 1))) * 1000.0,
        rmse_cs_shell=float(np.sqrt(shell_sq["cs"] / max(shell_n["cs"], 1))) * 1000.0,
        # Slope of the predicted axial correction against d(Pb-Pb). The hub hopping carries a
        # d-dependence a cage-centred state has no reason to reproduce, so this is fine
        # structure that can separate them when axial_red alone cannot.
        slope_vs_d=(float(np.polyfit(np.array(d_list), np.array(cor), 1)[0])
                    if len(d_list) > 2 else float("nan")),
    )


def fresh_model(arch_path, base_path, seed, device, response=None, madelung=None,
                eps_inf=4.0, z_init=None, counting=False, counting_overrides=None):
    """Architecture from a current-code model, base weights from Stage-A, head freshly drawn.

    Three sources rather than one, deliberately:
      * the ARCHITECTURE comes from a model built by the current code, so the head config
        matches R2' exactly and the two are comparable. The pre-existing r1_h*.model files
        predate `gauge_penalty` and cannot even be forwarded;
      * the BASE comes from Stage-A, which is what "frozen Stage-A base" means and is not the
        drifted base an R2 run ends with;
      * the HEAD is re-drawn under the cell's seed, so the seed spread the verdict rule
        compares against reflects initialisation and not merely data order.
    """
    from mace.tools.scripts_utils import extract_config_mace_model

    if response is not None:
        raise ValueError(
            "the response channel was deleted by Edit 2 -- it is the same physics as the "
            "Madelung on-site term and running both double counts. Pass "
            "madelung=<pristine stoichiometry> instead. This raises rather than ignoring "
            "the argument so an archived harness cannot silently train a term-less model.")

    arch = torch.load(arch_path, map_location="cpu", weights_only=False)
    torch.manual_seed(seed)
    cfg = extract_config_mace_model(arch)
    if counting:
        # Edit 4 REPLACES the spectral head; install_local_head/install_bounded_elements must
        # not also run, or the model would carry two heads both writing delta_sr.
        cfg["counting_head"] = True
        # Counting-head knobs, applied to the CONFIG so they travel through
        # `extract_config_mace_model` on the way back out. Setting them on the head after
        # construction would build the head at the old value and leave the model attribute
        # -- the one the config round-trip reads -- disagreeing with the object that trains.
        for key, value in (counting_overrides or {}).items():
            if not key.startswith("counting_"):
                raise ValueError(f"counting_overrides key {key!r} is not a counting knob")
            cfg[key] = value
    if madelung is not None:
        # Edit 1 replaces the response channel. `madelung` is the pristine stoichiometry in
        # the model's own species order; the Ewald kernel is built for the on-site term even
        # with the long-range ENERGY branch still staged off.
        cfg["madelung_on_site"] = True
        cfg["madelung_composition"] = list(madelung)
        cfg["madelung_eps_inf"] = float(eps_inf)
        cfg["madelung_z_init"] = list(z_init) if z_init is not None else None
    model = arch.__class__(**cfg)

    base = torch.load(base_path, map_location="cpu", weights_only=False)
    base_sd, sd = base.state_dict(), model.state_dict()
    copied = []
    for k in sd:
        if k in base_sd and not is_correction_param(k) and sd[k].shape == base_sd[k].shape:
            sd[k] = base_sd[k].clone()
            copied.append(k)
    model.load_state_dict(sd)
    if not copied:
        raise SystemExit(f"no base weights copied from {base_path}; wrong base model?")
    return model.to(device), len(copied)


def run_cell(arch_path, base_path, mask, loss_arm, seed, batches, frame_masks, device,
             epochs, lr, response=False):
    model, _ = fresh_model(arch_path, base_path, seed, device, response)
    model.train()
    for n, p in model.named_parameters():
        p.requires_grad_(is_correction_param(n))
    head_params = [p for n, p in model.named_parameters() if is_correction_param(n)]
    opt = torch.optim.AdamW(head_params, lr=lr)

    for _ in range(epochs):
        for batch, frames in batches:
            fm = [frame_masks[id(a)] for a in frames]
            d = batch.to_dict()
            # Fresh leaf every step. The batches are built once and shared across cells, and
            # an earlier no_grad forward (the hub-coupling assertion) leaves requires_grad
            # set on the same tensor, which breaks the autograd path for the force.
            d["positions"] = batch.positions.detach().clone().requires_grad_(True)
            d["_clamp_mask"] = None if mask == "free" else stacked(fm, mask, device)
            out = model(d, training=True, compute_force=True)
            diff = (out["forces"] - batch.forces) ** 2
            if loss_arm == "nbhd":
                keep = stacked(fm, "nbhd12", device)
                loss = diff[keep].mean()
            else:
                loss = diff.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    metrics = evaluate(model, batches, frame_masks, device)
    metrics.update(mask=mask, loss=loss_arm, seed=seed, arch=str(arch_path))
    return metrics


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", type=Path, required=True,
                    help="current-code model supplying the head architecture")
    ap.add_argument("--base", type=Path,
                    default=Path.home() / "runs" / "e0_base_s1" / "e0_base_s1.model",
                    help="Stage-A base whose weights are frozen")
    ap.add_argument("--head", default="h3")
    ap.add_argument("--response", action="store_true",
                    help="step 2: enable the carrier-field response channel")
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "valid.xyz")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--cutoff", type=float, default=None)
    ap.add_argument("--masks", nargs="+", default=list(ALL_MASKS))
    ap.add_argument("--losses", nargs="+", default=["full", "nbhd"])
    ap.add_argument("--seeds-2atom", type=int, default=5)
    ap.add_argument("--seeds-other", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    model0 = torch.load(args.arch, map_location="cpu", weights_only=False)
    cutoff = graph_cutoff_for(model0, args.cutoff)
    print(f"  graph cutoff {cutoff:.1f} A (model r_max {float(model0.r_max):.1f})",
          flush=True)
    del model0

    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    rng = np.random.default_rng(0)
    frames_all = read(args.data, ":")

    def counters(a):
        c = a.info.get("carrier_counts")
        if isinstance(c, str):
            c = [int(x) for x in c.split()]
        return np.asarray(c, dtype=int)

    charged = [a for a in frames_all if counters(a).any()][: args.frames]
    keep, frame_masks = [], {}
    for a in charged:
        m = masks_for(a, rng)
        if m is not None:
            frame_masks[id(a)] = m
            keep.append(a)
    print(f"  {len(keep)} charged frames usable", flush=True)

    batches = make_batches(keep, z_table, cutoff, args.batch_size, args.device)

    model_probe, n_copied = fresh_model(args.arch, args.base, 0, args.device,
                                        args.response)
    print(f"  base weights copied: {n_copied} tensors", flush=True)
    for two in ("hub2", "lig2"):
        if two in args.masks:
            f = coupling_fraction(model_probe, batches, frame_masks, args.device, two)
            print(f"  {two}: coupled in {f:.0%} of checked frames", flush=True)
            if two == "lig2" and f < 0.9:
                raise SystemExit(
                    f"lig2 is coupled in only {f:.0%} of frames -- it is two isolated atoms, "
                    "so hub2-vs-lig2 would compare coupling rather than site choice.")
    del model_probe

    rows = []
    for mask in args.masks:
        n_seeds = args.seeds_2atom if mask in FIVE_SEED else args.seeds_other
        for loss_arm in args.losses:
            for seed in range(1, n_seeds + 1):
                r = run_cell(args.arch, args.base, mask, loss_arm, seed, batches,
                             frame_masks, args.device, args.epochs, args.lr,
                             args.response)
                r["head"] = args.head
                rows.append(r)
                print(f"  {args.head:3s} {mask:9s} {loss_arm:5s} s{seed}  "
                      f"axial_red {r['axial_red']:+.3f}  r {r['pearson']:+.3f}  "
                      f"ratio {r['ratio']:.2f}  rmse_nbhd {r['rmse_nbhd']:.1f}  "
                      f"lig {r['rmse_lig_shell']:.0f} cs {r['rmse_cs_shell']:.0f} "
                      f"slope {r['slope_vs_d']:+.3f} Pb {r['pb_fraction']:.2f}",
                      flush=True)
                args.out.write_text(json.dumps(rows, indent=2, default=float))

    print(f"\nwrote {args.out}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
