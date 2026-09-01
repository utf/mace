"""D1: is the axial hub force carried by on-site gradients of NON-hub atoms?

R2 ended 0/20: under a frozen base every head fits the residual -- including the 7-12x axial
hub signal T2 established -- with site-energy spreads 100-300x the unsupervised channels and
no spatial contraction at all. The proposed mechanism is that it can, because eps_i is an MLP
over neighbourhood features with a 5 A reach. The head's force on a hub Pb is

    F_hub = sum_i alpha_i d(eps_i)/dR_hub  +  sum_ij psi_i psi_j d(t_ij)/dR_hub

so a delocalised alpha still recovers a concentrated force by making eps_i steeply sensitive
to R_hub for every atom whose features see the hub. Large eps spread, extended state, forces
fit. If that is what is happening, no amount of extra physics on top of a flexible on-site
term will fix it, and the on-site term has to become rigid instead.

Method. Hellmann-Feynman, contracted against the H the head ACTUALLY assembled (captured via
the diagnostic `internals` hook) rather than a formula re-derived here -- for H3 the H
includes the sigma term, and a re-derivation that forgot it would quietly misattribute its
contribution. With beta_ij = sum_k w_k psi_ki psi_kj (so beta_ii = alpha_i):

    E_HF      = sum_ij beta_ij H_ij           == sum_k w_k lambda_k, exactly
    S_on_hub  = sum_{i in hub}  beta_ii H_ii
    S_on_non  = sum_{i not hub} beta_ii H_ii
    S_hop     = sum_{i != j}    beta_ij H_ij

beta and w are DETACHED, so differentiating each scalar w.r.t. the hub positions gives that
term's Hellmann-Feynman force. The full autograd force of the head's energy is reported as a
fourth column: the difference is the softmax-weight variation, -Cov_w(lambda, dlambda/dR)/T_s,
which is ~0 when one state dominates but need not be for these delocalised, near-degenerate
solutions. Reporting the residual is how we know the decomposition is complete rather than
assuming it.

Everything is differentiated through the FULL forward with positions requiring grad. The
d(eps)/dR_hub pathway runs through the trunk, so rebuilding eps from cached or detached
features would autograd to zero and make the leak invisible -- reading as leak_frac ~ 0 and
sending the decision tree down the wrong branch.

Sign convention for the axial projection matches r0_pair_force.py exactly, so the numbers sit
next to T2's +0.1035 eV/A: positive = the two hub Pb pushed apart.

The vacancy assignment is used to identify the hub and to project. It never enters the model.
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

from mace import tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo, make_batch  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402

SPECIES = {17: "Cl", 55: "Cs", 82: "Pb"}


def load_model(path: Path, device: str, config_from: Path | None = None):
    """`.model` loads directly; a raw `.pt` checkpoint needs a config to rebuild from.

    Half the R2 seeds were stopped before their final save and have only checkpoints. The
    anneal is a training-loop schedule, not a constructor argument, so an anneal seed can be
    rebuilt from a same-head noanneal sibling's config; hop_scale rides along in the state
    dict and is reported per row.
    """
    if path.suffix == ".model":
        return torch.load(path, map_location=device, weights_only=False).to(device).eval()
    from mace.tools.scripts_utils import extract_config_mace_model

    if config_from is None:
        raise SystemExit(f"{path} is a raw checkpoint; pass --config-from a sibling .model")
    template = torch.load(config_from, map_location="cpu", weights_only=False)
    model = template.__class__(**extract_config_mace_model(template))
    state = torch.load(path, map_location=device, weights_only=False)
    sd = state.get("model", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  warning: {len(missing)} missing keys, first {missing[:3]}")
    return model.to(device).eval()


def axial(vec_a, vec_b, axis):
    """0.5 * (F_b - F_a) . axis -- r0_pair_force's convention."""
    return 0.5 * (float(np.dot(vec_b, axis)) - float(np.dot(vec_a, axis)))


def analyse_frame(model, atoms, z_table, cutoff, device):
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return None
    a, b = int(site.shell[0]), int(site.shell[1])

    pos = atoms.get_positions()
    vec, dist = get_distances(pos[a][None], pos[b][None], cell=atoms.get_cell(),
                              pbc=atoms.pbc)
    axis = vec[0, 0] / max(float(dist[0, 0]), 1e-12)
    d_hub = float(dist[0, 0])

    batch = make_batch([atoms], z_table, cutoff, device)
    data = batch.to_dict()
    data["positions"] = data["positions"].detach().requires_grad_(True)
    positions = data["positions"]

    grabbed: dict = {}
    head = model.spectral
    original = head.forward

    def wrapped(*args, **kwargs):
        kwargs["internals"] = grabbed
        return original(*args, **kwargs)

    head.forward = wrapped
    try:
        model(data, training=False, compute_force=False)
    finally:
        head.forward = original
    if not grabbed:
        raise SystemExit("head internals were not captured -- is this a spectral-head model?")

    H, psi, w = grabbed["H"], grabbed["psi"], grabbed["w"]
    eps_all, local = grabbed["eps"], grabbed["local"]

    counts = data["carrier_counts"].reshape(1, -1) if "carrier_counts" in data else None
    if counts is None:
        raise SystemExit("frame carries no carrier_counts")
    c = int(torch.argmax(counts.reshape(-1)).item())        # the one supervised channel

    n = len(atoms)
    slots = local[:n].long()
    psi_c = psi[0, c][slots]                                # [n, m], real atoms only
    w_c = w[0, c]                                           # [m]
    beta = (psi_c * w_c.unsqueeze(0)) @ psi_c.T             # [n, n], beta_ii = alpha_i
    beta = beta.detach()

    H_c = H[0, c][slots][:, slots]                          # [n, n], still in the graph

    alpha = torch.diagonal(beta).clone()
    alpha_sum = float(alpha.sum())
    n_eff = float(1.0 / (alpha / max(alpha_sum, 1e-30)).pow(2).sum())

    z = np.array(atoms.get_atomic_numbers())
    a_np = alpha.detach().cpu().numpy()
    mass = {SPECIES.get(int(k), str(k)): float(a_np[z == k].sum() / max(a_np.sum(), 1e-30))
            for k in sorted(set(z.tolist()))}

    # Within-species site spread of eps -- the diagnostic that was missing. A head that
    # discriminates only between sublattices shows ~0 here while the overall spread is large.
    eps_c = eps_all[:n, c].detach().cpu().numpy()
    within = {SPECIES.get(int(k), str(k)): float(eps_c[z == k].std())
              for k in sorted(set(z.tolist()))}

    hub = torch.zeros(n, dtype=torch.bool)
    hub[[a, b]] = True
    diagH = torch.diagonal(H_c)
    d_beta = torch.diagonal(beta)

    # F1: the hopping force nets two physically distinct pieces and they must be separated.
    #
    #   hub-hub   the bond-removal signature. For a bonding-signed H the ground state is
    #             nodeless (Perron-Frobenius), so beta_ab >= 0; with t'(d) < 0 the force
    #             -dE/dd = 2 beta_ab t'(d) is INWARD for any occupancy. The head cannot write
    #             the physically required +t_hub(d) at all, and its only moves are to hold
    #             t_hub at the decay prior or to one-side the state so beta_ab ~ 0.
    #   hub-ligand  a hub Pb pulled toward its five remaining ligands: net OUTWARD by
    #             missing-neighbour asymmetry, and only weakly dependent on d(Pb-Pb).
    #
    # Pooled as one "hop" term these cancel and the evasion is invisible, which is why S1
    # read as "the hopping force already points the right way".
    off = beta * H_c
    off = off - torch.diag(torch.diagonal(off))
    hub_d = hub.to(off.device)
    both = hub_d.unsqueeze(1) & hub_d.unsqueeze(0)      # the a-b edge (and b-a)
    one = hub_d.unsqueeze(1) ^ hub_d.unsqueeze(0)       # exactly one endpoint is a hub Pb
    parts = {
        "on_hub": (d_beta[hub] * diagH[hub]).sum(),
        "on_non": (d_beta[~hub] * diagH[~hub]).sum(),
        "hop": off.sum(),
        "hop_hubhub": (off * both).sum(),
        "hop_hublig": (off * one).sum(),
        "hop_rest": (off * ~(both | one)).sum(),
    }
    out = {}
    for name, scalar in parts.items():
        g = torch.autograd.grad(scalar, positions, retain_graph=True, allow_unused=True)[0]
        f = (-g).detach().cpu().numpy()                     # force = -dE/dR
        out[f"axial_{name}"] = axial(f[a], f[b], axis)

    # Full autograd of the head energy: catches the softmax-weight variation the three
    # Hellmann-Feynman terms omit.
    e_full = (w[0, c].detach() * grabbed["lam"][0, c]).sum()
    g_full = torch.autograd.grad(e_full, positions, retain_graph=False, allow_unused=True)[0]
    f_full = (-g_full).detach().cpu().numpy()
    out["axial_full"] = axial(f_full[a], f_full[b], axis)
    out["residual"] = out["axial_full"] - (out["axial_on_hub"] + out["axial_on_non"]
                                           + out["axial_hop"])

    # F1: two-sidedness of the hub pair. Suppressing beta_ab by driving the amplitude onto one
    # of the two Pb is the other way out of the inward hub-hub force, and it is invisible in
    # N_eff -- a state on one Pb plus its shell is just as "localised" as one on both.
    aa, bb = float(alpha[a]), float(alpha[b])
    out.update(n_eff=n_eff, d_hub=d_hub, channel=c,
               alpha_hub_a=aa, alpha_hub_b=bb,
               two_sided=float(min(aa, bb) / max(max(aa, bb), 1e-30)),
               beta_ab=float(beta[a, b]),
               hub_hop=float(abs(H_c[a, b].detach())),
               eps_std_all=float(eps_c.std()),
               mass=mass, within=within, n_atoms=n)
    return out



def graph_cutoff_for(model, override=None):
    """Neighbour-list radius the head was TRAINED with.

    run_train builds the graph at the spectral Hamiltonian's range (10 A here) and filters the
    trunk back to r_max inside the model. An analysis that rebuilds batches at r_max instead
    silently drops every 5-10 A edge, so the head is handed a truncated H: the hub pair at a
    median 5.43 A then has no edge at all and reports |H_ab| = 0 with no hopping term, which
    is an artefact of the analysis and not a property of the model. Derived from the model so
    it cannot drift from what training used.
    """
    if override:
        return float(override)
    return max(float(model.r_max), float(getattr(model, "spectral_r_cut", 0.0) or 0.0))


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--config-from", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=here / "dataset_cf" / "eval_qp1.xyz")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--cutoff", type=float, default=None,
                    help="override; default = the model's own graph cutoff")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    _assert_repo()
    model = load_model(args.model, args.device, args.config_from)
    cutoff = graph_cutoff_for(model, args.cutoff)
    print(f"  graph cutoff {cutoff:.1f} A (model r_max {float(model.r_max):.1f})")
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    frames = read(args.data, ":")[: args.limit]

    rows = [r for r in (analyse_frame(model, a, z_table, cutoff, args.device)
                        for a in frames) if r is not None]
    if not rows:
        raise SystemExit("no frames analysable")

    def med(key):
        return float(np.median([r[key] for r in rows]))

    tot = np.array([abs(r["axial_on_hub"]) + abs(r["axial_on_non"]) + abs(r["axial_hop"])
                    for r in rows])
    per_frame = np.array([abs(r["axial_on_non"]) for r in rows]) / np.maximum(tot, 1e-30)

    hop_frac = np.array([abs(r["axial_hop"]) for r in rows]) / np.maximum(tot, 1e-30)
    summary = dict(
        label=args.label or args.model.stem,
        model=str(args.model), frames=len(rows),
        n_eff=med("n_eff"),
        axial_on_hub=med("axial_on_hub"), axial_on_non=med("axial_on_non"),
        axial_hop=med("axial_hop"), axial_full=med("axial_full"),
        residual=med("residual"),
        leak_frac=float(np.median(per_frame)), hop_frac=float(np.median(hop_frac)),
        hub_hop=med("hub_hop"), eps_std_all=med("eps_std_all"), d_hub=med("d_hub"),
        # F1: the split that makes the evasion visible, plus the two-sidedness that is the
        # other way of suppressing beta_ab without changing N_eff.
        axial_hop_hubhub=med("axial_hop_hubhub"),
        axial_hop_hublig=med("axial_hop_hublig"),
        axial_hop_rest=med("axial_hop_rest"),
        two_sided=med("two_sided"), beta_ab=med("beta_ab"),
        alpha_hub_a=med("alpha_hub_a"), alpha_hub_b=med("alpha_hub_b"),
        mass={k: float(np.median([r["mass"].get(k, 0.0) for r in rows]))
              for k in rows[0]["mass"]},
        within={k: float(np.median([r["within"].get(k, 0.0) for r in rows]))
                for k in rows[0]["within"]},
    )

    print(f"\n=== D1 {summary['label']}  ({len(rows)} frames) ===")
    print("  leak_frac = median over frames of |axial_on_non| / "
          "(|axial_on_hub| + |axial_on_non| + |axial_hop|)")
    print(f"  N_eff {summary['n_eff']:.1f}   eps std (all sites) "
          f"{summary['eps_std_all']:.4f} eV")
    print("  alpha species mass: " + "  ".join(
        f"{k} {v:.3f}" for k, v in summary["mass"].items()))
    print("  WITHIN-species eps std: " + "  ".join(
        f"{k} {v:.4f}" for k, v in summary["within"].items()))
    print("  axial force decomposition (eV/A, + = hub Pb pushed apart):")
    print(f"     on-site hub      {summary['axial_on_hub']:+.4f}")
    print(f"     on-site NON-hub  {summary['axial_on_non']:+.4f}   <-- the leak")
    print(f"     hopping          {summary['axial_hop']:+.4f}")
    print(f"     full (autograd)  {summary['axial_full']:+.4f}")
    print(f"     residual (softmax-weight variation) {summary['residual']:+.4f}")
    print(f"  leak_frac {summary['leak_frac']:.3f}   hop_frac {summary['hop_frac']:.3f}   "
          f"|H_ab| {summary['hub_hop']:.4f} eV at d {summary['d_hub']:.2f} A")

    if args.out:
        args.out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2,
                                       default=float))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
