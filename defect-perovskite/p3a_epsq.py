"""Plan P Phase 3a §2.2b/c/d, analysis of the 3x3x3 finite-displacement set (p3a_fc333.py): real-space force constants
and dipole-displacement tensors, Fourier-interpolated D(q) and the q-dependent Born charges Z*_k(q) (the form factor
built in), eps_ion(q) from the longitudinal projection along x, y, z at the registered wavevectors, the form factor of
the strongest IR mode, and the dispersion along Gamma-X/Y/Z (bare and with the nonanalytic term at eps_inf 4.0).

Conventions: for a displacement wave u_b = w_k e^{iq.R_b}/sqrt(M_k),
    D(q)_{k'a,kb} = (M_k' M_k)^-1/2 sum_{b' in k'} G[k,b,b',a] e^{-iq.(R_b' - R_k)},   G = d2E/du_{b'a} du_{0k b}
    Z*_k(q)_{ab}  = sum_{b'} Lam[k,b,b',a] e^{-iq.(R_b' - R_k)},                       Lam = dp_{b'a}/du_{0k b}
    Ztilde_m(q)_a = sum_{kb} Z*_k(q)_{ab} e_m(q)_{kb}/sqrt(M_k);  eps_ion(q) = (4 pi C/Omega) sum_m Ztilde Ztilde^*/w_m^2
    eps_ion_L(q) = qhat . eps_ion(q) . qhat  (the bare D(q): the base has no long-range dipole term).
Registered q: 0; 2 pi/L for L = 47.3, 43.0, 32.8, 28.6 A (the ladder cells' Omega^(1/3)); 0.26 A^-1; plus the 3x3x3
commensurate points n 2 pi/L_axis (n = 1, 2) where the Fourier sum is exact.

    python defect-perovskite/p3a_epsq.py --raw ~/runs/dscc/p3a/fc333_raw.npz --lst ~/runs/dscc/p3a/p3a_lst.npz --out ~/runs/dscc/p3a/p3a_epsq.json
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace.modules.dscc.ewald import COULOMB   # noqa: E402
from p3a_common import MEV                      # noqa: E402
from ase.data import atomic_masses              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True); ap.add_argument("--lst", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--eps_inf", type=float, default=4.0); ap.add_argument("--r_local", type=float, default=12.0)
ap.add_argument("--L_list", type=float, nargs="+", default=[47.3, 43.0, 32.8, 28.6]); ap.add_argument("--q_extra", type=float, nargs="+", default=[0.26])
args = ap.parse_args()
d = np.load(args.raw); G = d["G"]; Lam = d["Lam"]; done = d["done"]; R = d["positions"]; cellB = d["cell"]; numbers = d["numbers"]
assert done.all(), f"raw set incomplete: {int(done.sum())}/{len(done)}"
lst = np.load(args.lst); Z0 = lst["Z"]; w2_gamma = lst["w2"]; unit_cell = lst["cell"]; unit_numbers = lst["numbers"]
N = G.shape[0]; NB = G.shape[2]; M = np.array([atomic_masses[n] for n in unit_numbers]); Omega = abs(np.linalg.det(unit_cell)); CONV = 4 * math.pi * COULOMB / Omega
kappa_of = np.arange(NB) % N                        # ASE repeat order: image-major, atom-minor
# minimum-image pair vectors r[k, b'] = R_b' - R_k in the supercell (orthorhombic; the receptive field 12 A < half the box)
inv = np.linalg.inv(cellB)
r = R[None, :, :] - R[:N, None, :]
frac = r @ inv; frac -= np.round(frac); r = frac @ cellB; dist = np.linalg.norm(r, axis=-1)
far = dist > args.r_local
Gt = np.transpose(G, (0, 2, 1, 3)); Lt = np.transpose(Lam, (0, 2, 1, 3))          # [k, b', b, a]
locality = {"max_abs_G_beyond_r_local": float(np.abs(Gt[far]).max()), "max_abs_G": float(np.abs(G).max()),
            "max_abs_Lam_beyond_r_local": float(np.abs(Lt[far]).max()), "max_abs_Lam": float(np.abs(Lam).max()),
            "half_box_A": (0.5 * np.linalg.norm(cellB, axis=1)).tolist(), "r_local": args.r_local}
# acoustic sum rule on the real-space constants: correct the self block so every row sums to zero
asr_before = float(np.abs(G.sum(2)).max())
for k in range(N):
    G[k, :, k, :] -= G[k].sum(1)
# Z* from the dipole tensor at q = 0 against the autograd Z*
Zfd = Lam.sum(2).transpose(0, 2, 1)                 # Z*_k[a, b] = sum_b' Lam[k, b, b', a]
z_check = {"max_abs_diff_Zfd_vs_autograd": float(np.abs(Zfd - Z0).max()), "Z_trace3_by_species_fd": {s: float(np.mean([np.trace(Zfd[k]) / 3 for k in range(N) if unit_numbers[k] == n])) for s, n in (("Cl", 17), ("Cs", 55), ("Pb", 82))}}
Msq = np.sqrt(M)


def dyn(q):
    """D(q) (3N x 3N, Hermitian-symmetrised), Z*_k(q) [N, 3, 3] complex."""
    ph = np.exp(-1j * (r @ q))                       # [N, NB]
    D = np.zeros((N, 3, N, 3), complex); Zq = np.zeros((N, 3, 3), complex)
    for k in range(N):
        gk = G[k] * ph[k][None, :, None]             # [3(b), NB, 3(a)]
        lk = Lam[k] * ph[k][None, :, None]
        for kp in range(N):
            sel = kappa_of == kp
            D[kp, :, k, :] = gk[:, sel, :].sum(1).T / (Msq[kp] * Msq[k])   # [a, b]
        Zq[k] = lk.sum(1).T                                                # [a, b]
    D = D.reshape(3 * N, 3 * N); herm = float(np.abs(D - D.conj().T).max()); D = 0.5 * (D + D.conj().T)
    return D, Zq, herm


def modes_eps(q, qhat, eps_inf, nac=False):
    D, Zq, herm = dyn(q)
    if nac:
        zl = np.einsum("a,kab->kb", qhat, Zq).reshape(-1)                       # qhat . Z*_k(q), per (k, b)
        D = D + CONV / eps_inf * np.outer(zl.conj(), zl) / np.outer(np.repeat(Msq, 3), np.repeat(Msq, 3))
        D = 0.5 * (D + D.conj().T)
    w2, e = np.linalg.eigh(D)
    zt = np.einsum("kab,kbm->am", Zq, e.reshape(N, 3, -1) / Msq[:, None, None])  # [3, 3N]
    eps = np.zeros((3, 3)); n_soft = 0; contrib = []
    for m in range(3 * N):
        if w2[m] <= 1e-6:
            n_soft += 1; continue
        c = np.real(np.outer(zt[:, m], zt[:, m].conj())) / w2[m]; eps += c; contrib.append((float(math.sqrt(w2[m]) * MEV), float(qhat @ c @ qhat * CONV)))
    eps *= CONV
    return {"w2": w2, "eps_tensor": eps, "eps_L": float(qhat @ eps @ qhat), "eps_poly": float(np.trace(eps) / 3), "n_soft": n_soft, "herm": herm,
            "top": sorted(contrib, key=lambda t: -t[1])[:6], "Zq": Zq}


res = {"raw": args.raw, "lst": args.lst, "eps_inf": args.eps_inf, "Omega": Omega, "locality": locality, "asr_residual_before": asr_before, "z_check": z_check, "axes": {}}
# q = 0 check against the Gamma Hessian of the unit cell
m0 = modes_eps(np.zeros(3), np.array([0.0, 1.0, 0.0]), args.eps_inf)
w_sc = np.sqrt(np.clip(m0["w2"], 0, None)) * MEV; w_g = np.sqrt(np.clip(w2_gamma, 0, None)) * MEV
res["q0_check"] = {"max_abs_dw_meV_vs_gamma_hessian": float(np.abs(np.sort(w_sc) - np.sort(w_g)).max()), "n_soft": m0["n_soft"], "eps_ion_poly_q0": m0["eps_poly"],
                   "eps_ion_diag_q0": np.diag(m0["eps_tensor"]).tolist(), "hermiticity_defect": m0["herm"]}
print(f"q=0: max |dw| vs Gamma Hessian {res['q0_check']['max_abs_dw_meV_vs_gamma_hessian']:.3f} meV; eps_ion poly {m0['eps_poly']:.2f} diag {np.round(np.diag(m0['eps_tensor']), 2).tolist()}; "
      f"locality max|G| beyond {args.r_local} A {locality['max_abs_G_beyond_r_local']:.2e} (max {locality['max_abs_G']:.2f}); Z* fd vs autograd {z_check['max_abs_diff_Zfd_vs_autograd']:.1e}", flush=True)
# the strongest IR mode at Gamma (for the form factor)
e_g = lst["e"]; keep = [m for m in range(3 * N) if w2_gamma[m] > 1e-6]
zt_g = {m: np.einsum("kab,kb->a", Z0, e_g[:, m].reshape(N, 3) / Msq[:, None]) for m in keep}
m_star = max(keep, key=lambda m: (zt_g[m] ** 2).sum() / w2_gamma[m]); vec_star = e_g[:, m_star].reshape(N, 3) / Msq[:, None]
res["strongest_mode"] = {"m": int(m_star), "omega_meV": float(math.sqrt(w2_gamma[m_star]) * MEV), "Ztilde": zt_g[m_star].tolist()}
for axis, name in enumerate("xyz"):
    Laxis = float(np.linalg.norm(cellB[axis])); qhat = np.eye(3)[axis]
    qlist = [("0", 0.0)] + [(f"n{n}", n * 2 * math.pi / Laxis) for n in (1, 2)] + [(f"L{L}", 2 * math.pi / L) for L in args.L_list] + [(f"q{qe}", qe) for qe in args.q_extra]
    rows = []
    for tag, qn in sorted(qlist, key=lambda t: t[1]):
        q = qhat * qn; me = modes_eps(q, qhat, args.eps_inf)
        # form factor of the strongest Gamma mode at this q: |qhat . sum_k Z*_k(q) vec_k| / |qhat . sum_k Z*_k(0) vec_k|
        s_q = qhat @ np.einsum("kab,kb->a", me["Zq"], vec_star); s_0 = qhat @ np.einsum("kab,kb->a", Z0, vec_star)
        ff = float(abs(s_q) / abs(s_0)) if abs(s_0) > 1e-12 else None
        rows.append({"tag": tag, "q": qn, "lambda": (2 * math.pi / qn if qn > 0 else None), "eps_ion_L": me["eps_L"], "eps_L": args.eps_inf + me["eps_L"], "eps_ion_poly": me["eps_poly"],
                     "n_soft": me["n_soft"], "hermiticity_defect": me["herm"], "form_factor_strongest_mode": ff, "top_modes_meV_contrib": me["top"]})
        print(f"  {name} {tag:>6} q={qn:.4f}: eps_ion_L {me['eps_L']:.2f} (poly {me['eps_poly']:.2f}), eps_L {args.eps_inf + me['eps_L']:.2f}, f {ff if ff is None else round(ff, 3)}, soft {me['n_soft']}", flush=True)
    # dispersion Gamma -> zone boundary (xi = 0 .. 0.5 of the UNIT cell's reciprocal vector), bare and with the NA term
    b_len = 2 * math.pi / float(np.linalg.norm(unit_cell[axis])); disp = []
    for xi in np.linspace(0, 0.5, 11):
        q = qhat * xi * b_len
        bare = modes_eps(q, qhat, args.eps_inf); na = modes_eps(q if xi > 0 else qhat * 1e-4 * b_len, qhat, args.eps_inf, nac=True)
        disp.append({"xi": float(xi), "q": float(xi * b_len), "omega_bare_meV": (np.sqrt(np.clip(bare["w2"], 0, None)) * MEV).tolist(),
                     "omega_na_meV": (np.sqrt(np.clip(na["w2"], 0, None)) * MEV).tolist(), "n_imag_bare": int((bare["w2"] < -1e-6).sum())})
    res["axes"][name] = {"L_supercell": Laxis, "b_unit": b_len, "epsq": rows, "dispersion": disp}
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1)
print("saved", args.out)
