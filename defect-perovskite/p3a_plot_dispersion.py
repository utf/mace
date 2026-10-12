"""Phase 3a §2.2d: dispersion along Gamma-X/Y/Z (bare and with the nonanalytic term) and eps_L(q) per axis, from p3a_epsq.json."""
import json, sys
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1])); out = sys.argv[2]
fig, ax = plt.subplots(1, 4, figsize=(18, 4.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
for i, name in enumerate("xyz"):
    disp = d["axes"][name]["dispersion"]; q = np.array([r["q"] for r in disp])
    wb = np.array([r["omega_bare_meV"] for r in disp]); wn = np.array([r["omega_na_meV"] for r in disp])
    ax[i].plot(q, wb, c="tab:blue", lw=0.6, alpha=0.7); ax[i].plot(q, wn, c="tab:red", lw=0.6, alpha=0.5, ls="--")
    ax[i].plot([], [], c="tab:blue", label="bare (base alone)"); ax[i].plot([], [], c="tab:red", ls="--", label="with NA term (ε∞ 4.0)")
    ax[i].set_xlabel(f"q along {name} (Å⁻¹)"); ax[i].set_ylabel("ω (meV)"); ax[i].set_title(f"Γ → {'XYZ'[i]}"); ax[i].set_ylim(0, 32); ax[i].legend(fontsize=8, loc="upper right")
    n_imag = sum(r["n_imag_bare"] for r in disp); ax[i].text(0.02, 0.02, f"imaginary modes on path: {n_imag}", transform=ax[i].transAxes, fontsize=8)
for name, c in zip("xyz", ("tab:blue", "tab:orange", "tab:green")):
    rows = d["axes"][name]["epsq"]; q = [r["q"] for r in rows]; e = [r["eps_L"] for r in rows]
    ax[3].plot(q, e, "o-", c=c, label=f"ε_L(q), q ∥ {name}")
ax[3].plot([0, 0.13, 0.261], [23.4, 15.4, 8.8], "s:", c="gray", label="base v2 (P0, q ∥ y; ε∞ 4 + ε_ion)")
for L in (47.3, 43.0, 32.8, 28.6):
    ax[3].axvline(2 * np.pi / L, c="k", ls=":", lw=0.6)
ax[3].set_xlabel("q (Å⁻¹)"); ax[3].set_ylabel("ε_L = ε∞ + ε_ion,L"); ax[3].set_title("longitudinal ε(q); dotted = 2π/L of the ladder cells"); ax[3].legend(fontsize=8); ax[3].set_ylim(0, 26)
fig.suptitle("Base v8 w01 lattice response from the 3×3×3 finite-displacement set"); fig.tight_layout(); fig.savefig(out, dpi=130); print("saved", out)
