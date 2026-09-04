#!/usr/bin/env python3
"""§1.2's measured negative: what an Ewald geometry precompute would and would not remove.

The spec asked for a per-frame store of `A_per` and `A_iso` so that `phi = A_per Z` becomes a
matvec. This times the pieces first. A stored kernel removes the VALUE of phi; every Ewald
quantity in the training step -- phi in H, E_LR, arm B's image potential -- needs a live
position derivative, which a stored matrix cannot supply. The store was therefore not built,
and these four numbers are the reason.

Promoted from a scratch script because the report cites its numbers.
"""

import time

import torch

from mace.modules.latent_ewald import LatentEwald


def bench(fn, n=10):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    torch.set_default_dtype(torch.float64)
    dev = "cuda"
    B, N = 8, 79
    ew = LatentEwald({"sigma": 1.0}).to(dev)
    a = 14.2
    pos = (torch.rand(B * N, 3, device=dev) * a).requires_grad_(True)
    cell = torch.eye(3, device=dev).reshape(1, 3, 3).repeat(B, 1, 1) * a
    batch = torch.arange(B, device=dev).repeat_interleave(N)
    Z = torch.randn(B * N, device=dev, requires_grad=True)
    q = torch.randn(B * N, device=dev)

    def value():
        with torch.no_grad():
            ew.energy(Z.detach(), pos.detach(), cell, batch).sum()

    def phi_only():
        # what `site_potential` does, WITHOUT create_graph: dE/dq alone
        qq = Z.detach().clone().requires_grad_(True)
        e = ew.energy(qq, pos.detach(), cell, batch).sum()
        torch.autograd.grad(e, qq)[0]

    def phi_with_graph():
        qq = Z.clone().requires_grad_(True)
        e = ew.energy(qq, pos, cell, batch).sum()
        return torch.autograd.grad(e, qq, create_graph=True)[0]

    def phi_then_force():
        phi = phi_with_graph()
        loss = (q * phi).sum()
        torch.autograd.grad(loss, pos, retain_graph=False)[0]

    def cross_energy_force():
        # the alternative: g = q^T A Z as three plain energies, then ONE backward to R
        qd = q.detach()
        g = (ew.energy(qd + Z, pos, cell, batch)
             - ew.energy(qd, pos, cell, batch)
             - ew.energy(Z, pos, cell, batch)).sum()
        torch.autograd.grad(g, pos, retain_graph=False)[0]

    for name, fn in (("energy value (no grad)", value),
                     ("dE/dq, no create_graph", phi_only),
                     ("dE/dq, create_graph", phi_with_graph),
                     ("dE/dq + d/dR of q.phi", phi_then_force),
                     ("3 energies + d/dR", cross_energy_force)):
        print(f"  {name:26s} {bench(fn) * 1e3:8.2f} ms")

    # How big is A?  N x N per frame.
    print(f"\n  A per frame: {N}x{N} = {N * N * 8 / 1024:.0f} kB float64, "
          f"{N * N * 4 / 1024:.0f} kB float32; 2877 frames -> "
          f"{2877 * N * N * 8 / 1e6:.0f} MB / {2877 * N * N * 4 / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
