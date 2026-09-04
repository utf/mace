"""A TEST-ONLY occupation policy, on synthetic Hamiltonians only. Plan v8 section 3.

WHAT IT IS. A policy whose occupations are a KNOWN vector `f` supplied in the state's
payload and assigned in eigenvalue order: `P = U diag(f) U^T`, `F = sum_k f_k eps_k`. That
is enough to verify the things the plan asks a mock to verify -- schema validation, the
per-spin count/charge consistency, dispatch, cache-key separation, that equality to
`S_ref`'s PHYSICAL key (not charge alone) controls reference identity, and the matrix-level
contract `dF/dH_ab = P_ba` -- with an occupation the count fill would never produce, so a
test that passed because the mock quietly reduced to the count fill cannot pass.

WHAT IT IS NOT. It carries no physical state, no donor/acceptor reading, no continuation rule
and no force or stress claim, and it is never exercised through the production energy path:
`MACEDefect` and `CountingHead` refuse its key at construction, and `test_electronic_state`
asserts on the AST that no module under `mace/` registers a policy. It lives under `tests/`
so that importing it from production code would be visible as such.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from mace.modules import defect_state as ds

MOCK_KEY = "mock_known_occupation"


def mock_state(f_maj: Sequence[float], f_min: Sequence[float], q_formal: int,
               delta_n, state_id: str = "") -> ds.ElectronicStateSpec:
    """A mock state carrying its occupation vectors in the payload."""
    return ds.ElectronicStateSpec(
        q_formal=q_formal, delta_n=tuple(delta_n), occupation_policy=MOCK_KEY,
        occupation_payload={"f_maj": [float(x) for x in f_maj],
                            "f_min": [float(x) for x in f_min]},
        state_id=state_id)


def _occupations(spec: ds.ElectronicStateSpec, n_states: int):
    payload = dict(spec.occupation_payload)
    f_maj = torch.tensor(payload["f_maj"], dtype=torch.float64)
    f_min = torch.tensor(payload["f_min"], dtype=torch.float64)
    if f_maj.numel() != n_states or f_min.numel() != n_states:
        raise ValueError(f"the payload's occupation vectors have {f_maj.numel()} and "
                         f"{f_min.numel()} entries for a {n_states}-state Hamiltonian")
    if bool((f_maj < 0).any() or (f_maj > 1).any() or (f_min < 0).any() or (f_min > 1).any()):
        raise ValueError("occupations lie in [0, 1]")
    return f_maj, f_min


def check_counts(spec: ds.ElectronicStateSpec, ref: ds.ElectronicStateSpec,
                 n_states: int, n_total: int) -> None:
    """The exact per-spin electron-count / formal-charge consistency of section 2.1:
    `Tr[P_sigma - P_ref,sigma] = delta_n_sigma` and `Q - Q_ref = -sum delta_n`, where the
    reference fill is the count fill's integer split of `n_total`."""
    spec.check_against(ref)
    f_maj, f_min = _occupations(spec, n_states)
    n_maj_ref, n_min_ref = float((n_total + 1) // 2), float(n_total // 2)
    got = (float(f_maj.sum()) - n_maj_ref, float(f_min.sum()) - n_min_ref)
    if any(abs(g - d) > 1e-9 for g, d in zip(got, spec.delta_n)):
        raise ValueError(f"Tr[P_sigma - P_ref,sigma] = {got} but delta_n = {spec.delta_n}")


class MockKnownOccupation(ds.OccupationPolicy):
    """`solve` returns `(F, lam, U, P_now, P_ref)` in the shape of `head_energy_hf`, with
    `F = sum_sigma sum_k f_k eps_k` -- an entropy-free band energy of a FIXED occupation, so
    that `dF/dH_ab = P_ba` holds exactly by Hellmann-Feynman on each eigenvalue."""

    key = MOCK_KEY
    version = 1

    def solve(self, H, n_total, spec: ds.ElectronicStateSpec, t_el=None,
              response_out=None, internals=None):
        if not isinstance(spec, ds.ElectronicStateSpec) or spec.occupation_policy != MOCK_KEY:
            raise ValueError("the mock policy solves only a mock state")
        lam, U = torch.linalg.eigh(H.double())
        f_maj, f_min = _occupations(spec, int(lam.numel()))
        n_maj_ref, n_min_ref = (n_total + 1) // 2, n_total // 2
        # The reference under this policy is the integer count fill at zero temperature --
        # a known vector too, so the mock stays free of the production bisection.
        f_ref_maj = (torch.arange(lam.numel()) < n_maj_ref).double()
        f_ref_min = (torch.arange(lam.numel()) < n_min_ref).double()
        lam_d, U_d = lam.detach(), U.detach()
        energy = ((f_maj + f_min - f_ref_maj - f_ref_min) * lam).sum()
        p_now = (U_d * (f_maj + f_min)) @ U_d.T
        p_ref = (U_d * (f_ref_maj + f_ref_min)) @ U_d.T
        if internals is not None:
            internals["occupations"] = torch.stack([f_maj, f_min, f_ref_maj, f_ref_min])
        return energy, lam_d, U_d, p_now, p_ref

    def solve_batched(self, *a, **k):
        raise NotImplementedError("the mock has no batched path; it is not a solver")


def install() -> MockKnownOccupation:
    """Register the mock for the current process. Idempotent."""
    policy = MockKnownOccupation()
    ds.register_test_policy(policy)
    return policy
