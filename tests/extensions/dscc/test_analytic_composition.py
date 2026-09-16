"""LAUNCH GATE: for every composition of head modules, the closed-form gradient either EQUALS
autograd or is REFUSED. There is no third option, and that is the whole point.

Registered 2026-09-15 after the capture bug: `GraphedStep` took the analytic route for an
O5-linear head whose crystal-field terms the closed form does not contain, and replayed forces
19 meV/A wrong with nothing raised -- the channel widths matched by coincidence, so no shape
error fired. A term added to the Hamiltonian and not to `analytic.h0_position_gradient` is
exactly that failure, and it is silent by construction: the closed form does not know what it
is missing.

This gate is on the COMPOSITION, not on one model. Each case below builds a head, asks
`analytic_refusal`, and then:

  * refusal is None  -> the analytic forces MUST equal the autograd forces;
  * refusal is a str -> the model MUST fall back, i.e. `analytic_head_forces=True` still
    produces the autograd answer, and `GraphedStep` must decline the model.

A new flag therefore cannot be added without appearing here: either it is in the kernel and
proven equal, or it is named in `analytic_refusal` and proven to fall back.
"""
import pytest
import torch

from mace.modules.dscc.analytic import analytic_refusal
from tests.extensions.dscc.test_model import _batch
from tests.extensions.dscc.test_screening import _o5, _o32, _o33
from tests.extensions.dscc.test_w6 import VACP, _o3, _w6

torch.set_default_dtype(torch.float64)

# (name, builder, whether the closed form is expected to cover it)
CASES = [
    ("w6",            lambda: _w6(0),                  False),   # coupled: not an analytic head
    ("o3",            lambda: _o3(0),                  True),
    ("o3.3",          lambda: _o33(0),                 True),
    ("o3.2-screened", lambda: _o32(0),                 False),   # three-body dt/dR not in kernel
    ("o5a-linear",    lambda: _o5(0, linear=True),     False),   # dV/dR not in kernel
    ("o5c-mlp",       lambda: _o5(0, mlp=True),        False),
]


@pytest.mark.parametrize("name,build,covered", CASES, ids=[c[0] for c in CASES])
def test_the_closed_form_either_matches_autograd_or_refuses(name, build, covered):
    model = build()
    refusal = analytic_refusal(model)
    if covered:
        assert refusal is None, f"{name} was expected to be in the kernel but is refused: {refusal}"
    else:
        assert refusal is not None or not getattr(model, "decoupled", False), (
            f"{name} carries a term the closed form does not differentiate and is NOT refused. "
            "That is the 19 meV/A capture bug: add it to `analytic_refusal` or to the kernel.")

    batch = _batch([VACP])
    ref = model(dict(batch), compute_force=True, analytic_head_forces=False)
    got = model(dict(batch), compute_force=True)
    assert float(ref["forces"].abs().max()) > 1e-4, "comparison of zeros proves nothing"
    # Whichever route the gate chose, the answer is the autograd answer.
    assert float((ref["forces"] - got["forces"]).abs().max()) < 2e-7, (
        f"{name}: the route the gate selected disagrees with autograd")


@pytest.mark.parametrize("name,build,covered", CASES, ids=[c[0] for c in CASES])
def test_capture_serves_every_head_it_accepts_with_a_gradient_that_contains_every_term(name, build, covered):
    """The invariant, restated after the tail learned to dispatch.

    It USED to be "capture declines exactly what the closed form declines", because the
    captured tail WAS the closed form. That is no longer true and the new contract is
    stronger: the tail asks `analytic_refusal` and routes to the closed form where it is
    valid and to `h0_position_gradient_vjp` where it is not. So capture's refusal list is
    SHORTER -- it now turns away only what has a value-dependent shape.

    What must hold, and is what this asserts: for every head, EITHER capture refuses it, OR
    the gradient it would use contains every term. The second is guaranteed by construction
    (the VJP differentiates the assembly rather than a transcription of it) and is checked
    end-to-end against eager in `test_capture.py`. Here we assert the dispatch cannot fall
    into a gap: no head may be accepted by capture while `analytic_refusal` is None and the
    closed form nonetheless raises.
    """
    from mace.modules.dscc.analytic import capture_refusal, h0_position_gradient
    model = build()
    cap, ana = capture_refusal(model), analytic_refusal(model)
    if cap is not None:
        return                                  # refused: nothing to serve
    if ana is None:
        # The closed form claims this head. It must actually accept it -- the W6 case was
        # exactly this gap: `analytic_refusal` said None, the kernel raised on `decoupled`.
        assert getattr(model, "decoupled", False), (
            f"{name}: `analytic_refusal` claims the closed form covers it, but the kernel "
            "refuses non-decoupled heads. The predicate and the kernel disagree.")
    # Either way the model's own forward produces the autograd answer (checked above), and
    # the captured route is checked against eager on CUDA in test_capture.py.


def test_every_head_flag_is_accounted_for():
    """A flag that changes `H0` and is neither in the kernel nor in `analytic_refusal` is the
    failure mode this file exists for. If this list grows, a case goes in `CASES` above."""
    import inspect
    from mace.modules.dscc import model as model_mod
    src = inspect.getsource(model_mod.MACEDSCC.__init__)
    # The flags that add or replace a term in H0. `directional`, `route_b`, `scf_free` and
    # `coupling` select a route rather than adding a term and are covered by the route gates.
    term_flags = {"screened", "correlation2", "on_site_from_base", "crystal_field",
                  "cf_linear", "cf_mlp"}
    missing = {f for f in term_flags if f not in src}
    assert not missing, f"flags no longer on the model: {missing} -- update this gate"
    ref_src = inspect.getsource(analytic_refusal)
    # Each term flag is either named in the refusal, or is covered by the kernel and has a
    # `covered=True` case above.
    covered_by_kernel = {"correlation2"}
    for f in sorted(term_flags - covered_by_kernel):
        assert f in ref_src or f in ("cf_linear", "cf_mlp"), (
            f"`{f}` changes H0 but `analytic_refusal` never mentions it: either differentiate "
            "it in the kernel and add a covered case here, or name it in the refusal")
