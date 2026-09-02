"""The cross-path test: train, evaluate and capture must agree about the forward pass.

This is the regression test for the fourth instance of one bug. `train` applied the clamp
mask, `evaluate` and `capture` did not, and a two-atom clamp reported N_eff = 35 -- the
number of a delocalised state, from a model that had been trained on two atoms. Nothing was
wrong with any of the three functions individually; they simply disagreed about what the
forward pass was.

So the test does not check that each path handles a clamp correctly. It checks that the three
paths, given ONE context object, produce the SAME carrier state -- which is the property that
actually failed, and the one no per-function test can express.
"""

import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from e3nn import o3

from mace import data, modules, tools
from mace.data.defects import prepare_defect_configurations
from mace.modules.defect_context import EPS_INF_DEFAULT, ForwardContext
from mace.modules.defect_models import MACEDefect
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)

Z_TABLE = tools.AtomicNumberTable([6, 14])
CUTOFF = 4.0


def build_model(seed=0, **overrides):
    torch.manual_seed(seed)
    config = dict(
        r_max=CUTOFF, num_bessel=6, num_polynomial_cutoff=5, max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=len(Z_TABLE),
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([1.0, 3.0]), avg_num_neighbors=4,
        atomic_numbers=Z_TABLE.zs, correlation=2,
        atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        carrier_feature_dim=8, counter_embedding_dim=8, carrier_mlp_hidden=16,
        use_long_range=False, spectral_head=True, spectral_r_cut=6.0,
        spectral_first_shell=True,
    )
    config.update(overrides)
    return MACEDefect(**config)


def make_batch(n_frames=2, seed=0):
    rng = np.random.default_rng(seed)
    atoms_list = []
    for _ in range(n_frames):
        n = 8
        pos = rng.uniform(0.0, 6.0, size=(n, 3))
        symbols = ["C"] * (n // 2) + ["Si"] * (n - n // 2)
        a = Atoms(symbols=symbols, positions=pos, cell=np.eye(3) * 8.0, pbc=True)
        a.info["charge"] = 1.0
        a.info["spin_multiplicity"] = 2
        a.arrays["forces"] = rng.normal(scale=0.1, size=(n, 3))
        a.info["energy"] = float(rng.normal())
        atoms_list.append(a)
    configs = prepare_defect_configurations(
        [data.config_from_atoms(a) for a in atoms_list])
    dataset = [data.AtomicData.from_config(c, z_table=Z_TABLE, cutoff=6.0)
               for c in configs]
    loader = torch_geometric.dataloader.DataLoader(dataset, batch_size=n_frames,
                                                   shuffle=False)
    return next(iter(loader)), atoms_list


def neff_of(out, channel=2):
    """Participation ratio of the carrier amplitude -- the quantity that read 35."""
    alpha = out["carrier_alpha"][:, channel].detach().cpu().numpy()
    s2 = float((alpha ** 2).sum())
    return 1.0 / s2 if s2 > 0 else float("nan")


class TestProductionRefusesMasks:
    def test_production_refuses_a_clamp(self):
        model = build_model()
        with pytest.raises(ValueError, match="refuses clamp_mask"):
            ForwardContext.production(model, clamp="hub2")

    def test_production_refuses_a_probe_loss_mask(self):
        model = build_model()
        with pytest.raises(ValueError, match="refuses clamp_mask"):
            ForwardContext.production(model, probe_loss_mask="hub2")

    def test_a_clamp_without_the_means_to_build_it_is_refused(self):
        """The silent-drop failure: a clamp named but no way to turn it into a mask."""
        model = build_model()
        with pytest.raises(ValueError, match="needs frame_masks and stack_fn"):
            ForwardContext.diagnostic(model, clamp="hub2")

    def test_diagnostic_context_is_labelled(self):
        model = build_model()
        assert ForwardContext.diagnostic(model).is_diagnostic is True
        assert ForwardContext.production(model).is_diagnostic is False
        assert ForwardContext.production(model).as_dict()["is_diagnostic"] is False


class TestGraphSpec:
    def test_cutoff_is_the_larger_of_r_max_and_r_couple(self):
        """The fault that voided R1, R2 and Test 1: a graph built at r_max while the head
        was configured for spectral_r_cut."""
        model = build_model(spectral_r_cut=9.0)
        ctx = ForwardContext.production(model)
        assert ctx.cutoff == pytest.approx(9.0)
        assert ctx.r_max == pytest.approx(CUTOFF)

    def test_eps_inf_comes_from_the_context_not_the_model(self):
        """model.eps_inf_init is 6.5 on the retained checkpoints while both launchers pass
        4.0. The context is the single source."""
        model = build_model(eps_inf_init=6.5)
        ctx = ForwardContext.production(model)
        assert ctx.eps_inf == pytest.approx(EPS_INF_DEFAULT)
        assert ctx.eps_inf != pytest.approx(float(model.eps_inf_init))


class TestCrossPathAgreement:
    """One context, three paths, one carrier state."""

    @staticmethod
    def _paths(model, ctx, batch, frames):
        """The three call shapes, written exactly as train/evaluate/capture write them."""
        # train: positions carry a graph, forces requested
        d_train = ctx.forward_dict(batch, frames, requires_grad=True)
        out_train = model(d_train, training=True, compute_force=True)

        # evaluate: eval mode, forces still requested
        model.eval()
        d_eval = ctx.forward_dict(batch, frames, requires_grad=True)
        with torch.enable_grad():
            out_eval = model(d_eval, training=False, compute_force=True)

        # capture: no grad, no forces, head internals hooked
        grabbed = {}
        head = model.spectral
        original = head.forward
        head.forward = lambda *a, **k: original(*a, **dict(k, internals=grabbed))
        try:
            with torch.no_grad():
                d_cap = ctx.forward_dict(batch, frames, requires_grad=False)
                out_cap = model(d_cap, training=False, compute_force=False)
        finally:
            head.forward = original
        model.train()
        return out_train, out_eval, out_cap, grabbed

    def test_unclamped_paths_agree(self):
        model = build_model()
        batch, frames = make_batch()
        ctx = ForwardContext.production(model)
        a, b, c, _ = self._paths(model, ctx, batch, frames)
        assert neff_of(a) == pytest.approx(neff_of(b), rel=1e-9)
        assert neff_of(a) == pytest.approx(neff_of(c), rel=1e-9)

    def test_clamped_paths_agree(self):
        """The regression. Under a clamp the three paths previously disagreed, and the
        disagreement was invisible because each path was individually correct."""
        model = build_model()
        batch, frames = make_batch()
        n_nodes = int(batch.positions.shape[0])

        # A two-atom clamp, expressed the way the harness expresses it.
        def stack_fn(per_frame, which, device):
            mask = torch.zeros(n_nodes, dtype=torch.bool)
            offset = 0
            for m in per_frame:
                idx = np.where(m[which])[0]
                mask[offset + idx] = True
                offset += m[which].size
            return mask.to(device)

        frame_masks = {}
        for f in frames:
            m = np.zeros(len(f), dtype=bool)
            m[:2] = True
            frame_masks[id(f)] = {"hub2": m}

        ctx = ForwardContext.diagnostic(model, clamp="hub2", frame_masks=frame_masks,
                                        stack_fn=stack_fn)
        a, b, c, _ = self._paths(model, ctx, batch, frames)
        assert neff_of(a) == pytest.approx(neff_of(b), rel=1e-9)
        assert neff_of(a) == pytest.approx(neff_of(c), rel=1e-9)

    def test_the_clamp_actually_bites(self):
        """Guard against the test above passing because the clamp does nothing at all:
        clamped and unclamped must differ."""
        model = build_model()
        batch, frames = make_batch()
        n_nodes = int(batch.positions.shape[0])

        def stack_fn(per_frame, which, device):
            mask = torch.zeros(n_nodes, dtype=torch.bool)
            offset = 0
            for m in per_frame:
                mask[offset + np.where(m[which])[0]] = True
                offset += m[which].size
            return mask.to(device)

        frame_masks = {}
        for f in frames:
            m = np.zeros(len(f), dtype=bool)
            m[:2] = True
            frame_masks[id(f)] = {"hub2": m}

        free = ForwardContext.production(model)
        clamped = ForwardContext.diagnostic(model, clamp="hub2", frame_masks=frame_masks,
                                            stack_fn=stack_fn)
        model.eval()
        with torch.no_grad():
            out_free = model(free.forward_dict(batch, frames), training=False,
                             compute_force=False)
            out_clamped = model(clamped.forward_dict(batch, frames), training=False,
                                compute_force=False)
        assert neff_of(out_free) != pytest.approx(neff_of(out_clamped), rel=1e-6)

    def test_a_clamped_context_without_frames_raises(self):
        """Forwarding a clamped context unclamped is the N_eff-35 bug; it must be loud."""
        model = build_model()
        batch, frames = make_batch()
        frame_masks = {id(f): {"hub2": np.zeros(len(f), dtype=bool)} for f in frames}
        ctx = ForwardContext.diagnostic(model, clamp="hub2", frame_masks=frame_masks,
                                        stack_fn=lambda *a: None)
        with pytest.raises(ValueError, match="no frames"):
            ctx.forward_dict(batch)
