"""How much of the trunk gradient actually flows through E_LR[q^host]?

2b(i) detaches q^host from the shared features on the hypothesis that fitting it drives
species salience into the representation the logit head reads. That hypothesis is only
actionable if the pathway carries appreciable gradient. Measured on a 6-atom toy the detach
changed the trunk gradient by 0.09%, which would make A1 a near-null intervention -- so
measure it on the real system, at the real state, before spending the runs.

Uses the A0 checkpoint (epoch 4-ish, cuEq layout) and real 79-atom V_Cl+ frames.
"""

import sys

sys.path.insert(0, "/home/alex/src/mace/.claude/worktrees/size-extensivity")
sys.path.insert(0, "/home/alex/src/mace/.claude/worktrees/size-extensivity/defect-perovskite")
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from ase.io import read  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)
from mace import data as mace_data  # noqa: E402
from mace import tools  # noqa: E402
from mace.cli.convert_e3nn_cueq import run as to_cueq  # noqa: E402
from mace.tools import torch_geometric  # noqa: E402

CKPT = sorted(Path("/home/alex/runs/perov_A0_s1/checkpoints").glob("*.pt"))
if not CKPT:
    raise SystemExit("no A0 checkpoint yet")
state = torch.load(CKPT[-1], map_location="cpu", weights_only=False)
print(f"checkpoint {CKPT[-1].name}")

reference = torch.load(
    "/home/alex/runs/perov_lr_s1/perov_lr_s1.model", map_location="cpu", weights_only=False
)
model = to_cueq(reference, device="cpu")
model.use_polarisation = False
model.host_carrier_coupling = False
model.latent_charges.use_polarisation = False
missing, unexpected = model.load_state_dict(state["model"], strict=False)
assert not missing, missing
model = model.train()

z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
frames = [
    a
    for a in read(
        "/home/alex/src/mace/.claude/worktrees/size-extensivity/"
        "defect-perovskite/dataset_pbe/train.xyz",
        ":",
    )
    if tuple(int(v) for v in a.info.get("carrier_counts", [])) == (0, 0, 1, 0)
    and len(a) == 79
][:3]

keyspec = mace_data.KeySpecification(
    info_keys={
        "carrier_counts": "carrier_counts",
        "host": "host",
        "multiplicity": "multiplicity",
        "m_s_ref_doubled": "m_s_ref_doubled",
    },
    arrays_keys={},
)

# Every trunk parameter, not one: the pathway could be concentrated anywhere.
trunk = [p for n, p in model.named_parameters()
         if p.requires_grad and ("interactions" in n or "products" in n)]
print(f"trunk tensors: {len(trunk)}\n")
print(f"{'frame':>6s} {'|g| base':>15s} {'|g| via q_host':>15s} {'share':>10s}")

for number, atoms in enumerate(frames):
    config = mace_data.config_from_atoms(atoms, key_specification=keyspec)
    mace_data.canonicalise_config_counters(config)
    atomic = mace_data.AtomicData.from_config(config, z_table=z_table, cutoff=5.0)
    batch = next(
        iter(torch_geometric.dataloader.DataLoader([atomic], batch_size=1, shuffle=False))
    ).to_dict()

    # Measure the pathway DIRECTLY rather than through the flag. StructuredLatentCharges
    # is @compile_mode("script"), so setting host_charge_detached at runtime on a converted
    # model need not reach the compiled forward -- and an exactly 0.000% difference is the
    # signature of a flag being ignored, not of a pathway carrying nothing.
    #
    # q^host leaves the forward with its graph intact, so E_LR[q^host] can be rebuilt and
    # differentiated on its own. That is the pathway, with no flag involved.
    out = model(batch, training=True, compute_force=False)
    grads_total = torch.autograd.grad(
        out["base_energy"].sum(), trunk, retain_graph=True, allow_unused=True
    )
    total = float(
        torch.sqrt(sum((g.detach() ** 2).sum() for g in grads_total if g is not None))
    )

    print(f"   base_energy.requires_grad = {out['base_energy'].requires_grad}")
    print(f"   q_host.requires_grad      = {out['latent_charges_host'].requires_grad}")
    print(f"   host_charge_detached      = {model.latent_charges.host_charge_detached}")
    qh = out["latent_charges_host"]
    print(f"   |q_host| max = {float(qh.abs().max()):.3e}  rms = {float(qh.pow(2).mean().sqrt()):.3e}")
    print(f"   E_LR[q_host] = {float(model.latent_ewald.energy(qh, batch['positions'], batch['cell'].view(-1,3,3), batch['batch']).sum()):.6e} eV")
    q_host = out["latent_charges_host"]
    energy_host = model.latent_ewald.energy(
        q_host, batch["positions"], batch["cell"].view(-1, 3, 3), batch["batch"]
    )
    grads_host = torch.autograd.grad(
        energy_host.sum(), trunk, retain_graph=False, allow_unused=True
    )
    host = float(
        torch.sqrt(sum((g.detach() ** 2).sum() for g in grads_host if g is not None))
    )
    print(f"{number:6d} {total:15.6e} {host:15.6e} {host / max(total, 1e-30) * 100:9.3f}%")

print()
print("Share of the base-energy trunk gradient carried by E_LR[q^host]. If it is small,")
print("2b(i) removes almost nothing and A1 ~ A0 is expected rather than informative.")
