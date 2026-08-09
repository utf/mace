"""Does a checkpoint pickled before the gauge buffer existed still load and run?

``__setstate__`` has to *register* the buffer, not merely set an attribute, or it stays
out of the state dict and out of ``.to()``. That path only fires on a genuinely old
pickle, which no unit test can produce -- so this runs it against a real one.
"""

import torch

from mace import data, tools
from mace.data.defects import load_band_edges
from mace.tools import torch_geometric

MODEL = "/home/alex/runs/bfull_128ch_L1_s1/bfull_128ch_L1_s1.model"
DATA = "/home/alex/src/mace/defect-example/dataset_full"

# float32, not double: these checkpoints carry cuEq-converted TorchScript submodules
# compiled at the training dtype, and promoting them mismatches inside the scripted graph.
torch.set_default_dtype(torch.float32)
model = torch.load(MODEL, map_location="cpu", weights_only=False).float().eval()
buffer = getattr(model, "gauge_counters", "ABSENT")
print("gauge_counters:", buffer if isinstance(buffer, str) else tuple(buffer.shape))
print("is a registered buffer:", "gauge_counters" in dict(model.named_buffers()))

spec = data.KeySpecification()
spec.info_keys.update(
    {
        "energy": "REF_energy",
        "carrier_counts": "carrier_counts",
        "multiplicity": "multiplicity",
        "host": "host",
        "pair_id": "pair_id",
        "stress": "REF_stress",
        "head": "head",
    }
)
spec.arrays_keys.update({"forces": "REF_forces"})
_, configs = data.load_from_xyz(
    file_path=f"{DATA}/valid.xyz",
    key_specification=spec,
    band_edges=load_band_edges(f"{DATA}/band_edges.json"),
)
dataset = [
    data.AtomicData.from_config(c, z_table=tools.AtomicNumberTable([6, 14]), cutoff=4.0)
    for c in configs[:2]
]
batch = next(
    iter(torch_geometric.dataloader.DataLoader(dataset, batch_size=2, shuffle=False))
)
out = model(batch.to_dict(), training=False, compute_force=False)
print("forward OK; gauge_mean_u =", out.get("gauge_mean_u"))
print("carrier_logits shape =", tuple(out["carrier_logits"].shape))
