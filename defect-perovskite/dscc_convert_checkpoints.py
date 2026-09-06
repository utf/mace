"""Re-save trained MACEDSCC checkpoints so that they reference only `mace.modules.dscc`
classes and the plain base (`aprime_prod_base.pt`): needed before the deletion sweep
removes the v8 modules the old pickles reference. Verified per run on one frame.
    python defect-perovskite/dscc_convert_checkpoints.py --runs /home/alex/runs/dscc/dscc_arm1_* --base /home/alex/runs/aprime_prod/aprime_prod_base.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mace import tools                                            # noqa: E402
from mace.modules.dscc import data as dd                          # noqa: E402
from mace.modules.dscc.kernels import KernelConfig               # noqa: E402
from mace.modules.dscc.model import MACEDSCC                      # noqa: E402
from mace.modules.dscc.scf import ScfOptions                      # noqa: E402
from mace.tools import torch_geometric                            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--base", default="/home/alex/runs/aprime_prod/aprime_prod_base.pt")
    ap.add_argument("--dataset", default=str(HERE / "dataset_pbe"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="the checkpoints were saved from the training device; compare there")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    dev = args.device
    base = torch.load(args.base, weights_only=False, map_location=dev).double().to(dev)
    import ase.io
    frame = [a for a in ase.io.read(f"{args.dataset}/train.xyz", ":") if int(a.info["cell_charge"]) != 0][0]
    frame.info.setdefault("carrier_counts", np.array([0, 0, 1, 0]))
    z_table = tools.AtomicNumberTable([17, 55, 82])
    for run in args.runs:
        run = Path(run)
        old = torch.load(run / "model.pt", weights_only=False, map_location=dev).to(dev)
        extra = old.get_extra_state()
        rec = json.load(open(run / "run_record.json"))["config"]
        new = MACEDSCC(base, r_cut=extra["r_cut"], kernel=KernelConfig(**extra["kernel"]), sigma_s=extra["sigma_s"],
                       coupling=extra["coupling"], route_b=extra["route_b"], directional=bool(rec["directional"]),
                       r_split=extra["r_split"], lambda_max=extra["lambda_max"], hidden=old.h0.sk.hop[0].out_features,
                       scf=ScfOptions(**extra["scf"]))
        new = new.to(dev)
        state = {k: v for k, v in old.state_dict().items() if not k.startswith("base.")}
        missing, unexpected = new.load_state_dict(state, strict=False)
        missing = [m for m in missing if not m.startswith("base.")]
        assert not missing and not unexpected, (missing, unexpected)
        d = dd.atomic_data([frame], z_table, extra["r_cut"])[0]
        batch = next(iter(torch_geometric.dataloader.DataLoader([d], batch_size=1))).to(dev).to_dict()
        a = old(dict(batch), compute_force=True); b = new(dict(batch), compute_force=True)
        de = float((a["energy"] - b["energy"]).abs().max()); df = float((a["forces"] - b["forces"]).abs().max())
        assert de < 1e-9 and df < 1e-9, (run.name, de, df)
        new = new.cpu()
        torch.save(new, run / "model_dscc.pt")
        torch.save({"h0": new.h0.state_dict(), "q0_pristine": new.q0_pristine, "pristine_atoms": new.pristine_atoms},
                   run / "h0_state.pt")
        print(run.name, "converted; energy/force diff", f"{de:.1e}", f"{df:.1e}", flush=True)


if __name__ == "__main__":
    main()
