"""D0: the distillation set (registered enumeration in DSCC_V5_IMPLEMENTATION_SPEC.md, "D0 enumeration").

  --stage build   generate every structure family (CPU), write dataset_d0/unlabelled.xyz + manifest
  --stage label   MH-1 `omat_pbe` teacher labels (energy, forces, stress; float64, batched by size),
                  the three isolated atoms, the 90/10 split by parent -> dataset_d0/{train,valid}.xyz
Every structure carries: config_type = d0_family, d0_parent (index into the parent list, -1 for
generated), d0_seed, d0_source_charge, cell_charge 0, carrier_counts [0,0,0,0],
config_stress_weight (1 on `strained`, 0 elsewhere).
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read, write
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["build", "label"], required=True)
ap.add_argument("--out_dir", default=str(HERE / "dataset_d0"))
ap.add_argument("--teacher", default=os.path.expanduser("~/.cache/mace/macemh1model"))
ap.add_argument("--head", default="omat_pbe")
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--split_seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0, help="debug: label only this many structures")
ap.add_argument("--precision", choices=["float32", "float64"], default="float32",
                help="teacher inference precision (float32 + cueq: the teacher's own training precision; float64 is ~30x slower on the A4000)")
ap.add_argument("--cueq", type=int, default=1, help="convert the teacher to cuEquivariance for inference")
args = ap.parse_args()
OUT = Path(args.out_dir); OUT.mkdir(exist_ok=True)


def tag(a, family, parent, seed, src_q, sw):
    a.info = {k: v for k, v in a.info.items() if k in ("host",)}
    a.info.update({"config_type": family, "d0_family": family, "d0_parent": int(parent), "d0_seed": int(seed),
                   "d0_source_charge": int(src_q), "cell_charge": 0, "config_stress_weight": float(sw)})
    a.info["carrier_counts"] = np.zeros(4, dtype=np.int64)
    for k in list(a.arrays):
        if k not in ("numbers", "positions"):
            del a.arrays[k]
    return a


def build():
    pbe = read(f"{HERE}/dataset_pbe/train.xyz", ":") + read(f"{HERE}/dataset_pbe/valid.xyz", ":")
    charged = [a for a in pbe if int(a.info.get("cell_charge", 0)) != 0]
    neutral = read(f"{HERE}/dataset_e0/train.xyz", ":") + read(f"{HERE}/dataset_e0/valid.xyz", ":")
    neutral = [a for a in neutral if a.info.get("config_type") != "IsolatedAtom"]
    parents, out = [], []
    for a in charged:
        parents.append(tag(a.copy(), "charged_relabelled", len(parents), 0, int(a.info["cell_charge"]), 0.0))
    for a in neutral:
        parents.append(tag(a.copy(), "neutral", len(parents), 0, 0, 0.0))
    out += parents
    # perturbed: 9 copies per parent, 3 at each sigma
    for p, a in enumerate(parents):
        for c, sigma in enumerate([0.05] * 3 + [0.10] * 3 + [0.20] * 3):
            seed = 1000 * p + c; rng = np.random.default_rng(seed)
            b = a.copy(); b.positions = b.positions + rng.normal(0.0, sigma, b.positions.shape)
            out.append(tag(b, "perturbed", p, seed, a.info["d0_source_charge"], 0.0)); out[-1].info["d0_sigma"] = sigma
    # strained: 110 parents of each of three kinds, every k-th by index
    kinds = {"pristine80": [p for p, a in enumerate(parents) if a.info["d0_family"] == "neutral" and len(a) == 80],
             "neutral_vac": [p for p, a in enumerate(parents) if a.info["d0_family"] == "neutral" and len(a) != 80],
             "charged": [p for p, a in enumerate(parents) if a.info["d0_family"] == "charged_relabelled"]}
    modes = [("iso", +0.02), ("iso", -0.02), ("iso", +0.04), ("iso", -0.04), ("shear_xy", +0.03), ("shear_xy", -0.03)]
    for kind, idx in kinds.items():
        pick = idx[:: max(1, len(idx) // 110)][:110]
        for p in pick:
            for m, (mode, e) in enumerate(modes):
                a = parents[p].copy()
                F = np.eye(3)
                if mode == "iso":
                    F *= (1.0 + e)
                else:
                    F[0, 1] = e
                cell = np.array(a.get_cell()) @ F.T
                a.set_cell(cell, scale_atoms=True)
                out.append(tag(a, "strained", p, m, parents[p].info["d0_source_charge"], 1.0)); out[-1].info["d0_strain"] = f"{mode}{e:+.2f}"
    # tilings of the static pristine cell with one Cl removed, thermal displacements
    from mace.modules.dscc.ladder import tiling_ladder
    doc = json.load(open(HERE / "static_pristine_cell.json"))
    unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
    for t in ((1, 1, 2), (2, 2, 1), (2, 2, 2)):
        base = tiling_ladder(unit, [t], vacancy_species=17, vacancy_index=0)[0]
        for i in range(100):
            seed = 5000 + 100 * sum(t) + i; rng = np.random.default_rng(seed)
            b = base.copy(); b.positions = b.positions + rng.normal(0.0, 0.10, b.positions.shape)
            out.append(tag(b, "tilings", -1, seed, 0, 0.0)); out[-1].info["d0_tiling"] = ",".join(map(str, t))
    # polar sublattice displacements of the static 80-atom cell
    Z = {"Cs": 55, "Pb": 82, "Cl": 17}
    pats = [({s: +1.0}, s) for s in ("Cs", "Pb", "Cl")] + [({"Pb": +1.0, "Cl": -1.0}, "Pb-Cl"), ({"Cs": +1.0, "Cl": -1.0}, "Cs-Cl"), ({"Cs": +1.0, "Pb": +1.0, "Cl": -1.0}, "CsPb-Cl")]
    nums = unit.get_atomic_numbers(); k = 0
    for pat, name in pats:
        for ax in range(3):
            for d in (0.02, 0.04, 0.06, 0.08, 0.10):
                for sgn in (+1, -1):
                    if name == "CsPb-Cl" and d not in (0.04, 0.08, 0.10):
                        continue
                    b = unit.copy(); pos = b.get_positions()
                    for s, w in pat.items():
                        pos[nums == Z[s], ax] += sgn * w * d
                    b.set_positions(pos)
                    out.append(tag(b, "polar", -1, k, 0, 0.0)); out[-1].info["d0_polar"] = f"{name}:{'xyz'[ax]}:{sgn * d:+.2f}"; k += 1
    counts = {}
    for a in out:
        counts[a.info["d0_family"]] = counts.get(a.info["d0_family"], 0) + 1
    write(OUT / "unlabelled.xyz", out)
    man = {"stage": "build", "n_parents": len(parents), "counts": counts, "total": len(out), "built": time.strftime("%F %T")}
    json.dump(man, open(OUT / "manifest.json", "w"), indent=1)
    print(json.dumps(man, indent=1))


def label():
    import torch
    from mace import data as mace_data
    from mace.tools import AtomicNumberTable, torch_geometric
    dt = torch.float32 if args.precision == "float32" else torch.float64
    torch.set_default_dtype(dt)
    model = torch.load(args.teacher, map_location="cpu", weights_only=False).to(dt)
    if args.cueq and args.device.startswith("cuda"):
        from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
        model = run_e3nn_to_cueq(model, device=args.device)
    model = model.to(args.device).eval()
    heads = list(model.heads); hidx = heads.index(args.head)
    zt = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    sha = hashlib.sha256(open(args.teacher, "rb").read()).hexdigest()
    frames = read(OUT / "unlabelled.xyz", ":")
    if args.limit:
        frames = frames[: args.limit]
    iso = [Atoms(numbers=[z], positions=[[10.0, 10.5, 11.0]], cell=np.diag([20.0, 21.0, 22.0]), pbc=True) for z in (17, 55, 82)]
    for a in iso:
        tag(a, "IsolatedAtom", -1, 0, 0, 0.0); a.info["config_weight"] = 1000.0
    todo = frames + iso
    by_size = {}
    for i, a in enumerate(todo):
        by_size.setdefault(len(a), []).append(i)
    t0 = time.time(); done = 0
    for n, idx in by_size.items():
        for s in range(0, len(idx), args.batch_size):
            chunk = idx[s:s + args.batch_size]
            cfgs = [mace_data.config_from_atoms(todo[i]) for i in chunk]
            ds = [mace_data.AtomicData.from_config(c, z_table=zt, cutoff=float(model.r_max), heads=heads) for c in cfgs]
            batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=len(ds)))).to(args.device).to_dict()
            batch["head"] = torch.full((len(chunk),), hidx, dtype=torch.long, device=args.device)
            out = model(batch, training=False, compute_force=True, compute_stress=True)
            ptr = batch["ptr"].cpu().numpy()
            for g, i in enumerate(chunk):
                a = todo[i]
                a.info["REF_energy"] = float(out["energy"][g])
                a.arrays["REF_forces"] = out["forces"][ptr[g]:ptr[g + 1]].detach().cpu().double().numpy()
                a.info["REF_stress"] = out["stress"][g].detach().cpu().double().numpy().reshape(3, 3)
            done += len(chunk)
            if done % 2000 < args.batch_size:
                print(f"labelled {done}/{len(todo)} [{time.time() - t0:.0f} s]", flush=True)
    e0_pred = {int(a.numbers[0]): float(a.info["REF_energy"]) for a in iso}
    zs = [int(z) for z in model.atomic_numbers]; ae = model.atomic_energies_fn.atomic_energies
    e0_row = {z: float(ae[hidx, zs.index(z)]) for z in (17, 55, 82)}
    # split 90/10 by parent geometry, stratified by family, seed
    rng = np.random.default_rng(args.split_seed)
    parents = sorted({a.info["d0_parent"] for a in frames if a.info["d0_parent"] >= 0})
    fam_of_parent = {a.info["d0_parent"]: a.info["d0_family"] for a in frames if a.info["d0_parent"] >= 0 and a.info["d0_family"] in ("charged_relabelled", "neutral")}
    held_parent = set()
    for fam in ("charged_relabelled", "neutral"):
        ps = [p for p in parents if fam_of_parent.get(p) == fam]; rng.shuffle(ps)
        held_parent |= set(ps[: len(ps) // 10])
    held_gen = {}
    for fam in ("tilings", "polar"):
        ids = [i for i, a in enumerate(frames) if a.info["d0_family"] == fam]; rng.shuffle(ids)
        held_gen[fam] = set(ids[: len(ids) // 10])
    train, valid = [], []
    for i, a in enumerate(frames):
        held = (a.info["d0_parent"] in held_parent) if a.info["d0_parent"] >= 0 else (i in held_gen.get(a.info["d0_family"], set()))
        (valid if held else train).append(a)
    train += iso
    write(OUT / "train.xyz", train); write(OUT / "valid.xyz", valid)
    counts = {}
    for name, lst in (("train", train), ("valid", valid)):
        for a in lst:
            counts.setdefault(name, {}); counts[name][a.info["d0_family"]] = counts[name].get(a.info["d0_family"], 0) + 1
    man = json.load(open(OUT / "manifest.json"))
    man.update({"stage": "label", "teacher": args.teacher, "teacher_sha256": sha, "head": args.head, "precision": args.precision, "cueq": bool(args.cueq),
                "E0_predicted": e0_pred, "E0_stored_row": e0_row, "split_seed": args.split_seed, "split": counts,
                "n_held_parents": len(held_parent), "labelled": time.strftime("%F %T"), "seconds": time.time() - t0})
    json.dump(man, open(OUT / "manifest.json", "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("E0_predicted", "E0_stored_row", "split", "seconds")}, indent=1))


build() if args.stage == "build" else label()
