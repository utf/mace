"""Plan section 4 / 6: states, strata and geometry-state groups formed before the split;
size-grouped batches."""
import numpy as np
import pytest
import torch
from ase import Atoms

from mace.modules.dscc import data as dd


def _atoms(n_cl=3, charge=0, counts=(0, 0, 0, 0), seed=0, source="10"):
    rng = np.random.default_rng(seed)
    atoms = Atoms("Cs1Pb1" + f"Cl{n_cl}", positions=rng.random((2 + n_cl, 3)) * 5.0,
                  cell=np.eye(3) * 5.6, pbc=True)
    atoms.info.update({"carrier_counts": np.array(counts), "cell_charge": charge,
                       "config_type": "toy", "source_dir": source, "REF_energy": -1.0})
    atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))
    return atoms


class TestMeta:
    def test_state_stratum_and_group(self):
        a = _atoms(counts=(0, 0, 1, 0), charge=1)
        m = dd.frame_meta(0, a, "CsPbCl3", pristine_atoms=5)
        assert m.state.Q == 1 and m.n_ref == 1 + 4 + 21 and m.stratum[2] == 1
        assert m.stratum[0] == "10" and m.stratum[4] == dd.CELL_CONVENTION and m.stratum[5] == 1
        # The same geometry under another state is the same group.
        b = a.copy(); b.info.update({"carrier_counts": np.array([0, 0, 0, 0]), "cell_charge": 0})
        assert dd.frame_meta(1, b, "CsPbCl3", 5).group == m.group
        c = _atoms(seed=1)
        assert dd.frame_meta(2, c, "CsPbCl3", 5).group != m.group
        with pytest.raises(ValueError):
            dd.frame_meta(3, _atoms(counts=(0, 0, 1, 0), charge=0), "CsPbCl3", 5)

    def test_split_keeps_groups_whole(self):
        frames = []
        for s in range(20):
            a = _atoms(seed=s); frames.append(a)
            b = a.copy(); b.info.update({"carrier_counts": np.array([0, 0, 1, 0]), "cell_charge": 1})
            frames.append(b)
        metas = [dd.frame_meta(i, f, "CsPbCl3", 5) for i, f in enumerate(frames)]
        folds = dd.split_by_group(metas, (0.6, 0.2, 0.2), seed=3)
        assert sum(len(f) for f in folds) == 40 and all(len(f) % 2 == 0 for f in folds)
        dd.assert_no_group_split(metas, folds)
        with pytest.raises(ValueError):
            dd.assert_no_group_split(metas, [[0], [1]])


class TestBatches:
    def test_size_grouped_sampler(self):
        sizes = [79] * 7 + [159] * 3 + [80] * 4
        sampler = dd.SizeGroupedSampler(sizes, batch_size=4, seed=1)
        batches = list(sampler)
        assert sorted(i for b in batches for i in b) == list(range(14))
        for b in batches:
            assert len({sizes[i] for i in b}) == 1 and len(b) <= 4
        sampler.set_epoch(1)
        assert list(sampler) != batches

    def test_atomic_data_carries_state_and_key(self):
        from mace import tools
        z_table = tools.AtomicNumberTable([17, 55, 82])
        ds = dd.atomic_data([_atoms(counts=(0, 0, 1, 0), charge=1)], z_table, r_cut=6.0)
        d = ds[0]
        assert d.carrier_counts.reshape(-1).tolist() == [0.0, 0.0, 1.0, 0.0]
        assert int(d.frame_key) != 0 and int(d.frame_index) == 0
        assert float(d.energy) == -1.0 and d.edge_index.shape[0] == 2
