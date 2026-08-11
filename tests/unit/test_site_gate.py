"""The seed anneal must gate on SITE structure, not on gap magnitude.

A gap magnitude cannot distinguish a species gap from a site gap, and a species gap is free
to produce. Measured on the collapsed run: logits separated Cs from Pb and Cl by 17.4 while
varying within Cs by 0.097 -- a factor of 180 -- and the raw gap read 24.5 at epoch 1, so the
seed was withdrawn immediately and attention locked onto the wrong sublattice for the rest of
training. The short-range sibling reached only 6.5 by epoch 5, kept its seed through epoch 6,
and localised correctly in exactly that window.

``site_logit_structure`` must therefore read ~0 for a pure species ordering however large,
and rise when real site structure appears.
"""

import pytest
import torch

from mace.modules.defect_seed import site_logit_structure


@pytest.fixture(autouse=True)
def _double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _cell(n_per_species=(16, 16, 47)):
    """One graph, three species, in the CsPbCl3 proportions of a 79-atom V_Cl cell."""
    species = torch.cat(
        [torch.full((n,), z, dtype=torch.long) for z, n in enumerate(n_per_species)]
    )
    node_attrs = torch.zeros(len(species), len(n_per_species))
    node_attrs[torch.arange(len(species)), species] = 1.0
    batch = torch.zeros(len(species), dtype=torch.long)
    return species, node_attrs, batch


def test_pure_species_ordering_reads_zero():
    """The exact failure: a huge elemental split with no within-species variation."""
    species, node_attrs, batch = _cell()
    logits = torch.zeros(len(species), 4)
    # Cs +16.54, Pb -2.39, Cl -0.90 -- the measured collapsed logits, constant within species.
    for index, value in enumerate((16.54, -2.39, -0.90)):
        logits[species == index] = value
    site = site_logit_structure(logits, node_attrs, batch, 1)
    assert float(site.abs().max()) < 1e-9, (
        f"a pure species ordering must contribute nothing, got {site.tolist()}"
    )


def test_site_structure_is_detected():
    """Two atoms raised above their own species reads as real structure."""
    species, node_attrs, batch = _cell()
    logits = torch.zeros(len(species), 4)
    for index, value in enumerate((16.54, -2.39, -0.90)):
        logits[species == index] = value
    # Two Pb lifted by 5 -- the vacancy shell.
    shell = torch.nonzero(species == 1).flatten()[:2]
    logits[shell] += 5.0
    site = site_logit_structure(logits, node_attrs, batch, 1)
    assert float(site.min()) > 0.1, f"site structure not detected: {site.tolist()}"


def test_it_is_blind_to_the_size_of_the_species_split():
    """Scaling the elemental ordering by 100x must not move the readout at all."""
    species, node_attrs, batch = _cell()
    results = []
    for scale in (1.0, 100.0):
        logits = torch.zeros(len(species), 4)
        for index, value in enumerate((16.54, -2.39, -0.90)):
            logits[species == index] = value * scale
        shell = torch.nonzero(species == 1).flatten()[:2]
        logits[shell] += 5.0
        results.append(site_logit_structure(logits, node_attrs, batch, 1))
    assert torch.allclose(results[0], results[1], atol=1e-9), (
        f"readout moved with the species split: {results[0].tolist()} vs "
        f"{results[1].tolist()}"
    )


def test_raw_gap_would_have_been_fooled():
    """Guard the premise: max-minus-mean DOES track the species split, which is the bug."""
    species, node_attrs, batch = _cell()
    gaps = []
    for scale in (1.0, 100.0):
        logits = torch.zeros(len(species), 4)
        for index, value in enumerate((16.54, -2.39, -0.90)):
            logits[species == index] = value * scale
        gaps.append(float((logits.max(dim=0).values - logits.mean(dim=0)).max()))
    assert gaps[1] > 10 * gaps[0], (
        "if the raw gap did not track the species split there would be nothing to fix"
    )


def test_single_atom_species_do_not_dilute_the_average():
    """A species present once has no spread to measure and must not be averaged in."""
    species, node_attrs, batch = _cell(n_per_species=(16, 16, 1))
    logits = torch.zeros(len(species), 4)
    for index, value in enumerate((16.54, -2.39, -0.90)):
        logits[species == index] = value
    shell = torch.nonzero(species == 1).flatten()[:2]
    logits[shell] += 5.0
    with_singleton = site_logit_structure(logits, node_attrs, batch, 1)

    keep = species != 2
    node_attrs_two = node_attrs[keep][:, :2]
    without = site_logit_structure(
        logits[keep], node_attrs_two, batch[keep], 1
    )
    assert torch.allclose(with_singleton, without, atol=1e-9), (
        f"the singleton species changed the readout: {with_singleton.tolist()} vs "
        f"{without.tolist()}"
    )


def test_batches_are_independent():
    species, node_attrs, _ = _cell()
    logits = torch.zeros(len(species), 4)
    for index, value in enumerate((16.54, -2.39, -0.90)):
        logits[species == index] = value
    shell = torch.nonzero(species == 1).flatten()[:2]
    flat = site_logit_structure(logits, node_attrs, torch.zeros(len(species), dtype=torch.long), 1)
    structured = logits.clone()
    structured[shell] += 5.0

    both = torch.cat([logits, structured])
    attrs = torch.cat([node_attrs, node_attrs])
    batch = torch.cat(
        [torch.zeros(len(species), dtype=torch.long), torch.ones(len(species), dtype=torch.long)]
    )
    result = site_logit_structure(both, attrs, batch, 2)
    assert float(result[0].abs().max()) < 1e-9, "flat graph contaminated by its neighbour"
    assert float(result[1].min()) > 0.1, "structured graph washed out by its neighbour"
    assert torch.allclose(result[0], flat.squeeze(0), atol=1e-9)
