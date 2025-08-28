import pytest
import anndata as ad

from cytodatagen.populations import ControlPopulation, ControlPopulationBuilder


@pytest.fixture
def n_marker():
    return 10


@pytest.fixture
def markers(n_marker):
    return [f"cd_{i}" for i in range(n_marker)]


@pytest.fixture
def name_control():
    return "control_pop"


def test_control_pop_builder(name_control, markers):
    builder = ControlPopulationBuilder(name_control, markers)
    pop = builder.build()
    assert isinstance(pop, ControlPopulation)
    anndata = pop.sample()
    assert isinstance(anndata, ad.AnnData)


def test_signal_pop_builder():
    pass
