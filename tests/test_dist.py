

import numpy as np
import pytest

from cytodatagen.dists import JoinedDistribution, MultivariateNormal


@pytest.fixture
def n():
    return 10


@pytest.fixture
def m():
    return 5


@pytest.fixture
def mean(m):
    return np.arange(m)


@pytest.fixture
def cov(m):
    return np.eye(m)


@pytest.fixture
def dist(mean, cov) -> MultivariateNormal:
    return MultivariateNormal(mean, cov)


def test_multivariate_normal(m, n, dist):
    x = dist.sample(rng=19)
    assert isinstance(x, np.ndarray)
    assert x.shape == (m,)
    x = dist.sample(n=n, rng=19)
    assert isinstance(x, np.ndarray)
    assert x.shape == (n, m)


def test_joined_dist(m, n, dist):
    dist = JoinedDistribution(dists=[dist, dist])
    x = dist.sample(rng=19)
    assert isinstance(x, np.ndarray)
    assert x.shape == (2 * m,)
    x = dist.sample(n=n, rng=19)
    assert isinstance(x, np.ndarray)
    assert x.shape == (n, 2 * m)
