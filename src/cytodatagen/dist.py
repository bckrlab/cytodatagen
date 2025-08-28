import abc
import numpy as np
import numpy.typing as npt
import scipy.stats as stats


class Distribution(abc.ABC):
    """Base class for random distributions, since scipy's new API still seems to be a bit unstable."""

    @abc.abstractmethod
    def sample(self, n: int = 1, rng=None) -> np.ndarray:
        pass

    def __call__(self, n: int = 1, rng=None) -> np.ndarray:
        return self.sample(n, rng=rng)


class MultivariateNormal(Distribution):
    """Multivariate Normal Distribution."""

    def __init__(self, mean, cov):
        super().__init__()
        self.mean = mean
        self.cov = cov

    def sample(self, n: int = 1, rng=None):
        rng = np.random.default_rng(rng)
        # for consistency, we want to return shape (m) if n==1, else (n,m)
        size = None if n == 1 else n
        x = rng.multivariate_normal(mean=self.mean, cov=self.cov, size=size)
        return x


class JoinedDistribution(Distribution):
    """Join distributions together."""

    def __init__(self, dists: list[Distribution]):
        self.dists = dists

    def sample(self, n=1, rng=None):
        rng = np.random.default_rng(rng)
        xs = [dist.sample(n, rng=rng) for dist in self.dists]
        if n == 1:
            xs = np.concatenate(xs)
        else:
            xs = np.concatenate(xs, axis=1)
        return xs
