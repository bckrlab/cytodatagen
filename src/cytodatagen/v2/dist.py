import abc
import pandas as pd
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
        x = rng.multivariate_normal(mean=self.mean, cov=self.cov, size=n)
        return x


class JoinedDistribution(Distribution):
    """Join distributions together."""

    def __init__(self, dists: list[Distribution]):
        self.dists = dists

    def sample(self, n=1, rng=None):
        rng = np.random.default_rng(rng)
        xs = [dist.sample(n, rng=rng) for dist in self.dists]
        return np.stack(xs)


class MarkerDistribution(abc.ABC):
    """Samples named marker values from their distribution."""

    @abc.abstractmethod
    def sample(self, n: int = 1, rng=None) -> pd.DataFrame:
        """Samples n new marker values and returns a DataFrame."""
        pass

    @property
    @abc.abstractmethod
    def markers(self) -> list:
        """Returns the marker names of this distribution."""
        pass


class NamedMarkerDistribution(MarkerDistribution):
    def __init__(self, markers: str | npt.ArrayLike, dist: Distribution):
        if isinstance(markers, str):
            markers = [markers]
        self._markers = np.asarray(markers)
        self.dist = dist

    def sample(self, n=1, rng=None):
        rng = np.random.default_rng(rng)
        x = np.asarray(self.dist.sample(n))
        if x.ndim == 1:
            x = x.reshape(n, 1)
        if x.shape[-1] != len(self.markers):
            raise RuntimeError("dimensions of data and markers doesn't match")
        df = pd.DataFrame(x, columns=self.markers)
        return df

    @property
    def markers(self):
        return self._markers.tolist()


class JoinedMarkerDistribution(MarkerDistribution):
    """For joining independent markers with different distributions together."""

    def __init__(self, dists: list[MarkerDistribution]):
        self.dists = dists

    def sample(self, n=1, rng=None):
        rng = np.random.default_rng(rng)
        dfs = [dist.sample(n, rng) for dist in self.dists]
        return pd.concat(dfs, axis=1)

    @property
    def markers(self):
        markers = [m for dist in self.dists for m in dist.markers]
        return markers
