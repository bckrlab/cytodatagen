import numpy as np
import numpy.typing as npt
import pandas as pd
import abc

from cytodatagen.dist import Distribution


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
            x = x.reshape(n, -1)
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
