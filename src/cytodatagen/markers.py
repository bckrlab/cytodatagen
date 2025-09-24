import numpy as np
import numpy.typing as npt
import pandas as pd
import abc

from cytodatagen.dists import Distribution, MultivariateNormal
from cytodatagen.registry import dist_registry
from cytodatagen.registry import mdist_registry


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

    def to_dict(self) -> dict:
        raise NotImplementedError()

    @classmethod
    def from_dict(self, data: dict):
        raise NotImplementedError()


@mdist_registry.register_class("named_mdist")
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

    def to_dict(self):
        data = {
            "_target_": "named_mdist",
            "markers": self.markers,
            "dist": self.dist.to_dict()
        }
        return data

    @classmethod
    def from_dict(cls, data):
        dist = data["dist"]
        key = dist["_target_"]
        dist = dist_registry.get(key).from_dict(dist)
        return cls(data["markers"], dist)


@mdist_registry.register_class("joined_mdist")
class JoinedMarkerDistribution(MarkerDistribution):
    """For joining independent markers with different distributions together."""

    def __init__(self, mdists: list[MarkerDistribution]):
        self.mdists = mdists

    def sample(self, n=1, rng=None):
        rng = np.random.default_rng(rng)
        dfs = [dist.sample(n, rng) for dist in self.mdists]
        return pd.concat(dfs, axis=1)

    @property
    def markers(self):
        markers = [m for dist in self.mdists for m in dist.markers]
        return markers

    def to_dict(self):
        mdists = [mdist.to_dict() for mdist in self.mdists]
        data = {
            "_target_": "joined",
            "mdists": mdists
        }
        return super().to_dict()
