import abc
import json
from typing import Self

import pandas as pd
from cytodatagen.io import NumpyJSONEncoder
from cytodatagen.markers import MarkerDistribution, NamedMarkerDistribution
from cytodatagen.dist import MultivariateNormal


import anndata as ad
import numpy as np
import numpy.typing as npt


class CellPopulation(abc.ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def sample(self, n: int = 1, rng=None) -> ad.AnnData:
        """Samples n cells from the corresponding marker distributions."""
        rng = np.random.default_rng(rng)
        df = self._sample(n, rng)
        df.index = [f"cell_{i}" for i in range(len(df))]
        adata = ad.AnnData(df)
        adata.obs["ct_name"] = self.name
        return adata

    @abc.abstractmethod
    def _sample(self, n: int = 1, rng=None) -> pd.DataFrame:
        pass

    def to_dict(self) -> dict:
        d = {
            "name": self.name
        }
        return d


class DistributionPopulation(CellPopulation):
    def __init__(self, name: str, dist: MarkerDistribution):
        """A population of cells with specific marker distributions."""
        super().__init__(name)
        self.dist = dist

    def _sample(self, n: int = 1, rng=None) -> ad.AnnData:
        """Samples n cells from the corresponding marker distributions."""
        rng = np.random.default_rng(rng)
        df = self.dist.sample(n, rng)
        return df


class ControlPopulation(CellPopulation):
    """Populations with marker values following a multivariate normal distribution."""

    def __init__(
        self, name: str, markers: list[str], mean: np.ndarray = None, var: np.ndarray = None, cov_norm: np.ndarray = None,
    ):
        super().__init__(name)
        self.markers = markers
        self._mean = mean.copy()
        self._var = var.copy()
        self._cov_norm = cov_norm.copy()

    def _sample(self, n=1, rng=None) -> pd.DataFrame:
        rng = np.random.default_rng(rng)
        df = self.dist.sample(n, rng)
        return df

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def var(self) -> np.ndarray:
        return self._var.copy()

    @property
    def cov_norm(self) -> np.ndarray:
        return self._cov_norm.copy()

    @property
    def cov(self) -> np.ndarray:
        D = np.diag(self.var)
        return D @ self.cov_norm @ D

    @property
    def dist(self) -> MarkerDistribution:
        dist = MultivariateNormal(self.mean, self.cov)
        mdist = NamedMarkerDistribution(markers=self.markers, dist=dist)
        return mdist

    def to_dict(self):
        d = {
            "name": self.name,
            "markers": self.markers,
            "mean": self.mean,
            "var": self.var,
            "cov_norm": self.cov_norm,
        }
        return d


class ControlPopulationBuilder:
    """Builds a new MultivariateNormalControl from Parameters."""

    def __init__(
        self, name: str, markers: list[str],
        mean_loc: float = 5.0, mean_scale: float = 1.0,
        scale_low: float = 1.0, scale_high: float = 1.0
    ):
        self.name = name
        self.markers = markers
        self.mean_loc = mean_loc
        self.mean_scale = mean_scale
        self.scale_low = scale_low
        self.scale_high = scale_high

    def build(self, rng=None) -> ControlPopulation:
        rng = np.random.default_rng(rng)
        mean = rng.normal(self.mean_loc, self.mean_scale, size=len(self.markers))
        var = rng.uniform(low=self.scale_low, high=self.scale_high, size=len(self.markers))
        cov_norm = rng.normal(size=(len(self.markers), len(self.markers)))
        cov_norm = cov_norm @ cov_norm.T
        # norm diagonals, so that we have more control via var
        D = np.diag(cov_norm)
        cov_norm = cov_norm / D
        return ControlPopulation(self.name, self.markers, mean, var, cov_norm)


class SignalPopulation(CellPopulation):
    """Derives a population with signal shift from a MultivariateNormalControl."""

    def __init__(self, control: ControlPopulation, signal_markers: npt.ArrayLike, mean: np.ndarray, var: np.ndarray):
        super().__init__(control.name)
        self.control = control
        self.signal_markers = signal_markers
        self._mean = mean.copy()
        self._var = var.copy()

    def _sample(self, n=1, rng=None):
        rng = np.random.default_rng(rng)
        df = self.dist.sample(n, rng)
        return df

    @property
    def mean(self):
        mean = self.control.mean
        mean[self.signal_markers] = self._mean[self.signal_markers]
        return mean

    @property
    def var(self):
        var = self.control.var
        var[self.signal_markers] = self._var[self.signal_markers]

    @property
    def cov_norm(self):
        return self.control.cov_norm

    @property
    def cov(self):
        D = np.diag(self.var)
        return D @ self.cov_norm @ D

    def dist(self) -> MarkerDistribution:
        dist = MultivariateNormal(self.mean, self.cov)
        mdist = NamedMarkerDistribution(markers=self.markers, dist=dist)
        return mdist

    def to_dict(self):
        d = {
            "name": self.name,
            "signal_markers"
            "control": self.control.to_dict(),
            "mean": self.mean,
            "var": self.var
        }
        return d


class SignalPopulationBuilder:
    def __init__(
        self, control: ControlPopulation, signal_markers: npt.ArrayLike,
        mean_loc: float = 5.0, mean_scale: float = 1.0,
        scale_low: float = 1.0, scale_high: float = 1.0
    ):
        self.control = control
        self.signal_markers = signal_markers
        self.mean_loc = mean_loc
        self.mean_scale = mean_scale
        self.scale_low = scale_low
        self.scale_high = scale_high

    def build(self, rng=None) -> Self:
        rng = np.random.default_rng(rng)
        markers = self.control.markers
        mean = rng.normal(self.mean_loc, self.mean_scale, size=len(markers))
        var = rng.uniform(low=self.scale_low, high=self.scale_high, size=len(markers))
        return SignalPopulation(self.control, self.signal_markers, mean, var)
