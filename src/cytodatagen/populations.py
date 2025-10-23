"""
Provides cell populations to sample cells with marker values from as AnnData objects.
"""

import abc
import pandas as pd
import anndata as ad
import numpy as np
import numpy.typing as npt

from typing import Self
from cytodatagen.markers import MarkerDistribution, NamedMarkerDistribution
from cytodatagen.dists import MultivariateNormal
from cytodatagen.registry import pop_registry, mdist_registry


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
        adata.obs["pop_name"] = self.name
        return adata

    @abc.abstractmethod
    def _sample(self, n: int = 1, rng=None) -> pd.DataFrame:
        pass

    def to_dict(self) -> dict:
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, data: dict):
        raise NotImplementedError()


@pop_registry.register_class("dist_pop")
class DistributionPopulation(CellPopulation):

    dists = {
        "mv_normal": MultivariateNormal
    }

    def __init__(self, name: str, mdist: MarkerDistribution):
        """A population of cells with specific marker distributions."""
        super().__init__(name)
        self.mdist = mdist

    def _sample(self, n: int = 1, rng=None):
        """Samples n cells from the corresponding marker distributions."""
        rng = np.random.default_rng(rng)
        df = self.mdist.sample(n, rng)
        return df

    def to_dict(self):
        data = {
            "_target_": "dist_pop",
            "name": self.name,
            "mdist": self.mdist.to_dict()
        }
        return data

    @classmethod
    def from_dict(cls, data):
        mdist_data = data["mdist"]
        key = mdist_data["_target_"]
        mdist = mdist_registry.get(key).from_dict(mdist_data)
        return cls(data["name"], mdist)


@pop_registry.register_class("control_pop")
class ControlPopulation(CellPopulation):
    """Populations with marker values following a multivariate normal distribution."""

    def __init__(
        self, name: str, markers: list[str], mean: npt.ArrayLike, var: npt.ArrayLike, cov_norm: npt.ArrayLike,
    ):
        super().__init__(name)
        self.markers = markers
        self._mean = np.asarray(mean).copy()
        self._var = np.asarray(var).copy()
        self._cov_norm = np.asarray(cov_norm).copy()

    def _sample(self, n=1, rng=None) -> pd.DataFrame:
        rng = np.random.default_rng(rng)
        df = self.mdist.sample(n, rng)
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
    def mdist(self) -> MarkerDistribution:
        dist = MultivariateNormal(self.mean, self.cov)
        mdist = NamedMarkerDistribution(markers=self.markers, dist=dist)
        return mdist

    def to_dict(self):
        data = {
            "_target_": "control_pop",
            "name": self.name,
            "markers": self.markers,
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "cov_norm": self.cov_norm.tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        pop = cls(
            name=data["name"],
            markers=data["markers"],
            mean=data["mean"],
            var=data["var"],
            cov_norm=data["cov_norm"]
        )
        return pop


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
        # watch out that cov_norm is still symmetric and PSD!
        D = np.diag(cov_norm)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
        cov_norm = D_inv_sqrt @ cov_norm @ D_inv_sqrt
        return ControlPopulation(self.name, self.markers, mean, var, cov_norm)


@pop_registry.register_class("signal_pop")
class SignalPopulation(CellPopulation):
    """Derives a population with signal shift from a MultivariateNormalControl."""

    def __init__(self, control: ControlPopulation, signal_markers: npt.ArrayLike, mean_signal: npt.ArrayLike, var_signal: npt.ArrayLike):
        super().__init__(control.name)
        self.control = control
        self.signal_markers = signal_markers
        self.mean_signal = np.asarray(mean_signal).copy()
        self.var_signal = np.asarray(var_signal).copy()

    def _sample(self, n=1, rng=None):
        rng = np.random.default_rng(rng)
        df = self.mdist.sample(n, rng)
        return df

    @property
    def mean(self):
        mean = self.control.mean
        mean[self.signal_markers] = self.mean_signal[self.signal_markers]
        return mean

    @property
    def var(self):
        var = self.control.var
        var[self.signal_markers] = self.var_signal[self.signal_markers]
        return var

    @property
    def cov_norm(self):
        return self.control.cov_norm

    @property
    def cov(self):
        D = np.diag(self.var)
        return D @ self.cov_norm @ D

    @property
    def mdist(self) -> MarkerDistribution:
        dist = MultivariateNormal(self.mean, self.cov)
        mdist = NamedMarkerDistribution(markers=self.control.markers, dist=dist)
        return mdist

    def to_dict(self):
        data = {
            "_target_": "signal_pop",
            "control": self.control.to_dict(),
            "signal_markers": self.signal_markers,
            "mean_signal": self.mean_signal.tolist(),
            "var_signal": self.var_signal.tolist()
        }
        return data

    @classmethod
    def from_dict(cls, data):
        control = ControlPopulation.from_dict(data["control"])
        pop = cls(
            control=control,
            signal_markers=data["signal_markers"],
            mean_signal=data["mean_signal"],
            var_signal=data["var_signal"]
        )
        return pop


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

    def build(self, rng=None) -> SignalPopulation:
        rng = np.random.default_rng(rng)
        markers = self.control.markers
        mean = rng.normal(self.mean_loc, self.mean_scale, size=len(markers))
        var = rng.uniform(low=self.scale_low, high=self.scale_high, size=len(markers))
        return SignalPopulation(self.control, self.signal_markers, mean, var)
