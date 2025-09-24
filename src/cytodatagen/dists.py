import abc
import numpy as np
import numpy.typing as npt
import scipy.stats as stats

from cytodatagen.registry import dist_registry


class Distribution(abc.ABC):
    """Base class for random distributions, since scipy's new API still seems to be a bit unstable."""

    @abc.abstractmethod
    def sample(self, n: int = 1, rng=None) -> np.ndarray:
        pass

    def __call__(self, n: int = 1, rng=None) -> np.ndarray:
        return self.sample(n, rng=rng)

    def to_dict(self) -> dict:
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, data):
        raise NotImplementedError()


@dist_registry.register_class("mv_normal")
class MultivariateNormal(Distribution):
    """Multivariate Normal Distribution."""

    def __init__(self, mean: npt.ArrayLike, cov: npt.ArrayLike):
        super().__init__()
        self.mean = np.asarray(mean)
        self.cov = np.asarray(cov)

    def sample(self, n: int = 1, rng=None):
        rng = np.random.default_rng(rng)
        # for consistency, we want to return shape (m) if n==1, else (n,m)
        size = None if n == 1 else n
        x = rng.multivariate_normal(mean=self.mean, cov=self.cov, size=size)
        return x

    def to_dict(self):
        data = {
            "_target_": "mv_normal",
            "mean": self.mean.tolist(),
            "cov": self.mean.tolist()
        }
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(data["mean"], data["cov"])


@dist_registry.register_class("joined_dist")
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

    def to_dict(self):
        dists = [dist.to_dict() for dist in self.dists]
        data = {
            "_target_": "joined_dist",
            "dists": dists
        }
        return data

    @classmethod
    def from_dict(cls, data):
        dists = []
        for dist in data["dists"]:
            key = dist["_target_"]
            d = dist_registry.get(key).from_dict(dist)
            dists.append(d)
        return cls(dists)
