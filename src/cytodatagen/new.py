import abc
import pandas as pd
import numpy as np
import scipy.stats as stats
import anndata as ad


class MarkerDist(abc.ABC):
    @abc.abstractmethod
    def sample(self, n: int = 1, rng=None) -> pd.DataFrame:
        """Samples n new marker values and returns a DataFrame."""
        pass

    @property
    @abc.abstractmethod
    def markers(self) -> list:
        """Returns the marker names of this distribution."""
        pass


class StatsDist(MarkerDist):
    """Specifies the distribution of markers using scipy's distribution class."""

    def __init__(self, markers, dist):
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


class JoinedDist(MarkerDist):
    def __init__(self, dists: list[MarkerDist]):
        self.dists = dists

    def sample(self, n=1, rng=None):
        rng = np.random.default_rng(rng)
        dfs = []
        for dist in self.dists:
            df = dist.sample(n, rng)
            dfs.append(df)
        return pd.concat(dfs, axis=1)

    @property
    def markers(self):
        markers = [m for dist in self.dists for m in dist.markers]
        return markers


class CellPopulation:
    def __init__(self, name: str, dist: MarkerDist):
        """A population of cells with specific marker distributions."""
        self.name = name
        self.dist = dist

    def sample(self, n: int = 1, rng=None) -> ad.AnnData:
        """Samples n cells from the corresponding marker distributions."""
        rng = np.random.default_rng(rng)
        df = self.dist.sample(n, rng)
        adata = ad.AnnData(df)
        adata.obs["ct_name"] = self.name
        return adata


class SubjectClass:
    def __init__(self, name: str, alpha, populations: list[CellPopulation]):
        self.name = name
        self.alpha = np.asarray(alpha)
        self.populations = populations
        if len(self.alpha) != len(self.populations):
            raise ValueError("length mismatch of alpha prior and cell populations")

    def sample(self, n: int = 10_000, rng=None) -> ad.AnnData:
        """Samples a new subject."""
        rng = np.random.default_rng(rng)
        adatas = []
        dist = self.sample_dist(n, rng)
        for n, population in zip(dist, self.populations):
            adata = population.sample(n, rng)
            adatas.append(adata)
        adata = ad.concat(adatas, axis=0)
        adata.obs["label"] = self.name
        adata.obs = adata.obs.add_prefix("cell_", axis=0)
        return adata

    def sample_dist(self, n: int = 10_000, rng=None) -> np.ndarray:
        """Samples cell type proportions from a Dirichlet distribution."""
        rng = np.random.default_rng(rng)
        dist = np.floor(rng.dirichlet(self.alpha) * n).astype(int)
        remainder = n - dist.sum()
        if remainder > 0:
            leftover = rng.choice(np.arange(len(self.alpha)), remainder, replace=True)
            dist = dist + np.bincount(leftover)
        assert dist.sum() == n
        return dist


class Transform(abc.ABC):
    @abc.abstractmethod
    def apply(self, adata: ad.AnnData, rng=None) -> ad.AnnData:
        pass


class SinhTransform(Transform):
    def __init__(self, cofactor: float | np.ndarray = 5.0):
        super().__init__()
        self.cofactor = cofactor

    def apply(self, adata: ad.AnnData, rng=None) -> ad.AnnData:
        adata.X = self.cofactor * np.sinh(adata.X)
        return adata


class ExpTransform(Transform):
    def __init__(self):
        super().__init__()

    def apply(self, adata: ad.AnnData, rng=None) -> ad.AnnData:
        adata.X = np.exp(adata.X)
        return adata


class NoiseTransform(Transform):
    def __init__(self, snr_db: float = 20):
        super().__init__()
        self.snr_db = snr_db

    def apply(self, adata: ad.AnnData, rng=None) -> ad.AnnData:
        rng = np.random.default_rng(rng)
        signal_scale = adata.X.std(axis=0)
        noise_scale = signal_scale / self.snr
        noise = rng.normal(scale=noise_scale)
        adata.X = adata.X + noise
        adata.layers["noise"] = noise
        return adata

    @property
    def snr(self):
        return np.power(10, self.snr_db / 10.0)


class BatchTransform(Transform):
    def __init__(self, n_batch: int, scale: float = 1.0):
        super().__init__()
        self.n_batch = n_batch
        self.scale = scale

    def apply(self, adata: ad.AnnData, rng=None) -> ad.AnnData:
        rng = np.random.default_rng(rng)
        for i, (subject_id) in enumerate(adata.obs["subject_id"].unique()):
            mask = (adata.obs["subject_id"] == subject_id)
            shift = rng.normal(scale=self.scale)
            adata[:, mask] = adata[:, mask] + shift
            adata.obs.loc[mask]["batch_id"] = i
            adata.obs.loc[mask]["batch_shift"] = shift
        return adata


class CytoDataGenBuilder:
    """Assits in constructing a new CytoDataGen object from a config."""

    def __init__(self):
        pass

    def build(self):
        pass


class CytoDataGen:
    def __init__(self, classes: list[SubjectClass]):
        self.classes = classes
        pass

    def generate(self, n_samples_per_class: int, rng=None) -> ad.AnnData:
        rng = np.random.default_rng(rng)
        adatas = []

        subject_id = 0
        for cls in self.classes:
            for i in range(n_samples_per_class):
                adata = cls.sample(rng=rng)
                adata.obs["subject_id"] = subject_id
                adata.obs = adata.obs.add_prefix(f"subject_{subject_id}_")
                adatas.append(adata)
                subject_id += 1

        adata = ad.concat(adatas)
        return adata


if __name__ == "__main__":
    dist = StatsDist(["cd_1", "cd_2", "cd_3"], dist=stats.Normal(mu=[0, 1, 2]))
    print(dist.sample(10).head())
    pop = CellPopulation("a_cell", dist=dist)
    sclass = SubjectClass("positive", alpha=[2], populations=[pop])
    s = sclass.sample()
    print(s.obs.head())
