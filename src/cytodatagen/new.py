import abc
import pandas as pd
import numpy as np
import numpy.typing as npt
import scipy.stats as stats
import anndata as ad
import dataclasses as dc


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
        # sum of dist might not match n due to rounding errors
        if remainder > 0:
            leftover = rng.choice(np.arange(len(self.alpha)), remainder, replace=True)
            values, counts = np.unique(leftover, return_counts=True)
            dist[values] += counts
        assert dist.sum() == n
        return dist


class Transform(abc.ABC):
    @abc.abstractmethod
    def apply(self, adata: ad.AnnData, rng=None) -> ad.AnnData:
        pass

    def __call__(self, adata: ad.AnnData, rng=None):
        return self.apply(adata, rng)


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
        # randomly assign subjects to batches
        subject_ids = adata.obs["subject_id"]
        ids = subject_ids.unique()
        batch_ids = rng.permuted(np.arange(len(ids)) % self.n_batch)
        batch_shifts = rng.normal(scale=self.scale, size=self.n_batch)[batch_ids]
        df = pd.DataFrame({"batch_id": batch_ids, "batch_shift": batch_shifts}, index=ids)
        # apply batch shifts
        adata.X += df.loc[subject_ids]["batch_shift"].to_numpy()[:, np.newaxis]
        adata.obs["batch_id"] = df.loc[subject_ids]["batch_id"].to_numpy()
        adata.obs["batch_shift"] = df.loc[subject_ids]["batch_shift"].to_numpy()
        return adata


class ComposedTransform(Transform):
    def __init__(self, transforms: list[Transform]):
        super().__init__()
        self.transforms = transforms

    def apply(self, adata, rng=None):
        for xform in self.transforms:
            adata = xform(adata, rng=rng)
        return adata


@dc.dataclass
class CytoDataGenBuilderConfig:
    n_class: int = 2
    n_markers: int = 30
    marker_names: list[str] = None
    class_names: list[str] = None
    ct_names: list[str] = None
    n_ct: int = 5
    n_signal_markers: int = 3
    n_signal_ct: int = 2
    ct_alpha: float | npt.ArrayLike = 5.0
    ct_mean_loc: float = 3
    ct_mean_scale: float = 1.0
    ct_scale: float = 1.0
    transforms: dict = dc.field(default_factory=dict)


class CytoDataGenBuilder:
    """Assists in constructing a new CytoDataGen object from a config."""

    _xforms = {
        "sinh": SinhTransform,
        "exp": ExpTransform,
        "noise": NoiseTransform,
        "batch": BatchTransform
    }

    def __init__(self, config: CytoDataGenBuilderConfig):
        self.config = config

    def check_config(self):
        if self.config.marker_names is not None and len(self.config.marker_names) != self.config.n_markers:
            raise ValueError("length of marker_names does not match n_markers")

    def build(self, rng=None):
        rng = np.random.default_rng(rng)
        self.check_config()
        marker_names = self.build_marker_names()
        ct_names = self.build_ct_names()
        xform = self.build_transform()
        # generate mu and sigma of control class
        mu = self.build_class_mu(rng)[np.newaxis].repeat(self.config.n_class, axis=0)
        sigma = self.build_class_sigma(rng)[np.newaxis].repeat(self.config.n_class, axis=0)
        # generate signals
        mu_signal = np.stack([self.build_class_mu(rng) for i in range(self.config.n_class)])
        sigma_signal = np.stack([self.build_class_sigma(rng) for i in range(self.config.n_class)])
        signal_class = np.arange(1, self.config.n_class)
        signal_ct = np.stack([rng.choice(np.arange(self.config.n_ct), size=self.config.n_signal_ct, replace=False) for i in range(self.config.n_class)])[:, :, np.newaxis]
        signal_markers = np.stack([rng.choice(np.arange(self.config.n_markers), size=(self.config.n_signal_markers, self.config.n_signal_ct), replace=False) for i in range(self.config.n_class)])
        # replace parameters with signal distributions
        # numpy will automatically broadcast the signal indices to (n_signal_class, n_signal_ct, n_signal_markers)
        mu_idx = (signal_class, signal_ct[signal_class], signal_markers[signal_class])
        mu[mu_idx] = mu_signal[mu_idx]
        # signal markers in signal_ct will also express different covariance
        sigma_idx = (signal_class, signal_ct[signal_class], signal_markers[signal_class], signal_markers[signal_class])
        sigma[sigma_idx] = sigma_signal[sigma_idx]

        alpha = self.config.ct_alpha
        alpha = np.array([alpha for i in range(self.config.n_ct)]) if np.isscalar(alpha) else np.asarray(alpha)
        alpha = alpha[np.newaxis].repeat(self.config.n_class, axis=0)

        classes = []
        for i in range(self.config.n_class):
            pops = []
            for j in range(self.config.n_ct):
                dist = stats.Normal(mu=mu[i, j], sigma=sigma[i, j])
                mdist = StatsDist(marker_names, dist)
                pop = CellPopulation(ct_names[j], mdist)
                pops.append(pop)
            subject_class = SubjectClass(f"class_{i}", alpha[i], pops)
            classes.append(subject_class)
        generator = CytoDataGen(classes=classes, transform=xform)
        return generator

    def build_class_mu(self, rng=None):
        rng = np.random.default_rng(rng)
        size = [self.config.n_ct, self.config.n_markers]
        mu = rng.normal(loc=self.config.ct_mean_loc, scale=self.config.ct_mean_scale, size=size)
        return mu

    def build_class_sigma(self, rng=None):
        rng = np.random.default_rng(rng)
        size = [self.config.n_ct, self.config.n_markers, self.config.n_markers]
        sigma = rng.normal(size=size)
        # trick to make sure sigma is valid covariance matrix
        sigma = sigma @ sigma.T
        diag = np.diag(sigma)
        sigma = sigma / np.outer(diag, diag)
        sigma = self.config.ct_scale * sigma * self.config.ct_scale
        return sigma

    def build_transform(self):
        xforms = []
        for key, value in self.config.transforms.items():
            xform_cls = self._xforms[key]
            xform = xform_cls(**value)
            xforms.append(xform)
        xform = ComposedTransform(xforms)
        return xform

    def build_marker_names(self):
        marker_names = self.config.marker_names
        if marker_names is None:
            marker_names = [f"cd_{i}" for i in self.config.n_markers]
        return marker_names

    def build_ct_names(self, rng=None):
        rng = np.random.default_rng()
        ct_names = [f"ct_{i}" for i in self.config.n_ct]
        return ct_names


class CytoDataGen:

    def __init__(self, classes: list[SubjectClass], transform: Transform = None):
        self.classes = classes
        self.transform = transform

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

        if self.transform is not None:
            adata = self.transform(adata, rng=rng)

        return adata


if __name__ == "__main__":
    dist = StatsDist(["cd_1", "cd_2", "cd_3"], dist=stats.Normal(mu=[0, 1, 2]))
    print(dist.sample(10).head())
    pop = CellPopulation("a_cell", dist=dist)
    sclass = SubjectClass("positive", alpha=[2], populations=[pop])
    s = sclass.sample()
    print(s.obs.head())
    s2 = sclass.sample()
    adata = ad.concat({"s1": s, "s2": s2}, label="subject_id")
    xform = BatchTransform(n_batch=2)
    adata = xform.apply(adata)
    print(adata.obs.head())
    print(adata.obs.describe())
