import abc
import pandas as pd
import numpy as np
import numpy.typing as npt
import scipy.stats as stats
import anndata as ad
import dataclasses as dc


from .transforms import BatchTransform, ComposedTransform, ExpTransform, SinhTransform, Transform, NoiseTransform
from .dist import MultivariateNormal, MarkerDistribution, NamedMarkerDistribution
from tqdm import tqdm


class CellPopulation:
    def __init__(self, name: str, dist: MarkerDistribution):
        """A population of cells with specific marker distributions."""
        self.name = name
        self.dist = dist

    def sample(self, n: int = 1, rng=None) -> ad.AnnData:
        """Samples n cells from the corresponding marker distributions."""
        rng = np.random.default_rng(rng)
        df = self.dist.sample(n, rng)
        df.index = [f"cell_{i}" for i in range(len(df))]
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
        adatas = {}
        dist = self.sample_dist(n, rng)
        for n, population in zip(dist, self.populations):
            adata = population.sample(n, rng)
            adatas[population.name] = adata
        adata = ad.concat(adatas, axis=0, index_unique="_")
        adata.obs["label"] = self.name
        adata.obs.index = [f"cell_{i}" for i in range(len(adata.obs))]
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


@dc.dataclass
class CytoDataGenBuilderConfig:
    n_class: int = 2
    n_samples_per_class: int = 30
    n_markers: int = 30
    marker_names: list[str] = None
    class_names: list[str] = None
    ct_names: list[str] = None
    n_cells_min: int = 10_000
    n_cells_max: int = 10_000
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
        signal_markers = np.stack([rng.choice(np.arange(self.config.n_markers), size=(self.config.n_signal_ct, self.config.n_signal_markers), replace=False) for i in range(self.config.n_class)])

        # replace parameters with signal distributions
        # numpy will automatically broadcast the signal indices to (n_signal_class, n_signal_ct, n_signal_markers)
        mu_idx = (signal_class, signal_ct[signal_class], signal_markers[signal_class])
        mu[mu_idx] = mu_signal[mu_idx]
        # signal markers in signal_ct will also express different covariance
        sigma_idx = (signal_class, signal_ct[signal_class], signal_markers[signal_class], signal_markers[signal_class])
        # a bit tricky since we need to ensure cov is still PSD
        sigma[sigma_idx] = sigma[sigma_idx] + sigma_signal[sigma_idx]

        alpha = self.config.ct_alpha
        alpha = np.array([alpha for i in range(self.config.n_ct)]) if np.isscalar(alpha) else np.asarray(alpha)
        alpha = alpha[np.newaxis].repeat(self.config.n_class, axis=0)

        classes = []
        for i in range(self.config.n_class):
            pops = []
            for j in range(self.config.n_ct):
                dist = MultivariateNormal(mean=mu[i, j], cov=sigma[i, j])
                mdist = NamedMarkerDistribution(marker_names, dist)
                pop = CellPopulation(ct_names[j], mdist)
                pops.append(pop)
            subject_class = SubjectClass(f"class_{i}", alpha[i], pops)
            classes.append(subject_class)

        generator = CytoDataGen(
            n_samples_per_class=self.config.n_samples_per_class,
            n_cells_min=self.config.n_cells_min,
            n_cells_max=self.config.n_cells_max,
            classes=classes,
            transform=xform
        )
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
        sigma = sigma @ sigma.transpose(0, 2, 1)
        # normalize diagnoals
        for i, x in enumerate(sigma):
            diag = np.diag(x)
            sigma[i] = x / np.outer(diag, diag)
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
            marker_names = [f"cd_{i}" for i in range(self.config.n_markers)]
        return marker_names

    def build_ct_names(self, rng=None):
        rng = np.random.default_rng()
        ct_names = [f"ct_{i}" for i in range(self.config.n_ct)]
        return ct_names


class CytoDataGen:

    def __init__(self, classes: list[SubjectClass], n_samples_per_class: int = 30, n_cells_min=10_000, n_cells_max: int = 10_000, transform: Transform = None):
        self.classes = classes
        self.transform = transform
        self.n_samples_per_class = n_samples_per_class
        self.n_cells_min = n_cells_min
        self.n_cells_max = n_cells_max

    def generate(self, rng=None, with_progress=True) -> ad.AnnData:
        rng = np.random.default_rng(rng)
        adatas = []

        subject_id = 0
        total = len(self.classes) * self.n_samples_per_class
        pbar = tqdm(total=total, disable=not with_progress)
        for cls in self.classes:
            pbar.set_description(f"class {cls.name}")
            for i in range(self.n_samples_per_class):
                n = rng.integers(self.n_cells_min, self.n_cells_max, endpoint=True)
                adata = cls.sample(n=n, rng=rng)
                adata.obs["subject_id"] = subject_id
                adata.obs = adata.obs.add_prefix(f"subject_{subject_id}_", axis=0)
                adatas.append(adata)
                subject_id += 1
                pbar.update()
        pbar.close()

        adata = ad.concat(adatas)

        if self.transform is not None:
            adata = self.transform(adata, rng=rng)

        return adata
