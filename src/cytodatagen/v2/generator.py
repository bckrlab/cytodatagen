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


class Subject:
    def __init__(self, name: str, alpha, populations: list[CellPopulation]):
        self.name = name
        self.alpha = np.asarray(alpha)
        self.populations = populations
        if len(self.alpha) != len(self.populations):
            raise ValueError("length mismatch of alpha prior and cell populations")

    def sample(self, n: int = 10_000, rng=None) -> ad.AnnData:
        """Samples n cells from the populations with proportions given by alpha."""
        rng = np.random.default_rng(rng)
        adatas = {}
        dist = self.sample_dist(n, rng)
        for i, (n, population) in enumerate(zip(dist, self.populations)):
            adata = population.sample(n, rng)
            adata.obs["ct_id"] = i
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
    n_marker: int = 30
    marker_names: list[str] = None
    class_names: list[str] = None
    ct_names: list[str] = None
    n_cells_min: int = 10_000
    n_cells_max: int = 10_000
    n_ct: int = 5
    n_signal_marker: int = 3
    n_signal_ct: int = 2
    ct_alpha: float | npt.ArrayLike = 5.0
    ct_mean_loc: float = 3
    ct_mean_scale: float = 1.0
    ct_scale_min: float = 0.5
    ct_scale_max: float = 2.0
    transforms: dict = dc.field(default_factory=dict)


@dc.dataclass
class SubjectBuilderConfig:
    n_marker: int = 30
    n_signal_marker: int = 3
    marker_names: list[str] = None
    ct_names: list[str] = None
    n_ct: int = 5
    n_signal_ct: int = 2
    ct_alpha: float | npt.ArrayLike = 5.0
    ct_mean_loc: float = 3
    ct_mean_scale: float = 1.0
    ct_scale_min: float = 0.5
    ct_scale_max: float = 2.0


class MultivariateNormalSubjectBuilder:
    def __init__(self, config: SubjectBuilderConfig):
        self.config = config

    def init_params(self, rng=None):
        rng = np.random.default_rng(rng)
        self.params = {}
        self.mean_control = self._make_mean(rng=rng)
        self.var_control = self._make_var(rng=rng)
        self.cov_norm = self._make_cov_norm(rng=rng)

    def _make_mean(self, rng=None):
        rng = np.random.default_rng(rng)
        size = [self.config.n_ct, self.config.n_marker]
        return rng.normal(loc=self.config.ct_mean_loc, scale=self.config.ct_mean_scale, size=size)

    def _make_cov_norm(self, rng=None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        cov_size = [self.config.n_ct, self.config.n_marker, self.config.n_marker]
        cov = rng.normal(size=cov_size)
        # trick to make sure sigma is valid covariance matrix
        cov = cov @ cov.transpose(0, 2, 1)
        # normalize diagonals for each cell type
        for i, x in enumerate(cov):
            diag = np.diag(x)
            cov[i] = x / np.outer(diag, diag)
        return cov

    def _make_var(self, rng=None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        size = (self.config.n_ct, self.config.n_marker)
        return rng.uniform(low=self.config.ct_scale_min, high=self.config.ct_scale_max, size=size)

    def _make_cov(self, sigma: np.ndarray):
        """Scale normed covariance matrix with sigma, so that each variable has variance sigma."""
        # utilize numpy broadcasting
        return sigma[:, :, np.newaxis] * self.cov_norm * sigma[:, np.newaxis, :]

    def _make_alpha(self) -> np.ndarray:
        alpha = self.config.ct_alpha
        if np.isscalar(alpha):
            alpha = np.repeat(alpha, self.config.n_ct)
        else:
            alpha = alpha.copy()
        return alpha

    def build_control(self, name: str, rng=None) -> Subject:
        """Initialize multivariate normal parameters and build a control subject."""
        rng = np.random.default_rng()
        self.init_params(rng)
        cov = self._make_cov(self.var_control)
        alpha = self._make_alpha()
        mean = self.mean_control.copy()
        params = dict(alpha=alpha, mean=mean, cov=cov)
        self.params[name] = params
        return self._build_subject(name, alpha, mean, cov)

    def build_signal(self, name: str, rng=None) -> Subject:
        """Build a signal subject with slightly modified control parameters."""
        rng = np.random.default_rng()
        # get signal cell types and markers
        signal_ct = rng.choice(np.arange(self.config.n_ct), self.config.n_signal_ct, replace=False)
        signal_marker = np.stack([rng.choice(np.arange(self.config.n_marker), self.config.n_signal_marker, replace=False) for i in signal_ct])
        # generate params for signal shifts
        mask = (signal_ct[:, np.newaxis], signal_marker)
        mean_signal = self._make_mean(rng=rng)
        var_signal = self._make_var(rng=rng)
        mean = self.mean_control.copy()
        mean[mask] = mean_signal[mask]
        var = self.var_control.copy()
        var[mask] = var_signal[mask]
        cov = self._make_cov(var)
        # change alpha for signal ct
        alpha = self._make_alpha()
        alpha[signal_ct] = np.roll(alpha[signal_ct], 1)
        params = dict(alpha=alpha, mean=mean, cov=cov, signal_ct=signal_ct, signal_marker=signal_marker)
        self.params[name] = params
        return self._build_subject(name, alpha, mean, cov)

    def _build_subject(self, name: str, alpha, mean, cov) -> Subject:
        populations = []
        for i in range(self.config.n_ct):
            dist = MultivariateNormal(mean=mean[i], cov=cov[i])
            mdist = NamedMarkerDistribution(self.config.marker_names, dist)
            pop = CellPopulation(self.config.ct_names[i], mdist)
            populations.append(pop)
        return Subject(name, alpha, populations)


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
        if self.config.marker_names is not None and len(self.config.marker_names) != self.config.n_marker:
            raise ValueError("length of marker_names does not match n_markers")

    def build(self, rng=None):
        rng = np.random.default_rng(rng)
        self.check_config()
        marker_names = self.build_marker_names()
        ct_names = self.build_ct_names()
        xform = self.build_transform()

        # TODO: this is a bit redundant...
        builder_config = SubjectBuilderConfig(
            n_marker=self.config.n_marker,
            n_signal_marker=self.config.n_signal_marker,
            marker_names=marker_names,
            ct_names=ct_names,
            n_ct=self.config.n_ct,
            n_signal_ct=self.config.n_signal_ct,
            ct_alpha=self.config.ct_alpha,
            ct_mean_loc=self.config.ct_mean_loc,
            ct_mean_scale=self.config.ct_mean_scale,
            ct_scale_min=self.config.ct_scale_min,
            ct_scale_max=self.config.ct_scale_max
        )

        subject_builder = MultivariateNormalSubjectBuilder(builder_config)

        # build control and signal classes
        classes = [subject_builder.build_control("control", rng=rng)]
        for i in range(self.config.n_class - 1):
            signal_class = subject_builder.build_signal(f"signal_{i}", rng=rng)
            classes.append(signal_class)

        # instantiate and return generator
        generator = CytoDataGen(
            n_samples_per_class=self.config.n_samples_per_class,
            n_cells_min=self.config.n_cells_min,
            n_cells_max=self.config.n_cells_max,
            classes=classes,
            transform=xform
        )

        return generator

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
            marker_names = [f"cd_{i}" for i in range(self.config.n_marker)]
        return marker_names

    def build_ct_names(self, rng=None):
        rng = np.random.default_rng()
        ct_names = [f"ct_{i}" for i in range(self.config.n_ct)]
        return ct_names


class CytoDataGen:

    def __init__(self, classes: list[Subject], n_samples_per_class: int = 30, n_cells_min=10_000, n_cells_max: int = 10_000, transform: Transform = None):
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
