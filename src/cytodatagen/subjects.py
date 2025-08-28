import dataclasses as dc
import anndata as ad
import numpy as np
import numpy.typing as npt

from cytodatagen.dist import MultivariateNormal
from cytodatagen.markers import NamedMarkerDistribution
from cytodatagen.populations import DistributionPopulation


class Subject:
    def __init__(self, name: str, alpha, populations: list[DistributionPopulation]):
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
            pop = DistributionPopulation(self.config.ct_names[i], mdist)
            populations.append(pop)
        return Subject(name, alpha, populations)
