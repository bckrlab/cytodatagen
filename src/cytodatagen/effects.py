"""
Effects that alter either the cell type composition or the marker expression values of generated samples.
"""

import abc
import numpy as np
import numpy.typing as npt
import anndata as ad

from typing import Union

"""
cofactor (float): cofactor for the sinh transform, if xform='sinh'
with_composition_effect (bool): classes will have different cell type compositions by switching ct_alpha values (requires n_signal_ct>=2)
with_expression_effect (bool): classes will have different marker expressions
with_batch_shift (bool): samples with different batch_ids will have a batch_shift applied to them
n_batch (int): number of batches with different batch effects
batch_scale (float): sd for sampling post-xform shifts for each batch
marker_snr_db (float|ArrayLike): signal to noise ratio in decibell, for noise that is applied post-xform
xform ("none"|"exp"|"sinh"): inverse transform that is applied to the marker expressions
"""


class Effect(abc.ABC):
    def __init__(self):
        super().__init__()
        self.params = dict()

    def fit_params(self, n_class: int, n_samples_per_class: int, n_markers: int, n_ct: int, rng=None):
        pass


class CompositionEffect(Effect):

    @abc.abstractmethod
    def apply(self, class_id: int, sample_id: int, ct_alpha: np.ndarray, rng=None) -> np.ndarray:
        pass

    def __call__(self, class_id: int, sample_id: int, ct_alpha, rng=None) -> np.ndarray:
        return self.apply(class_id, sample_id, ct_alpha, rng=rng)


class SwitchCompositionEffect(CompositionEffect):
    def __init__(self, n_switch_ct: int, p_switch: float = 0.9):
        """
        Args:
            n_switch_ct (int): number of cell types that will have their alpha switched
            p_switch (float): probability of a non-control sample being affected 
        """
        super().__init__()
        self.n_switch_ct = n_switch_ct
        self.p_switch = p_switch

    def fit_params(self, n_class: int, n_samples_per_class: int, n_markers: int, n_ct: int, rng=None):
        rng = np.random.default_rng(rng)
        self.params["switch_from_ct"] = np.stack([
            rng.choice(n_ct, size=self.n_switch_ct, replace=False) for i in range(n_class - 1)
        ])

        self.params["switch_to_ct"] = np.roll(self.params["switch_from_ct"], 1, axis=1)

    def apply(self, class_id, sample_id, ct_alpha, rng=None):
        rng = np.random.default_rng(rng)
        ct_alpha = np.array(ct_alpha)
        if class_id == 0 or rng.random() < self.p_switch:
            return ct_alpha
        switch_from = self.params["switch_from_ct"][class_id]
        switch_to = self.params["switch_to_ct"][class_id]
        ct_alpha[switch_from] = ct_alpha[switch_to]

        return ct_alpha


class ExpressionEffect(Effect):
    """Base class for effects that alter expression values."""

    @abc.abstractmethod
    def apply(self, class_id: int, sample_id: int, adata: ad.AnnData, rng=None) -> ad.AnnData:
        pass

    def __call__(self, class_id: int, sample_id: int, adata: ad.AnnData, rng=None) -> ad.AnnData:
        return self.apply(class_id, sample_id, adata, rng=rng)


class BatchEffect(ExpressionEffect):
    def __init__(self, n_batch=3, scale: float = 1.0):
        """
        Args:
            n_batch: number of batches in each class
            scale: scale for sampling the shifts in each batch from a normal distribution
        """
        super().__init__()
        self.n_batch = n_batch
        self.scale = scale

    def fit_params(self, n_class, n_samples_per_class, n_markers, n_ct, rng=None):
        rng = np.random.default_rng(rng)
        self.params["batch_ids"] = np.tile(np.arange(n_samples_per_class) % self.n_batch, n_class)
        self.params["batch_shifts"] = rng.normal(scale=self.scale, size=self.n_batch)

    def apply(self, class_id, sample_id, adata: ad.AnnData, rng=None):
        batch_id = self.params["batch_ids"][sample_id]
        batch_shift = self.params["batch_shifts"][batch_id]
        adata.X += batch_shift
        adata.obs["batch_id"] = batch_id
        adata.obs["batch_shift"] = batch_shift
        return adata


class ClassSignalEffect(ExpressionEffect):
    """Shifts expression values of signal markers in signal cell types for each non-control class."""

    def __init__(self, n_signal_markers: int = 3, n_signal_ct: int = 1, p_cell: float = 0.5, p_sample: float = 0.9):
        """
        Args:        
            n_signal_markers (int): number of markers that carry sample class information
            n_signal_ct (int): number of cell types that carry sample class information
            p_cell (float): probability of cells of a signal cell type to be affected by expression effects
            p_sample (float): probability of a sample to be affected by composition and expression effects
        """
        super().__init__()
        self.n_signal_markers = n_signal_markers
        self.n_signal_ct = n_signal_ct
        self.p_cell = p_cell
        self.p_sample = p_sample

    def fit_params(self, n_class, n_samples_per_class, n_markers, n_ct, rng=None):
        rng = np.random.default_rng(rng)
        self.params["signal_markers"] = np.stack([
            rng.choice(np.arange(n_markers), self.n_signal_markers, replace=False) for i in range(n_class - 1)
        ])

        self.params["signal_cts"] = np.stack([
            rng.choice(np.arange(n_ct), self.n_signal_ct, replace=False) for i in range(n_class - 1)
        ])

    def apply(self, class_id, sample_id, adata, rng=None):
        rng = np.random.default_rng(rng)
        if not rng.random() < self.p_sample or class_id == 0:
            return adata
        signal_cts = self.params["signal_cts"][class_id]
        signal_markers = self.params["signal_markers"][class_id]
        mask = adata.obs["ct"] == signal_cts

        return adata


class ExpXformEffect(ExpressionEffect):
    """
    Effect that applies an exponential function as inverse of the logarithmic transform.
    f(x) := exp(x) 
    """

    def apply(self, class_id, sample_id, adata, rng=None):
        adata.X = np.exp(adata.X)
        return adata


class SinhXformEffect(ExpressionEffect):
    def __init__(self, cofactor: float = 5.0):
        """
        Inverse of the popular arsinh transform of cytometry data:
        f(x) := cofactor * sinh(x)

        Args:
            cofactor (float): cofactor parameter of the arsinh transform. Recommended values are 5.0 for CyTOF, and 150.0 for flow cytometry.
        """
        super().__init__()
        self.cofactor = cofactor

    def apply(self, class_id, sample_id, adata, rng=None):
        adata.X = self.cofactor * np.sinh(adata.X)
        return adata


class NoiseEffect(ExpressionEffect):
    def __init__(self, marker_snr_db: Union[float | npt.ArrayLike] = 20.0):
        """
        Adds Gaussian noise to each channel given a Signal-To-Noise Ratio in Decibell.

        Args:
            marker_snr_db (float|ArrayLike): Signal To Noise Ratio for each channel in Decibell.
            E.g., a SNR of 20 implies sd(noise) = sd(signal) / 100.

        For more information regarding the marker_snr_db, please checkout:

        @article{
            10.1002/cyto.a.23250,
            author = {Giesecke, Claudia and Feher, Kristen and von Volkmann, Konrad and Kirsch, Jenny and Radbruch, Andreas and Kaiser, Toralf},
            title = {Determination of background, signal-to-noise, and dynamic range of a flow cytometer: {A} novel practical method for instrument characterization and standardization},
            shorttitle = {Determination of background, signal-to-noise, and dynamic range of a flow cytometer},
            year = {2017},
            url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/cyto.a.23250},
            doi = {10.1002/cyto.a.23250},
            journal = {Cytometry Part A},
        }
        """
        super().__init__()
        self.marker_snr_db = marker_snr_db

    def apply(self, class_id, sample_id, adata, rng=None):
        """adds noise to a numpy array according to SNR given by the config's marker_snr_db"""
        rng = np.random.default_rng(rng)

        # compute linear snr from config
        marker_snr_db = self.marker_snr_db
        if np.isscalar(marker_snr_db):
            marker_snr_db = np.full(adata.X.shape, marker_snr_db)
        marker_snr = np.power(10, marker_snr_db / 10)

        # compute marker signal and resulting noise parameters
        signal_scale = adata.X.std(axis=0)
        noise_scale = signal_scale / marker_snr

        # sample and add noise
        noise = rng.normal(scale=noise_scale, size=adata.X.shape)
        adata.X = adata.X + noise
        return adata
