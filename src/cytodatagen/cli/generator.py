import pandas as pd
import numpy as np
import numpy.typing as npt
import anndata as ad
import dataclasses as dc

from cytodatagen.subjects import MultivariateNormalSubjectBuilder, Subject, SubjectBuilderConfig
from cytodatagen.transforms import BatchTransform, ComposedTransform, ExpTransform, SinhTransform, Transform, NoiseTransform, TransformBuilder
from tqdm import tqdm


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


class CytoDataGenBuilder:
    """Assists in constructing a new CytoDataGen object from a config."""

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

        # build transforms
        xform_builder = TransformBuilder(self.config.transforms)
        xform = xform_builder.build()

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
