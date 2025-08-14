#!/usr/bin/env python

"""
Standalone script to generate synthetic cytometry data.
"""

import numpy as np
import anndata as ad
import logging
import pandas as pd
import numpy.typing as npt
import dataclasses as dc

from typing import Literal, Union
from pathlib import Path
from tqdm import tqdm, trange

from cytodatagen.effects import BatchEffect, ClassSignalEffect, SwitchCompositionEffect, CompositionEffect, ExpXformEffect, ExpressionEffect, NoiseEffect, SinhXformEffect

logger = logging.getLogger(__name__)

_config_args_doc = """
    Args:
        n_class (int): number of classes, including one control class without composition or expressoin effects
        n_samples_per_class (int): number of samples in each class
        n_cells_min (int): minimum number of cells per sample
        n_cells_max (int): maximum number of cels per sample
        n_markers (int): number of markers, i.e., cell features
        n_ct (int): total number of distinct cell types
        ct_alpha (float|ArrayLike): alpha for sampling the cell type counts in each sample from a Dirichlet distribution
        ct_mean_loc (float): mean to sample the cell type means from
        ct_mean_scale (float): sd for sampling the cell type means
        composition_effects (dict): specification {effect_key: kwargs} of composition effects
        expression_effects (dict): specification {effect_key: kwargs} of expression effects
        seed (int): seed for the random generator
"""


@dc.dataclass
class CytoDataGeneratorConfig:
    n_class: int = 2
    n_samples_per_class: int = 30
    n_cells_min: int = 1024
    n_cells_max: int = 1024
    n_markers: int = 30
    n_ct: int = 5
    ct_alpha: Union[float, npt.ArrayLike] = 1.0
    ct_mean_loc: float = 3
    ct_mean_scale: float = 1.0
    expression_effects: dict = dc.field(default_factory=dict)
    composition_effects: dict = dc.field(default_factory=dict)
    seed: int = 19


CytoDataGeneratorConfig.__doc__ = """
Configuration for the CytoDataGenerator.

{args}
""".format(args=_config_args_doc)


class CytoDataGenerator:

    CT_NAME_1 = ["big", "hungry", "mean", "nihilistic", "blue", "red", "green", "nasty", "wobbly", "tiny", "angry", "happy", "cheeky", "creepy"]
    CT_NAME_2 = ["stem", "blood", "skin", "bone", "muscle", "fat", "nerve", "endothelial", "pancreatic", "immune"]

    COMPOSITION_EFFECTS = {
        "switch": SwitchCompositionEffect
    }

    EXPRESSION_EFFECTS = {
        "batch": BatchEffect,
        "noise": NoiseEffect,
        "sinh": SinhXformEffect,
        "exp": ExpXformEffect,
        "signal": ClassSignalEffect,
    }

    def __init__(
        self,
        config: CytoDataGeneratorConfig
    ):
        self.config = config
        self.rng: np.random.Generator = None
        self.params: dict = None
        self.composition_effects: list[CompositionEffect] = None
        self.expression_effects: list[ExpressionEffect] = None

    def generate(self) -> ad.AnnData:
        self.parse_config()
        self.generate_params()

        # generate samples
        samples = []
        labels = []
        for class_id in range(self.config.n_class):
            class_samples = self.generate_class(class_id)
            samples.extend(class_samples)
            labels.extend([class_id for sample in class_samples])
        adata = ad.concat(samples)

        # add metadata
        adata.uns["config"] = dc.asdict(self.config)
        adata.uns["labels"] = labels
        adata.uns["params"] = self.params
        return adata

    def parse_config(self):
        self.rng = np.random.default_rng(self.config.seed)
        if self.config.n_cells_min > self.config.n_cells_max:
            raise ValueError("n_cells_min should be less or equal n_cells_max")

        self.composition_effects = []
        for key, value in self.config.composition_effects.items():
            cls = self.COMPOSITION_EFFECTS[key]
            if isinstance(value, CompositionEffect):
                effect = value
            else:
                effect = cls(**value)
            self.composition_effects.append(effect)

        self.expression_effects = []
        for key, value in self.config.expression_effects.items():
            cls = self.EXPRESSION_EFFECTS[key]
            if isinstance(value, ExpressionEffect):
                effect = value
            else:
                effect = cls(**value)
            self.expression_effects.append(effect)

    def generate_params(self) -> dict:
        """Generates all parameters, e.g., for cell type distributions."""
        self.params = dict()

        # generate cell type data
        self.params["ct_names"] = np.asarray([self.get_ct_name() for i in range(self.config.n_ct)])
        self.params["ct_means"] = np.stack([self.rng.normal(self.config.ct_mean_loc, self.config.ct_mean_scale, size=self.config.n_markers) for i in range(self.config.n_ct)])
        self.params["ct_covs"] = [self.rng.normal(size=(self.config.n_markers, self.config.n_markers)) for i in range(self.config.n_ct)]
        self.params["ct_covs"] = np.stack([cov @ cov.T for cov in self.params["ct_covs"]])

        for effect in self.composition_effects:
            effect.fit_params(self.config.n_class, self.config.n_samples_per_class, self.config.n_markers, self.config.n_ct, rng=self.rng)

        for effect in self.expression_effects:
            effect.fit_params(self.config.n_class, self.config.n_samples_per_class, self.config.n_markers, self.config.n_ct, rng=self.rng)

    def generate_class(self, class_id: int):
        samples = []

        n = self.config.n_samples_per_class
        for i in range(n):
            sample_id = n * class_id + i
            sample = self.generate_sample(class_id, sample_id)
            samples.append(sample)
        return samples

    def generate_sample(self, class_id: int, sample_id: int) -> ad.AnnData:
        """generates a new sample for the given cell type means/variances and class label."""
        alpha = self.config.ct_alpha
        alpha = np.repeat(alpha, self.config.n_ct) if np.isscalar(alpha) else np.array(alpha)
        alpha = self.apply_composition_effects(class_id, sample_id, alpha)
        ct_counts = self.generate_ct_counts(alpha)
        ct_ids = np.repeat(np.arange(self.config.n_ct), ct_counts)
        cells = []
        for i in range(self.config.n_ct):
            ct_cells = self.rng.multivariate_normal(self.params["ct_means"][i], cov=self.params["ct_covs"][i], size=ct_counts[i])
            cells.append(ct_cells)
        x = np.concatenate(cells)

        obs = pd.DataFrame({
            "sample_id": sample_id,
            "class_id": class_id,
            "ct": ct_ids,
            "ct_name": self.params["ct_names"][ct_ids],
        }, index=[f"sample_{sample_id}_cell_{i}" for i in range(len(x))])

        adata = ad.AnnData(X=x, obs=obs, vidx=self.get_var_names())
        adata = self.apply_expression_effects(class_id, sample_id, adata)

        return adata

    def apply_composition_effects(self, class_id: int, sample_id: int, alpha: np.ndarray) -> np.ndarray:
        for effect in self.composition_effects:
            alpha = effect.apply(class_id, sample_id, alpha, rng=self.rng)
        return alpha

    def apply_expression_effects(self, class_id: int, sample_id: int, adata: ad.AnnData) -> ad.AnnData:
        for effect in self.expression_effects:
            adata = effect.apply(class_id, sample_id, adata, rng=self.rng)
        return adata

    def generate_ct_counts(self, alpha: np.ndarray) -> np.ndarray:
        """samples the counts of each cell type in a sample from a dirichlet distribution"""
        # generate cell type counts
        alpha = np.repeat(alpha, self.config.n_ct) if np.isscalar(alpha) else np.asarray(alpha)
        n_cells = self.rng.integers(self.config.n_cells_min, self.config.n_cells_max, endpoint=True)
        ct_counts = np.floor(self.rng.dirichlet(alpha) * n_cells).astype(int)
        # distribute remainder due to floating point errors
        ct_remainder = n_cells - ct_counts.sum()
        ids, counts = np.unique(self.rng.choice(np.arange(self.config.n_ct), size=ct_remainder), return_counts=True)
        ct_counts[ids] += counts
        return ct_counts

    def __call__(self, *args, **kwds):
        return self.generate()

    def get_ct_name(self) -> str:
        name_1 = self.rng.choice(self.CT_NAME_1)
        name_2 = self.rng.choice(self.CT_NAME_2)
        return f"{name_1}_{name_2}_cell"

    def get_var_names(self) -> np.ndarray:
        return np.array([f"cd_{i}" for i in range(self.config.n_markers)])


def make_cyto_data(
    n_class: int = 2,
    n_samples_per_class: int = 30,
    n_cells_min: int = 1024,
    n_cells_max: int = 1024,
    n_markers: int = 30,
    n_ct: int = 5,
    ct_mean_loc: float = 3,
    ct_mean_scale: float = 1.0,
    ct_alpha: Union[float, npt.ArrayLike] = 1.0,
    composition_effects: dict = None,
    expression_effects: dict = None,
    seed=19
) -> ad.AnnData:
    composition_effects = dict() if composition_effects is None else composition_effects
    expression_effects = dict() if expression_effects is None else expression_effects

    # a bit awkward to repeat args like that...
    config = CytoDataGeneratorConfig(
        n_class=n_class,
        n_samples_per_class=n_samples_per_class,
        n_cells_min=n_cells_min,
        n_cells_max=n_cells_max,
        n_markers=n_markers,
        n_ct=n_ct,
        ct_mean_loc=ct_mean_loc,
        ct_mean_scale=ct_mean_scale,
        ct_alpha=ct_alpha,
        composition_effects=composition_effects,
        expression_effects=expression_effects,
        seed=seed
    )

    generator = CytoDataGenerator(config)

    adata = generator.generate()

    return adata


make_cyto_data.__doc__ = """
Procedural interface for generating synthetic cytometry data.

{args}
""".format(args=_config_args_doc)
