#!/usr/bin/env python

import argparse
import dataclasses as dc
import json
import anndata as ad
import numpy as np
import logging

from cytodatagen.subjects import Subject
from cytodatagen.transforms import ComposedTransform, Transform
from cytodatagen.utils.io import write_fcs, write_h5ad, write_parquet
from pathlib import Path
from tqdm import tqdm

# 1. reconstruct subjects from json
# 2. add transforms
# 3. generate cells

logger = logging.getLogger(__name__)


@dc.dataclass
class DataGeneratorConfig:
    n_samples_per_subject: int = 10
    n_cells_min: int = 10_000
    n_cells_max: int = 10_000


class DataGenerator:
    def __init__(self, config: DataGeneratorConfig, subjects: list[Subject], transforms: list[Transform]):
        self.config = config
        self.subjects = subjects
        self.transforms = transforms

    def generate(self, rng=None):
        rng = np.random.default_rng(rng)
        samples = []
        with tqdm(total=len(self.subjects) * self.config.n_samples_per_subject) as pbar:
            for subject in self.subjects:
                for i in range(self.config.n_samples_per_subject):
                    n_cells = rng.uniform(self.config.n_cells_min, self.config.n_cells_max)
                    sample = subject.sample(n=n_cells, rng=rng)
                    samples.append(sample)
                    pbar.update()
        keys = np.arange(len(samples))
        samples = ad.concat(samples, axis=0, label="sample_id", keys=keys, index_unique="_sample_")
        # apply transforms
        for transform in self.transforms:
            samples = transform.apply(samples)
        return samples


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=Path, help="path to the subjects.json file")
    parser.add_argument("--transforms", type=Path, help="path to the transforms.json file")
    parser.add_argument("--n-samples-per-subject", type=int, default=10)
    parser.add_argument("--n-cells-min", type=int, default=10_000)
    parser.add_argument("--n-cells-max", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--format", choices=["h5ad", "fcs", "parquet"], default="h5ad")
    parser.add_argument("-o", "--output", type=Path, default="cytodata", help="output directory path")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)

    config = DataGeneratorConfig(
        n_samples_per_subject=args.n_samples_per_subject,
        n_cells_min=args.n_cells_min,
        n_cells_max=args.n_cells_max
    )

    with open(args.subjects, "r") as subjects_json:
        subjects_data = json.load(subjects_json)

    subjects = [Subject.from_dict(data) for data in subjects_data]

    with open(args.transforms, "r") as transforms_json:
        transforms_data = json.load(transforms_json)

    transforms = [ComposedTransform.from_dict({"transforms": transforms_data})]

    data_gen = DataGenerator(config, subjects, transforms)

    rng = np.random.default_rng(args.seed)
    adata = data_gen.generate(rng=rng)

    if args.format == "h5ad":
        write_h5ad(args.output, adata)
    elif args.format == "fcs":
        write_fcs(args.output, adata, sample_id="sample_id")
    elif args.format == "parquet":
        write_parquet(args.output, adata, sample_id="sample_id")


if __name__ == "__main__":
    main()
