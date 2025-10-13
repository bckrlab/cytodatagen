"""
Provides the Subject class for sampling cells.
Also implements a CLI for generating control and signal subjects and store them as subjects.json.
"""

import argparse
import json
import anndata as ad
import numpy as np
import numpy.typing as npt

from cytodatagen.cli.subjects import SubjectGeneratorConfig
from cytodatagen.cli.subjects import SubjectGenerator
from cytodatagen.dists import MultivariateNormal
from cytodatagen.markers import NamedMarkerDistribution
from cytodatagen.populations import DistributionPopulation
from cytodatagen.registry import pop_registry
from pathlib import Path


class Subject:
    def __init__(self, name: str, alpha: npt.ArrayLike, populations: list[DistributionPopulation]):
        self.name = name
        self.alpha = np.asarray(alpha)
        self.populations = populations
        if len(self.alpha) != len(self.populations):
            raise ValueError("length mismatch of alpha prior and cell populations")

    def sample(self, n: int = 10_000, rng=None) -> ad.AnnData:
        """Samples n cells from the populations with proportions given by alpha."""
        rng = np.random.default_rng(rng)
        adatas = {}
        dist = self.sample_pop_proportions(n, rng)
        for i, (n, population) in enumerate(zip(dist, self.populations)):
            adata = population.sample(n, rng)
            adata.obs["pop_id"] = i
            adatas[population.name] = adata
        adata = ad.concat(adatas, axis=0, index_unique="_")
        adata.obs["subject"] = self.name
        adata.obs.index = [f"cell_{i}" for i in range(len(adata.obs))]
        return adata

    def sample_pop_proportions(self, n: int = 10_000, rng=None) -> np.ndarray:
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

    def to_dict(self) -> dict:
        data = {
            "_target_": "subject",
            "name": self.name,
            "alpha": self.alpha.tolist(),
            "populations": [pop.to_dict() for pop in self.populations]
        }
        return data

    @classmethod
    def from_dict(cls, data):
        pops = []
        pops_data = data["populations"]
        for pop_data in pops_data:
            key = pop_data["_target_"]
            pop = pop_registry.get(key).from_dict(pop_data)
            pops.append(pop)
        return cls(data["name"], data["alpha"], pops)


if __name__ == "__main__":
    # generate subjects
    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=Path, help="config.json file")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("-o", "--output", type=Path, default="subjects.json")

    args = parser.parse_args()

    with open(args.config, "r") as config_json:
        config = json.load(config_json)

    subject_gen_config = SubjectGeneratorConfig(**config)

    subject_gen = SubjectGenerator(subject_gen_config)

    subjects = subject_gen.make_subjects(rng=args.seed)

    subject_data = [subject.to_dict() for subject in subjects]

    with open(args.output, "w") as json_file:
        json.dump(subject_data, json_file, indent=4)
