#!/usr/env/bin python

"""
Generates a subjects.json that specifies subjects and their multivariate normal distributed cell populations.
"""


import argparse
import dataclasses as dc

import json
import numpy as np

from cytodatagen.markers import MarkerDistribution, NamedMarkerDistribution
from cytodatagen.populations import DistributionPopulation, MultivariateNormal, NormalPopulation

from pathlib import Path


@dc.dataclass
class SubjectGeneratorConfig:
    n_subjects: int = 2
    subject_names: list = dc.field(default_factory=list)

    n_marker: int = 30
    marker_names: list = dc.field(default_factory=list)

    n_pop: int = 5
    pop_names: list = dc.field(default_factory=list)

    n_signal_marker: int = 3
    signal_markers: list = dc.field(default_factory=list)

    n_signal_pop: int = 3
    signal_pops: list = dc.field(default_factory=list)

    alphas: dict = dc.field()
    alpha_low: int = 10
    alpha_high: int = 100

    mean_loc: int = 5
    mean_scale: int = 3

    scale_low: int = 5
    scale_high: int = 10


class SubjectGenerator:
    def __init__(self, config: SubjectGeneratorConfig):
        self.config = config

    def make_subjects(self):
        pass

    def make_control_subject(self):
        pass

    def make_signal_subject(self):
        pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=Path)

    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    config = SubjectGeneratorConfig(**config)

    subgen = SubjectGenerator(config)
    subjects = subgen.make_subjects()


if __name__ == "__main__":
    main()
