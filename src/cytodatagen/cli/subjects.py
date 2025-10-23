#!/usr/bin/env python

"""
CLI for generating subjects.
"""

import argparse
import dataclasses as dc
import json
import numpy as np
import numpy.typing as npt

from pathlib import Path
from cytodatagen.populations import ControlPopulation, ControlPopulationBuilder, SignalPopulationBuilder
from cytodatagen.subjects import Subject


class ControlSubjectBuilder:
    """
    Creates a new Subject with ControlPopulations.
    """

    def __init__(
        self,
        name: str,
        alpha: npt.ArrayLike,
        pop_names: list[str],
        marker_names: list[str],
        mean_loc: float = 5.0,
        mean_scale: float = 1.0,
        scale_low: float = 1.0,
        scale_high: float = 1.0
    ):
        self.name = name
        self.alpha = alpha
        self.pop_names = pop_names
        self.marker_names = marker_names
        self.mean_loc = mean_loc
        self.mean_scale = mean_scale
        self.scale_low = scale_low
        self.scale_high = scale_high

    def build(self, rng=None) -> Subject:
        rng = np.random.default_rng(rng)
        populations = []
        for pop_name in self.pop_names:
            pop_builder = ControlPopulationBuilder(
                name=pop_name,
                markers=self.marker_names,
                mean_loc=self.mean_loc,
                mean_scale=self.mean_scale,
                scale_low=self.scale_low,
                scale_high=self.scale_high
            )
            population = pop_builder.build(rng=rng)
            populations.append(population)
        subject = Subject(name=self.name, alpha=self.alpha, populations=populations)
        return subject


class SignalSubjectBuilder:
    """
    Creates a new Subject with SignalPopulations by adding signals to ControlPopulations. 
    """

    def __init__(
        self,
        name: str,
        alpha: npt.ArrayLike,
        control_populations: list[ControlPopulation],
        signal_markers: dict,
        signal_pops: list[str],
        mean_loc: float = 5,
        mean_scale: float = 1,
        scale_low: float = 1,
        scale_high: float = 1,
    ):
        self.name = name
        self.alpha = alpha
        self.control_populations = control_populations
        self.signal_markers = signal_markers
        self.signal_pops = signal_pops
        self.mean_loc = mean_loc
        self.mean_scale = mean_scale
        self.scale_low = scale_low
        self.scale_high = scale_high

    def build(self, rng=None) -> Subject:
        rng = np.random.default_rng(rng)
        pops = []
        for control_pop in self.control_populations:
            if control_pop.name in self.signal_pops:
                signal_markers = self.signal_markers[control_pop.name]
                # map marker names to ids
                signal_markers = [
                    i for i, marker in enumerate(control_pop.markers) if marker in signal_markers
                ]

                pop_builder = SignalPopulationBuilder(
                    control_pop,
                    signal_markers,
                    mean_loc=self.mean_loc,
                    mean_scale=self.mean_scale,
                    scale_low=self.scale_low,
                    scale_high=self.scale_high
                )
                pop = pop_builder.build(rng=rng)
            else:
                pop = control_pop
            pops.append(pop)
        return Subject(self.name, self.alpha, pops)


@dc.dataclass
class SubjectGeneratorConfig:
    n_subject: int = 2
    subject_names: list[str] = None
    n_pop: int = 5
    pop_names: list[str] = None
    n_marker: int = 30
    marker_names: list[str] = None
    n_signal_pop: int = 1
    signal_pop: dict = None
    n_signal_marker: int = 3
    signal_marker: dict = None
    alpha: dict = None
    alpha_low: float = 1.0
    alpha_high: float = 100
    mean_loc: float = 5
    mean_scale: float = 1
    scale_low: float = 1
    scale_high: float = 1


class SubjectGenerator:
    def __init__(self, config: SubjectGeneratorConfig):
        self.config = config

    def make_subjects(self, rng=None) -> list[Subject]:
        rng = np.random.default_rng(rng)
        subject_names = self.make_subject_names()
        pop_names = self.make_pop_names()
        marker_names = self.make_marker_names()
        alpha = self.make_alpha(subject_names, rng)
        signal_pop = self.make_signal_pop(subject_names, pop_names, rng)
        signal_marker = self.make_signal_marker(subject_names, pop_names, marker_names, signal_pop, rng)
        subjects = []
        control_name = subject_names[0]
        control_builder = ControlSubjectBuilder(
            name=control_name,
            alpha=alpha[control_name],
            pop_names=pop_names,
            marker_names=marker_names,
            mean_loc=self.config.mean_loc,
            mean_scale=self.config.mean_scale,
            scale_low=self.config.scale_low,
            scale_high=self.config.scale_high
        )
        control_subject = control_builder.build(rng=rng)
        control_pops = control_subject.populations
        subjects.append(control_subject)

        for i, subject_name in enumerate(subject_names[1:]):
            signal_builder = SignalSubjectBuilder(
                name=subject_name,
                alpha=alpha[subject_name],
                control_populations=control_pops,
                signal_markers=signal_marker[subject_name],
                signal_pops=signal_pop[subject_name],
                mean_loc=self.config.mean_loc,
                mean_scale=self.config.mean_scale,
                scale_low=self.config.scale_low,
                scale_high=self.config.scale_high
            )
            signal_subject = signal_builder.build(rng=rng)
            subjects.append(signal_subject)
        return subjects

    def make_subject_names(self):
        subject_names = self.config.subject_names
        if subject_names is not None:
            if self.config.n_subject != len(subject_names):
                raise ValueError(f"got {len(subject_names)} names for {self.config.n_subject} subjects")
        else:
            subject_names = ["control"] + [f"signal_{i + 1}" for i in range(len(self.config.n_subject - 1))]
        return subject_names

    def make_pop_names(self):
        pop_names = self.config.pop_names
        if pop_names is not None:
            if self.config.n_pop != len(pop_names):
                raise ValueError(f"got {len(pop_names)} names for {self.config.n_pop} populations")
        else:
            pop_names = [f"pop_{i + 1}" for i in range(self.config.n_pop)]
        return pop_names

    def make_marker_names(self):
        marker_names = self.config.marker_names
        if marker_names is not None:
            if self.config.n_marker != len(marker_names):
                raise ValueError(f"got {len(marker_names)} names for {self.config.n_marker} markerulations")
        else:
            marker_names = [f"marker_{i + 1}" for i in range(self.config.n_marker)]
        return marker_names

    def make_alpha(self, subject_names: list[str], rng=None):
        rng = np.random.default_rng(rng)
        alpha = self.config.alpha
        if alpha is not None:
            for subject in subject_names:
                if subject not in alpha:
                    raise ValueError(f"missing alpha for subject {subject}")
        else:
            alpha = dict()
            for subject in subject_names:
                alpha[subject] = rng.uniform(self.config.alpha_low, self.config.alpha_high, size=self.config.n_pop).tolist()
        return alpha

    def make_signal_pop(self, subject_names: list[str], pop_names: list[str], rng=None):
        rng = np.random.default_rng(rng)
        signal_pop = self.config.signal_pop
        if signal_pop is not None:
            for subject in signal_pop.keys():
                if subject not in subject_names:
                    raise ValueError(f"got signal_pop for unknown subject {subject}")
                for pop in signal_pop[subject]:
                    if pop not in pop_names:
                        raise ValueError(f"got unknown signal_pop {pop}")
        else:
            signal_pop = dict()
            for subject in subject_names:
                signal_pop[subject] = rng.choice(pop_names, size=self.config.n_signal_pop, replace=False)
        return signal_pop

    def make_signal_marker(self, subject_names: list[str], pop_names: list[str], marker_names: list[str], signal_pop: dict, rng=None):
        rng = np.random.default_rng(rng)
        signal_marker = self.config.signal_marker
        if signal_marker is not None:
            for subject in signal_marker.keys():
                if subject not in subject_names:
                    raise ValueError(f"got signal_marker for unknown subject {subject}")
                for pop in signal_marker[subject].keys():
                    if pop not in pop_names:
                        raise ValueError(f"got signal_marker for unkown population {pop}")
                    for marker in signal_marker[subject][pop]:
                        if marker not in marker_names:
                            raise ValueError(f"got unknown signal_marker {marker}")
        else:
            signal_marker = dict()
            for subject in subject_names:
                signal_marker[subject] = dict()
                for pop in signal_pop[subject]:
                    signal_marker[subject][pop] = rng.choice(
                        marker_names, size=self.config.n_signal_marker, replace=False
                    )
        return signal_marker


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
