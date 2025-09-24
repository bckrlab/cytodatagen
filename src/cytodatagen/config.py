#!/usr/bin/env python

"""
Generates config.json specification for multivariate control and signal subjects.
"""

import numpy as np
import datetime as dt
import logging
import json
import argparse
import dataclasses as dc

from cytodatagen.generator import CytoDataGenBuilder, CytoDataGenBuilderConfig
from cytodatagen.io import write_fcs, write_h5ad, write_parquet
from pathlib import Path

from cytodatagen.transforms import TransformBuilder

logger = logging.getLogger(__name__)


@dc.dataclass
class ConfigGeneratorConfig:
    """this is getting ridiculous"""
    ct_names: list = dc.field(default_factory=list)
    marker_names: list = dc.field(default_factory=list)
    subject_names: list = dc.field(default_factory=list)

    n_subject: int = 2
    n_marker: int = 30
    n_ct: int = 5

    n_samples_per_subject: int = 30
    n_cells_low: int = 10_000
    n_cells_high: int = 10_000
    n_signal_ct: int = 3
    n_signal_marker: int = 3

    n_batch: int = 5
    preconfig: dict = dc.field(default_factory=dict)

    xforms: dict = dc.field(default_factory=dict)
    seed: int = 19


# accept given signal_ct / signal_marker or generate new ones
# accept given alphas or generate new ones
# accept given ct names / marker names or generate new ones
# accept given means / variances or generate new ones

class ConfigGenerator:
    """
    Generates a config.json file for generating data.

    Instantiates Control and Signal Subjects.
    Assembles transforms and their parameters.
    Writes Cytodatagen parameters.
    """

    def __init__(self, config: ConfigGeneratorConfig):
        self.config = config

    def make_config(self) -> dict:
        pass

    def check_config(self):
        if self.config.ct_names and len(self.config.ct_names) != self.config.n_ct:
            raise ValueError(f"length missmatch between ct_names ({len(self.config.ct_names)}) and n_ct ({self.config.n_ct})")

        if self.config.marker_names and len(self.config.marker_names) != self.config.n_marker:
            raise ValueError(f"length missmatch between marker_names ({len(self.config.marker_names)}) and n_marker ({self.config.n_marker})")

        if self.config.subject_names and len(self.config.subject_names) != self.config.n_subject:
            raise ValueError(f"length missmatch between subject_names ({len(self.config.subject_names)}) and n_subject ({self.config.n_subject})")

        ct_names = self.make_ct_names()
        marker_names = self.make_marker_names()
        subject_names = self.make_subject_names()

        subjects = []

    def write_config(self, file_path: Path) -> dict:
        file_path = Path(file_path)
        if file_path.suffix.lower() == ".json":
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path.mkdir(parents=True, exist_ok=True)
            file_path /= "config.json"

        with open(file_path) as config_file:
            json.dump({}, config_file)

    def make_transforms(self, args) -> dict:
        xforms = {}

        if args.add_batch_xform:
            xforms["batch"] = {
                "n_batch": args.n_batch,
                "scale": args.batch_scale
            }

        if args.add_sinh_xform:
            xforms["sinh"] = {
                "cofactor": args.cofactor
            }

        if args.add_exp_xform:
            xforms["exp"] = {}

        if args.add_noise_xform:
            xforms["noise"] = {
                "marker_snr_db": args.marker_snr_db
            }

        return xforms


"""config.json
{
    subject_names: []
    marker_names: []
    ct_names: []
    signal_markers: []
    signal_ct: []
    alphas: []

    subjects: [
        {
            class_names: "control subject",
            name: 
            kwargs: {
                name:
                mean: 
                ...
            }
        },
        ...
    ],
    xforms: {},
    seed: 19,

}
"""


def setup_logging(path: Path, verbose: bool = False):
    t = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = path / f"{t}.log"
    log_level = logging.INFO if verbose else logging.WARNING

    logging.basicConfig(filename=log_path, level=log_level)

    logger.info("writing log file to: %s", log_path)


def main():
    parser = argparse.ArgumentParser(description="generate config.json for control and signal subjects with a multivariate normal distribution.")

    parser.add_argument("--n-class", type=int, default=2)
    parser.add_argument("--n-samples-per-class", type=int, default=30)
    parser.add_argument("--n-cells-min", type=int, default=10_000)
    parser.add_argument("--n-cells-max", type=int, default=10_000)
    parser.add_argument("--n-ct", type=int, default=5)
    parser.add_argument("--ct-mean-loc", type=float, default=5)
    parser.add_argument("--ct-mean-scale", type=float, default=1)
    parser.add_argument("--ct-scale-min", type=float, default=0.5)
    parser.add_argument("--ct-scale-max", type=float, default=2.0)
    parser.add_argument("--ct-alpha", type=float, default=5)
    parser.add_argument("--n-marker", type=int, default=30)

    parser.add_argument("--n-signal-marker", type=int, default=3)
    parser.add_argument("--n-signal-ct", type=int, default=3)

    parser.add_argument("--add-batch-xform", action="store_true")
    parser.add_argument("--n-batch", type=int, default=3)
    parser.add_argument("--batch-scale", type=float, default=1)

    parser.add_argument("--add-noise-xform", action="store_true")
    parser.add_argument("--marker-snr-db", type=float, default=20)

    parser.add_argument("--add-sinh-xform", action="store_true")
    parser.add_argument("--cofactor", type=int, default=5)

    parser.add_argument("--add-exp-xform", action="store_true")

    parser.add_argument("--seed", type=int, default=19, help="random seed")
    parser.add_argument("-f", "--file", type=Path, default=None, help="parse config from json file instead")

    parser.add_argument("-o", "--output", type=Path, default="config.json", help="output json file")

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    setup_logging(args.output, args.verbose)
    logger.info("args: %s", args)

    if args.file is not None:
        logger.info("reading config from json file: other config options will be ignored")
        with open(args.file) as config_file:
            config = json.load(config_file)
    else:
        xforms = make_transforms(args)

        config = dict(
            n_class=args.n_class,
            n_samples_per_class=args.n_samples_per_class,
            n_cells_min=args.n_cells_min,
            n_cells_max=args.n_cells_max,
            n_marker=args.n_marker,
            n_signal_marker=args.n_signal_marker,
            n_ct=args.n_ct,
            n_signal_ct=args.n_signal_ct,
            ct_alpha=args.ct_alpha,
            ct_mean_loc=args.ct_mean_loc,
            ct_mean_scale=args.ct_mean_scale,
            ct_scale_min=args.ct_scale_min,
            ct_scale_max=args.ct_scale_max,
            transforms=xforms
        )

    logger.info("config: %s", json.dumps(config))

    rng = np.random.default_rng(args.seed)

    config = CytoDataGenBuilderConfig(**config)
    builder = CytoDataGenBuilder(config)
    gen = builder.build(rng=rng)
    adata = gen.generate(rng=rng)

    config = ConfigGeneratorConfig()
    config_gen = ConfigGenerator(config)

    config_gen.write_config(args.output)


if __name__ == "__main__":
    main()
