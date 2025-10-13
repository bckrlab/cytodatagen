#!/usr/bin/env python

import argparse
import json
import logging
import datetime as dt
import argparse

from pathlib import Path

import numpy as np
from cytodatagen.io import write_fcs, write_h5ad, write_parquet
from cytodatagen.generator import CytoDataGenBuilder, CytoDataGenBuilderConfig


logger = logging.getLogger(__name__)


def setup_logging(path: Path, verbose: bool = False):
    t = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = path / f"{t}.log"
    log_level = logging.INFO if verbose else logging.WARNING

    logging.basicConfig(filename=log_path, level=log_level)

    logger.info("writing log file to: %s", log_path)


def make_transforms(args) -> dict:
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


def main():
    parser = argparse.ArgumentParser()

    config_group = parser.add_argument_group("config")

    config_group.add_argument("--n-class", type=int, default=2)
    config_group.add_argument("--n-samples-per-class", type=int, default=30)
    config_group.add_argument("--n-cells-min", type=int, default=10_000)
    config_group.add_argument("--n-cells-max", type=int, default=10_000)
    config_group.add_argument("--n-ct", type=int, default=5)
    config_group.add_argument("--ct-mean-loc", type=float, default=5)
    config_group.add_argument("--ct-mean-scale", type=float, default=1)
    config_group.add_argument("--ct-scale-min", type=float, default=0.5)
    config_group.add_argument("--ct-scale-max", type=float, default=2.0)
    config_group.add_argument("--ct-alpha", type=float, default=5)
    config_group.add_argument("--n-marker", type=int, default=30)

    config_group.add_argument("--n-signal-marker", type=int, default=3)
    config_group.add_argument("--n-signal-ct", type=int, default=3)

    config_group.add_argument("--add-batch-xform", action="store_true")
    config_group.add_argument("--n-batch", type=int, default=3)
    config_group.add_argument("--batch-scale", type=float, default=1)

    config_group.add_argument("--add-noise-xform", action="store_true")
    config_group.add_argument("--marker-snr-db", type=float, default=20)

    config_group.add_argument("--add-sinh-xform", action="store_true")
    config_group.add_argument("--cofactor", type=int, default=5)

    config_group.add_argument("--add-exp-xform", action="store_true")

    config_group.add_argument("--seed", type=int, default=19, help="random seed")
    config_group.add_argument("-f", "--file", type=Path, default=None, help="parse config from json file instead")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--format", choices=["h5ad", "fcs"], default="h5ad")
    output_group.add_argument("-o", "--output", type=Path, default="cytodata", help="output directory path")
    output_group.add_argument("-v", "--verbose", action="store_true")

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

    if args.format == "h5ad":
        write_h5ad(args.output, adata)
    elif args.format == "fcs":
        write_fcs(args.output, adata, sample_id="sample_id")
    elif args.format == "parquet":
        write_parquet(args.output, adata, sample_id="sample_id")


if __name__ == "__main__":
    main()
