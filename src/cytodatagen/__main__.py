#!/usr/bin/env python

import json
import logging
import datetime as dt
import argparse

from pathlib import Path

from cytodatagen.generator import make_cyto_data
from cytodatagen.io import write_fcs, write_h5ad


logger = logging.getLogger(__name__)


def setup_logging(path: Path, verbose: bool = False):
    t = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = path / f"{t}.log"
    log_level = logging.INFO if verbose else logging.WARNING

    logging.basicConfig(filename=log_path, level=log_level)

    logger.info("writing log file to: %s", log_path)


def make_composition_effects(args):
    composition_effects = dict()
    if args.add_switch_effect:
        composition_effects["switch"] = {
            "p_switch": args.p_switch
        }
    return composition_effects


def make_expression_effects(args):
    expression_effects = dict()
    if args.add_signal_effect:
        expression_effects["signal"] = {
            "p_cell": args.p_cell,
            "p_sample": args.p_sample,
            "n_signal_markers": args.n_signal_markers,
            "n_signal_ct": args.n_signal_ct
        }

    if args.add_batch_effect:
        expression_effects["batch"] = {
            "n_batch": args.n_batch,
            "scale": args.batch_scale
        }

    if args.add_sinh_effect:
        expression_effects["sinh"] = {
            "cofactor": args.cofactor
        }

    if args.add_exp_effect:
        expression_effects["exp"] = {}

    if args.add_noise_effect:
        expression_effects["noise"] = {
            "marker_snr_db": args.marker_snr_db
        }
    return expression_effects


def main():
    parser = argparse.ArgumentParser()

    config_group = parser.add_argument_group("config")

    config_group.add_argument("--n-class", type=int, default=2)
    config_group.add_argument("--n-samples-per-class", type=int, default=30)
    config_group.add_argument("--n-cells-min", type=int, default=1024)
    config_group.add_argument("--n-cells-max", type=int, default=1024)
    config_group.add_argument("--n-ct", type=int, default=5)
    config_group.add_argument("--ct-mean-loc", type=float, default=5)
    config_group.add_argument("--ct-mean-scale", type=float, default=1)
    config_group.add_argument("--ct-alpha", type=float, default=1)
    config_group.add_argument("--n-markers", type=int, default=30)

    config_group.add_argument("--add-switch-effect", action="store_true")
    config_group.add_argument("--p-switch", type=float, default=0.9)

    config_group.add_argument("--add-batch-effect", action="store_true")
    config_group.add_argument("--n-batch", type=int, default=3)
    config_group.add_argument("--batch-scale", type=float, default=1)

    config_group.add_argument("--add-noise-effect", action="store_true")
    config_group.add_argument("--marker-snr-db", type=float, default=20)

    config_group.add_argument("--add-sinh-effect", action="store_true")
    config_group.add_argument("--cofactor", type=int, default=5)

    config_group.add_argument("--add-exp-effect", action="store_true")

    config_group.add_argument("--add-signal-effect", action="store_true")
    config_group.add_argument("--n-signal-markers", type=int, default=3)
    config_group.add_argument("--n-signal-ct", type=int, default=3)
    config_group.add_argument("--p-cells", type=float, default=0.5)
    config_group.add_argument("--p-samples", type=float, default=0.5)

    config_group.add_argument("--seed", type=int, default=19, help="random seed")
    config_group.add_argument("-f", "--file", type=Path, default=None, help="parse config from json file instead")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--format", choices=["h5ad", "fcs"], default="h5ad")
    output_group.add_argument("-o", "--output", type=Path, default="cytodata")
    output_group.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    setup_logging(args.output, args.verbose)

    if args.file is not None:
        logger.info("reading config from json file: other config options will be ignored")
        with open(args.file) as config_file:
            config = json.load(config_file)
    else:
        composition_effects = make_composition_effects(args)
        expression_effects = make_expression_effects(args)

        config = dict(
            n_class=args.n_class,
            n_samples_per_class=args.n_samples_per_class,
            n_cells_min=args.n_cells_min,
            n_cells_max=args.n_cells_max,
            n_markers=args.n_markers,
            n_ct=args.n_ct,
            ct_mean_loc=args.ct_mean_loc,
            ct_mean_scale=args.ct_mean_scale,
            ct_alpha=args.ct_alpha,
            composition_effects=composition_effects,
            expression_effects=expression_effects,
            seed=args.seed
        )

    logger.info("config: %s", json.dumps(config))
    adata = make_cyto_data(**config)

    args.output.mkdir(parents=True, exist_ok=True)

    if args.format == "h5ad":
        write_h5ad(args.output, adata)
    elif args.format == "fcs":
        write_fcs(args.output, adata, sample_id="sample_id")


if __name__ == "__main__":
    main()
