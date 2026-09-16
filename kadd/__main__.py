"""CLI entry point.

Usage:
    python -m kadd classify-train <paramcard.yml> [--verbose]
    python -m kadd classify-train --type swin3d --size xlight [--verbose]

The --type/--size form autodetects the matching config under config_files/
(classify_<type>_<size>.yml, see kadd/util/config_registry.py) instead of
requiring a hand-typed path — only works for configs following that naming
convention (the legacy classify_swin3d_v1.yml/v2.yml predate it and still
need an explicit path).

Mirrors the `python -m den_pie density-train <paramcard.yml>` shape from
poehlers/DenPIE.
"""
import argparse
import logging
import socket

import torch

from .classify.model import ALL_SIZES
from .util.config_registry import TYPES, resolve_config
from .util.logger import init_logger, separator
from .util.parse import find_run, log_yaml, parse, setup_dir


def main():
    parser = argparse.ArgumentParser(prog="kadd")
    subparsers = parser.add_subparsers(required=True)

    classify_train_parser = subparsers.add_parser("classify-train")
    classify_train_parser.add_argument("paramcard", nargs="?", default=None,
                                        help="Path to a config YAML. Omit if using --type/--size instead.")
    classify_train_parser.add_argument("--type", choices=TYPES,
                                        help="Encoder type — used with --size to autodetect a config "
                                             "under config_files/ instead of passing paramcard.")
    classify_train_parser.add_argument("--size", choices=ALL_SIZES,
                                        help="Model size tier — used with --type (see above). Valid values "
                                             "depend on --type (e.g. 'xmedium' is cnn3d-only); an invalid "
                                             "combo is caught by --type/--size resolution below, not here.")
    classify_train_parser.add_argument("--config-dir", default="config_files",
                                        help="Directory to search when resolving --type/--size (default: config_files)")
    classify_train_parser.add_argument("--verbose", action="store_true")
    classify_train_parser.set_defaults(func=classify_train)

    args = parser.parse_args()

    if args.paramcard and (args.type or args.size):
        parser.error("pass either paramcard or --type/--size, not both")
    if not args.paramcard and not (args.type and args.size):
        parser.error("classify-train requires either paramcard or both --type and --size")
    if not args.paramcard:
        try:
            args.paramcard = resolve_config(args.type, args.size, args.config_dir)
        except ValueError as e:
            parser.error(str(e))

    args.func(args)


def classify_init(params: dict):
    """Build model + data + device for classify mode."""
    from .classify.data import KneeDataLoader
    from .classify.model import build_model

    device = torch.device('cuda' if torch.cuda.is_available()
                           else 'mps' if torch.backends.mps.is_available()
                           else 'cpu')
    logging.info(f"Device: {device}")

    data_loader = KneeDataLoader(params)
    model = build_model(params)

    return model, data_loader.data, device


def classify_train(args: argparse.Namespace) -> None:
    from .classify.train import ClassifierTraining

    params = parse(args.paramcard)
    run_name = setup_dir(args.paramcard)
    params['name'] = run_name
    init_logger(fn=run_name, verbose=args.verbose)
    logging.info(f'{socket.gethostname()}: classify-train starting')
    log_yaml(params, source=args.paramcard)
    separator()

    model, data, device = classify_init(params)
    ClassifierTraining(params, model, data, device).main()


if __name__ == "__main__":
    main()
