"""YAML config parsing and output-directory helpers.

Adapted from the run-directory convention used in poehlers/DenPIE
(`output/<name>_<timestamp>/` + a `<name>_latest` symlink), reimplemented here
rather than copied verbatim (DenPIE's own util/parse.py wasn't inspected —
only its usage from __main__.py was).
"""
import logging
import os
import shutil
from datetime import datetime

import yaml


def parse(paramcard: str) -> dict:
    """Load a YAML config file into a dict."""
    with open(paramcard, 'r') as f:
        return yaml.safe_load(f)


def setup_dir(paramcard: str) -> str:
    """Create output/<name>_<timestamp>/ for a new run, plus a <name>_latest
    symlink pointing at it. Returns the run directory path.
    """
    params = parse(paramcard)
    name = params['name']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"output/{name}_{timestamp}"
    os.makedirs(run_name, exist_ok=True)

    latest_link = f"output/{name}_latest"
    if os.path.islink(latest_link):
        os.remove(latest_link)
    elif os.path.exists(latest_link):
        if os.path.isdir(latest_link):
            shutil.rmtree(latest_link)
        else:
            os.remove(latest_link)
    try:
        os.symlink(os.path.basename(run_name), latest_link)
    except OSError:
        logging.warning(f"Could not create symlink {latest_link} -> {run_name}")

    return run_name


def find_run(name: str):
    """Locate an existing run directory for `name` via its _latest symlink.

    Returns (run_name, run_exists).
    """
    latest_link = f"output/{name}_latest"
    if os.path.exists(latest_link):
        return latest_link, True
    return f"output/{name}", False


def log_yaml(params: dict, source: str) -> None:
    """Copy the config file used for this run into the run directory."""
    dest = os.path.join(params['name'], os.path.basename(source))
    shutil.copy(source, dest)
    logging.info(f"Config copied to {dest}")
