#!/usr/bin/env python3
"""Download the RSNA Knee Abnormality Detection competition data via kagglehub.

Non-interactive by design (does NOT call kagglehub.login(), which opens a
browser OAuth flow) — checks for an existing credential and exits with an
actionable message if none is found, rather than hanging or prompting.

Manual prerequisites (cannot be done by this script):
  1. Have a Kaggle account, and click "Join Competition" / accept the rules
     on the competition page. Client libraries cannot bypass this — Kaggle
     returns a 403 (competitions.participate denied) until you've done it.
  2. Get a credential: download kaggle.json from kaggle.com/settings to
     ~/.kaggle/kaggle.json, OR set KAGGLE_API_TOKEN, OR set KAGGLE_USERNAME
     + KAGGLE_KEY.

Usage:
    python scripts/download_data.py [--output-dir DIR]
"""
import argparse
import inspect
import os
import subprocess
import sys

import kagglehub

# Overview URL the user originally gave (singular "abnormality"). RSNA's own
# Rules-doc "Competition Website" field says the plural
# "rsna-knee-abnormalities-detection", likely a typo in their boilerplate —
# try singular first, fall back to plural.
SLUG_CANDIDATES = [
    "rsna-knee-abnormality-detection",
    "rsna-knee-abnormalities-detection",
]

DEFAULT_OUTPUT_DIR = "data"

# kagglehub/kagglesdk only ever check KAGGLE_API_TOKEN (env var — either a
# literal token or a path to a file containing one), ~/.kaggle/access_token,
# ~/.kaggle/access_token.txt, or ~/.kaggle/kaggle.json (confirmed by reading
# kagglesdk.kaggle_env.get_access_token_from_env directly). A project-local
# token file is NOT auto-discovered, so if one exists here, point
# KAGGLE_API_TOKEN at it explicitly rather than requiring it live in $HOME.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_TOKEN_PATH = os.path.join(_PROJECT_ROOT, ".kaggle", "access_token")


def _has_credentials() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if os.path.exists(os.path.expanduser("~/.kaggle/access_token")):
        return True
    if os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        return True
    if os.path.exists(LOCAL_TOKEN_PATH):
        os.environ["KAGGLE_API_TOKEN"] = LOCAL_TOKEN_PATH
        return True
    return False


def _download(slug: str, output_dir: str) -> str:
    kwargs = {}
    sig = inspect.signature(kagglehub.competition_download)
    if "output_dir" in sig.parameters:
        kwargs["output_dir"] = output_dir
        path = kagglehub.competition_download(slug, **kwargs)
    else:
        # Older kagglehub without output_dir support: fall back to steering
        # its cache via KAGGLEHUB_CACHE so a large download doesn't silently
        # land on the internal disk.
        os.environ["KAGGLEHUB_CACHE"] = output_dir
        path = kagglehub.competition_download(slug)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                     help=f"Where the competition files land (default: {DEFAULT_OUTPUT_DIR}, "
                          "on the same /Volumes/T7 external volume as the rest of this project)")
    args = ap.parse_args()

    if not _has_credentials():
        print(
            "ERROR: No Kaggle credentials found.\n"
            "  1. Go to kaggle.com/settings -> API -> Create New Token, which downloads kaggle.json.\n"
            "  2. Put it at ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME + KAGGLE_KEY, "
            "or KAGGLE_API_TOKEN).\n"
            "  3. Also make sure you've clicked \"Join Competition\" on the competition page — "
            "a credential alone isn't enough.\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    last_err = None
    for slug in SLUG_CANDIDATES:
        try:
            print(f"Trying competition slug: {slug}")
            # Preview via the `kaggle` CLI (kaggle competitions files) before
            # the actual kagglehub download — this is a lightweight metadata
            # call that both validates the slug and shows total file sizes,
            # so a large multi-GB download isn't started blind.
            print(f"  Files for {slug} (kaggle competitions files):")
            subprocess.run(["kaggle", "competitions", "files", slug])
            path = _download(slug, args.output_dir)
            print(f"Downloaded to: {path}")
            break
        except Exception as e:  # noqa: BLE001 — surfaced to the user below
            msg = str(e)
            if "competitions.participate" in msg or "403" in msg:
                print(
                    "ERROR: Kaggle denied this download (competitions.participate). "
                    "This means your account hasn't joined/accepted the rules for "
                    f"'{slug}' yet — go to kaggle.com/c/{slug} and click "
                    "\"Join Competition\", then re-run this script.",
                    file=sys.stderr,
                )
                sys.exit(1)
            last_err = e
            print(f"  ... failed ({e}), trying next slug candidate if any")
    else:
        print(f"ERROR: all competition slug candidates failed. Last error: {last_err}",
              file=sys.stderr)
        sys.exit(1)

    print("\nDownload directory contents (top level):")
    subprocess.run(["ls", "-la", args.output_dir])
    print("\nDisk usage of the download target volume:")
    subprocess.run(["df", "-h", args.output_dir])
    print("\nSize of the downloaded data:")
    subprocess.run(["du", "-sh", args.output_dir])
    print(
        "\nIf train_series/ / test_series/ ended up nested under a subfolder "
        "kagglehub created, move/symlink them so the layout matches what "
        "config_files/classify_swin3d_v1.yml expects: "
        "data/train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm"
    )


if __name__ == "__main__":
    main()
