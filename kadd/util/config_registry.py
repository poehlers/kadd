"""Autodetects available (type, size) config files under config_files/, so the
CLI can accept `--type swin3d --size xlight` instead of a hand-typed path.

Depends entirely on the naming convention `classify_<type>_<size>.yml`
(e.g. `classify_cnn3d_xlight.yml`) — every config file added under this
convention is automatically discoverable with no code change; anything named
differently (e.g. classify_swin3d_v1.yml/classify_swin3d_v2.yml, which predate
this convention and are kept for reproducibility, see CLAUDE.md) is simply not
part of the --type/--size registry and must still be run by explicit path.
"""
import glob
import os
import re

from kadd.classify.model import CNN3D_SIZES, SIZES

TYPES = ('swin3d', 'cnn3d')

# Per-type size vocabularies — swin3d and cnn3d don't share one (cnn3d grew
# an 'xmedium' tier swin3d has no architecture for; see model.py's SIZES/
# CNN3D_SIZES comments).
_VALID_SIZES = {'swin3d': SIZES, 'cnn3d': CNN3D_SIZES}

_CONFIG_NAME_RE = re.compile(r'^classify_([a-z0-9]+)_([a-z0-9]+)\.yml$')

DEFAULT_CONFIG_DIR = 'config_files'


def discover_configs(config_dir: str = DEFAULT_CONFIG_DIR) -> dict:
    """Scan config_dir for classify_<type>_<size>.yml files.

    Returns {(type, size): path}. Only filenames whose type is in TYPES and
    whose size is valid FOR THAT TYPE (_VALID_SIZES) are included — this is
    what excludes the legacy classify_swin3d_v1.yml/v2.yml (their second
    segment, 'v1'/'v2', isn't a recognized size) from being misread as a
    bogus size tier, and what keeps e.g. 'xmedium' being valid for cnn3d
    from also being (wrongly) treated as valid for swin3d.
    """
    registry = {}
    for path in sorted(glob.glob(os.path.join(config_dir, 'classify_*.yml'))):
        m = _CONFIG_NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        type_, size = m.group(1), m.group(2)
        if size in _VALID_SIZES.get(type_, ()):
            registry[(type_, size)] = path
    return registry


def resolve_config(type_: str, size: str, config_dir: str = DEFAULT_CONFIG_DIR) -> str:
    """Resolve --type/--size to a config file path, or raise a clear error
    listing what's actually available in config_dir.
    """
    registry = discover_configs(config_dir)
    path = registry.get((type_, size))
    if path is None:
        available = ', '.join(f"{t}/{s}" for t, s in sorted(registry)) or '(none found)'
        raise ValueError(
            f"No config found for type={type_!r} size={size!r} under {config_dir}/ "
            f"(expected a file named classify_{type_}_{size}.yml). "
            f"Available type/size combos: {available}"
        )
    return path
