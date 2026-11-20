"""Find the `friendly` package (https://github.com/LSSTDESC/friendly) without having to install it.

`add_friendly_to_path()` makes ``import friendly`` work. It uses, in this order:

1. the directory given by the environment variable ``FRIENDLY_PATH``;
2. a `friendly` that is already importable (installed with pip, or already on ``sys.path``);
3. a clone of friendly next to this copaings clone, i.e. ``<folder containing copaings>/friendly``;
4. the copy on the SLAC file system, ``DEFAULT_FRIENDLY_PATH``.

The directory to give is the root of the friendly git clone, the one that *contains* the ``friendly/`` package folder.
"""
import importlib.util
import os
import sys

COPAINGS_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FRIENDLY_PATH = '/sdf/home/m/mesquita/WORK/friendly'


def _is_friendly_clone(path):
    """True if `path` is a directory that contains the ``friendly`` package folder."""
    return bool(path) and os.path.isfile(os.path.join(path, 'friendly', 'ellipses.py'))


def add_friendly_to_path():
    """Make ``import friendly`` work (see the module docstring for where friendly is looked for).

    Returns
    -------
    str or None
        Directory added to ``sys.path``, or None if friendly was already importable.

    Raises
    ------
    ImportError
        If friendly is found nowhere.
    """
    from_env = os.environ.get('FRIENDLY_PATH', '')
    if _is_friendly_clone(from_env):
        if from_env not in sys.path:
            sys.path.insert(0, from_env)
        return from_env

    if importlib.util.find_spec('friendly') is not None:
        return None

    next_to_copaings = os.path.join(os.path.dirname(COPAINGS_REPO), 'friendly')
    for path in (next_to_copaings, DEFAULT_FRIENDLY_PATH):
        if _is_friendly_clone(path):
            sys.path.insert(0, path)
            return path

    raise ImportError('friendly not found. Clone it next to copaings '
                      f'(git clone https://github.com/LSSTDESC/friendly.git {next_to_copaings}) '
                      'or set FRIENDLY_PATH to the directory of your friendly clone.')
