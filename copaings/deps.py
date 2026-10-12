"""Access to the `friendly` package (and, optionally, Mesquita's my_blending_lib).

Where to find them:
  * `friendly` installed in the environment (pip install -e <path to friendly>): nothing to do, or
  * set the environment variable FRIENDLY_PATH to the directory that contains the `friendly` package, or
  * fall back to the SLAC default below.
`my_blending_lib` is only needed for the slow reference EOT path (fast_eot: false) and imports the whole LSST stack;
set BLENDING_LIB_PATH if it is not in the SLAC default location.
"""
import os
import sys

_DEFAULT_FRIENDLY = '/sdf/home/m/mesquita/WORK/friendly/'
_DEFAULT_BLENDING_LIB = '/sdf/home/m/mesquita/WORK/commissioning/'

for _env, _default in (('BLENDING_LIB_PATH', _DEFAULT_BLENDING_LIB), ('FRIENDLY_PATH', _DEFAULT_FRIENDLY)):
    _p = os.environ.get(_env, _default)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def mbl():
    """Import and return Mesquita's ``my_blending_lib``.

    It imports the whole LSST stack, so it is only used by the slow reference ellipse overlap test
    (``fast_eot: false``). Set ``BLENDING_LIB_PATH`` if it is not at the default location.

    Returns
    -------
    module
    """
    import my_blending_lib
    return my_blending_lib


def matching():
    """Import and return ``friendly.matching`` (friends-of-friends matching).

    Returns
    -------
    module
    """
    import friendly.matching as matching
    return matching
