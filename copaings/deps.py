"""Access to the `friendly` package.

`friendly`: https://github.com/LSSTDESC/friendly

Where to find it:
  * installed in the environment (``pip install -e <path to friendly>``): nothing to do, or
  * set the environment variable FRIENDLY_PATH to the directory that contains the `friendly` package, or
  * fall back to the SLAC default below.
"""
import os
import sys

_DEFAULT_FRIENDLY = '/sdf/home/m/mesquita/WORK/friendly/'

_p = os.environ.get('FRIENDLY_PATH', _DEFAULT_FRIENDLY)
if os.path.isdir(_p) and _p not in sys.path:
    sys.path.insert(0, _p)


def matching():
    """Import and return ``friendly.matching`` (friends-of-friends matching), from https://github.com/LSSTDESC/friendly.

    Returns
    -------
    module
    """
    import friendly.matching as matching
    return matching
