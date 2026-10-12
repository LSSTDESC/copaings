"""compute_Sb must also work when a true object and an LSST object of a group share the same row number."""
import numpy as np
import pandas as pd
import pytest

from copaings import deps  # noqa: F401

pytest.importorskip('friendly')
from copaings.sb_core import compute_Sb  # noqa: E402

PS = 0.2


def _cats():
    rng = np.random.default_rng(4)
    n = 12
    ra = 53.1 + rng.uniform(0, 4, n) / 3600
    dec = -28.0 + rng.uniform(0, 4, n) / 3600
    lsst = pd.DataFrame({'ra': ra, 'dec': dec, 'x': (53.1 - ra) * np.cos(np.radians(dec)) * 3600 / PS,
                         'y': (dec + 28.0) * 3600 / PS, 'Ixx_pixel_i': rng.uniform(3, 6, n),
                         'Iyy_pixel_i': rng.uniform(3, 6, n), 'Ixy_pixel_i': rng.uniform(-1, 1, n), 'psf_fwhm_i': 0.8,
                         'mag_i_cModel': rng.uniform(22, 25, n), 'blendedness': 0.1})
    true = pd.DataFrame({'ra': 53.1 + rng.uniform(0, 4, n) / 3600, 'dec': -28.0 + rng.uniform(0, 4, n) / 3600,
                         'size_true': rng.uniform(0.2, 0.6, n), 'size_minor_true': rng.uniform(0.1, 0.3, n),
                         'position_angle_true': rng.uniform(0, 180, n), 'mag_i': rng.uniform(22, 25, n)})
    return lsst, true


def test_group_with_colliding_ids():
    lsst, true = _cats()
    lsst['sb'] = np.nan
    # true object 3 and LSST object 3 have the same row number but are different objects
    compute_Sb([[[3, 4, 5], [3, 6]]], true, lsst, Sb_col_name_in_obs_cat='sb')
    assert np.isfinite(lsst.loc[[3, 6], 'sb']).all()
    assert lsst['sb'].notna().sum() == 2  # only the group's LSST objects are written


def test_collision_free_group_unchanged():
    lsst, true = _cats()
    lsst['sb'] = np.nan
    compute_Sb([[[0, 1], [7, 8]]], true, lsst, Sb_col_name_in_obs_cat='sb')
    assert np.isfinite(lsst.loc[[7, 8], 'sb']).all()
