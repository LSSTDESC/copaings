"""HST footprint: rows without F814W coverage are dropped, and the footprint follows the HST objects (no hull)."""
import numpy as np
import pandas as pd
from astropy.table import Table

from copaings.catalogs import _read_hst_hlf, near_points


def test_near_points_radius():
    survey = np.array([[53.0, -28.0]])
    one_arcsec = 1 / 3600
    tests = np.array([
        [53.0, -28.0 + 4 * one_arcsec],    # 4" away: in
        [53.0, -28.0 + 6 * one_arcsec],    # 6" away: out
        [53.0 + 4.5 * one_arcsec / np.cos(np.radians(28.0)), -28.0],   # 4.5" away along ra: in
    ])
    inside = near_points(tests, survey, 5.0)
    assert list(inside) == [True, False, True]


def test_near_points_gap_inside_hull():
    # two clumps of objects: the middle is inside their convex hull, but has no object nearby
    rng = np.random.default_rng(1)
    clump_a = rng.uniform([53.0, -28.0], [53.01, -27.99], size=(200, 2))
    clump_b = rng.uniform([53.05, -28.0], [53.06, -27.99], size=(200, 2))
    survey = np.vstack([clump_a, clump_b])
    tests = np.array([[53.005, -27.995], [53.03, -27.995]])
    inside = near_points(tests, survey, 5.0)
    assert list(inside) == [True, False]


def test_hlf_rows_without_f814w_coverage_are_dropped(tmp_path):
    n = 4
    table = Table({
        'ra': np.full(n, 53.1), 'dec': np.full(n, -27.8), 'x': np.zeros(n), 'y': np.zeros(n),
        'f_f814w': [10.0, -99.0, -2.0, 5.0],   # covered, no coverage, negative flux, covered but zero weight
        'e_f814w': np.ones(n),
        'w_f814w': [0.5, 0.0, 0.5, 0.0],
        'class_star': np.zeros(n), 'a_image': np.ones(n), 'b_image': np.ones(n), 'theta_J2000': np.zeros(n),
        'flux_radius': np.ones(n), 'fwhm_image': np.ones(n),
    })
    path = tmp_path / 'hlf.fits'
    table.write(path)
    df = _read_hst_hlf(str(path))
    assert len(df) == 1
    assert isinstance(df, pd.DataFrame)
