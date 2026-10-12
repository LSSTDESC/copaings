"""The KD-tree ellipse overlap test must agree with the all-pairs one; candidate pairs must contain every overlap."""
import numpy as np
import pandas as pd
import pytest

from copaings import deps  # noqa: F401  (puts `friendly` on sys.path)

pytest.importorskip('friendly')
from copaings.eot_fast import EllipseCache, _candidate_pairs, split_group  # noqa: E402

PS = 0.2
RA0, DEC0 = 53.1, -28.0


def _radec_to_xy(ra, dec):  # the common tangent frame built by `prep`
    return (RA0 - ra) * np.cos(np.radians(dec)) * 3600 / PS, (dec - DEC0) * 3600 / PS


def _catalogs(rng, n_lsst=150, n_true=120, scale_arcsec=12.0):  # >100 LSST objects, needed to recognise the common frame
    ra = RA0 + rng.uniform(0, scale_arcsec, n_lsst) / 3600
    dec = DEC0 + rng.uniform(0, scale_arcsec, n_lsst) / 3600
    x, y = _radec_to_xy(ra, dec)
    ixx = rng.uniform(2, 8, n_lsst)
    iyy = rng.uniform(2, 8, n_lsst)
    lsst = pd.DataFrame({'ra': ra, 'dec': dec, 'x': x, 'y': y, 'Ixx_pixel_i': ixx, 'Iyy_pixel_i': iyy,
                         'Ixy_pixel_i': rng.uniform(-1, 1, n_lsst), 'psf_fwhm_i': 0.8})
    true = pd.DataFrame({'ra': RA0 + rng.uniform(0, scale_arcsec, n_true) / 3600,
                         'dec': DEC0 + rng.uniform(0, scale_arcsec, n_true) / 3600,
                         'size_true': rng.uniform(0.1, 0.8, n_true), 'size_minor_true': rng.uniform(0.05, 0.1, n_true),
                         'position_angle_true': rng.uniform(0, 180, n_true)})
    return lsst, true


def _canon(comps):
    return sorted((tuple(sorted(c[0])), tuple(sorted(c[1]))) for c in comps)


def test_kd_path_matches_all_pairs():
    rng = np.random.default_rng(1)
    lsst, true = _catalogs(rng)
    cache = EllipseCache(lsst, true)
    assert cache.abs_frame is not None  # the common frame is recognised
    for _ in range(25):
        obs = rng.choice(len(lsst), 6, replace=False)
        # no true index equal to an LSST index: the all-pairs path drops those (friendly's "duplicate id" rule), the KD path does not
        pool = np.setdiff1d(np.arange(len(true)), obs)
        grp = [list(rng.choice(pool, 8, replace=False)), list(obs)]
        dense = split_group(cache, [list(grp[0]), list(grp[1])], dense_max=10**9)
        kd = split_group(cache, [list(grp[0]), list(grp[1])], dense_max=0)
        assert _canon(dense) == _canon(kd)


def test_candidate_pairs_contain_every_overlap():
    rng = np.random.default_rng(2)
    n = 400
    x, y = rng.uniform(0, 200, n), rng.uniform(0, 200, n)
    r = np.where(rng.random(n) < 0.1, rng.uniform(10, 40, n), rng.uniform(1, 6, n))  # a few large ellipses
    i, j = _candidate_pairs(x, y, r)
    got = {tuple(sorted(p)) for p in zip(i.tolist(), j.tolist())}
    iu, ju = np.triu_indices(n, 1)
    want = {(a, b) for a, b in zip(iu.tolist(), ju.tolist()) if np.hypot(x[a] - x[b], y[a] - y[b]) <= r[a] + r[b]}
    assert got == want
