"""HST-only cases (COSMOS): config checks and the COSMOS ACS reader."""
import numpy as np
import pandas as pd
import pytest

from copaings.catalogs import build_hst
from copaings.config import Case


def test_telescopes_checked_and_ordered():
    assert Case(name='t', telescopes=['HST']).telescopes == ['HST']
    assert Case(name='t', telescopes=['HST', 'Euclid']).telescopes == ['Euclid', 'HST']
    with pytest.raises(ValueError):
        Case(name='t', telescopes=[])
    with pytest.raises(ValueError):
        Case(name='t', telescopes=['JWST'])
    with pytest.raises(ValueError):
        Case(name='t', hst_catalog='other')


def test_cosmos_acs_reader(tmp_path):
    path = tmp_path / 'acs.parq'
    pd.DataFrame({
        'ra': [150.0, 150.1, 150.2, 150.3], 'dec': [2.0, 2.1, 2.2, 2.3],
        'mag_auto': [22.0, 24.0, 99.0, 23.0], 'magerr_auto': [0.01, 0.05, 9.9, 0.02],
        'a_world': [1e-4, 2e-4, 1e-4, 1e-4], 'b_world': [5e-5, 2e-4, 1e-4, 1e-4], 'theta_world': [30.0, -10.0, 0, 0],
        'class_star': [0.0, 0.1, 0.0, 0.99], 'mu_class': [1, 1, 1, 2],
        'flux_radius': [3.0, 4.0, 3.0, 2.0], 'fwhm_image': [5.0, 6.0, 5.0, 3.0],
    }).to_parquet(path)
    df = build_hst(Case(name='t', telescopes=['HST'], hst_catalog='cosmos_acs', hst_file=str(path)))
    assert len(df) == 2  # mag 99 and the star (mu_class 2) dropped
    np.testing.assert_allclose(df['F814_MAG'], [22.0, 24.0], atol=1e-6)  # flux rebuilt from MAG_AUTO round-trips
    np.testing.assert_allclose(df['size_true'], [0.36, 0.72])  # a_world deg -> arcsec
    np.testing.assert_allclose(df['ellipticity'], [0.5, 0.0])
    np.testing.assert_allclose(df['position_angle_true'], [-30.0, 10.0])
