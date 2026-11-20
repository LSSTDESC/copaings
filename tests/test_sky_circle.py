"""sky_circle: exact great-circle cut shared by the three catalogs."""
import numpy as np

from copaings.catalogs import in_sky_circle
from copaings.config import Case


def test_in_sky_circle():
    ra0, dec0, radius = 53.0, -28.15, 1.6
    # center, just inside / outside along dec, and along ra (where 1 deg of ra is cos(dec) deg on the sky)
    ra = np.array([53.0, 53.0, 53.0, 53.0 + 1.55 / np.cos(np.deg2rad(dec0)), 53.0 + 1.65 / np.cos(np.deg2rad(dec0))])
    dec = np.array([-28.15, -28.15 + 1.59, -28.15 - 1.61, -28.15, -28.15])
    assert in_sky_circle(ra, dec, ra0, dec0, radius).tolist() == [True, True, False, True, False]


def test_case_default_no_circle():
    assert Case(name='t').sky_circle == []
