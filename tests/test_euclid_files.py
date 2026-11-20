"""Euclid tile files: Q1 (healpix-named) and DR1 (TILE_INDEX-named) folders."""
import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from copaings.catalogs import build_euclid, euclid_files, euclid_tiles
from copaings.config import Case


def _write_tile(folder, release, tile, ra, dec, n=5):
    """Write a small MER-like tile catalog around (ra, dec)."""
    rng = np.random.default_rng(tile % 1000)
    Table({
        'RIGHT_ASCENSION': ra + rng.uniform(-0.01, 0.01, n),
        'DECLINATION': dec + rng.uniform(-0.01, 0.01, n),
        'POSITION_ANGLE': rng.uniform(0, 180, n),
        'SEMIMAJOR_AXIS': rng.uniform(2, 5, n),
        'ELLIPTICITY': rng.uniform(0, 0.5, n),
        'FLUX_VIS_1FWHM_APER': rng.uniform(0.1, 10, n),
        'FLUXERR_VIS_1FWHM_APER': np.full(n, 0.01),
        'EXTENDED_FLAG': np.ones(n, dtype=np.int32),
    }).write(folder / f'euclid_{release}_mer_final_{tile}.fits')


def _lsst(ra, dec):
    return pd.DataFrame({'ra': [ra], 'dec': [dec]})


def test_q1_auto_uses_healpix(tmp_path):
    import healpy as hp
    ra, dec = 53.0, -28.0
    pix = int(hp.ang2pix(64, ra, dec, nest=True, lonlat=True))
    _write_tile(tmp_path, 'q1', pix, ra, dec)
    _write_tile(tmp_path, 'q1', 10207, 60.0, -48.0)  # elsewhere, no LSST object there
    (tmp_path / 'notes.txt').write_text('ignored')
    case = Case(name='t', euclid_dir=str(tmp_path))
    assert list(euclid_files(case)) == sorted([pix, 10207])
    assert euclid_tiles(case, _lsst(ra, dec)) == [pix]
    assert euclid_tiles(Case(name='t', euclid_dir=str(tmp_path), euclid_region='all'), _lsst(ra, dec)) == sorted([pix, 10207])


def test_dr1_auto_reads_all(tmp_path):
    tiles = [101370732, 101370733]
    _write_tile(tmp_path, 'dr1', tiles[0], 53.0, -28.0)
    _write_tile(tmp_path, 'dr1', tiles[1], 53.5, -28.5)
    case = Case(name='t', euclid_dir=str(tmp_path))
    assert euclid_tiles(case, _lsst(53.0, -28.0)) == tiles
    df = build_euclid(case, _lsst(53.0, -28.0))
    assert len(df) == 10
    assert np.isfinite(df[['ra', 'dec', 'vis_mag', 'size_true']].values).all()
    with pytest.raises(ValueError):
        euclid_tiles(Case(name='t', euclid_dir=str(tmp_path), euclid_region='notebook'), _lsst(53.0, -28.0))


def test_mixed_or_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        euclid_files(Case(name='t', euclid_dir=str(tmp_path)))
    _write_tile(tmp_path, 'q1', 10207, 53.0, -28.0)
    _write_tile(tmp_path, 'dr1', 101370732, 53.0, -28.0)
    with pytest.raises(ValueError):
        euclid_files(Case(name='t', euclid_dir=str(tmp_path)))
