"""Build the LSST / Euclid / HST catalogs (logic lifted from the notebook, stage `prep`)."""
import gc
import json
import os
import re

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from scipy.spatial import ConvexHull, Delaunay, cKDTree

USDF_REPO_INDEX = '/sdf/group/rubin/shared/data-repos.yaml'  # butler repository names at USDF (dp1, dp2_prep, ...)

EUCLID_PIXEL_SCALE = 0.1  # arcsec / pix
# HLF GOODS-S v2.1 60mas mosaic: 0.06 arcsec / pix (Whitaker et al. 2019, ApJS 244, 16); checked on the catalog
# by fitting x, y against ra, dec
HST_PIXEL_SCALE = 0.06

# (ra, dec) of the Euclid Q1 healpix tiles (nside=64) covering ECDFS
EUCLID_TILE_CENTERS = [(53.25, -27.9), (52.75, -28.4), (52.75, -27.4), (52.25, -28.0), (53.6, -28.5), (53.5, -29.0)]
EUCLID_BOX = dict(ra=(52.25, 54.00), dec=(-28.75, -27.25))


def nJy2mag(flux):
    """Convert a flux to an AB magnitude.

    Parameters
    ----------
    flux : array_like
        Flux [nJy].

    Returns
    -------
    numpy.ndarray
        AB magnitude. Zero flux gives inf and negative flux gives NaN (callers silence the numpy warnings).
    """
    return u.nJy.to(u.ABmag, flux)


def mag_err(flux, flux_err):
    """Magnitude error from a flux and its error.

    Parameters
    ----------
    flux, flux_err : array_like
        Flux and its 1-sigma error, in the same units.

    Returns
    -------
    numpy.ndarray
        ``1.085736 * flux_err / flux`` [mag] (1.0857 = 2.5 / ln 10).
    """
    return 1.085736 * flux_err / flux


def native_endian(df):
    """Convert every column of a dataframe to native byte order, in place.

    FITS tables are big-endian, which pandas >= 2.3 cannot hash (``drop_duplicates``, ``groupby``) or hand to
    scikit-learn.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to convert; modified in place.

    Returns
    -------
    pandas.DataFrame
        The same dataframe.
    """
    for col in df.columns:
        arr = np.asarray(df[col])
        if arr.dtype.byteorder not in ('=', '|'):
            df[col] = pd.Series(arr.astype(arr.dtype.newbyteorder('=')), index=df.index)
    return df


def add_ellipticity_lsst(df):
    """Add the ellipticity of the LSST objects from their i-band second moments, in place.

    Parameters
    ----------
    df : pandas.DataFrame
        LSST catalog with ``Ixx_pixel_i``, ``Iyy_pixel_i`` and ``Ixy_pixel_i``. The columns ``e1`` (``(Ixx-Iyy)/(Ixx+Iyy)``),
        ``e2`` (``2 Ixy/(Ixx+Iyy)``) and ``ellipticity`` (``hypot(e1, e2)``) are added.
    """
    tr = df['Ixx_pixel_i'] + df['Iyy_pixel_i']
    df['e1'] = (df['Ixx_pixel_i'] - df['Iyy_pixel_i']) / tr
    df['e2'] = 2.0 * df['Ixy_pixel_i'] / tr
    df['ellipticity'] = np.hypot(df['e1'], df['e2'])


# ----------------------------------------------------------------------------- LSST
def use_usdf_butler_defaults():
    """Set the USDF butler environment variables that are not already set.

    The USDF Jupyter kernels set them, but a terminal or a slurm job may not:

    * ``DAF_BUTLER_REPOSITORY_INDEX``: where repository names such as ``'dp2_prep'`` are defined;
    * ``PGPASSFILE`` and ``PGUSER``: the credentials of the butler database (``~/.lsst/postgres-credentials.txt``).

    Outside USDF (files absent) nothing is changed.
    """
    import os
    if os.path.exists(USDF_REPO_INDEX):
        os.environ.setdefault('DAF_BUTLER_REPOSITORY_INDEX', USDF_REPO_INDEX)
    credentials = os.path.expanduser('~/.lsst/postgres-credentials.txt')
    if os.path.exists(credentials) and 'PGPASSFILE' not in os.environ:
        os.environ['PGPASSFILE'] = credentials
        os.environ.setdefault('PGUSER', 'rubin')


def fetch_lsst_raw(case, columns=None):
    """Read the LSST object table of every tract of the case from the butler.

    Only the columns that are needed are read: the DP2 object table has about 1200 columns (about 7 GB per tract as a
    dataframe). If the column selection is not supported the full table is read and then reduced.

    Parameters
    ----------
    case : Case
        Uses `repo`, `collections`, `skymap`, `dataset_type`, `tracts` and `columns`.
    columns : list of str, optional
        Butler columns to read; default all the columns of `case.columns`.

    Returns
    -------
    pandas.DataFrame
        Butler columns of all tracts concatenated, plus ``tract_id``. A tract that cannot be read is skipped with a
        message.

    Raises
    ------
    RuntimeError
        If no tract could be read.
    """
    from lsst.daf.butler import Butler
    use_usdf_butler_defaults()
    butler = Butler(case.repo, collections=case.collections)
    wanted = list(dict.fromkeys(columns or case.columns.values()))
    out = []
    for tract in case.tracts:
        data_id = {'skymap': case.skymap, 'tract': tract}
        try:
            try:
                available = set(butler.get(f'{case.dataset_type}.columns', data_id))
                params = {'columns': [c for c in wanted if c in available]}
                df = butler.get(case.dataset_type, data_id, parameters=params).to_pandas()
            except Exception as e:  # no column component / parameters support: read everything
                print(f'tract {tract}: column selection failed ({type(e).__name__}), reading the full table')
                df = butler.get(case.dataset_type, data_id).to_pandas()
                df = df[[c for c in wanted if c in df.columns]]
            df['tract_id'] = tract
            out.append(df)
            print(f'Added tract {tract}: {len(df):,} rows, {df.shape[1]} columns')
        except Exception as e:
            print(f'Skipping tract {tract}: {e}')
    if not out:
        raise RuntimeError('no tract could be read')
    return pd.concat(out)


def lsst_all_radec(case, raw=None):
    """Positions of every LSST galaxy of the tracts, before any sky cut (for the sky maps).

    Parameters
    ----------
    case : Case
    raw : pandas.DataFrame, optional
        Output of `fetch_lsst_raw`; when not given only the ra, dec and i-extendedness columns are read from the butler.

    Returns
    -------
    pandas.DataFrame
        Columns ``ra``, ``dec`` [deg] of the extended objects (``i_extendedness == 1``, as in `build_lsst`).
    """
    cols = case.columns
    names = {cols[k]: k for k in ('ra', 'dec', 'i_extendedness')}
    if raw is None:
        raw = fetch_lsst_raw(case, columns=list(names))
    df = raw[list(names)].rename(columns=names)
    df = df[df['i_extendedness'] == 1]
    return pd.DataFrame({'ra': df['ra'].values, 'dec': df['dec'].values})


# internal names that may legitimately be absent from the butler table (DP2 has no per-object pixel centroid / i_fwhm)
OPTIONAL_LSST = ('x', 'y', 'psf_fwhm_i_from_butler', 'u_cModelFlux', 'z_cModelFlux', 'y_cModelFlux')


def build_lsst(case, raw):
    """Build the LSST catalog used by the pipeline from the raw butler table.

    Renames the columns to the internal names, drops point sources (``i_extendedness != 1``), computes the PSF size,
    the ellipticity, the cModel magnitudes in ugrizy and, by default, the pixel positions in a common tangent plane.
    Columns that may legitimately be absent (``x``, ``y``, ``psf_fwhm_i_from_butler``, u, z and y fluxes) become NaN.

    Parameters
    ----------
    case : Case
        Column map, pixel scale and `common_frame_xy`.
    raw : pandas.DataFrame
        Output of `fetch_lsst_raw`.

    Returns
    -------
    pandas.DataFrame
        Fresh RangeIndex. Notable columns: ``ra``, ``dec`` [deg], ``x``, ``y`` [pixel, common frame],
        ``Ixx_pixel_i``, ``Iyy_pixel_i``, ``Ixy_pixel_i`` [pixel2], ``psf_fwhm_i`` [arcsec], ``e1``, ``e2``,
        ``ellipticity``, ``{b}_cModel_mag`` (AB), ``i_blendedness``, ``tract_id``.

    Raises
    ------
    KeyError
        If a required column is missing from the table (all missing columns are listed).
    ValueError
        If the table has no pixel centroids and `common_frame_xy` is False.
    """
    cols = case.columns
    missing = [f'{new} <- {old}' for new, old in cols.items() if old not in raw.columns and new not in OPTIONAL_LSST]
    if missing:
        raise KeyError('columns missing from the butler table, set them in `lsst_columns` of the case yaml:\n  '
                       + '\n  '.join(missing))
    df = pd.DataFrame({new: raw[old].values for new, old in cols.items() if old in raw.columns})
    df['tract_id'] = raw['tract_id'].values
    has_xy = 'x' in df.columns and 'y' in df.columns
    if not has_xy and not case.common_frame_xy:
        raise ValueError('table has no pixel centroids: common_frame_xy must be true')
    if 'psf_fwhm_i_from_butler' not in df.columns:
        df['psf_fwhm_i_from_butler'] = np.nan
    for b in 'uzy':  # bands used by the VIS / F814 tree models; absent bands become all-NaN
        if f'{b}_cModelFlux' not in df.columns:
            df[f'{b}_cModelFlux'] = np.nan
    ixx, iyy = df['psf_ixx'], df['psf_iyy']
    sigma = np.sqrt((ixx + iyy) * case.lsst_pixel_scale**2 / 2)
    df['psf_fwhm_i'] = 2 * np.sqrt(2 * np.log(2)) * sigma  # arcsec

    df = df[df['i_extendedness'] == 1].reset_index(drop=True)  # drop point sources
    if has_xy:
        df['x_tract'], df['y_tract'] = df['x'], df['y']
    if case.common_frame_xy or not has_xy:
        # same orientation convention as friendly.ellipses.deg2pix: x grows towards decreasing RA, y towards +dec
        ra0, dec0 = df['ra'].mean(), df['dec'].mean()
        df['x'] = (ra0 - df['ra']) * np.cos(np.radians(df['dec'])) * 3600 / case.lsst_pixel_scale
        df['y'] = (df['dec'] - dec0) * 3600 / case.lsst_pixel_scale
    add_ellipticity_lsst(df)
    with np.errstate(divide='ignore', invalid='ignore'):
        for b in 'ugrizy':
            df[f'{b}_cModel_mag'] = nJy2mag(df[f'{b}_cModelFlux'])
    return df


# ----------------------------------------------------------------------------- Euclid
EUCLID_FILE_RE = re.compile(r'^euclid_(?P<release>[a-z0-9]+)_mer_final_(?P<tile>\d+)\.fits$')
N_HEALPIX_64 = 12 * 64 ** 2  # Q1 tiles are named by their healpix pixel (nside = 64); DR1 tiles by TILE_INDEX


def euclid_files(case):
    """Find the Euclid MER tile catalogs in `euclid_dir`.

    Parameters
    ----------
    case : Case
        Uses `euclid_dir`, a folder of files named ``euclid_<release>_mer_final_<tile>.fits``, for example
        ``euclid_q1_mer_final_10207.fits`` (Q1, tile = healpix pixel) or ``euclid_dr1_mer_final_101370732.fits``
        (DR1, tile = MER TILE_INDEX). Other files are ignored.

    Returns
    -------
    dict
        ``{tile: path}``, sorted by tile.

    Raises
    ------
    FileNotFoundError
        If the folder holds no such file.
    ValueError
        If it holds files of several releases (the same sources would be read twice).
    """
    files, releases = {}, set()
    for name in os.listdir(case.euclid_dir):
        m = EUCLID_FILE_RE.match(name)
        if m:
            files[int(m['tile'])] = os.path.join(case.euclid_dir, name)
            releases.add(m['release'])
    if not files:
        raise FileNotFoundError(f'no euclid_<release>_mer_final_<tile>.fits file in {case.euclid_dir}')
    if len(releases) > 1:
        raise ValueError(f'{case.euclid_dir} mixes Euclid releases {sorted(releases)}: keep one per folder')
    return dict(sorted(files.items()))


def euclid_release(case):
    """Euclid data release of the files in `euclid_dir` (``'Q1'``, ``'DR1'``...), from their names; None if unknown."""
    try:
        names = os.listdir(case.euclid_dir)
    except OSError:
        return None
    found = set()
    for name in names:
        m = EUCLID_FILE_RE.match(name)
        if m:
            found.add(m['release'].upper())
    if len(found) == 1:
        return found.pop()
    return None


def _healpix_named(tiles):
    """True if every tile number is a healpix pixel at nside 64 (Q1 naming)."""
    return all(0 <= t < N_HEALPIX_64 for t in tiles)


def euclid_tiles(case, lsst):
    """Euclid MER tiles to read for a case.

    Parameters
    ----------
    case : Case
        Uses `euclid_region` and `euclid_dir`.
    lsst : pandas.DataFrame
        LSST catalog with ``ra`` and ``dec`` [deg].

    Returns
    -------
    list of int
        Tile numbers (keys of `euclid_files`). ``'auto'``: for healpix-named files (Q1), the tiles that contain LSST
        objects; otherwise (DR1, named by TILE_INDEX) every file of the folder. ``'all'``: every file of the folder.
        ``'notebook'``: the six hand-picked ECDFS healpix tiles of `EUCLID_TILE_CENTERS`.

    Raises
    ------
    ValueError
        If `euclid_region` is not ``'auto'``, ``'all'`` or ``'notebook'``, or if ``'notebook'`` is used with files that
        are not healpix-named.
    """
    import healpy as hp
    available = list(euclid_files(case))
    if case.euclid_region == 'all':
        return available
    if case.euclid_region == 'notebook':
        if not _healpix_named(available):
            raise ValueError("euclid_region 'notebook' needs healpix-named (Q1) tile files")
        return sorted({int(hp.ang2pix(64, np.radians(90.0 - dec), np.radians(ra), nest=True))
                       for ra, dec in EUCLID_TILE_CENTERS})
    if case.euclid_region != 'auto':
        raise ValueError("euclid_region must be 'auto', 'all' or 'notebook'")
    if not _healpix_named(available):
        print(f'Euclid: tile files of {case.euclid_dir} are not named by healpix pixel (e.g. DR1): reading all of them')
        return available
    lsst_tiles = np.unique(hp.ang2pix(64, lsst['ra'].values, lsst['dec'].values, nest=True, lonlat=True))
    return sorted(set(int(t) for t in lsst_tiles) & set(available))


def build_euclid(case, lsst):
    """Build the Euclid catalog from the MER tiles.

    Parameters
    ----------
    case : Case
        Uses `euclid_dir` and `euclid_region`.
    lsst : pandas.DataFrame
        LSST catalog, used to choose the tiles (see `euclid_tiles`).

    Returns
    -------
    pandas.DataFrame
        Fresh RangeIndex, native byte order, duplicate positions removed. Columns: ``ra``, ``dec`` [deg],
        ``position_angle_true`` [deg], ``size_true`` and ``size_minor_true`` (semi-major / semi-minor axis [arcsec]),
        ``ellipticity``, ``e1``, ``e2``, ``vis_flux`` and ``vis_fluxErr`` (1-FWHM aperture flux [nJy]), ``vis_mag``,
        ``vis_magErr``, ``is_extended``.
    """
    files = euclid_files(case)
    pix = [p for p in euclid_tiles(case, lsst) if p in files]  # 'notebook' tiles may lack a file
    if not pix:
        raise FileNotFoundError(f'no Euclid tile to read in {case.euclid_dir} (euclid_region={case.euclid_region!r})')
    print(f'Euclid: reading {len(pix)} tiles {pix}')
    parts = []
    for p in pix:
        with fits.open(files[p]) as hd:
            d = hd[1].data
            parts.append(pd.DataFrame({
                'ra': d['RIGHT_ASCENSION'],
                'dec': d['DECLINATION'],
                'position_angle_true': d['POSITION_ANGLE'] + 90,
                'size_true': d['SEMIMAJOR_AXIS'] * EUCLID_PIXEL_SCALE,  # arcsec
                'ellipticity': d['ELLIPTICITY'],
                'vis_flux': d['FLUX_VIS_1FWHM_APER'] * 1e3,  # uJy -> nJy
                'vis_fluxErr': d['FLUXERR_VIS_1FWHM_APER'] * 1e3,
                'is_extended': d['EXTENDED_FLAG'],
            }))
    df = native_endian(pd.concat(parts, ignore_index=True))  # FITS columns are big-endian; pandas >= 2.3 cannot hash them
    del parts
    gc.collect()

    if case.euclid_region == 'notebook':
        (r0, r1), (d0, d1) = EUCLID_BOX['ra'], EUCLID_BOX['dec']
        df = df[(df['ra'] <= r1) & (df['ra'] >= r0) & (df['dec'] <= d1) & (df['dec'] >= d0)].reset_index(drop=True)
    df = df.drop_duplicates(subset=['ra', 'dec']).reset_index(drop=True)  # tiles may overlap slightly at edges

    df['size_minor_true'] = df['size_true'] * (1 - df['ellipticity'])
    with np.errstate(divide='ignore', invalid='ignore'):
        df['vis_mag'] = nJy2mag(df['vis_flux'])
        df['vis_magErr'] = mag_err(df['vis_flux'], df['vis_fluxErr'])
    theta = np.radians(df['position_angle_true'])
    df['e1'] = df['ellipticity'] * np.cos(2 * theta)
    df['e2'] = df['ellipticity'] * np.sin(2 * theta)
    return native_endian(df)


def occupied_area(ra, dec, cell=0.02):
    """Sky area covered by a set of objects.

    Counts the occupied cells of a grid that is uniform in (ra * cos(dec0), dec). Unlike a convex hull this excludes
    holes and gaps, which matters for patchy coverage.

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].
    cell : float, default 0.02
        Cell size [deg].

    Returns
    -------
    float
        Area [deg2].
    """
    cd = np.cos(np.radians(np.median(dec)))
    ix = np.floor(np.asarray(ra) * cd / cell).astype(np.int64)
    iy = np.floor(np.asarray(dec) / cell).astype(np.int64)
    return len(np.unique(ix * 1_000_000 + iy)) * cell * cell


# ----------------------------------------------------------------------------- HST
def build_hst(case):
    """Build the HST catalog of the case.

    Parameters
    ----------
    case : Case
        Uses `hst_file` and `hst_catalog` (``'hlf_goodss'``: HLF GOODS-S, ECDFS; ``'cosmos_acs'``: COSMOS ACS F814W
        catalog of Leauthaud et al. 2007).

    Returns
    -------
    pandas.DataFrame
        Fresh RangeIndex, native byte order, stars removed and objects without F814W flux dropped. Columns: ``ra``,
        ``dec`` [deg], ``size_true`` and ``size_minor_true`` (semi-major / semi-minor axis [arcsec]),
        ``position_angle_true`` [deg], ``ellipticity``, ``e1``, ``e2``, ``F814_FLUX`` and ``F814_FLUXERR`` [nJy],
        ``F814_MAG``, ``F814_MAGERR``, plus ``class_star``, ``flux_radius`` and ``fwhm_image`` (HLF also ``x``, ``y``).
    """
    df = _read_hst_cosmos_acs(case.hst_file) if case.hst_catalog == 'cosmos_acs' else _read_hst_hlf(case.hst_file)
    e = 1 - df['size_minor_true'] / df['size_true']
    theta = np.radians(df['position_angle_true'])
    df['ellipticity'] = e
    df['e1'] = e * np.cos(2 * theta)
    df['e2'] = e * np.sin(2 * theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['F814_MAG'] = nJy2mag(df['F814_FLUX'])
        df['F814_MAGERR'] = mag_err(df['F814_FLUX'], df['F814_FLUXERR'])
    return native_endian(df)


def _read_hst_hlf(path):
    """HLF GOODS-S catalog (fits): galaxies (``class_star <= 0.6``) with an F814W flux, sizes in arcsec."""
    with fits.open(path) as hd:
        df = native_endian(pd.DataFrame(hd[1].data))
    cols = ['ra', 'dec', 'x', 'y', 'f_f814w', 'e_f814w', 'w_f814w', 'class_star', 'a_image', 'b_image',
            'theta_J2000', 'flux_radius', 'fwhm_image']
    df = df[cols].rename(columns={'f_f814w': 'F814_FLUX', 'e_f814w': 'F814_FLUXERR', 'a_image': 'size_true',
                                  'b_image': 'size_minor_true', 'theta_J2000': 'position_angle_true'})
    df = df.dropna(subset=['F814_FLUX'])
    # Rows without F814W coverage have f_f814w = -99 and w_f814w = 0 (not NaN, so dropna keeps them). They would
    # widen the HST footprint over sky with no F814W data, so drop them here, before any footprint is built.
    # Rows with f <= 0 would get no magnitude and be removed by the magnitude cut anyway.
    covered = (df['w_f814w'] > 0) & (df['F814_FLUX'] > 0)
    df = df[covered]
    # HLF fluxes are scaled to an AB zero point of 25 (mag = 25 - 2.5 log10 f), not given in uJy
    # (Whitaker et al. 2019, ApJS 244, 16, "The Hubble Legacy Field GOODS-S Photometric Catalog").
    # To nJy (AB zero point 31.4): f_nJy = f * 10**(0.4 * (31.4 - 25))
    zp25_to_nJy = 10 ** (0.4 * (31.4 - 25))
    for c in ('F814_FLUX', 'F814_FLUXERR'):
        df[c] = df[c] * zp25_to_nJy
    df = df[df['class_star'] <= 0.6].reset_index(drop=True)  # 0 = galaxy, 1 = star
    df['size_true'] *= HST_PIXEL_SCALE
    df['size_minor_true'] *= HST_PIXEL_SCALE
    df['position_angle_true'] = -df['position_angle_true']
    return df


def _read_hst_cosmos_acs(path):
    """COSMOS ACS F814W catalog (Leauthaud et al. 2007, parquet): galaxies (``mu_class == 1``) with a valid MAG_AUTO.

    The ACS tiles are rotated, so sizes come from ``a_world``, ``b_world`` [deg] and the angle from ``theta_world``
    (counter-clockwise from east), which `position_angle_true` = -theta_world puts in the same convention as the HLF
    catalog (-theta_J2000 of north-up images). Fluxes are rebuilt from MAG_AUTO (AB).
    """
    cols = ['ra', 'dec', 'mag_auto', 'magerr_auto', 'a_world', 'b_world', 'theta_world', 'class_star', 'mu_class',
            'flux_radius', 'fwhm_image']
    df = pd.read_parquet(path, columns=cols)
    df = df[(df['mu_class'] == 1) & (df['mag_auto'] < 50) & (df['a_world'] > 0)].reset_index(drop=True)  # 99 = no mag
    flux = 10 ** ((31.4 - df['mag_auto'].values) / 2.5)  # AB mag -> nJy
    out = pd.DataFrame({
        'ra': df['ra'].values,
        'dec': df['dec'].values,
        'size_true': df['a_world'].values * 3600,  # arcsec
        'size_minor_true': df['b_world'].values * 3600,
        'position_angle_true': -df['theta_world'].values,
        'F814_FLUX': flux,
        'F814_FLUXERR': flux * np.log(10) / 2.5 * df['magerr_auto'].values,
        'class_star': df['class_star'].values,
        'flux_radius': df['flux_radius'].values,
        'fwhm_image': df['fwhm_image'].values,
    })
    return native_endian(out)


# ----------------------------------------------------------------------------- footprints / assembly
def _radec(df):
    """Stack the sky positions of a catalog.

    Parameters
    ----------
    df : pandas.DataFrame
        Catalog with ``ra`` and ``dec``.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 2) with columns ra, dec.
    """
    return np.stack((df['ra'].values, df['dec'].values), axis=1)


def in_hull(p, points):
    """Test which points lie inside the convex hull of another set of points.

    Only the hull vertices are triangulated, which gives the same region as triangulating every point but is cheap
    for millions of points.

    Parameters
    ----------
    p : numpy.ndarray
        Points to test, shape (N, 2).
    points : numpy.ndarray
        Points whose convex hull is used, shape (M, 2).

    Returns
    -------
    numpy.ndarray
        Boolean array of length N.
    """
    verts = points[ConvexHull(points).vertices]
    return Delaunay(verts).find_simplex(p) >= 0


def _unit_vectors(radec):
    """Unit vectors on the sphere of (ra, dec) positions [deg], shape (N, 3)."""
    ra = np.radians(radec[:, 0])
    dec = np.radians(radec[:, 1])
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])


def near_points(p, points, radius_arcsec):
    """Test which points have at least one point of another set within a given distance on the sky.

    Used as a survey footprint that follows the real shape of the survey (gaps, jagged edges), unlike a convex hull:
    a position is covered if a survey object lies within `radius_arcsec`. With a dense survey (HST: about 250 galaxies
    per arcmin2), a circle of a few arcsec inside the covered area almost never contains no object.

    Parameters
    ----------
    p : numpy.ndarray
        Positions to test, shape (N, 2), columns ra, dec [deg].
    points : numpy.ndarray
        Positions of the survey objects, shape (M, 2), columns ra, dec [deg].
    radius_arcsec : float
        Distance [arcsec] (exact great-circle distance).

    Returns
    -------
    numpy.ndarray
        Boolean array of length N.
    """
    tree = cKDTree(_unit_vectors(points))
    chord = 2 * np.sin(np.radians(radius_arcsec / 3600) / 2)  # straight-line distance between unit vectors
    dist, _ = tree.query(_unit_vectors(p), k=1, distance_upper_bound=chord)
    return np.isfinite(dist)  # no neighbour within the radius: dist = inf


def in_sky_circle(ra, dec, ra0, dec0, radius):
    """Test which positions lie within a circle on the sky (exact great-circle distance).

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].
    ra0, dec0 : float
        Center of the circle [deg].
    radius : float
        Radius of the circle [deg].

    Returns
    -------
    numpy.ndarray
        Boolean array, True inside the circle (edge included).
    """
    ra, dec = np.deg2rad(np.asarray(ra, float)), np.deg2rad(np.asarray(dec, float))
    r0, d0 = np.deg2rad(ra0), np.deg2rad(dec0)
    cos_sep = np.sin(d0) * np.sin(dec) + np.cos(d0) * np.cos(dec) * np.cos(ra - r0)
    return cos_sep >= np.cos(np.deg2rad(radius))


def assemble(case):
    """Build the three catalogs of a case and the metadata computed with them.

    Applies, in this order: the optional `sky_box` and `sky_circle`, `only_hst_field`, the footprint and area computation, and the
    magnitude cut. All returned catalogs have a fresh RangeIndex, so the group indices of later stages are row
    positions in these tables.

    Parameters
    ----------
    case : Case

    Returns
    -------
    lsst : pandas.DataFrame
        LSST catalog, with ``in_<survey>_footprint`` for each survey of `telescopes` and the columns friendly expects
        (``blendedness``). Euclid footprint: convex hull of the Euclid objects. HST footprint: an HST galaxy within
        `hst_footprint_radius` arcsec (convex hull if it is 0).
    trues : dict of pandas.DataFrame
        ``{survey: catalog}`` for the surveys of `case.telescopes` (``'Euclid'``, ``'HST'``), each with
        ``in_lsst_footprint`` and ``mag_i``.
    meta : dict
        ``n_raw`` and ``n_after_cut`` (objects per survey before and after the magnitude cut), ``area_hull_deg2`` and
        ``area_deg2`` (convex-hull and occupied-cell sky area per survey).
    lsst_all : pandas.DataFrame
        ``ra``, ``dec`` of every LSST galaxy of the tracts, before the sky cuts (see `lsst_all_radec`).
    """
    def in_box(df):
        """Keep the rows of a catalog that lie inside the case's `sky_box` (all rows if no box is set)."""
        if not case.sky_box:
            return df
        r0, r1, d0, d1 = case.sky_box
        return df[(df['ra'] >= r0) & (df['ra'] <= r1) & (df['dec'] >= d0) & (df['dec'] <= d1)].reset_index(drop=True)

    def in_circle(df):
        """Keep the rows of a catalog that lie inside the case's `sky_circle` (all rows if no circle is set)."""
        if not case.sky_circle:
            return df
        ra0, dec0, radius = case.sky_circle
        return df[in_sky_circle(df['ra'].values, df['dec'].values, ra0, dec0, radius)].reset_index(drop=True)

    def crop(df):
        """Apply `sky_box` then `sky_circle`."""
        return in_circle(in_box(df))

    raw = fetch_lsst_raw(case)
    lsst_all = lsst_all_radec(case, raw)  # whole footprint, before the sky cuts (sky maps)
    lsst = crop(build_lsst(case, raw))
    del raw
    trues = {}
    if 'Euclid' in case.telescopes:
        trues['Euclid'] = crop(build_euclid(case, lsst))  # tiles chosen from the (possibly cropped) LSST positions
    if 'HST' in case.telescopes:
        trues['HST'] = crop(build_hst(case))

    if case.only_hst_field and 'Euclid' in trues and 'HST' in trues:
        hst, euclid = trues['HST'], trues['Euclid']
        r0, r1, d0, d1 = hst['ra'].min(), hst['ra'].max(), hst['dec'].min(), hst['dec'].max()
        euclid = euclid[(euclid['ra'] >= r0) & (euclid['ra'] <= r1) & (euclid['dec'] >= d0) & (euclid['dec'] <= d1)]
        trues['Euclid'] = euclid.reset_index(drop=True)

    meta = {'n_raw': {'LSST': len(lsst), **{t: len(d) for t, d in trues.items()}}, 'sky_circle': list(case.sky_circle),
            'telescopes': list(trues)}

    # footprints, computed before the magnitude cut, as in the notebook
    ll = _radec(lsst)
    pts = {t: _radec(d) for t, d in trues.items()}
    meta['area_hull_deg2'] = {'LSST': ConvexHull(ll).volume, **{t: ConvexHull(p).volume for t, p in pts.items()}}
    # sky area actually covered (holes excluded): used to normalise the surface-density plots
    meta['area_deg2'] = {'LSST': occupied_area(lsst['ra'], lsst['dec']),
                         **{t: occupied_area(d['ra'], d['dec']) for t, d in trues.items()}}
    for t, d in trues.items():
        d['in_lsst_footprint'] = in_hull(pts[t], ll)
        if t == 'HST' and case.hst_footprint_radius > 0:
            # HST fields are jagged: a convex hull would include sky with no HST data
            lsst[f'in_{t.lower()}_footprint'] = near_points(ll, pts[t], case.hst_footprint_radius)
        else:
            lsst[f'in_{t.lower()}_footprint'] = in_hull(ll, pts[t])  # convex hull of the survey (may include gaps)
    meta['hst_footprint_radius_arcsec'] = case.hst_footprint_radius

    # friendly wants these column names
    lsst['vis_blendedness'] = lsst['i_blendedness']
    lsst['blendedness'] = lsst['i_blendedness']
    own_mag = {'Euclid': 'vis_mag', 'HST': 'F814_MAG'}
    for t, d in trues.items():
        d['mag_i'] = d[own_mag[t]]

    m = case.mag_cut
    lsst = lsst[lsst['i_cModel_mag'] <= m].reset_index(drop=True)
    trues = {t: d[d[own_mag[t]] <= m].reset_index(drop=True) for t, d in trues.items()}
    meta['n_after_cut'] = {'LSST': len(lsst), **{t: len(d) for t, d in trues.items()}}
    return lsst, trues, meta, lsst_all


def save_meta(path, meta):
    """Write the metadata dictionary of `assemble` to a json file.

    Parameters
    ----------
    path : str
        Output file.
    meta : dict
        Json-serialisable metadata.
    """
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


def load_meta(path):
    """Read a metadata json file written by `save_meta`.

    Parameters
    ----------
    path : str
        Input file.

    Returns
    -------
    dict
    """
    with open(path) as f:
        return json.load(f)
