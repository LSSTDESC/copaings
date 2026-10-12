"""Build the LSST / Euclid / HST catalogs (logic lifted from the notebook, stage `prep`)."""
import gc
import json

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from scipy.spatial import ConvexHull, Delaunay

EUCLID_PIXEL_SCALE = 0.1  # arcsec / pix
HST_PIXEL_SCALE = 0.05

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
def fetch_lsst_raw(case):
    """Read the LSST object table of every tract of the case from the butler.

    Only the columns that are needed are read: the DP2 object table has about 1200 columns (about 7 GB per tract as a
    dataframe). If the column selection is not supported the full table is read and then reduced.

    Parameters
    ----------
    case : Case
        Uses `repo`, `collections`, `skymap`, `dataset_type`, `tracts` and `columns`.

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
    butler = Butler(case.repo, collections=case.collections)
    wanted = list(dict.fromkeys(case.columns.values()))
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
def euclid_available_tiles(case):
    """List the Euclid MER tiles that have a catalog file.

    Parameters
    ----------
    case : Case
        Uses `euclid_dir`; files are named ``euclid_q1_mer_final_<healpix>.fits``.

    Returns
    -------
    list of int
        Sorted healpix pixel numbers (nside = 64, NESTED).
    """
    import os
    import re
    pat = re.compile(r'euclid_q1_mer_final_(\d+)\.fits$')
    return sorted(int(m.group(1)) for f in os.listdir(case.euclid_dir) if (m := pat.match(f)))


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
        Healpix pixel numbers (nside = 64, NESTED). ``'auto'``: tiles that contain LSST objects and have a file.
        ``'notebook'``: the six hand-picked ECDFS tiles of `EUCLID_TILE_CENTERS`.

    Raises
    ------
    ValueError
        If `euclid_region` is neither ``'auto'`` nor ``'notebook'``.
    """
    import healpy as hp
    if case.euclid_region == 'notebook':
        return sorted({int(hp.ang2pix(64, np.radians(90.0 - dec), np.radians(ra), nest=True))
                       for ra, dec in EUCLID_TILE_CENTERS})
    if case.euclid_region != 'auto':
        raise ValueError("euclid_region must be 'auto' or 'notebook'")
    lsst_tiles = np.unique(hp.ang2pix(64, lsst['ra'].values, lsst['dec'].values, nest=True, lonlat=True))
    return sorted(set(int(t) for t in lsst_tiles) & set(euclid_available_tiles(case)))


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
    pix = euclid_tiles(case, lsst)
    print(f'Euclid: reading {len(pix)} tiles {pix}')
    parts = []
    for p in pix:
        with fits.open(f'{case.euclid_dir}/euclid_q1_mer_final_{p}.fits') as hd:
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
    """Build the HST catalog from the HLF GOODS-S catalog.

    Parameters
    ----------
    case : Case
        Uses `hst_file`.

    Returns
    -------
    pandas.DataFrame
        Fresh RangeIndex, native byte order, stars removed (``class_star > 0.6``) and objects without F814W flux
        dropped. Columns: ``ra``, ``dec`` [deg], ``x``, ``y``, ``size_true`` and ``size_minor_true`` [arcsec],
        ``position_angle_true`` [deg], ``ellipticity``, ``e1``, ``e2``, ``F814_FLUX`` and ``F814_FLUXERR`` [nJy],
        ``F814_MAG``, ``F814_MAGERR``, plus the SExtractor columns ``class_star``, ``flux_radius``, ``fwhm_image``.
    """
    with fits.open(case.hst_file) as hd:
        df = native_endian(pd.DataFrame(hd[1].data))
    cols = ['ra', 'dec', 'x', 'y', 'f_f814w', 'e_f814w', 'w_f814w', 'class_star', 'a_image', 'b_image',
            'theta_J2000', 'flux_radius', 'fwhm_image']
    df = df[cols].rename(columns={'f_f814w': 'F814_FLUX', 'e_f814w': 'F814_FLUXERR', 'a_image': 'size_true',
                                  'b_image': 'size_minor_true', 'theta_J2000': 'position_angle_true'})
    df = df.dropna(subset=['F814_FLUX'])
    for c in ('F814_FLUX', 'F814_FLUXERR'):
        df[c] = df[c] * 1e3  # uJy -> nJy
    df = df[df['class_star'] <= 0.6].reset_index(drop=True)  # 0 = galaxy, 1 = star
    df['size_true'] *= HST_PIXEL_SCALE
    df['size_minor_true'] *= HST_PIXEL_SCALE
    df['position_angle_true'] = -df['position_angle_true']

    e = 1 - df['size_minor_true'] / df['size_true']
    theta = np.radians(df['position_angle_true'])
    df['ellipticity'] = e
    df['e1'] = e * np.cos(2 * theta)
    df['e2'] = e * np.sin(2 * theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['F814_MAG'] = nJy2mag(df['F814_FLUX'])
        df['F814_MAGERR'] = mag_err(df['F814_FLUX'], df['F814_FLUXERR'])
    return native_endian(df)


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


def assemble(case):
    """Build the three catalogs of a case and the metadata computed with them.

    Applies, in this order: the optional `sky_box`, `only_hst_field`, the footprint and area computation, and the
    magnitude cut. All returned catalogs have a fresh RangeIndex, so the group indices of later stages are row
    positions in these tables.

    Parameters
    ----------
    case : Case

    Returns
    -------
    lsst, euclid, hst : pandas.DataFrame
        The catalogs. ``lsst`` also has ``in_hst_footprint`` and ``in_euclid_footprint``, ``euclid`` has
        ``in_lsst_footprint`` (convex-hull tests), and the columns friendly expects (``blendedness``, ``mag_i``).
    meta : dict
        ``n_raw`` and ``n_after_cut`` (objects per survey before and after the magnitude cut), ``area_hull_deg2`` and
        ``area_deg2`` (convex-hull and occupied-cell sky area per survey).
    """
    def in_box(df):
        """Keep the rows of a catalog that lie inside the case's `sky_box` (all rows if no box is set)."""
        if not case.sky_box:
            return df
        r0, r1, d0, d1 = case.sky_box
        return df[(df['ra'] >= r0) & (df['ra'] <= r1) & (df['dec'] >= d0) & (df['dec'] <= d1)].reset_index(drop=True)

    lsst = in_box(build_lsst(case, fetch_lsst_raw(case)))
    euclid = in_box(build_euclid(case, lsst))  # tiles chosen from the (possibly cropped) LSST positions
    hst = in_box(build_hst(case))

    if case.only_hst_field:
        r0, r1, d0, d1 = hst['ra'].min(), hst['ra'].max(), hst['dec'].min(), hst['dec'].max()
        euclid = euclid[(euclid['ra'] >= r0) & (euclid['ra'] <= r1) & (euclid['dec'] >= d0) & (euclid['dec'] <= d1)]
        euclid = euclid.reset_index(drop=True)

    meta = {'n_raw': {'LSST': len(lsst), 'Euclid': len(euclid), 'HST': len(hst)}}

    # footprints, computed before the magnitude cut, as in the notebook
    ll, ee, hh = _radec(lsst), _radec(euclid), _radec(hst)
    meta['area_hull_deg2'] = {'LSST': ConvexHull(ll).volume, 'Euclid': ConvexHull(ee).volume, 'HST': ConvexHull(hh).volume}
    # sky area actually covered (holes excluded): used to normalise the surface-density plots
    meta['area_deg2'] = {'LSST': occupied_area(lsst['ra'], lsst['dec']), 'Euclid': occupied_area(euclid['ra'], euclid['dec']),
                         'HST': occupied_area(hst['ra'], hst['dec'])}
    euclid['in_lsst_footprint'] = in_hull(ee, ll)
    lsst['in_hst_footprint'] = in_hull(ll, hh)
    lsst['in_euclid_footprint'] = in_hull(ll, ee)  # convex hull of the Euclid tiles read (may include gaps)

    # friendly wants these column names
    lsst['vis_blendedness'] = lsst['i_blendedness']
    lsst['blendedness'] = lsst['i_blendedness']
    euclid['mag_i'] = euclid['vis_mag']
    hst['mag_i'] = hst['F814_MAG']

    m = case.mag_cut
    lsst = lsst[lsst['i_cModel_mag'] <= m].reset_index(drop=True)
    euclid = euclid[euclid['vis_mag'] <= m].reset_index(drop=True)
    hst = hst[hst['F814_MAG'] <= m].reset_index(drop=True)
    meta['n_after_cut'] = {'LSST': len(lsst), 'Euclid': len(euclid), 'HST': len(hst)}
    return lsst, euclid, hst, meta


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
