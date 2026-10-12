"""Fast ellipse overlap test (EOT): same result as friendly's graph-based test, without the unused work.

friendly's group2graph builds a full graph per group (Monte Carlo overlap area on a 1000x1000 grid for every edge,
purity, magnitudes, blending entropy) although the EOT only needs the connected components of the "ellipses
intersect" graph. Here we do only that: ellipse parameters in numpy, batched intersection test, scipy components.

Groups with more than DENSE_MAX members (large FoF linking lengths percolate into groups of 1e5+ objects) do not test
all pairs: candidate pairs come from a KD-tree (two ellipses can only intersect if their centres are closer than the sum
of their semi-major axes) and only those are tested exactly. Objects whose ellipse is undefined (NaN moments) count as
overlapping with everything in a small group, like in friendly; in a big group they link to neighbours within
`nan_link_px` pixels instead.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import friendly.ellipses as ellipses  # deps.py puts friendly on sys.path

FWHM2SIGMA = 2 * np.sqrt(2 * np.log(2))
DENSE_MAX = 400  # groups up to this size test all pairs
PAIR_CHUNK = 400_000
SMALL_R_PX = 8.0  # ellipses with semi-major axis below this (pixels) use the cheap small-small search


class EllipseCache:
    """Ellipse parameters of the LSST catalog and the columns of the true catalog needed to place its ellipses.

    Computed once per job. The LSST ellipses come from the i-band second moments; the true-catalog ellipses are built
    per group by `true_params`.

    Parameters
    ----------
    lsst : pandas.DataFrame
        LSST catalog with ``ra``, ``dec``, ``x``, ``y``, ``Ixx_pixel_i``, ``Iyy_pixel_i``, ``Ixy_pixel_i``, ``psf_fwhm_i``.
    true_cat : pandas.DataFrame
        True catalog with ``ra``, ``dec``, ``size_true``, ``size_minor_true``, ``position_angle_true``.
    pixel_scale : float, default 0.2
        LSST pixel scale [arcsec/pixel] (friendly hardcodes 0.2).
    nan_link_px : float, default 10.0
        In groups handled by the KD-tree path, an object with an undefined ellipse is linked to neighbours closer than
        this [pixel].

    Attributes
    ----------
    lparam : numpy.ndarray
        Conic parameters (A, B, C, D, E, F) of every LSST ellipse, shape (N, 6).
    lamax : numpy.ndarray
        Semi-major axis of every LSST ellipse [pixel] (NaN where undefined).
    abs_frame : tuple of float or None
        Centre (ra0, dec0) of the common tangent plane of the LSST ``x``, ``y``, if every object agrees on it.
    """

    def __init__(self, lsst, true_cat, pixel_scale=0.2, nan_link_px=10.0):
        """Compute the LSST ellipse parameters and keep the true-catalog columns (see the class docstring)."""
        self.ps = pixel_scale  # friendly.ellipses.deg2pix hardcodes 0.2
        self.nan_link_px = nan_link_px
        self.lx, self.ly = lsst['x'].values, lsst['y'].values
        self.lra, self.ldec = lsst['ra'].values, lsst['dec'].values
        self.fwhm = lsst['psf_fwhm_i'].values
        a, b, th = ellipses.moments2ab(lsst['Ixx_pixel_i'].values, lsst['Iyy_pixel_i'].values, lsst['Ixy_pixel_i'].values)
        self.lparam = np.column_stack(ellipses.ab2AB(self.lx, self.ly, a, b, th))  # (N, 6)
        self.lamax = np.where(np.isfinite(a) & np.isfinite(b), np.maximum(a, b), np.nan)  # semi-major axis, pixels
        # Common tangent-plane frame of the LSST x, y (built from ra/dec in prep): recover its centre so that true objects
        # can be placed in the same frame, whatever the size of the group (valid only if every object agrees on it).
        cd = np.cos(np.radians(self.ldec))
        with np.errstate(invalid='ignore', divide='ignore'):
            ra0 = self.lra + self.lx * pixel_scale / (3600 * cd)
            dec0 = self.ldec - self.ly * pixel_scale / 3600
        self.abs_frame = None
        if np.isfinite(ra0).sum() > 100 and np.nanstd(ra0) < 1e-6 and np.nanstd(dec0) < 1e-6:
            self.abs_frame = (float(np.nanmedian(ra0)), float(np.nanmedian(dec0)))
        self.tra, self.tdec = true_cat['ra'].values, true_cat['dec'].values
        self.tsize, self.tminor = true_cat['size_true'].values, true_cat['size_minor_true'].values
        self.tpa = true_cat['position_angle_true'].values

    def true_params(self, idx_t, idx_o, big=False):
        """Ellipses of the true objects of one group.

        Same as friendly's ``deg2pix`` followed by ``ab2AB``: the axes are broadened by the PSF in quadrature and the
        position angle is converted to the pixel frame. In a normal group the objects are placed relative to the first LSST
        object and use its PSF. In a big group (spanning arcminutes) they are placed in the common LSST frame and use the
        PSF of the nearest LSST member.

        Parameters
        ----------
        idx_t : array_like of int
            Rows of the true objects in the true catalog.
        idx_o : sequence of int
            Rows of the LSST objects of the group; the first one is the reference.
        big : bool, default False
            Use the big-group placement.

        Returns
        -------
        params : numpy.ndarray
            Conic parameters (A, B, C, D, E, F), shape (len(idx_t), 6).
        x, y : numpy.ndarray
            Centres [pixel].
        amax : numpy.ndarray
            Semi-major axes [pixel].
        """
        idx_t = np.asarray(idx_t)
        o = idx_o[0]
        if big:
            io = np.asarray(idx_o, dtype=int)
            ok = np.isfinite(self.fwhm[io])
            cdec = np.cos(np.radians(self.tdec[idx_t]))
            if ok.any():
                tree = cKDTree(np.column_stack([self.lra[io[ok]] * np.cos(np.radians(self.ldec[io[ok]])), self.ldec[io[ok]]]))
                _, nn = tree.query(np.column_stack([self.tra[idx_t] * cdec, self.tdec[idx_t]]))
                fwhm = self.fwhm[io[ok]][nn]
            else:
                fwhm = np.full(len(idx_t), np.nan)
        else:
            fwhm = self.fwhm[o]
        psf = np.sqrt(2) * fwhm / FWHM2SIGMA / 3600  # deg
        a = np.sqrt((self.tsize[idx_t] / 3600) ** 2 + psf ** 2) * 3600 / self.ps
        b = np.sqrt((self.tminor[idx_t] / 3600) ** 2 + psf ** 2) * 3600 / self.ps
        theta = 270 - self.tpa[idx_t]
        theta = np.where(theta > 180, theta - 180, theta)
        if big and self.abs_frame is not None:
            ra0, dec0 = self.abs_frame
            x = (ra0 - self.tra[idx_t]) * np.cos(np.radians(self.tdec[idx_t])) * 3600 / self.ps
            y = (self.tdec[idx_t] - dec0) * 3600 / self.ps
        else:
            dx = (self.lra[o] - self.tra[idx_t]) * np.cos(np.radians((self.tdec[idx_t] + self.ldec[o]) / 2)) * 3600 / self.ps
            dy = (self.ldec[o] - self.tdec[idx_t]) * 3600 / self.ps
            x, y = self.lx[o] + dx, self.ly[o] - dy
        return np.column_stack(ellipses.ab2AB(x, y, a, b, theta)), x, y, np.maximum(a, b)


def _det(r0, r1, r2):
    """Batched 3x3 determinant.

    Parameters
    ----------
    r0, r1, r2 : tuple of numpy.ndarray
        The three rows; each is a tuple of 3 arrays of shape (n,).

    Returns
    -------
    numpy.ndarray
        Determinants, shape (n,).
    """
    m = np.stack([np.stack(r0, -1), np.stack(r1, -1), np.stack(r2, -1)], -2)
    return np.linalg.det(m)


def is_overlapping_batch(p1, p2):
    """Test whether pairs of ellipses intersect.

    Vectorised version of ``friendly.ellipses.is_overlapping`` (the algebraic test on the conic matrices).

    Parameters
    ----------
    p1, p2 : numpy.ndarray
        Conic parameters (A, B, C, D, E, F) of the first and second ellipse of each pair, shape (n, 6).

    Returns
    -------
    numpy.ndarray
        Boolean array of length n, True where the two ellipses intersect.
    """
    A1, B1, C1, D1, E1, F1 = p1.T
    A2, B2, C2, D2, E2, F2 = p2.T
    l0 = _det((A2, C2, D2), (C2, B2, E2), (D2, E2, F2))
    l1 = (_det((A1, C2, D2), (C1, B2, E2), (D1, E2, F2))
          + _det((A2, C1, D2), (C2, B1, E2), (D2, E1, F2))
          + _det((A2, C2, D1), (C2, B2, E1), (D2, E2, F1))) / 3
    l2 = (_det((A2, C1, D1), (C2, B1, E1), (D2, E1, F1))
          + _det((A1, C2, D1), (C1, B2, E1), (D1, E2, F1))
          + _det((A1, C1, D2), (C1, B1, E2), (D1, E1, F2))) / 3
    l3 = _det((A1, C1, D1), (C1, B1, E1), (D1, E1, F1))

    d1 = l3 * l1 - l2 * l2
    d2 = l3 * l0 - l1 * l2
    d3 = l2 * l0 - l1 * l1
    disc = (2 * d1) * (2 * d3) - d2 * d2
    return ~((disc >= 0) & ((l1 > 0) | (l2 > 0)))


def _candidate_pairs(x, y, r):
    """Pairs of ellipses that could intersect, from a KD-tree search.

    Two ellipses can only intersect if their centres are closer than the sum of their semi-major axes. Small ellipses
    (axis up to `SMALL_R_PX`) are searched among themselves with a single radius; each large ellipse is searched against the
    small ones and against the other large ones.

    Parameters
    ----------
    x, y : numpy.ndarray
        Centres [pixel].
    r : numpy.ndarray
        Semi-major axes [pixel].

    Returns
    -------
    i, j : numpy.ndarray of int
        Indices of the candidate pairs, ``i != j``, each pair once.
    """
    xy = np.column_stack([x, y])
    small = r <= SMALL_R_PX
    out_i, out_j = [], []
    keep_pairs = lambda a, b: np.hypot(x[a] - x[b], y[a] - y[b]) <= r[a] + r[b]

    sid = np.flatnonzero(small)
    if len(sid) > 1:  # small - small
        p = cKDTree(xy[sid]).query_pairs(2 * r[sid].max(), output_type='ndarray')
        if len(p):
            a, b = sid[p[:, 0]], sid[p[:, 1]]
            k = keep_pairs(a, b)
            out_i.append(a[k]); out_j.append(b[k])
    lid = np.flatnonzero(~small)
    if len(lid):
        if len(sid):  # large - small
            ts = cKDTree(xy[sid])
            lists = ts.query_ball_point(xy[lid], r[lid] + SMALL_R_PX)
            for k, nb in zip(lid, lists):
                if len(nb):
                    b = sid[np.asarray(nb, dtype=int)]
                    kk = np.hypot(x[k] - x[b], y[k] - y[b]) <= r[k] + r[b]
                    out_i.append(np.full(kk.sum(), k)); out_j.append(b[kk])
        if len(lid) > 1:  # large - large
            p = cKDTree(xy[lid]).query_pairs(2 * r[lid].max(), output_type='ndarray')
            if len(p):
                a, b = lid[p[:, 0]], lid[p[:, 1]]
                k = keep_pairs(a, b)
                out_i.append(a[k]); out_j.append(b[k])
    if not out_i:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(out_i).astype(int), np.concatenate(out_j).astype(int)


def split_group(cache, grp, dense_max=DENSE_MAX):
    """Split a FoF group into groups of overlapping ellipses.

    Returns the connected components of the graph whose edges join intersecting ellipses. Groups up to `dense_max`
    members test all pairs and, like friendly's ``group2graph``, drop true objects whose row number equals the row
    number of an LSST object of the group. Bigger groups use `_candidate_pairs` and keep all objects.

    Parameters
    ----------
    cache : EllipseCache
    grp : sequence
        ``[true_indices, lsst_indices]`` (rows in the true and LSST catalogs).
    dense_max : int, default DENSE_MAX
        Group size above which the KD-tree path is used (0 = always).

    Returns
    -------
    list
        Components ``[true_indices, lsst_indices]``, ordered by first member. A component can lack a true or an LSST
        object; callers drop those.
    """
    idx_o = list(grp[1])
    big = len(grp[0]) + len(idx_o) > dense_max  # KD-tree path
    if big:
        idx_t = list(grp[0])  # the "same index = duplicate" rule below is meaningless in groups holding most of a catalog
    else:
        obs_set = set(idx_o)
        idx_t = [t for t in grp[0] if t not in obs_set]  # friendly drops "duplicate ids" (0 cases in small groups)
    nt, no = len(idx_t), len(idx_o)
    n = nt + no
    if nt == 0:
        return [[[], idx_o]] if no else []
    tparams, tx, ty, tr = cache.true_params(np.asarray(idx_t), idx_o, big=big)
    params = np.vstack([tparams, cache.lparam[idx_o]])
    if not big:
        ii, jj = np.triu_indices(n, k=1)
        hit = is_overlapping_batch(params[ii], params[jj]) if len(ii) else np.zeros(0, bool)
        ii, jj = ii[hit], jj[hit]
    else:
        x = np.concatenate([tx, cache.lx[idx_o]])
        y = np.concatenate([ty, cache.ly[idx_o]])
        r = np.concatenate([tr, cache.lamax[idx_o]])
        bad = ~np.isfinite(r) | ~np.isfinite(params).all(axis=1)
        r = np.where(bad, cache.nan_link_px, r)
        ci, cj = _candidate_pairs(x, y, r)
        hits = np.zeros(len(ci), bool)
        for s0 in range(0, len(ci), PAIR_CHUNK):
            sl = slice(s0, s0 + PAIR_CHUNK)
            hits[sl] = is_overlapping_batch(params[ci[sl]], params[cj[sl]])
        hits |= bad[ci] | bad[cj]  # undefined ellipse: linked to neighbours within nan_link_px
        ii, jj = ci[hits], cj[hits]
    g = coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    out = {}
    for k, lab in enumerate(labels):
        comp = out.setdefault(lab, [[], []])
        (comp[0] if k < nt else comp[1]).append(idx_t[k] if k < nt else idx_o[k - nt])
    return list(out.values())
