"""Ellipse overlap test (EOT): split each FoF group into groups of really overlapping ellipses.

Same result as friendly's ``group2graph``, but it only does what the EOT needs: compute the ellipses, test which
pairs of ellipses intersect (friendly's ``is_overlapping``, all pairs at once) and group the ellipses linked by a
chain of intersections. friendly's graph code also computes overlap areas, purities and entropies, which are not
needed here.

Very large FoF groups (long linking lengths percolate into groups of 1e5+ objects) cannot test all pairs. For groups
with more than ``dense_max`` members, only the pairs whose centres are closer than the sum of their semi-major axes
are tested (found with a KD-tree): other pairs cannot intersect. In those groups an object with an undefined
ellipse (NaN moments) is linked to its neighbours closer than `nan_link_px` pixels; in small groups it is linked to
every member, like in friendly.
"""
import numpy as np
from scipy.spatial import cKDTree

from . import deps

deps.add_friendly_to_path()
import friendly.ellipses as ellipses  # noqa: E402
from friendly.matching import connected_groups  # noqa: E402

FWHM_OVER_SIGMA = 2 * np.sqrt(2 * np.log(2))
DENSE_MAX = 400  # groups up to this size test all pairs
PAIR_CHUNK = 400_000  # pairs tested at once in big groups (limits memory)
SMALL_R_PX = 8.0  # ellipses with a semi-major axis up to this [pixel] are "small" in `candidate_pairs`


class EllipseCache:
    """Ellipses of the LSST catalog, and the columns of the true catalog needed to build its ellipses.

    Built once per job. The LSST ellipses come from the i-band second moments; the ellipses of the true objects
    depend on the group (they are placed relative to an LSST object and broadened by its PSF) and are built by
    `true_params`.

    Parameters
    ----------
    lsst : pandas.DataFrame
        LSST catalog with ``ra``, ``dec``, ``x``, ``y``, ``Ixx_pixel_i``, ``Iyy_pixel_i``, ``Ixy_pixel_i``, ``psf_fwhm_i``.
    true_cat : pandas.DataFrame
        True catalog with ``ra``, ``dec``, ``size_true``, ``size_minor_true``, ``position_angle_true``.
    pixel_scale : float, default 0.2
        LSST pixel scale [arcsec/pixel].
    nan_link_px : float, default 10.0
        In big groups, an object with an undefined ellipse is linked to its neighbours closer than this [pixel].

    Attributes
    ----------
    lparam : numpy.ndarray
        Conic parameters (A, B, C, D, E, F) of every LSST ellipse, shape (N, 6).
    lamax : numpy.ndarray
        Semi-major axis of every LSST ellipse [pixel] (NaN where undefined).
    abs_frame : tuple of float or None
        ``(ra0, dec0)``, centre of the tangent plane in which the LSST ``x``, ``y`` were computed by `prep`, or None if
        the ``x``, ``y`` are not in one common plane.
    """

    def __init__(self, lsst, true_cat, pixel_scale=0.2, nan_link_px=10.0):
        """Compute the LSST ellipses and keep the true-catalog columns (see the class docstring)."""
        self.ps = pixel_scale
        self.nan_link_px = nan_link_px

        # LSST catalog
        self.lx, self.ly = lsst['x'].to_numpy(), lsst['y'].to_numpy()
        self.lra, self.ldec = lsst['ra'].to_numpy(), lsst['dec'].to_numpy()
        self.fwhm = lsst['psf_fwhm_i'].to_numpy()
        with np.errstate(invalid='ignore', divide='ignore'):
            a, b, theta = ellipses.moments2ab(lsst['Ixx_pixel_i'].to_numpy(), lsst['Iyy_pixel_i'].to_numpy(),
                                              lsst['Ixy_pixel_i'].to_numpy())
        self.lparam = np.column_stack(ellipses.ab2AB(self.lx, self.ly, a, b, theta))
        self.lamax = np.where(np.isfinite(a) & np.isfinite(b), np.maximum(a, b), np.nan)

        # `prep` computes x = (ra0 - ra) cos(dec) / pixel_scale and y = (dec - dec0) / pixel_scale. Invert this
        # for every object: if they all give the same (ra0, dec0), the x, y are in one common plane and the true
        # objects of big groups can be placed in it directly.
        with np.errstate(invalid='ignore', divide='ignore'):
            ra0 = self.lra + self.lx * pixel_scale / (3600 * np.cos(np.radians(self.ldec)))
            dec0 = self.ldec - self.ly * pixel_scale / 3600
        self.abs_frame = None
        if np.isfinite(ra0).sum() > 100 and np.nanstd(ra0) < 1e-6 and np.nanstd(dec0) < 1e-6:
            self.abs_frame = (float(np.nanmedian(ra0)), float(np.nanmedian(dec0)))

        # true catalog
        self.tra, self.tdec = true_cat['ra'].to_numpy(), true_cat['dec'].to_numpy()
        self.tsize, self.tminor = true_cat['size_true'].to_numpy(), true_cat['size_minor_true'].to_numpy()
        self.tpa = true_cat['position_angle_true'].to_numpy()

    def true_params(self, true_rows, lsst_rows, big=False):
        """Ellipses of the true objects of one group, in the LSST pixel frame.

        Same as friendly's ``deg2pix`` followed by ``ab2AB``: the axes are broadened by the PSF and the position angle
        is converted to the pixel frame.

        * Normal group: the objects are placed relative to the first LSST object of the group and use its PSF.
        * Big group (spanning arcminutes): they are placed in the common LSST frame and use the PSF of their
          nearest LSST object of the group.

        Parameters
        ----------
        true_rows : sequence of int
            Rows of the true objects in the true catalog.
        lsst_rows : sequence of int
            Rows of the LSST objects of the group.
        big : bool, default False
            Use the big-group placement.

        Returns
        -------
        params : numpy.ndarray
            Conic parameters (A, B, C, D, E, F), shape (len(true_rows), 6).
        x, y : numpy.ndarray
            Centres [pixel].
        amax : numpy.ndarray
            Semi-major axes [pixel].
        """
        t = np.asarray(true_rows, dtype=int)
        ra, dec = self.tra[t], self.tdec[t]
        ref = lsst_rows[0]

        if big:
            fwhm = self._nearest_lsst_fwhm(ra, dec, np.asarray(lsst_rows, dtype=int))
        else:
            fwhm = self.fwhm[ref]
        psf = np.sqrt(2) * fwhm / FWHM_OVER_SIGMA / 3600  # deg

        a = np.sqrt((self.tsize[t] / 3600) ** 2 + psf ** 2) * 3600 / self.ps
        b = np.sqrt((self.tminor[t] / 3600) ** 2 + psf ** 2) * 3600 / self.ps
        theta = 270 - self.tpa[t]
        theta = np.where(theta > 180, theta - 180, theta)

        if big and self.abs_frame is not None:
            ra0, dec0 = self.abs_frame
            x = (ra0 - ra) * np.cos(np.radians(dec)) * 3600 / self.ps
            y = (dec - dec0) * 3600 / self.ps
        else:
            dx = (self.lra[ref] - ra) * np.cos(np.radians((dec + self.ldec[ref]) / 2)) * 3600 / self.ps
            dy = (self.ldec[ref] - dec) * 3600 / self.ps
            x, y = self.lx[ref] + dx, self.ly[ref] - dy

        params = np.column_stack(ellipses.ab2AB(x, y, a, b, theta))
        return params, x, y, np.maximum(a, b)

    def _nearest_lsst_fwhm(self, ra, dec, lsst_rows):
        """PSF FWHM of the nearest LSST object (among `lsst_rows` with a valid PSF) of each position."""
        ok = lsst_rows[np.isfinite(self.fwhm[lsst_rows])]
        if len(ok) == 0:
            return np.full(len(ra), np.nan)
        tree = cKDTree(np.column_stack([self.lra[ok] * np.cos(np.radians(self.ldec[ok])), self.ldec[ok]]))
        _, nearest = tree.query(np.column_stack([ra * np.cos(np.radians(dec)), dec]))
        return self.fwhm[ok][nearest]


def candidate_pairs(x, y, r):
    """Pairs of ellipses that could intersect: centres closer than the sum of their semi-major axes.

    A KD-tree finds the neighbours. One search with the largest possible distance (2 * max(r)) would return far
    too many pairs when a few ellipses are huge, so the ellipses are split into "small" (r <= SMALL_R_PX) and
    "large", and each kind of pair is searched with its own radius.

    Parameters
    ----------
    x, y : numpy.ndarray
        Centres [pixel].
    r : numpy.ndarray
        Semi-major axes [pixel].

    Returns
    -------
    first, second : numpy.ndarray of int
        Indices of the candidate pairs, each pair once, ``first != second``.
    """
    xy = np.column_stack([x, y])
    small = np.flatnonzero(r <= SMALL_R_PX)
    large = np.flatnonzero(r > SMALL_R_PX)
    first, second = [], []

    # small - small
    if len(small) > 1:
        pairs = cKDTree(xy[small]).query_pairs(2 * r[small].max(), output_type='ndarray')
        first.append(small[pairs[:, 0]])
        second.append(small[pairs[:, 1]])

    # large - small: around each large ellipse, the small ones closer than its axis + SMALL_R_PX
    if len(large) > 0 and len(small) > 0:
        tree = cKDTree(xy[small])
        neighbours = tree.query_ball_point(xy[large], r[large] + SMALL_R_PX)
        for k, near in zip(large, neighbours):
            first.append(np.full(len(near), k))
            second.append(small[np.asarray(near, dtype=int)])

    # large - large
    if len(large) > 1:
        pairs = cKDTree(xy[large]).query_pairs(2 * r[large].max(), output_type='ndarray')
        first.append(large[pairs[:, 0]])
        second.append(large[pairs[:, 1]])

    if not first:
        return np.zeros(0, int), np.zeros(0, int)
    first = np.concatenate(first).astype(int)
    second = np.concatenate(second).astype(int)

    close = np.hypot(x[first] - x[second], y[first] - y[second]) <= r[first] + r[second]
    return first[close], second[close]


def split_group(cache, grp, dense_max=DENSE_MAX):
    """Split a FoF group into groups of overlapping ellipses.

    The ellipses are the nodes of a graph, with an edge between two ellipses that intersect; the result is the list
    of connected parts of this graph. Groups up to `dense_max` members test all pairs and, like friendly's
    ``group2graph``, drop the true objects whose row number equals the row number of an LSST object of the group.
    Bigger groups test only the pairs of `candidate_pairs` and keep all objects.

    Parameters
    ----------
    cache : EllipseCache
    grp : sequence
        ``[true_rows, lsst_rows]`` (rows in the true and LSST catalogs).
    dense_max : int, default DENSE_MAX
        Group size above which only the candidate pairs are tested (0 = always).

    Returns
    -------
    list
        Groups ``[true_rows, lsst_rows]``, ordered by their first member. A group can lack a true or an LSST
        object; callers drop those.
    """
    true_rows = list(grp[0])
    lsst_rows = list(grp[1])
    big = len(true_rows) + len(lsst_rows) > dense_max

    if not big:
        lsst_set = set(lsst_rows)
        true_rows = [t for t in true_rows if t not in lsst_set]  # friendly's "duplicate id" rule
    if not true_rows:
        return [[[], lsst_rows]] if lsst_rows else []

    # graph nodes: the true objects first, then the LSST objects
    n_true = len(true_rows)
    n_nodes = n_true + len(lsst_rows)
    true_params, tx, ty, t_axis = cache.true_params(true_rows, lsst_rows, big=big)
    params = np.vstack([true_params, cache.lparam[lsst_rows]])

    with np.errstate(invalid='ignore', over='ignore'):
        if not big:
            first, second = np.triu_indices(n_nodes, k=1)  # every pair of nodes
            overlap = ellipses.is_overlapping(params[first], params[second])
        else:
            x = np.concatenate([tx, cache.lx[lsst_rows]])
            y = np.concatenate([ty, cache.ly[lsst_rows]])
            axis = np.concatenate([t_axis, cache.lamax[lsst_rows]])
            undefined = ~np.isfinite(axis) | ~np.isfinite(params).all(axis=1)
            axis = np.where(undefined, cache.nan_link_px, axis)

            first, second = candidate_pairs(x, y, axis)
            overlap = np.zeros(len(first), bool)
            for start in range(0, len(first), PAIR_CHUNK):
                chunk = slice(start, start + PAIR_CHUNK)
                overlap[chunk] = ellipses.is_overlapping(params[first[chunk]], params[second[chunk]])
            overlap |= undefined[first] | undefined[second]  # undefined ellipse: linked to its close neighbours

    group_of_node = connected_groups(n_nodes, first[overlap], second[overlap])

    groups = {}
    for node, g in enumerate(group_of_node):
        members = groups.setdefault(g, [[], []])
        if node < n_true:
            members[0].append(true_rows[node])
        else:
            members[1].append(lsst_rows[node - n_true])
    return list(groups.values())
