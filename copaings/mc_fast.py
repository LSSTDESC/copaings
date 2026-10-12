"""Drop-in replacement for friendly.ellipses.overlap_area_MC giving bit-identical results, much faster.

The original evaluates both ellipse equations on a fixed 1000x1000 lattice covering +-100 px around the pair, although
the ellipses only span a few pixels. Points outside the ellipses' bounding boxes can never count, so we evaluate the
same lattice points, restricted to the bounding box of the pair. Counts, and hence areas, are identical.
Falls back to the original for degenerate / non-finite parameters.
"""
import numpy as np

import friendly.ellipses as ellipses

_original = ellipses.overlap_area_MC


def _bbox(p):
    """Bounding box of an ellipse given by its conic parameters.

    Parameters
    ----------
    p : sequence of float
        Parameters (A, B, C, D, E, F) of the interior ``A x^2 + B y^2 + 2C xy + 2D x + 2E y + F < 0``.

    Returns
    -------
    tuple of float or None
        ``(xmin, xmax, ymin, ymax)``, or None if the parameters are non-finite or do not describe an ellipse.
    """
    A, B, C, D, E, F = (float(v) for v in p)
    det = A * B - C * C
    if not (np.isfinite([A, B, C, D, E, F]).all() and det > 0 and A > 0):
        return None
    inv_xx, inv_yy = B / det, A / det
    cx = -(inv_xx * D - C / det * E)
    cy = -(-C / det * D + inv_yy * E)
    k = -(F + D * cx + E * cy)  # (p-c)^T M (p-c) < k
    if not k > 0:
        return None
    hx, hy = np.sqrt(k * inv_xx), np.sqrt(k * inv_yy)
    return cx - hx, cx + hx, cy - hy, cy + hy


def overlap_area_MC(param1, param2, xlim=[0, 1], ylim=[0, 1], Nx=1000, Ny=1000):
    """Total and overlap area of two ellipses, by Monte Carlo counting on a lattice.

    Drop-in replacement for ``friendly.ellipses.overlap_area_MC`` with identical results: the same lattice points are
    tested, but only inside the bounding box of the two ellipses, since no point outside can count. Falls back to
    friendly's function for degenerate parameters.

    Parameters
    ----------
    param1, param2 : sequence of float
        Conic parameters (A, B, C, D, E, F) of the two ellipses.
    xlim, ylim : list of float
        Limits of the lattice (default ``[0, 1]``).
    Nx, Ny : int, default 1000
        Number of lattice points along x and y.

    Returns
    -------
    total_area : float
        Area of the union of the two ellipses.
    overlap_area : float
        Area of their intersection. Both are in the squared units of `xlim` and `ylim`.
    """
    b1, b2 = _bbox(param1), _bbox(param2)
    if b1 is None or b2 is None:
        return _original(param1, param2, xlim, ylim, Nx, Ny)
    x = np.linspace(xlim[0], xlim[1], Nx)
    y = np.linspace(ylim[0], ylim[1], Ny)
    dx, dy = (xlim[1] - xlim[0]) / (Nx - 1), (ylim[1] - ylim[0]) / (Ny - 1)
    xmin, xmax = min(b1[0], b2[0]) - 2 * dx, max(b1[1], b2[1]) + 2 * dx  # 2-cell safety margin
    ymin, ymax = min(b1[2], b2[2]) - 2 * dy, max(b1[3], b2[3]) + 2 * dy
    i0, i1 = np.searchsorted(x, xmin), np.searchsorted(x, xmax)
    j0, j1 = np.searchsorted(y, ymin), np.searchsorted(y, ymax)

    N_total = Nx * Ny
    A_total = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0])
    if i1 <= i0 or j1 <= j0:  # bounding boxes miss the lattice entirely
        return 0.0, 0.0
    xg, yg = np.meshgrid(x[i0:i1], y[j0:j1])
    Z1 = ellipses.ellipse_equation(xg, yg, param1) < 0
    Z2 = ellipses.ellipse_equation(xg, yg, param2) < 0
    N_1, N_2, N_overlap = np.sum(Z1), np.sum(Z2), np.sum(Z1 & Z2)
    return (N_1 + N_2 - N_overlap) * A_total / N_total, N_overlap * A_total / N_total


def install():
    """Make friendly's graph code use `overlap_area_MC` of this module.

    friendly looks the function up on its ``ellipses`` module at call time, so replacing the attribute is enough.
    """
    ellipses.overlap_area_MC = overlap_area_MC


def uninstall():
    """Restore friendly's original ``overlap_area_MC``."""
    ellipses.overlap_area_MC = _original
