"""Figures from the saved stage outputs. No heavy computation here: safe to rerun from a notebook or a login node."""

import json
import os
import pickle

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .catalogs import euclid_release, in_sky_circle
from .config import TELESCOPES, initial
from .style import BLUE_CMAP, styled, top_colorbar

COLORS = {"LSST": "#56B4E9", "Euclid": "#E69F00", "HST": "#009E73"}  # DESC palette
IMAG_BINS = np.arange(18.25, 31.1, 0.5)
HSC_BLENDEDNESS_CUT = 10 ** (-0.375)


# every figure is written in each format, in a sub-folder of that name
FIG_FORMATS = ("pdf", "png")


def save_figure(fig, out_dir, name, dpi=300):
    """Save a figure as ``<out_dir>/pdf/<stem>.pdf`` and ``<out_dir>/png/<stem>.png``, then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    out_dir : str
        Plot directory; the format sub-folders are created.
    name : str
        File name; its extension, if any, is ignored (``'detect_counts.pdf'`` and ``'detect_counts'`` are the same).
    dpi : int, default 300
        Resolution of the png (and of rasterized parts of the pdf).

    Returns
    -------
    list of str
        The written paths.
    """
    image_extensions = [".pdf", ".png", ".svg", ".eps", ".jpg", ".jpeg"]
    stem, ext = os.path.splitext(name)
    if ext.lower() not in image_extensions:  # e.g. '_imag_lt23.5' is not an extension
        stem = name
    paths = []
    for fmt in FIG_FORMATS:
        folder = os.path.join(out_dir, fmt)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{stem}.{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        paths.append(path)
    plt.close(fig)
    print("saved", ", ".join(paths))
    return paths


def _blendedness_label(col):
    """Axis label of a blendedness column, band in italic: ``'i_blendedness'`` -> ``'LSST $i$-blendedness'``."""
    band, _, rest = col.partition("_")
    if rest == "blendedness" and len(band) == 1:
        return rf"LSST ${band}$-blendedness"
    return col


# band of each true catalog (filter names: upright) and its magnitude column
TRUE_BAND = {"Euclid": "VIS", "HST": "F814W"}
TRUE_MAG = {"Euclid": "vis_mag", "HST": "F814_MAG"}


def _lsst_label(case):
    """Legend name of the LSST catalog: 'LSST DP2' / 'LSST DP1' when the butler collection or repo says so."""
    text = f"{case.collections} {case.repo}".upper()
    for release in ("DP2", "DP1"):
        if release in text:
            return f"LSST {release}"
    return "LSST"


def _mag_hist_panel(res, ax, surface_density=False):
    """Magnitude distributions of LSST and of the true catalogs converted to LSST i (saved linear relations).

    Normalised densities, or objects per arcmin2 per bin (occupied sky areas) when `surface_density` is set.
    """
    area = res.meta["area_deg2"]

    def hist(values, color, label, area_deg2):
        if surface_density:
            area_arcmin2 = area_deg2 * 3600
            weights = np.ones(len(values)) / area_arcmin2
        else:
            weights = None
        ax.hist(
            values,
            bins=IMAG_BINS,
            weights=weights,
            density=not surface_density,
            color=color,
            lw=2,
            label=label,
            histtype="step",
        )

    hist(
        res.lsst["i_cModel_mag"].dropna().values,
        COLORS["LSST"],
        _lsst_label(res.case),
        area["LSST"],
    )
    for tel in res.tels:
        slope, intercept = res.params[f"{tel.lower()}_to_i"]
        true_mag = res.trues[tel][TRUE_MAG[tel]].dropna().values
        i_mag = slope * true_mag + intercept
        hist(
            i_mag,
            COLORS[tel],
            rf"{_true_label(res.case, tel)} {TRUE_BAND[tel]} to $i$",
            area[tel],
        )
    if surface_density:
        ax.set_ylabel(r"Surface density [arcmin$^{-2}$]")
    else:
        ax.set_ylabel(r"Normalised count")  # [mag$^{-1}$]")
    ax.grid(alpha=0.3)


def _true_label(case, tel):
    """Legend name of a true catalog: 'Euclid Q1' / 'Euclid DR1' from the Euclid file names, 'HST' as is."""
    if tel == "Euclid":
        release = euclid_release(case)
        if release:
            return f"Euclid {release}"
        return "Euclid"
    return tel


def _fraction_ylim(ax):
    """Y axis of a fraction plot: from 0 to the top of the data, never above 1."""
    top = ax.get_ylim()[1]
    if top > 1.0:
        top = 1.0
    ax.set_ylim(0, top)


def _no_top_tick_label(ax):
    """Keep only the y ticks clearly below the top of the axis.

    In a two-panel figure with no gap, a tick label at the top edge of the bottom panel sits on the line between the
    panels and reads as a label of the top panel.
    """
    bottom, top = ax.get_ylim()
    ticks = []
    for t in mpl.ticker.MaxNLocator().tick_values(bottom, top):
        if bottom <= t < top - 0.03 * (top - bottom):
            ticks.append(t)
    ax.set_yticks(ticks)
    ax.set_ylim(bottom, top)


def compute_error_on_fraction(x, y):
    """Statistical error on a ratio of counts.

    Parameters
    ----------
    x, y : array_like
        Numerator and denominator counts (``x <= y``).

    Returns
    -------
    numpy.ndarray
        ``x / y * sqrt(1/x + 1/y)``, and 0 where `x` or `y` is 0.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = x / y * np.sqrt(1 / x + 1 / y)
    both_positive = (x > 0) & (y > 0)
    err = np.where(both_positive, err, 0.0)
    return err


class Results:
    """Everything the plots need, loaded from the output directory of a case.

    Parameters
    ----------
    case : Case
        A case whose stages have all run (needs ``lsst_sb.parquet``, the catalogs, ``fit_params.json`` and the groups).

    Attributes
    ----------
    case : Case
    lsst : pandas.DataFrame
        ``lsst_sb.parquet``: LSST catalog with the entropy and purity columns.
    meta : dict
        ``catalogs/meta.json`` (object counts and sky areas).
    params : dict
        ``catalogs/fit_params.json``.
    trues : dict of pandas.DataFrame
        True catalogs of the surveys of the run, ``{'Euclid': ..., 'HST': ...}``.
    tels : list of str
        Surveys of the run (`case.telescopes`, in the order Euclid, HST).
    groups : dict of list
        Groups after the ellipse overlap test, per survey.
    fof_groups : dict of list
        FoF groups before the overlap test, per survey.
    plot_dir : str
        ``<out_dir>/plots`` (created).
    """

    def __init__(self, case):
        """Load the outputs of a case (see the class docstring)."""
        self.case = case
        self.lsst = pd.read_parquet(case.path("lsst_sb.parquet"))
        with open(case.path("catalogs", "meta.json")) as f:
            self.meta = json.load(f)
        with open(case.path("catalogs", "fit_params.json")) as f:
            self.params = json.load(f)

        # surveys of this run, in the fixed order of TELESCOPES
        self.tels = []
        for t in TELESCOPES:
            if t in case.telescopes:
                self.tels.append(t)

        self.trues = {}
        self.groups = {}
        self.fof_groups = {}
        for t in self.tels:
            self.trues[t] = pd.read_parquet(
                case.path("catalogs", f"{t.lower()}.parquet")
            )
            self.groups[t] = _load(case.path("groups", f"{initial(t)}_no_overlap.pkl"))[
                "data"
            ]
            self.fof_groups[t] = _load(case.path("groups", f"{initial(t)}_fof.pkl"))
        self.plot_dir = os.path.join(case.out_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    def in_nonone(self, tel):
        """Mask of the LSST objects that belong to a group other than 1-1.

        These are the objects whose entropy is computed by the `sb` stage; 1-1 objects have S_b = 0 by convention.

        Parameters
        ----------
        tel : {'Euclid', 'HST'}

        Returns
        -------
        numpy.ndarray
            Boolean array over the rows of `lsst`.
        """
        m = np.zeros(len(self.lsst), bool)
        for g in self.groups[tel]:
            is_one_to_one = len(g[0]) == 1 and len(g[1]) == 1
            if not is_one_to_one:
                m[g[1]] = True
        return m

    def save(self, fig, name):
        """Save a figure as ``plot_dir/pdf/<stem>.pdf`` and ``plot_dir/png/<stem>.png`` and close it.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        name : str
            File name; its extension is ignored (see `save_figure`).
        """
        save_figure(fig, self.plot_dir, name)


def _load(p):
    """Unpickle a file.

    Parameters
    ----------
    p : str
        Path.

    Returns
    -------
    object
    """
    with open(p, "rb") as f:
        return pickle.load(f)


def _obj_table(groups):
    """Flatten groups into one row per LSST object.

    Parameters
    ----------
    groups : list
        Groups ``[true_indices, lsst_indices]``.

    Returns
    -------
    idx : numpy.ndarray of int
        LSST row of each object.
    n_true, n_obs : numpy.ndarray of int
        Number of true and of LSST objects in the group of each object.
    """
    rows = []
    n_true_per_group = []
    n_obs_per_group = []
    for g in groups:
        rows.extend(g[1])
        n_true_per_group.append(len(g[0]))
        n_obs_per_group.append(len(g[1]))

    # one entry per LSST object: each group's numbers are repeated once per LSST object of the group
    idx = np.array(rows, dtype=int)
    n_true = np.repeat(n_true_per_group, n_obs_per_group)
    n_obs = np.repeat(n_obs_per_group, n_obs_per_group)
    return idx, n_true, n_obs


def _case_names(cases):
    """Name of each case, the default legend labels."""
    names = []
    for c in cases:
        names.append(c.name)
    return names


def _common_telescopes(results):
    """Surveys present in every case of `results`, in the order of `TELESCOPES`."""
    tels = []
    for t in TELESCOPES:
        in_all_cases = True
        for r in results:
            if t not in r.tels:
                in_all_cases = False
        if in_all_cases:
            tels.append(t)
    return tels


def fraction_per_bin(values, flag, bins):
    """Fraction of the objects of each bin that satisfy a condition.

    Parameters
    ----------
    values : numpy.ndarray
        Binned quantity of every object (e.g. LSST i magnitude).
    flag : numpy.ndarray of bool
        Condition, same length as `values`.
    bins : numpy.ndarray
        Bin edges.

    Returns
    -------
    fraction : numpy.ndarray
        ``N(flag) / N`` per bin (NaN in empty bins).
    error : numpy.ndarray
        Error from `compute_error_on_fraction`.
    """
    n_total, _ = np.histogram(values, bins)
    n_selected, _ = np.histogram(values[flag], bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = n_selected / n_total
    error = compute_error_on_fraction(n_selected, n_total)
    return fraction, error


def unrecog_fraction(groups, lsst, bins=IMAG_BINS):
    """Fraction of LSST objects per magnitude bin that belong to an unrecognized blend.

    An unrecognized blend is a group with more true than LSST objects.

    Parameters
    ----------
    groups : list
        Groups ``[true_indices, lsst_indices]``.
    lsst : pandas.DataFrame
        LSST catalog with ``i_cModel_mag``.
    bins : numpy.ndarray, default IMAG_BINS
        Magnitude bin edges.

    Returns
    -------
    fraction, error : numpy.ndarray
        See `fraction_per_bin`.
    """
    idx, n_true, n_obs = _obj_table(groups)
    imag = lsst["i_cModel_mag"].values[idx]
    is_unrecognized = n_true > n_obs
    return fraction_per_bin(imag, is_unrecognized, bins)


# ----------------------------------------------------------------------------- plots
def _lsst_all_positions(res):
    """LSST galaxy positions over the whole footprint (``catalogs/lsst_all_radec.parquet``), or, for runs prepared
    before that file existed, the positions of the cut catalog (with a message: run the `footprint` stage).
    """
    path = res.case.path("catalogs", "lsst_all_radec.parquet", makedirs=False)
    if os.path.exists(path):
        return pd.read_parquet(path)
    print(
        f"{path} missing: sky maps show only the cut catalog (run `python -m copaings footprint <case>`)"
    )
    return res.lsst[["ra", "dec"]]


def _draw_circle(ax, circle, **kw):
    """Draw a `sky_circle` ``[ra0, dec0, radius]`` (an ellipse in ra/dec coordinates)."""
    ra0, dec0, radius = circle
    t = np.linspace(0, 2 * np.pi, 500)
    ra_radius = radius / np.cos(
        np.deg2rad(dec0)
    )  # a circle on the sky is wider in RA away from the equator
    ra = ra0 + ra_radius * np.cos(t)
    dec = dec0 + radius * np.sin(t)
    ax.plot(ra, dec, **kw)


def _sky_axes(ax, dec):
    """Common look of the sky maps: equal sky scale in ra and dec, labels."""
    # 1 deg of RA and of Dec equally long on the sky
    ax.set_aspect(1 / np.cos(np.deg2rad(np.median(dec))))
    ax.set_xlabel("ra [deg]")
    ax.set_ylabel("dec [deg]")


@styled
def sky_map(res, gridsize=75, name="sky_map_lsst_density"):
    """Figure of the LSST galaxy density on the sky, over the whole footprint (before the sky cuts).

    Hexagonal bins with the colour bar on top, the outlines of the true catalogs used (convex hulls of their positions,
    after the cuts) and the `sky_circle` of the case when one is set.

    Parameters
    ----------
    res : Results
    gridsize : int, default 75
        Number of hexagons along RA.
    name : str, default 'sky_map_lsst_density'
        Output file name in the plot directory.
    """
    from scipy.spatial import ConvexHull

    pos = _lsst_all_positions(res)
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(
        pos["ra"],
        pos["dec"],
        gridsize=gridsize,
        mincnt=1,
        cmap=BLUE_CMAP,
        rasterized=True,
    )
    for tel in res.tels:
        pts = res.trues[tel][["ra", "dec"]].dropna().values
        if len(pts) < 3:
            continue
        hull = ConvexHull(pts)
        v = pts[hull.vertices]
        v = np.vstack([v, v[:1]])  # back to the first point to close the outline
        ax.plot(v[:, 0], v[:, 1], color=COLORS[tel], lw=2, label=tel)
    if res.case.sky_circle:
        _draw_circle(
            ax,
            res.case.sky_circle,
            color="r",
            ls="--",
            lw=1.5,
            label=f"circle ({res.case.sky_circle[2]:g}$^\\circ$)",
        )
    _sky_axes(ax, pos["dec"])
    ax.legend(loc="upper right", fontsize=13)
    top_colorbar(fig, hb, ax, label="LSST count")
    res.save(fig, name)


@styled
def sky_scatter(res, step=10, size=0.5, name="sky_map_lsst_scatter"):
    """Figure of the LSST galaxies on the sky as tiny dots, which shows the depth pattern of the single visits.

    As in ``dp2_cut_circle.ipynb``: every `step`-th galaxy of the whole footprint, grey outside the `sky_circle` and
    in the LSST colour inside it, with the circle drawn.

    Parameters
    ----------
    res : Results
    step : int, default 10
        Plot one galaxy out of `step`.
    size : float, default 0.5
        Marker size [points^2].
    name : str, default 'sky_map_lsst_scatter'
        Output file name in the plot directory.
    """
    pos = _lsst_all_positions(res)
    ra = pos["ra"].values
    dec = pos["dec"].values
    circle = res.case.sky_circle
    if circle:
        ra0, dec0, radius = circle
        inside = in_sky_circle(ra, dec, ra0, dec0, radius)
    else:
        inside = np.ones(len(ra), bool)
    outside = ~inside
    fig, ax = plt.subplots(figsize=(8, 8))
    dots = dict(marker=".", s=size, alpha=0.75, rasterized=True)
    ax.scatter(ra[outside][::step], dec[outside][::step], color="lightgray", **dots)
    ax.scatter(ra[inside][::step], dec[inside][::step], color=COLORS["LSST"], **dots)
    if circle:
        _draw_circle(ax, circle, color="r", lw=1.5)
    _sky_axes(ax, dec)
    res.save(fig, name)


def _draw_all_surveys(res, ax, ra, dec, step, size, legend_loc="lower right"):
    """LSST (whole footprint, every `step`-th galaxy), the true catalogs used (Euclid every `step`-th, HST all, fainter)
    and a legend, as in the bottom panel of the ``dp2_cut_circle.ipynb`` sky map."""
    dots = dict(marker=".", alpha=0.75, rasterized=True)
    ax.scatter(ra[::step], dec[::step], s=size, color=COLORS["LSST"], **dots)
    # empty scatter: a big dot in the legend
    ax.scatter(
        [], [], marker=".", s=300, color=COLORS["LSST"], label=_lsst_label(res.case)
    )
    for tel in res.tels:
        t = res.trues[tel]
        tel_dots = dict(dots)
        if tel == "HST":  # small catalog: every object, fainter
            k = 1
            tel_dots["alpha"] = 0.375
        else:
            k = step
            tel_dots["alpha"] = 0.75
        ax.scatter(
            t["ra"].values[::k],
            t["dec"].values[::k],
            s=size,
            color=COLORS[tel],
            **tel_dots,
        )
        ax.scatter(
            [],
            [],
            marker=".",
            s=300,
            color=COLORS[tel],
            label=_true_label(res.case, tel),
        )
    ax.legend(loc=legend_loc, prop={"size": 14})


@styled
def sky_surveys(res, step=10, size=0.5, legend_room=1.0, name="sky_map_surveys"):
    """Sky map of every survey: the bottom panel of `sky_two_panel` alone.

    Parameters
    ----------
    res : Results
    step : int, default 10
        Plot one LSST / Euclid object out of `step` (HST: all).
    size : float, default 0.5
        Marker size [points^2].
    legend_room : float, default 1.0
        Extra RA range [deg] on the right of the data, where the legend sits.
    name : str, default 'sky_map_surveys'
        Output file name in the plot directory.
    """
    pos = _lsst_all_positions(res)
    ra = pos["ra"].values
    dec = pos["dec"].values
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_all_surveys(res, ax, ra, dec, step, size)
    pad = 0.1
    ax.set_xlim(
        ra.min() - pad, ra.max() + legend_room
    )  # empty room on the right for the legend
    ax.set_ylim(dec.min() - pad, dec.max() + pad)
    ax.set_aspect(1 / np.cos(np.deg2rad(np.median(dec))))
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    res.save(fig, name)


# axis limits of the ECDFS sky maps of dp2_cut_circle.ipynb: ((ra_min, ra_max), (dec_min, dec_max)) [deg]
ECDFS_SKY_LIMS = ((50.4, 55.5), (-30.7, -25.46))


def _sky_limits(ra, dec, lims=None, pad=0.1):
    """Sky-map limits: `lims` if given; the ECDFS notebook limits when the data are centred inside them; else the
    data range plus `pad` [deg]."""
    if lims is not None:
        return lims

    ra_min, ra_max = ECDFS_SKY_LIMS[0]
    dec_min, dec_max = ECDFS_SKY_LIMS[1]
    ra_center = np.median(ra)
    dec_center = np.median(dec)
    if ra_min < ra_center < ra_max and dec_min < dec_center < dec_max:
        return ECDFS_SKY_LIMS

    ra_lims = (ra.min() - pad, ra.max() + pad)
    dec_lims = (dec.min() - pad, dec.max() + pad)
    return (ra_lims, dec_lims)


@styled
def sky_two_panel(
    res, gridsize=75, step=10, size=0.5, lims=None, name="sky_map_two_panel"
):
    """Two-panel sky map: LSST density on top, every survey below.

    Top: LSST galaxy density over the whole footprint (hexagonal bins), colour bar on top with 'LSST count' inside.
    Bottom: LSST (whole footprint), the true catalogs used (after the cuts) and a legend at the top right, as in
    ``dp2_cut_circle.ipynb`` (LSST and Euclid one object out of `step`, HST all of them, fainter).

    Parameters
    ----------
    res : Results
    gridsize : int, default 75
        Number of hexagons along RA in the top panel.
    step : int, default 10
        Plot one LSST / Euclid object out of `step` in the bottom panel.
    size : float, default 0.5
        Marker size [points^2] of the bottom panel.
    lims : tuple, optional
        ``((ra_min, ra_max), (dec_min, dec_max))`` [deg]; default the notebook ECDFS limits (`ECDFS_SKY_LIMS`) for
        an ECDFS run, else the data range.
    name : str, default 'sky_map_two_panel'
        Output file name in the plot directory.
    """
    pos = _lsst_all_positions(res)
    ra = pos["ra"].values
    dec = pos["dec"].values
    ra_lims, dec_lims = _sky_limits(ra, dec, lims)
    dec_center = 0.5 * (dec_lims[0] + dec_lims[1])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 16), sharex=True, gridspec_kw={"hspace": 0}
    )
    ax1.set_anchor("S")  # the two panels touch
    ax2.set_anchor("N")

    # top: LSST density (hexagons sized on the whole footprint)
    hb = ax1.hexbin(
        ra, dec, gridsize=gridsize, mincnt=1, cmap=BLUE_CMAP, rasterized=True
    )

    # bottom: every survey
    _draw_all_surveys(res, ax2, ra, dec, step, size, legend_loc="upper right")

    for a in (ax1, ax2):
        a.set_xlim(ra_lims[0], ra_lims[1])
        a.set_ylim(dec_lims[0], dec_lims[1])
        a.set_aspect(1 / np.cos(np.deg2rad(dec_center)))
        a.set_ylabel("Dec [deg]")
    # no tick-label clash where the panels meet
    ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="lower"))
    ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="upper"))
    ax2.set_xlabel("RA [deg]")
    bar_height = 0.25 / fig.get_figheight()  # 0.25 inch
    top_colorbar(fig, hb, ax1, label="LSST count", height=bar_height, pad=0.0)
    res.save(fig, name)


@styled
def detect_counts(
    res,
    layout="horizontal",
    max_true=6,
    max_lsst=None,
    highlight=False,
    cell_fontsize=11,
    cell_fontfamily="sans-serif",
    show_counts=False,
    name="group_size_matrix",
):
    """Figure of the number of groups as a function of (true objects, LSST objects).

    One colour-coded matrix per survey with the percentage of groups in each cell (and their number with
    `show_counts`), and one colour bar with 'Count' written inside it.

    * ``layout='horizontal'``: panels side by side, rows = LSST objects, columns = true objects, colour bar on top.
    * ``layout='vertical'`` (as in ``dp2_cut_circle.ipynb``): panels stacked, rows = true objects, columns = LSST
      objects (labelled on top), colour bar at the bottom.

    Parameters
    ----------
    res : Results
    layout : {'horizontal', 'vertical'}, default 'horizontal'
    max_true : int, default 6
        Largest number of true objects shown; bigger groups are not counted.
    max_lsst : int, optional
        Largest number of LSST objects shown; default 6 (horizontal) or 5 (vertical).
    highlight : bool, default False
        Shade the cells with more true than LSST objects (unrecognized blends).
    cell_fontsize : float, default 11
        Font size of the percentages written in the cells (the counts use 3/4 of it).
    cell_fontfamily : str, default 'sans-serif'
        Font family of the cell text.
    show_counts : bool, default False
        Also write the number of groups at the bottom of each cell.
    name : str, default 'group_size_matrix'
        Output file name in the plot directory.
    """
    if layout not in ("horizontal", "vertical"):
        raise ValueError("layout must be 'horizontal' or 'vertical'")
    vertical = layout == "vertical"
    if max_lsst is None:
        if vertical:
            max_lsst = 5
        else:
            max_lsst = 6

    # one matrix per survey: number of groups with n_true true and n_lsst LSST objects
    mats = []
    for tel in res.tels:
        m = np.zeros((max_true + 1, max_lsst + 1), int)
        for g in res.groups[tel]:
            n_true = len(g[0])
            n_lsst = len(g[1])
            if n_true <= max_true and n_lsst <= max_lsst:
                m[n_true, n_lsst] += 1
        m = m[1:, 1:]  # no empty side: groups have at least one object of each
        if vertical:
            mats.append(m)  # rows = true objects
        else:
            mats.append(m.T)  # rows = LSST objects

    # common colour scale
    smallest = []
    largest = []
    for m in mats:
        smallest.append(m[m > 0].min())
        largest.append(m.max())
    vmin = min(smallest)
    vmax = max(largest)
    font = dict(
        ha="center",
        va="center",
        fontweight="bold",
        color="white",
        fontfamily=cell_fontfamily,
        path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
    )
    bottom_font = dict(font)  # same font, aligned on the bottom of the text
    bottom_font["va"] = "bottom"

    n_tel = len(res.tels)
    if vertical:
        fig, ax = plt.subplots(n_tel, 1, figsize=(7, 5.5 * n_tel), squeeze=False)
        ax = ax[:, 0]
        plt.subplots_adjust(hspace=0.001)
    else:
        fig, ax = plt.subplots(
            1, n_tel, figsize=(6 * n_tel, 8), sharey=True, squeeze=False
        )
        ax = ax[0]
        plt.subplots_adjust(wspace=0.0)
    for i in range(n_tel):
        a = ax[i]
        tel = res.tels[i]
        m = mats[i]
        im = a.matshow(m, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=BLUE_CMAP)
        total = m.sum()
        n_rows, n_cols = m.shape
        for row in range(n_rows):
            for col in range(n_cols):
                if vertical:
                    n_true = row
                    n_lsst = col
                else:
                    n_true = col
                    n_lsst = row
                if highlight and n_lsst < n_true:  # more true than LSST objects
                    a.add_patch(
                        plt.Rectangle(
                            (col - 0.5, row - 0.5),
                            1,
                            1,
                            fc="red",
                            ec="red",
                            alpha=0.2,
                            lw=1.5,
                        )
                    )
                if m[row, col] > 0:
                    a.text(
                        col,
                        row,
                        f"{100 * m[row, col] / total:.1f}%",
                        fontsize=cell_fontsize,
                        **font,
                    )
                    if show_counts:
                        # matshow y grows downwards: row + 0.4 = bottom of the cell
                        a.text(
                            col,
                            row + 0.4,
                            f"{m[row, col]:,}",
                            fontsize=0.7 * cell_fontsize,
                            **bottom_font,
                        )
        a.set_xticks(range(n_cols))
        a.set_xticklabels(range(1, n_cols + 1))
        a.set_yticks(range(n_rows))
        a.set_yticklabels(range(1, n_rows + 1))
        if vertical:
            a.set_ylabel(f"Detected {tel} objects")
            a.xaxis.set_label_position("top")
            a.tick_params(
                axis="x", top=True, bottom=True, labeltop=(i == 0), labelbottom=False
            )
        else:
            a.set_xlabel(f"Detected {tel} objects")
            a.xaxis.set_ticks_position("bottom")
            a.xaxis.set_label_position("bottom")
    if vertical:
        ax[0].set_xlabel("Detected LSST objects")
    else:
        ax[0].set_ylabel("Detected LSST objects")
    if highlight:
        ax[0].add_patch(
            plt.Rectangle(
                (0, 0), 0, 0, fc="red", ec="red", alpha=0.2, label="Unrec. Blends"
            )
        )
        ax[0].legend(loc="lower left")
    # 'Count' written inside the bar, white with a black outline (horizontal: 0.1 below the bar centre)
    h = 0.22 / fig.get_figheight()  # bar 0.22 inch tall whatever the number of panels
    if vertical:
        side = "bottom"
        pad = 0.001
        label_y = 0.5
    else:
        side = "top"
        pad = 0.0
        label_y = 0.4
    cbar = top_colorbar(
        fig,
        im,
        list(ax),
        label="Count",
        height=h,
        pad=pad,
        label_y=label_y,
        side=side,
    )
    cbar.ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())  # only the decades
    res.save(fig, name)


@styled
def unrecog_vs_mag(res, surface_density=False, also_fof_only=False, name=None):
    """Figure of the unrecognized-blend fraction versus LSST i magnitude.

    Top panel: magnitude distributions of the LSST catalog and of the true catalogs converted to i with the saved linear
    relations. Bottom panel: fraction of unrecognized blends per magnitude bin for each survey.

    Parameters
    ----------
    res : Results
    surface_density : bool, default False
        Top panel in objects per arcmin2 (using the occupied sky areas) instead of normalised densities.
    also_fof_only : bool, default False
        Also draw the fraction for the FoF groups before the ellipse overlap test.
    name : str, optional
        Output file name; default ``unrec_fraction_vs_imag`` or ``unrec_fraction_vs_imag_surface_density`` (pdf and png).
    """
    binsc = 0.5 * (IMAG_BINS[:-1] + IMAG_BINS[1:])
    ll = res.case.linking_length
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    fig.subplots_adjust(hspace=0)

    _mag_hist_panel(res, ax[0], surface_density)
    ax[0].legend()
    # its 0 would sit on the line between the panels and read as the bottom panel's top label
    ax[0].yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="lower"))

    for tel in res.tels:
        # one curve per kind of groups: (groups, line style, label)
        curves = []
        curves.append((res.groups[tel], "--", f'{tel} (FoF({ll:g}") + EOT)'))
        if also_fof_only:
            curves.append((res.fof_groups[tel], "-", f'{tel} (FoF({ll:g}"))'))
        for groups, ls, label in curves:
            f, e = unrecog_fraction(groups, res.lsst)
            ax[1].fill_between(binsc, f - e, f + e, alpha=0.1, color=COLORS[tel])
            ax[1].errorbar(
                binsc,
                f,
                yerr=e,
                marker=".",
                ms=10,
                alpha=0.8,
                ls=ls,
                color=COLORS[tel],
                label=label,
            )
    ax[1].set_xlabel(r"LSST $i$-mag")
    ax[1].set_ylabel("Fraction of unrecognized blends")
    ax[1].set_xlim(18.5, 28.2)
    _fraction_ylim(ax[1])
    _no_top_tick_label(ax[1])
    ax[1].grid(alpha=0.3)
    ax[1].legend(loc="upper left")
    if name is None:
        if surface_density:
            name = "unrec_fraction_vs_imag_surface_density"
        else:
            name = "unrec_fraction_vs_imag"
    res.save(fig, name)


def _sb_fraction(res, tel, x_col, bins, thr, extra_mask=None):
    """Fraction of the LSST objects in groups with S_b above a threshold, per bin of a column.

    Parameters
    ----------
    res : Results
    tel : {'Euclid', 'HST'}
    x_col : str
        Column of the LSST catalog that is binned.
    bins : numpy.ndarray
        Bin edges.
    thr : float
        Entropy threshold; the condition is ``S_b > thr``.
    extra_mask : callable, optional
        Function of the LSST sub-catalog returning a boolean mask of the objects to keep.

    Returns
    -------
    fraction, error : numpy.ndarray
        See `fraction_per_bin`.
    """
    idx, _, _ = _obj_table(res.groups[tel])
    sub = res.lsst.iloc[idx]
    if extra_mask is not None:
        sub = sub[extra_mask(sub)]
    x = sub[x_col].values
    is_high_sb = (
        sub[f"blending_entropy_{tel}"].values > thr
    )  # NaN S_b counts as not high
    ok = np.isfinite(x)
    return fraction_per_bin(x[ok], is_high_sb[ok], bins)


def sb_curve(res, tel, thr=0.2, bins=None, min_count=30):
    """Fraction of LSST objects with S_b above a threshold per magnitude bin.

    The population is all LSST objects that belong to a group of the ellipse overlap test (including 1-1 groups, whose
    S_b is 0).

    Parameters
    ----------
    res : Results
    tel : {'Euclid', 'HST'}
    thr : float, default 0.2
        Entropy threshold.
    bins : array_like, optional
        i-magnitude bin edges; default edges 18.25 to 28.75 in steps of 0.5.
    min_count : int, default 30
        Bins with fewer objects are returned as NaN.

    Returns
    -------
    centres : numpy.ndarray
        Bin centres.
    fraction, error : numpy.ndarray
        Fraction and its error (NaN in bins with fewer than `min_count` objects).
    counts : numpy.ndarray
        Number of objects per bin.
    """
    if bins is None:
        bins = np.arange(18.25, 29.1, 0.5)
    else:
        bins = np.asarray(bins)
    centres = 0.5 * (bins[:-1] + bins[1:])
    f, e = _sb_fraction(res, tel, "i_cModel_mag", bins, thr)

    idx, _, _ = _obj_table(res.groups[tel])
    n, _ = np.histogram(res.lsst["i_cModel_mag"].values[idx], bins)
    too_few = n < min_count
    f[too_few] = np.nan
    e[too_few] = np.nan
    return centres, f, e, n


@styled
def sb_vs_mag(res, thresholds=(0.0, 0.2), name="high_sb_fraction_vs_imag"):
    """Figure of the fraction of high-S_b objects versus LSST i magnitude.

    Parameters
    ----------
    res : Results
    thresholds : sequence of float, default (0.0, 0.2)
        Entropy thresholds, one curve per survey and threshold.
    name : str, default 'high_sb_fraction_vs_imag'
        Output file name in the plot directory.
    """
    bins = np.arange(18.25, 29.1, 0.5)
    binsc = 0.5 * (bins[:-1] + bins[1:])
    fig = plt.figure(figsize=(9, 5))
    for tel in res.tels:
        for k, thr in enumerate(thresholds):
            if k == 0:  # first threshold: faint dashed curve
                ls, marker, ms, alpha, fill_alpha = "--", "o", 4, 0.45, 0.01
            else:
                ls, marker, ms, alpha, fill_alpha = "-", "*", 9, 1, 0.4
            f, e = _sb_fraction(res, tel, "i_cModel_mag", bins, thr)
            plt.fill_between(binsc, f - e, f + e, alpha=fill_alpha, color=COLORS[tel])
            plt.errorbar(
                binsc,
                f,
                yerr=e,
                ls=ls,
                marker=marker,
                ms=ms,
                alpha=alpha,
                color=COLORS[tel],
                label=rf"{tel} $S_b > {thr}$",
            )
    plt.xlabel(r"LSST $i$-mag", fontsize=18)
    plt.ylabel("Fraction of LSST objects", fontsize=18)
    plt.xlim(18.5, 28.2)
    _fraction_ylim(plt.gca())
    plt.grid(alpha=0.2)
    plt.legend(loc="upper left")
    res.save(fig, name)


@styled
def mag_and_sb_vs_mag(
    res, thresholds=(0.0, 0.2), name="mag_and_high_sb_fraction_vs_imag"
):
    """Two-panel figure as in ``dp2_cut_circle.ipynb``: magnitude distributions (top), high-S_b fraction vs i (bottom).

    Top: LSST and the true catalogs converted to LSST i (densities). Bottom: for each survey, fraction of the LSST
    objects of the overlap-test groups with ``S_b > thresholds[0]`` (circles, faint) and ``> thresholds[1]``
    (squares); colours give the survey, markers the threshold.

    Parameters
    ----------
    res : Results
    thresholds : tuple of two float, default (0.0, 0.2)
        Background and interest entropy thresholds.
    name : str, default 'mag_and_high_sb_fraction_vs_imag'
        Output file name in the plot directory.
    """
    bins = np.arange(18.25, 29.1, 0.5)
    binsc = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    fig.subplots_adjust(hspace=0)

    _mag_hist_panel(res, ax[0])
    ax[0].legend(fontsize=15, loc="upper left")
    # its 0 would sit on the bottom panel's top label
    ax[0].yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="lower"))

    lo, hi = thresholds
    # one style per threshold: (threshold, marker, marker size, alpha, alpha of the error band)
    curve_styles = [
        (hi, "s", 4, 1.0, 0.15),
        (lo, "o", 5, 0.4, 0.1),
    ]
    for tel in res.tels:
        for thr, marker, ms, alpha, fill in curve_styles:
            f, e = _sb_fraction(res, tel, "i_cModel_mag", bins, thr)
            ax[1].fill_between(binsc, f - e, f + e, alpha=fill, color=COLORS[tel], lw=0)
            ax[1].errorbar(
                binsc,
                f,
                yerr=e,
                ls="--",
                marker=marker,
                ms=ms,
                color=COLORS[tel],
                alpha=alpha,
                capsize=0,
                elinewidth=1,
            )
    ax[1].plot([], [], ls="--", marker="o", color="k", ms=6, label=rf"$S_b > {lo:g}$")
    ax[1].plot([], [], ls="--", marker="s", color="k", ms=6, label=rf"$S_b > {hi:g}$")
    ax[1].legend(loc="upper left")
    ax[1].set_ylabel("Fraction of LSST objects")
    ax[1].set_xlabel(r"LSST $i$-mag")
    ax[1].set_xlim(18.5, 28.2)
    _fraction_ylim(ax[1])
    _no_top_tick_label(ax[1])
    ax[1].grid(alpha=0.3)
    res.save(fig, name)


@styled
def true_mag_fit_check(res, lim=(17, 30.5), bins=100, name="true_mag_fit_check"):
    """Check of the analytic fits of the survey magnitudes from the LSST bands, as in ``dp2_unrecog_bl.ipynb``.

    One panel per survey: density of the true magnitude (Euclid VIS, HST F814W) against the value fitted from the LSST
    bands for the 1-1 groups (the pairs the fit uses), with the 1:1 line; one colour bar on top of each panel
    (own count scale) with 'Count' inside. The
    fits are the stored ones (``color_fit`` of ``fit_params.json``), refitted when missing (see
    `color_fit.load_color_fit`).

    Parameters
    ----------
    res : Results
    lim : tuple of float, default (17, 30.5)
        Axis limits [mag].
    bins : int, default 100
        Number of density bins per axis.
    name : str, default 'true_mag_fit_check'
        Output file name in the plot directory.
    """
    from . import color_fit

    data = {}
    for tel in res.tels:
        fit = color_fit.load_color_fit(res.case, tel)
        ti, li = color_fit.pairs_11(res.groups[tel])
        true = res.trues[tel][TRUE_MAG[tel]].values[ti].astype(float)
        pred = color_fit.predict(fit, res.lsst.iloc[li])
        ok = np.isfinite(true) & np.isfinite(pred)
        data[tel] = (true[ok], pred[ok], fit)
    edges = np.linspace(lim[0], lim[1], bins + 1)

    n_tel = len(res.tels)
    fig, ax = plt.subplots(1, n_tel, figsize=(7 * n_tel, 7), sharey=True, squeeze=False)
    ax = ax[0]
    fig.subplots_adjust(wspace=-0.01)
    ims = []
    for a, tel in zip(ax, res.tels):
        true, pred, fit = data[tel]
        counts, _, _ = np.histogram2d(true, pred, bins=(edges, edges))
        vmax = counts.max()
        if vmax < 2:
            vmax = 2
        counts, xedges, yedges, im = a.hist2d(
            true,
            pred,
            bins=(edges, edges),
            norm=mpl.colors.LogNorm(vmin=1, vmax=vmax),  # own scale per survey
            cmap=BLUE_CMAP,
            cmin=1,
            rasterized=True,
        )
        a.axline((lim[0], lim[0]), slope=1, color="r", lw=0.9)
        a.set_xlim(lim[0], lim[1])
        a.set_ylim(lim[0], lim[1])
        a.set_aspect("equal")
        a.set_xticks(np.arange(18, lim[1], 2))
        a.grid(alpha=0.3)
        a.set_xlabel(
            f"{_true_label(res.case, tel)} {TRUE_BAND[tel]} magnitude", fontsize=22
        )
        ims.append(im)
        # a.text(
        #     0.04,
        #     0.96,
        #     f"{fit['n_pairs']:,} 1-1 pairs\nresidual NMAD {fit['resid_nmad']:.3f}",
        #     transform=a.transAxes,
        #     ha="left",
        #     va="top",
        #     fontsize=13,
        # )
    ax[0].set_yticks(np.arange(18, lim[1], 2))
    ax[0].set_ylabel(f"Fit from {_lsst_label(res.case)} bands", fontsize=22)
    for a in ax[1:]:
        a.tick_params(labelleft=False)
    # one colour bar per panel, each with its own count scale
    for a, im in zip(ax, ims):
        cbar = top_colorbar(
            fig,
            im,
            a,
            label="Count",
            height=0.25 / fig.get_figheight(),
            pad=0.0,
            label_y=0.4,
        )
        cbar.ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    res.save(fig, name)


@styled
def sb_vs_blendedness(
    res, thresholds=(0.0, 0.2), mag_max=None, col="i_blendedness", name=None
):
    """Figure of the fraction of high-S_b objects versus LSST blendedness.

    Parameters
    ----------
    res : Results
    thresholds : sequence of float, default (0.0, 0.2)
        Entropy thresholds.
    mag_max : float, optional
        Keep only objects brighter than this i magnitude.
    col : str, default 'i_blendedness'
        Blendedness column. Note that the DP2 values are not on the 0-1 scale of DP1.
    name : str, optional
        Output file name; default ``high_sb_fraction_vs_blendedness[_imag_lt<mag_max>]`` (pdf and png).
    """
    edges = np.arange(-0.25, 1.0, 0.05) + 0.125
    centers = 0.5 * (edges[:-1] + edges[1:])

    def brighter_than_mag_max(d):
        return d["i_cModel_mag"] < mag_max

    if mag_max is None:
        mask = None
    else:
        mask = brighter_than_mag_max

    fig = plt.figure(figsize=(8, 5))
    for tel in res.tels:
        for k, thr in enumerate(thresholds):
            if k == 0:  # first threshold: faint dashed curve
                ls, marker, ms, alpha = "--", "o", 4, 0.45
            else:
                ls, marker, ms, alpha = "-", "*", 9, 1
            f, e = _sb_fraction(res, tel, col, edges, thr, mask)
            plt.fill_between(centers, f - e, f + e, alpha=0.05, color=COLORS[tel])
            plt.errorbar(
                centers,
                f,
                yerr=e,
                ls=ls,
                marker=marker,
                ms=ms,
                alpha=alpha,
                color=COLORS[tel],
                label=rf"{tel} $S_b > {thr}$",
            )
    plt.axvline(HSC_BLENDEDNESS_CUT, ls="--", c="k", label="HSC cut")
    plt.xlabel(_blendedness_label(col))
    plt.ylabel("Fraction of LSST objects")
    plt.xlim(edges[0], edges[-1])
    _fraction_ylim(plt.gca())
    plt.grid(alpha=0.2)
    plt.legend(loc="upper right")
    if mag_max is not None:
        plt.title(rf"$i < {mag_max}$")
    if name is None:
        name = "high_sb_fraction_vs_blendedness"
        if mag_max:
            name = name + "_imag_lt" + str(mag_max)
    res.save(fig, name)


@styled
def sb_hist2d_blendedness(res, col="i_blendedness", name="sb_vs_blendedness_hist2d"):
    """Figure of the S_b versus blendedness density.

    Only objects whose entropy is computed (groups other than 1-1) are shown; S_b = 0 (1-1 groups) and NaN (no entropy)
    objects are not part of this plot.

    Parameters
    ----------
    res : Results
    col : str, default 'i_blendedness'
        Blendedness column.
    name : str, default 'sb_vs_blendedness_hist2d'
        Output file name in the plot directory.
    """
    xb = np.arange(-0.25, 1.0, 0.05) + 0.125
    yb = np.log(np.arange(1, 4, 0.25))
    # objects of the groups other than 1-1, and the largest bin count of all panels (common colour scale)
    data = {}
    vmax = 1
    for t in res.tels:
        d = res.lsst[res.in_nonone(t)]
        data[t] = d
        counts, _, _ = np.histogram2d(d[col], d[f"blending_entropy_{t}"], bins=[xb, yb])
        if counts.max() > vmax:
            vmax = counts.max()
    norm = mpl.colors.LogNorm(vmin=1, vmax=vmax)

    n_tel = len(res.tels)
    fig, ax = plt.subplots(1, n_tel, sharey=True, figsize=(7 * n_tel, 5), squeeze=False)
    ax = ax[0]
    fig.subplots_adjust(wspace=0)
    for a, t in zip(ax, res.tels):
        counts, xedges, yedges, im = a.hist2d(
            data[t][col].values,
            data[t][f"blending_entropy_{t}"].values,
            bins=(xb, yb),
            norm=norm,
            cmap=BLUE_CMAP,
        )
        a.set_title(t)
        a.set_xlabel(_blendedness_label(col))
    ax[0].set_ylabel("$S_b$")
    fig.colorbar(im, ax=ax, pad=0, label="Counts")
    res.save(fig, name)


@styled
def sb_euclid_vs_hst(res, mag_max=23.5, name="sb_euclid_vs_hst_hist2d"):
    """Figure comparing the Euclid and HST entropies of the same LSST objects.

    Parameters
    ----------
    res : Results
    mag_max : float, default 23.5
        Keep only objects brighter than this i magnitude.
    name : str, default 'sb_euclid_vs_hst_hist2d'
        Output file name in the plot directory.

    Notes
    -----
    Only objects whose entropy is computed for both surveys are used; if there are none the plot is skipped with a
    message.
    """
    if not {"Euclid", "HST"} <= set(res.tels):
        print("skipped", name, "(needs both Euclid and HST)")
        return
    # objects in a group other than 1-1 for both surveys, brighter than mag_max
    both = res.in_nonone("Euclid") & res.in_nonone("HST")
    bright = (res.lsst["i_cModel_mag"] < mag_max).values
    d = res.lsst[both & bright]
    if len(d) == 0:  # e.g. no LSST object has an S_b from both surveys
        print("skipped", name, "(no objects with S_b from both Euclid and HST)")
        return
    sb_euclid = d["blending_entropy_Euclid"]
    sb_hst = d["blending_entropy_HST"]

    # bins of 0.1 up to the largest S_b (S_b can exceed 1)
    hi = 1.0
    if sb_euclid.max() > hi:
        hi = float(sb_euclid.max())
    if sb_hst.max() > hi:
        hi = float(sb_hst.max())
    n_bins = int(round(hi / 0.1))
    b = np.linspace(0, hi, n_bins + 1)

    counts, _, _ = np.histogram2d(sb_euclid, sb_hst, bins=(b, b))
    if counts.max() == 0:
        print("skipped", name, "(no objects with S_b >= 0 from both surveys)")
        return
    fig = plt.figure(figsize=(9, 6))
    plt.hist2d(
        sb_euclid,
        sb_hst,
        bins=(b, b),
        norm=mpl.colors.LogNorm(),
        cmap=BLUE_CMAP,
    )
    plt.axline((0, 0), slope=1, color="red")
    plt.xlabel("Euclid $S_b$")
    plt.ylabel("HST $S_b$")
    plt.colorbar(pad=0.001, label="Count")
    res.save(fig, name)


def all_plots(case):
    """Make the standard figures of a case.

    Parameters
    ----------
    case : Case
        A case whose stages have all run. The figures are written to ``<out_dir>/plots``.
    """
    res = Results(case)
    sky_map(res)
    sky_scatter(res)
    sky_two_panel(res)
    sky_surveys(res)
    detect_counts(res, name="group_size_matrix")
    detect_counts(res, show_counts=True, name="group_size_matrix_with_counts")
    detect_counts(res, layout="vertical", name="group_size_matrix_vertical")
    detect_counts(
        res,
        layout="vertical",
        show_counts=True,
        name="group_size_matrix_vertical_with_counts",
    )
    unrecog_vs_mag(res)
    unrecog_vs_mag(res, surface_density=True)
    sb_vs_mag(res)
    mag_and_sb_vs_mag(res)
    true_mag_fit_check(res)
    sb_vs_blendedness(res)
    sb_vs_blendedness(res, mag_max=23.5)
    sb_hist2d_blendedness(res)
    sb_euclid_vs_hst(res)


# categorical slots 1-5 of the reference palette (light mode); adjacent-pair validated in that order
CASE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# secondary encoding: identity never rests on colour alone
CASE_MARKERS = ["o", "s", "^", "D", "v"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2de"
# bins with fewer LSST objects are not drawn (fractions too noisy to read)
MIN_COUNT = 30


@styled
def compare_sb(cases, out_dir, labels=None, thr=0.2, results=None, min_count=MIN_COUNT):
    """Compare the fraction of high-S_b objects versus i magnitude between cases.

    One curve per case; Euclid and HST in two panels sharing the y axis. Bins with fewer than `min_count` objects are
    not drawn.

    Parameters
    ----------
    cases : list of Case
    out_dir : str
        Output directory (created).
    labels : list of str, optional
        Legend labels; default the case names.
    thr : float, default 0.2
        Entropy threshold.
    results : list of Results, optional
        Already loaded results of the cases (avoids reading them again).
    min_count : int, default `MIN_COUNT`
        Minimum number of objects in a bin.

    Notes
    -----
    Writes ``pdf/compare_high_sb_fraction_vs_imag_gt<thr>.pdf``, ``png/...png`` and ``compare_high_sb_fraction_vs_imag_gt<thr>.csv`` (the plotted
    values and the number of objects per bin).
    """
    os.makedirs(out_dir, exist_ok=True)
    if not labels:
        labels = _case_names(cases)
    if not results:
        results = []
        for c in cases:
            results.append(Results(c))
    bins = np.arange(18.25, 29.1, 0.5)
    binsc = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    tels = _common_telescopes(results)
    fig, ax = plt.subplots(
        1, len(tels), figsize=(6 * len(tels), 5), sharey=True, squeeze=False
    )
    ax = ax[0]
    fig.subplots_adjust(wspace=0.04)
    for a, tel in zip(ax, tels):
        for k in range(len(results)):
            res = results[k]
            lab = labels[k]
            idx, _, _ = _obj_table(res.groups[tel])
            m = res.lsst["i_cModel_mag"].values[idx]
            is_high_sb = res.lsst[f"blending_entropy_{tel}"].values[idx] > thr
            ok = np.isfinite(m)
            f, e = fraction_per_bin(m[ok], is_high_sb[ok], bins)
            n, _ = np.histogram(m[ok], bins)
            keep = n >= min_count
            if k == 0:  # first case drawn on top
                zorder = 4
            else:
                zorder = 3
            a.errorbar(
                binsc[keep],
                f[keep],
                yerr=e[keep],
                color=CASE_COLORS[k],
                marker=CASE_MARKERS[k],
                ms=6,
                lw=2,
                elinewidth=1,
                capsize=0,
                mec="white",
                mew=0.8,
                label=lab,
                zorder=zorder,
            )
            # one csv row per drawn bin
            for j in range(len(binsc)):
                if keep[j]:
                    rows.append(
                        {
                            "survey": tel,
                            "case": lab,
                            "imag_center": binsc[j],
                            "n_objects": int(n[j]),
                            "frac_sb_gt": f[j],
                            "err": e[j],
                        }
                    )
        a.text(
            0.03,
            0.96,
            tel,
            transform=a.transAxes,
            va="top",
            fontsize=15,
            fontweight="bold",
            color=INK,
        )
        a.set_xlabel(r"LSST $i$-mag", fontsize=13, color=INK)
        a.set_xlim(18.5, 28.2)
        a.grid(color=GRID, lw=1, zorder=0)
        a.set_axisbelow(True)
        a.tick_params(colors=INK_MUTED, labelsize=11)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(INK_MUTED)
    _fraction_ylim(ax[0])  # y shared by both panels
    ax[0].set_ylabel(
        rf"Fraction of blends with $S_b > {thr:g}$", fontsize=13, color=INK
    )
    for a in ax[1:]:
        a.tick_params(left=False)
    leg = ax[0].legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.90),
        frameon=False,
        fontsize=11,
        handlelength=2.2,
    )
    for t in leg.get_texts():
        t.set_color(INK)
    thr_text = str(thr).replace(".", "p")  # 0.2 -> 0p2
    tag = "compare_high_sb_fraction_vs_imag_gt" + thr_text
    save_figure(fig, out_dir, tag, dpi=200)
    csv_path = os.path.join(out_dir, tag + ".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("saved", csv_path)


@styled
def compare(cases, out_dir, labels=None):
    """Compare cases: unrecognized-blend fraction and high-S_b fraction versus i magnitude.

    Parameters
    ----------
    cases : list of Case
        Cases whose stages have all run.
    out_dir : str
        Output directory (created).
    labels : list of str, optional
        Legend labels; default the case names.

    Notes
    -----
    Writes ``{pdf,png}/compare_unrec_fraction_vs_imag.*`` and, through `compare_sb`, the
    ``compare_high_sb_fraction_vs_imag_gt0p2`` figures and csv.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not labels:
        labels = _case_names(cases)
    results = []
    for c in cases:
        results.append(Results(c))
    compare_sb(cases, out_dir, labels, results=results)
    binsc = 0.5 * (IMAG_BINS[:-1] + IMAG_BINS[1:])
    cmap = plt.get_cmap("tab10")
    tels = _common_telescopes(results)
    fig, ax = plt.subplots(
        1, len(tels), figsize=(7 * len(tels), 5), sharey=True, squeeze=False
    )
    ax = ax[0]
    for a, tel in zip(ax, tels):
        for k in range(len(results)):
            res = results[k]
            f, e = unrecog_fraction(res.groups[tel], res.lsst)
            a.errorbar(binsc, f, yerr=e, marker=".", color=cmap(k), label=labels[k])
        a.set_title(tel)
        a.set_xlabel(r"LSST $i$-mag")
        a.set_xlim(18.5, 28.2)
        a.grid(alpha=0.3)
    _fraction_ylim(ax[0])  # y shared by both panels
    ax[0].set_ylabel("Fraction of unrecognized blends")
    ax[0].legend()
    save_figure(fig, out_dir, "compare_unrec_fraction_vs_imag")


@styled
def euclid_fit_check(
    cases,
    out_dir,
    labels=None,
    xlim=(18, 29),
    ylim=(18, 29),
    nbins=110,
    min_median_count=50,
    scale="log",
):
    """Check of the Euclid VIS to LSST i relation on the 1-1 groups, one panel per case.

    Density of VIS magnitude versus LSST i magnitude, with the saved linear relation (``euclid_to_i`` in
    ``fit_params.json``), the line i = VIS, and the binned median and mean of i. The saved relation is refit on the
    plotted arrays as a consistency check; the result (OK or MISMATCH) is printed.

    Parameters
    ----------
    cases : list of Case
    out_dir : str
        Output directory (created).
    labels : list of str, optional
        Panel labels; default the case names.
    xlim, ylim : tuple of float, default (18, 29)
        Axis limits [mag].
    nbins : int, default 110
        Number of density bins per axis.
    min_median_count : int, default 50
        Minimum number of pairs for a binned median or mean to be drawn.
    scale : {'log', 'linear'}, default 'log'
        Colour scale of the density.

    Notes
    -----
    Writes ``{pdf,png}/euclid_vis_vs_i_fit.*`` (``euclid_vis_vs_i_fit_linear`` for the linear scale).
    """
    os.makedirs(out_dir, exist_ok=True)
    if not labels:
        labels = _case_names(cases)
    # cases without Euclid have nothing to check
    euclid_cases = []
    euclid_labels = []
    for case, label in zip(cases, labels):
        if "Euclid" in case.telescopes:
            euclid_cases.append(case)
            euclid_labels.append(label)
    cases, labels = euclid_cases, euclid_labels
    if not cases:
        print("euclid_fit_check: no case with Euclid")
        return
    n = len(cases)
    ncol = 3
    nrow = int(np.ceil((n + 1) / ncol))  # last cell holds the legend
    from .color_fit import pairs_11

    fig, axes = plt.subplots(
        nrow, ncol, figsize=(5.2 * ncol, 4.9 * nrow), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    xb = np.linspace(xlim[0], xlim[1], nbins)
    yb = np.linspace(ylim[0], ylim[1], nbins)
    xs = np.linspace(xlim[0], xlim[1], 100)
    if scale == "log":
        norm = mpl.colors.LogNorm()
    else:
        norm = mpl.colors.Normalize(vmin=0)

    for k in range(n):
        a = axes[k]
        case = cases[k]
        lab = labels[k]
        with open(case.path("catalogs", "fit_params.json", makedirs=False)) as f:
            params = json.load(f)
        groups = _load(case.path("groups", "E_no_overlap.pkl", makedirs=False))["data"]
        ti, li = pairs_11(groups)

        euclid = pd.read_parquet(
            case.path("catalogs", "euclid.parquet", makedirs=False), columns=["vis_mag"]
        )
        lsst = pd.read_parquet(
            case.path("catalogs", "lsst.parquet", makedirs=False),
            columns=["i_cModel_mag"],
        )
        vis = euclid["vis_mag"].values[ti]
        imag = lsst["i_cModel_mag"].values[li]
        ok = np.isfinite(vis) & np.isfinite(imag)
        # float32 polyfit is inaccurate
        vis = vis[ok].astype(np.float64)
        imag = imag[ok].astype(np.float64)

        counts, xedges, yedges, im = a.hist2d(
            vis,
            imag,
            bins=[xb, yb],
            norm=norm,
            cmap=BLUE_CMAP,
            cmin=1,
        )

        # the saved relation, refitted on exactly the arrays drawn in the hist2d
        slope, icpt = params["euclid_to_i"]
        refit = np.polyfit(vis, imag, 1)
        if np.allclose(refit, [slope, icpt], atol=2e-3):
            status = "OK"
        else:
            status = "*** MISMATCH ***"
        print(
            f"{case.name}: saved fit ({slope:.4f}, {icpt:.3f}) "
            f"vs refit on the plotted arrays ({refit[0]:.4f}, {refit[1]:.3f})",
            status,
        )
        a.plot(
            xs,
            slope * xs + icpt,
            color=CASE_COLORS[1],
            lw=2.5,
            label=r"saved fit: $i = a\,\mathrm{VIS} + b$",
            zorder=4,
        )
        a.plot(
            xs,
            xs,
            color=INK_MUTED,
            lw=1,
            ls="--",
            label=r"$i$ = VIS",
            zorder=3,
        )
        # binned median and mean of i, in VIS bins of 0.5 mag with enough pairs
        mb = np.arange(xlim[0], xlim[1] + 0.01, 0.5)
        mc = 0.5 * (mb[:-1] + mb[1:])
        cnt, _ = np.histogram(vis, mb)
        med = np.full(len(mc), np.nan)
        mean = np.full(len(mc), np.nan)
        for j in range(len(mc)):
            if cnt[j] >= min_median_count:
                in_bin = (vis >= mb[j]) & (vis < mb[j + 1])
                med[j] = np.median(imag[in_bin])
                mean[j] = np.mean(imag[in_bin])
        a.plot(
            mc,
            med,
            color=INK,
            marker="o",
            ms=4,
            lw=1.2,
            mfc="white",
            label=r"binned median of $i$",
            zorder=5,
        )
        a.plot(
            mc,
            mean,
            color="#8a2be2",
            marker="s",
            ms=3.5,
            lw=1.2,
            ls=":",
            label=r"binned mean of $i$",
            zorder=5,
        )
        # a.text(
        #     0.03,
        #     0.97,
        #     f"{lab}\n{len(vis):,} 1-1 pairs\n" + rf"$a={slope:.3f},\ b={icpt:.2f}$",
        #     transform=a.transAxes,
        #     va="top",
        #     fontsize=11,
        #     color=INK,
        #     bbox=dict(fc="white", ec="none", alpha=0.85, pad=2),
        # )
        a.set_xlim(xlim[0], xlim[1])
        a.set_ylim(ylim[0], ylim[1])
        a.grid(color=GRID, lw=0.8, zorder=0)
        a.set_axisbelow(True)
        a.tick_params(colors=INK_MUTED, labelsize=10)
    # the empty cells after the last case: no axes, the legend in the first one
    handles, hl = axes[0].get_legend_handles_labels()
    for a in axes[n:]:
        a.axis("off")
    axes[n].legend(handles, hl, loc="center", frameon=False, fontsize=12)
    for a in axes[:n]:
        a.set_xlabel("Euclid VIS mag", fontsize=12, color=INK)
    for a in axes[::ncol]:  # first column
        a.set_ylabel(r"LSST $i$ mag (cModel)", fontsize=12, color=INK)
    fig.subplots_adjust(hspace=0.16, wspace=0.05)
    fig.colorbar(im, ax=axes.tolist(), label="pairs per bin", fraction=0.02, pad=0.02)
    if scale == "log":
        name = "euclid_vis_vs_i_fit"
    else:
        name = "euclid_vis_vs_i_fit_linear"
    save_figure(fig, out_dir, name, dpi=170)
