"""Figures from the saved stage outputs. No heavy computation here: safe to rerun from a notebook or a login node."""
import json
import os
import pickle

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TELESCOPES, initial

COLORS = {'LSST': '#56B4E9', 'Euclid': '#E69F00', 'HST': '#009E73'}  # DESC palette
IMAG_BINS = np.arange(18.25, 31.1, 0.5)
HSC_BLENDEDNESS_CUT = 10 ** (-0.375)


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
    x, y = np.asarray(x, float), np.asarray(y, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        err = x / y * np.sqrt(1 / x + 1 / y)
    return np.where((x > 0) & (y > 0), err, 0.0)


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
        True catalogs, ``{'Euclid': ..., 'HST': ...}``.
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
        self.lsst = pd.read_parquet(case.path('lsst_sb.parquet'))
        self.meta = json.load(open(case.path('catalogs', 'meta.json')))
        self.params = json.load(open(case.path('catalogs', 'fit_params.json')))
        self.trues = {t: pd.read_parquet(case.path('catalogs', f'{t.lower()}.parquet')) for t in TELESCOPES}
        self.groups = {t: _load(case.path('groups', f'{initial(t)}_no_overlap.pkl'))['data'] for t in TELESCOPES}
        self.fof_groups = {t: _load(case.path('groups', f'{initial(t)}_fof.pkl')) for t in TELESCOPES}
        self.plot_dir = os.path.join(case.out_dir, 'plots')
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
        ids = [g[1] for g in self.groups[tel] if not (len(g[0]) == 1 and len(g[1]) == 1)]
        if ids:
            m[np.concatenate(ids).astype(int)] = True
        return m

    def save(self, fig, name):
        """Save a figure in the plot directory and close it.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        name : str
            File name inside `plot_dir`; the extension selects the format.
        """
        path = os.path.join(self.plot_dir, name)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print('saved', path)


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
    with open(p, 'rb') as f:
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
    n_true = np.array([len(g[0]) for g in groups])
    n_obs = np.array([len(g[1]) for g in groups])
    idx = np.concatenate([g[1] for g in groups]).astype(int)
    return idx, np.repeat(n_true, n_obs), np.repeat(n_obs, n_obs)


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
    tot = np.histogram(values, bins)[0]
    sel = np.histogram(values[flag], bins)[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        return sel / tot, compute_error_on_fraction(sel, tot)


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
    idx, nt, no = _obj_table(groups)
    return fraction_per_bin(lsst['i_cModel_mag'].values[idx], nt > no, bins)


# ----------------------------------------------------------------------------- plots
def detect_counts(res, size=8, drop_zero=False, highlight=False, cbar_on_top=False, name='detect_counts.pdf'):
    """Figure of the number of groups as a function of (true objects, LSST objects).

    Two panels (Euclid, HST): a colour-coded matrix with the percentage of groups in each cell.

    Parameters
    ----------
    res : Results
    size : int, default 8
        Matrix size; groups with `size` or more objects of either kind are not counted.
    drop_zero : bool, default False
        Remove the empty row and column of 0 objects.
    highlight : bool, default False
        Shade the cells with more true than LSST objects (unrecognized blends).
    cbar_on_top : bool, default False
        Put the colour bar above the panels.
    name : str, default 'detect_counts.pdf'
        Output file name in the plot directory.
    """
    mats = []
    for tel in TELESCOPES:
        m = np.zeros((size, size), int)
        for g in res.groups[tel]:
            n, k = len(g[0]), len(g[1])
            if n < size and k < size:
                m[n, k] += 1
        mats.append(m[1:, 1:] if drop_zero else m)
    off = 1 if drop_zero else 0
    n_plot = mats[0].shape[0]
    vmin = min(m[m > 0].min() for m in mats)
    vmax = max(m.max() for m in mats)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    plt.subplots_adjust(wspace=0.0)
    for a, tel, m in zip(ax, TELESCOPES, mats):
        mt = m.T  # rows = LSST, cols = true survey
        im = a.matshow(mt, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
        total = mt.sum()
        for y in range(n_plot):
            for x in range(n_plot):
                if highlight and (y + off) < (x + off) and (y + off) > 0:
                    a.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fc='red', ec='red', alpha=.2, lw=1.5))
                if mt[y, x] > 0:
                    a.text(x, y, f'{100 * mt[y, x] / total:.1f}%', ha='center', va='center', fontsize=9,
                           fontweight='bold', color='white',
                           path_effects=[pe.withStroke(linewidth=1.2, foreground='black')])
        a.set_xlabel(f'Detected {tel} objects')
        a.set_xticks(range(n_plot))
        a.set_yticks(range(n_plot))
        a.set_xticklabels(range(off, n_plot + off))
        a.set_yticklabels(range(off, n_plot + off))
        a.xaxis.set_ticks_position('bottom')
        a.xaxis.set_label_position('bottom')
    ax[0].set_ylabel('Detected LSST objects')
    if highlight:
        ax[0].add_patch(plt.Rectangle((0, 0), 0, 0, fc='red', ec='red', alpha=.2, label='Unrec. Blends'))
        ax[0].legend(loc='lower left')
    if cbar_on_top:
        p0, p1 = ax[0].get_position(), ax[1].get_position()
        cax = fig.add_axes([p0.x0, p1.y1, p1.x1 - p0.x0, 0.028])
        fig.colorbar(im, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
    else:
        fig.colorbar(im, ax=ax, label='Count', fraction=0.15, pad=0, shrink=0.641)
    res.save(fig, name)


def unrecog_vs_mag(res, surface_density=False, also_fof_only=False, name=None):
    """Figure of the unrecognized-blend fraction versus LSST i magnitude.

    Top panel: magnitude distributions of the LSST catalog and of the true catalogs converted to i with the saved linear
    relations. Bottom panel: fraction of unrecognized blends per magnitude bin for each survey.

    Parameters
    ----------
    res : Results
    surface_density : bool, default False
        Top panel in objects per deg2 (using the occupied sky areas) instead of normalised densities.
    also_fof_only : bool, default False
        Also draw the fraction for the FoF groups before the ellipse overlap test.
    name : str, optional
        Output file name; default ``unrec_frac_and_mag.pdf`` or ``unrec_frac_and_mag_surface_density.pdf``.
    """
    bins, binsc = IMAG_BINS, 0.5 * (IMAG_BINS[:-1] + IMAG_BINS[1:])
    ll = res.case.linking_length
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    fig.subplots_adjust(hspace=0)

    def hist(a, values, color, label, area):
        """Draw a step histogram of magnitudes on an axis (normalised, or per deg2 when `surface_density` is set)."""
        w = np.ones(len(values)) / area if surface_density else None
        a.hist(values, bins=bins, weights=w, density=not surface_density, color=color, lw=2, label=label,
               histtype='step')

    area = res.meta['area_deg2']
    hist(ax[0], res.lsst['i_cModel_mag'].dropna().values, COLORS['LSST'], 'LSST', area['LSST'])
    for tel in TELESCOPES:
        s, c = res.params[f'{tel.lower()}_to_i']
        x = res.trues[tel][{'Euclid': 'vis_mag', 'HST': 'F814_MAG'}[tel]].dropna().values
        hist(ax[0], s * x + c, COLORS[tel], f'{tel} to i', area[tel])
    ax[0].set_ylabel(r'Surface density [deg$^{-2}$]' if surface_density else 'Density count')
    ax[0].legend()
    ax[0].grid(alpha=.3)

    for tel in TELESCOPES:
        curves = [(res.groups[tel], '--', f'{tel} (FoF({ll:g}") + EOT)')]
        if also_fof_only:
            curves.append((res.fof_groups[tel], '-', f'{tel} (FoF({ll:g}"))'))
        for groups, ls, label in curves:
            f, e = unrecog_fraction(groups, res.lsst)
            ax[1].fill_between(binsc, f - e, f + e, alpha=.1, color=COLORS[tel])
            ax[1].errorbar(binsc, f, yerr=e, marker='.', ms=10, alpha=.8, ls=ls, color=COLORS[tel], label=label)
    ax[1].set_xlabel('LSST i-mag')
    ax[1].set_ylabel('Fraction of unrecognized blends')
    ax[1].set_xlim(18.5, 28.2)
    ax[1].set_ylim(bottom=0)
    ax[1].grid(alpha=.3)
    ax[1].legend(loc='upper left')
    res.save(fig, name or ('unrec_frac_and_mag_surface_density.pdf' if surface_density else 'unrec_frac_and_mag.pdf'))


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
    ok = np.isfinite(x)
    return fraction_per_bin(x[ok], (sub[f'blending_entropy_{tel}'].values > thr)[ok], bins)


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
    bins = np.arange(18.25, 29.1, 0.5) if bins is None else np.asarray(bins)
    centres = 0.5 * (bins[:-1] + bins[1:])
    f, e = _sb_fraction(res, tel, 'i_cModel_mag', bins, thr)
    idx, _, _ = _obj_table(res.groups[tel])
    n = np.histogram(res.lsst['i_cModel_mag'].values[idx], bins)[0]
    f, e = np.where(n >= min_count, f, np.nan), np.where(n >= min_count, e, np.nan)
    return centres, f, e, n


def sb_vs_mag(res, thresholds=(0.0, 0.2), name='high_sb_unrec.pdf'):
    """Figure of the fraction of high-S_b objects versus LSST i magnitude.

    Parameters
    ----------
    res : Results
    thresholds : sequence of float, default (0.0, 0.2)
        Entropy thresholds, one curve per survey and threshold.
    name : str, default 'high_sb_unrec.pdf'
        Output file name in the plot directory.
    """
    bins = np.arange(18.25, 29.1, 0.5)
    binsc = 0.5 * (bins[:-1] + bins[1:])
    fig = plt.figure(figsize=(9, 5))
    for tel in TELESCOPES:
        for k, thr in enumerate(thresholds):
            f, e = _sb_fraction(res, tel, 'i_cModel_mag', bins, thr)
            plt.fill_between(binsc, f - e, f + e, alpha=.01 if k == 0 else .3, color=COLORS[tel])
            plt.errorbar(binsc, f, yerr=e, ls='--' if k == 0 else '-', marker='o' if k == 0 else '*',
                         ms=4 if k == 0 else 9, alpha=.45 if k == 0 else 1, color=COLORS[tel],
                         label=rf'{tel} $S_b > {thr}$')
    plt.xlabel('LSST i-mag', fontsize=18)
    plt.ylabel('Fraction of high $S_b$ blends', fontsize=18)
    plt.xlim(18.5, 28.2)
    plt.grid(alpha=.2)
    plt.legend(loc='upper left')
    res.save(fig, name)


def sb_vs_blendedness(res, thresholds=(0.0, 0.2), mag_max=None, col='i_blendedness', name=None):
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
        Output file name; default ``blendedness_high_sb_unrec[_mag<mag_max>].pdf``.
    """
    edges = np.arange(-.25, 1., 0.05) + 0.125
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (lambda d: d['i_cModel_mag'] < mag_max) if mag_max is not None else None
    fig = plt.figure(figsize=(8, 5))
    for tel in TELESCOPES:
        for k, thr in enumerate(thresholds):
            f, e = _sb_fraction(res, tel, col, edges, thr, mask)
            plt.fill_between(centers, f - e, f + e, alpha=.05, color=COLORS[tel])
            plt.errorbar(centers, f, yerr=e, ls='--' if k == 0 else '-', marker='o' if k == 0 else '*',
                         ms=4 if k == 0 else 9, alpha=.45 if k == 0 else 1, color=COLORS[tel],
                         label=rf'{tel} $S_b > {thr}$')
    plt.axvline(HSC_BLENDEDNESS_CUT, ls='--', c='k', label='HSC cut')
    plt.xlabel(col)
    plt.ylabel('Fraction of high $S_b$ blends')
    plt.xlim(edges[0], edges[-1])
    plt.grid(alpha=.2)
    plt.legend(loc='upper right')
    if mag_max is not None:
        plt.title(f'i < {mag_max}')
    res.save(fig, name or (f'blendedness_high_sb_unrec{"_mag" + str(mag_max) if mag_max else ""}.pdf'))


def sb_hist2d_blendedness(res, col='i_blendedness', name='sb_vs_blendedness_hist.pdf'):
    """Figure of the S_b versus blendedness density.

    Only objects whose entropy is computed (groups other than 1-1) are shown; S_b = 0 (1-1 groups) and -1 (no overlap)
    objects are not part of this plot.

    Parameters
    ----------
    res : Results
    col : str, default 'i_blendedness'
        Blendedness column.
    name : str, default 'sb_vs_blendedness_hist.pdf'
        Output file name in the plot directory.
    """
    xb = np.arange(-.25, 1., 0.05) + 0.125
    yb = np.log(np.arange(1, 4, 0.25))
    data = {t: res.lsst[res.in_nonone(t)] for t in TELESCOPES}
    H = [np.histogram2d(data[t][col], data[t][f'blending_entropy_{t}'], bins=[xb, yb])[0] for t in TELESCOPES]
    norm = mpl.colors.LogNorm(vmin=1, vmax=max(1, max(h.max() for h in H)))
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(14, 5))
    fig.subplots_adjust(wspace=0)
    for a, t in zip(ax, TELESCOPES):
        _, _, _, im = a.hist2d(data[t][col].values, data[t][f'blending_entropy_{t}'].values, bins=(xb, yb), norm=norm)
        a.set_title(t)
        a.set_xlabel('LSST i-blendedness')
    ax[0].set_ylabel('$S_b$')
    fig.colorbar(im, ax=ax, pad=0, label='Counts')
    res.save(fig, name)


def sb_euclid_vs_hst(res, mag_max=23.5, name='sb_euclid_vs_hst.pdf'):
    """Figure comparing the Euclid and HST entropies of the same LSST objects.

    Parameters
    ----------
    res : Results
    mag_max : float, default 23.5
        Keep only objects brighter than this i magnitude.
    name : str, default 'sb_euclid_vs_hst.pdf'
        Output file name in the plot directory.

    Notes
    -----
    Only objects whose entropy is computed for both surveys are used; if there are none the plot is skipped with a
    message.
    """
    both = res.in_nonone('Euclid') & res.in_nonone('HST')  # S_b computed for both surveys
    d = res.lsst[both & (res.lsst['i_cModel_mag'] < mag_max).values]
    if len(d) == 0:  # e.g. no LSST object has an S_b from both surveys
        print('skipped', name, '(no objects with S_b from both Euclid and HST)')
        return
    hi = max(1.0, float(d[['blending_entropy_Euclid', 'blending_entropy_HST']].max().max()))  # S_b can exceed 1
    b = np.linspace(0, hi, int(round(hi / 0.1)) + 1)
    if np.histogram2d(d['blending_entropy_Euclid'], d['blending_entropy_HST'], bins=(b, b))[0].max() == 0:
        print('skipped', name, '(no objects with S_b >= 0 from both surveys)')
        return
    fig = plt.figure(figsize=(9, 6))
    plt.hist2d(d['blending_entropy_Euclid'], d['blending_entropy_HST'], bins=(b, b), norm=mpl.colors.LogNorm())
    plt.axline((0, 0), slope=1, color='red')
    plt.xlabel('Euclid $S_b$')
    plt.ylabel('HST $S_b$')
    plt.colorbar(pad=0.001, label='Count')
    res.save(fig, name)


def all_plots(case):
    """Make the standard figures of a case.

    Parameters
    ----------
    case : Case
        A case whose stages have all run. The figures are written to ``<out_dir>/plots``.
    """
    res = Results(case)
    detect_counts(res)
    detect_counts(res, size=7, drop_zero=True, cbar_on_top=True, name='detect_counts_cbar_on_top.pdf')
    unrecog_vs_mag(res)
    unrecog_vs_mag(res, surface_density=True)
    sb_vs_mag(res)
    sb_vs_blendedness(res)
    sb_vs_blendedness(res, mag_max=23.5)
    sb_hist2d_blendedness(res)
    sb_euclid_vs_hst(res)


# categorical slots 1-5 of the reference palette (light mode); adjacent-pair validated in that order
CASE_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
CASE_MARKERS = ['o', 's', '^', 'D', 'v']  # secondary encoding: identity never rests on colour alone
INK, INK_MUTED, GRID = '#0b0b0b', '#52514e', '#e3e2de'
MIN_COUNT = 30  # bins with fewer LSST objects are not drawn (fractions too noisy to read)


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
    Writes ``compare_sb_gt<thr>.pdf``, ``.png`` and ``.csv`` (the plotted values and the number of objects per bin).
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = labels or [c.name for c in cases]
    results = results or [Results(c) for c in cases]
    bins = np.arange(18.25, 29.1, 0.5)
    binsc = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.subplots_adjust(wspace=0.04)
    for a, tel in zip(ax, TELESCOPES):
        for k, (res, lab) in enumerate(zip(results, labels)):
            idx, _, _ = _obj_table(res.groups[tel])
            m = res.lsst['i_cModel_mag'].values[idx]
            ok = np.isfinite(m)
            f, e = fraction_per_bin(m[ok], (res.lsst[f'blending_entropy_{tel}'].values[idx] > thr)[ok], bins)
            n = np.histogram(m[ok], bins)[0]
            keep = n >= min_count
            a.errorbar(binsc[keep], f[keep], yerr=e[keep], color=CASE_COLORS[k], marker=CASE_MARKERS[k], ms=6, lw=2,
                       elinewidth=1, capsize=0, mec='white', mew=0.8, label=lab, zorder=3 + (k == 0))
            rows += [dict(survey=tel, case=lab, imag_center=c, n_objects=int(nn), frac_sb_gt=ff, err=ee)
                     for c, nn, ff, ee, kp in zip(binsc, n, f, e, keep) if kp]
        a.text(0.03, 0.96, tel, transform=a.transAxes, va='top', fontsize=15, fontweight='bold', color=INK)
        a.set_xlabel('LSST i-mag', fontsize=13, color=INK)
        a.set_xlim(18.5, 28.2)
        a.grid(color=GRID, lw=1, zorder=0)
        a.set_axisbelow(True)
        a.tick_params(colors=INK_MUTED, labelsize=11)
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            a.spines[sp].set_color(INK_MUTED)
    ax[0].set_ylim(bottom=0)
    ax[0].set_ylabel(rf'Fraction of blends with $S_b > {thr:g}$', fontsize=13, color=INK)
    ax[1].tick_params(left=False)
    leg = ax[0].legend(loc='upper left', bbox_to_anchor=(0.0, 0.90), frameon=False, fontsize=11, handlelength=2.2)
    for t in leg.get_texts():
        t.set_color(INK)
    tag = f'compare_sb_gt{str(thr).replace(".", "p")}'
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{tag}.{ext}'), bbox_inches='tight', dpi=200)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f'{tag}.csv'), index=False)
    plt.close(fig)
    print('saved', os.path.join(out_dir, f'{tag}.pdf'), '(+ .png, .csv)')


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
    Writes ``compare_unrecog.pdf`` and, through `compare_sb`, ``compare_sb_gt0p2.{pdf,png,csv}``.
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = labels or [c.name for c in cases]
    results = [Results(c) for c in cases]
    compare_sb(cases, out_dir, labels, results=results)
    binsc = 0.5 * (IMAG_BINS[:-1] + IMAG_BINS[1:])
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for a, tel in zip(ax, TELESCOPES):
        for k, (res, lab) in enumerate(zip(results, labels)):
            f, e = unrecog_fraction(res.groups[tel], res.lsst)
            a.errorbar(binsc, f, yerr=e, marker='.', color=cmap(k), label=lab)
        a.set_title(tel)
        a.set_xlabel('LSST i-mag')
        a.set_xlim(18.5, 28.2)
        a.grid(alpha=.3)
    ax[0].set_ylabel('Fraction of unrecognized blends')
    ax[0].legend()
    path = os.path.join(out_dir, 'compare_unrecog.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print('saved', path)


def euclid_fit_check(cases, out_dir, labels=None, xlim=(18, 29), ylim=(18, 29), nbins=110, min_median_count=50,
                     scale='log'):
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
    Writes ``euclid_vis_vs_i_fit.{pdf,png}`` (``euclid_vis_vs_i_fit_linear`` for the linear scale).
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = labels or [c.name for c in cases]
    n = len(cases)
    ncol = 3
    nrow = int(np.ceil((n + 1) / ncol))  # last cell holds the legend
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.9 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    xb, yb = np.linspace(*xlim, nbins), np.linspace(*ylim, nbins)
    xs = np.linspace(*xlim, 100)
    for a, case, lab in zip(axes, cases, labels):
        params = json.load(open(case.path('catalogs', 'fit_params.json', makedirs=False)))
        groups = _load(case.path('groups', 'E_no_overlap.pkl', makedirs=False))['data']
        g11 = [g for g in groups if len(g[0]) == 1 and len(g[1]) == 1]
        ti = np.array([g[0][0] for g in g11], dtype=int)
        li = np.array([g[1][0] for g in g11], dtype=int)
        vis = pd.read_parquet(case.path('catalogs', 'euclid.parquet', makedirs=False), columns=['vis_mag'])['vis_mag'].values[ti]
        imag = pd.read_parquet(case.path('catalogs', 'lsst.parquet', makedirs=False), columns=['i_cModel_mag'])['i_cModel_mag'].values[li]
        ok = np.isfinite(vis) & np.isfinite(imag)
        vis, imag = vis[ok].astype(np.float64), imag[ok].astype(np.float64)  # float32 polyfit is inaccurate
        _, _, _, im = a.hist2d(vis, imag, bins=[xb, yb], norm=mpl.colors.LogNorm() if scale == 'log' else mpl.colors.Normalize(vmin=0),
                              cmap='Blues', cmin=1)
        slope, icpt = params['euclid_to_i']
        refit = np.polyfit(vis, imag, 1)  # exactly the arrays drawn in the hist2d below
        print(f'{case.name}: saved fit ({slope:.4f}, {icpt:.3f}) vs refit on the plotted arrays ({refit[0]:.4f}, {refit[1]:.3f})',
              'OK' if np.allclose(refit, [slope, icpt], atol=2e-3) else '*** MISMATCH ***')
        a.plot(xs, slope * xs + icpt, color=CASE_COLORS[1], lw=2.5, label=r'saved fit: $i = a\,\mathrm{VIS} + b$', zorder=4)
        a.plot(xs, xs, color=INK_MUTED, lw=1, ls='--', label='i = VIS', zorder=3)
        mb = np.arange(xlim[0], xlim[1] + 0.01, 0.5)
        mc = 0.5 * (mb[:-1] + mb[1:])
        cnt = np.histogram(vis, mb)[0]
        med = np.array([np.median(imag[(vis >= lo) & (vis < hi)]) if c >= min_median_count else np.nan
                        for lo, hi, c in zip(mb[:-1], mb[1:], cnt)])
        a.plot(mc, med, color=INK, marker='o', ms=4, lw=1.2, mfc='white', label='binned median of i', zorder=5)
        mean = np.array([np.mean(imag[(vis >= lo) & (vis < hi)]) if c >= min_median_count else np.nan
                         for lo, hi, c in zip(mb[:-1], mb[1:], cnt)])
        a.plot(mc, mean, color='#8a2be2', marker='s', ms=3.5, lw=1.2, ls=':', label='binned mean of i', zorder=5)
        a.text(0.03, 0.97, f'{lab}\n{len(vis):,} 1-1 pairs\n' + rf'$a={slope:.3f},\ b={icpt:.2f}$', transform=a.transAxes,
               va='top', fontsize=11, color=INK, bbox=dict(fc='white', ec='none', alpha=0.85, pad=2))
        a.set_xlim(*xlim)
        a.set_ylim(*ylim)
        a.grid(color=GRID, lw=0.8, zorder=0)
        a.set_axisbelow(True)
        a.tick_params(colors=INK_MUTED, labelsize=10)
    handles, hl = axes[0].get_legend_handles_labels()
    for a in axes[n:]:
        a.axis('off')
    axes[n].legend(handles, hl, loc='center', frameon=False, fontsize=12)
    for a in axes[:n]:
        if a.get_subplotspec().is_last_row() or a is axes[n - 1] or a.get_subplotspec().rowspan.stop == nrow or True:
            a.set_xlabel('Euclid VIS mag', fontsize=12, color=INK)
    for a in axes[::ncol]:
        a.set_ylabel('LSST i mag (cModel)', fontsize=12, color=INK)
    fig.subplots_adjust(hspace=0.16, wspace=0.05)
    fig.colorbar(im, ax=axes.tolist(), label='pairs per bin', fraction=0.02, pad=0.02)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'euclid_vis_vs_i_fit{"" if scale == "log" else "_linear"}.{ext}'),
                    bbox_inches='tight', dpi=170)
    plt.close(fig)
    print('saved', os.path.join(out_dir, f'euclid_vis_vs_i_fit{"" if scale == "log" else "_linear"}.pdf'), '(+ .png)')
