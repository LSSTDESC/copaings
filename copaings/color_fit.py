"""Analytic fits of the true-survey magnitudes from the LSST bands (as in ``dp2_cut_circle.ipynb``).

* Euclid VIS: piecewise linear in r with a break at r = 22 (continuous there) plus g-r, r-i and i-z colour terms,
  ``VIS = a0 + c_r r + c_gr (g-r) + c_ri (r-i) + c_iz (i-z)`` with ``(a0, c_r)`` = low / high below / above the break.
* HST F814W: straight line in LSST i, ``F814W = a i + b``, weighted by 1 / F814W error (errors below 1 mag).

Both are fitted on the 1-1 groups after the ellipse overlap test. The results are stored in the ``color_fit`` entry
of ``catalogs/fit_params.json`` with a label of the data they come from (survey release, LSST release), so they can be
fetched back later (`load_color_fit`) instead of being refitted.
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

R_BREAK = 22.0  # r magnitude of the slope change of the VIS fit


def vis_model(inputs, a0_low, cr_low, cr_high, c_gr, c_ri, c_iz):
    """VIS predicted from LSST ``inputs = (r, g-r, r-i, i-z)``; continuous at `R_BREAK`."""
    r, gr, ri, iz = inputs
    a0_high = a0_low + (cr_low - cr_high) * R_BREAK
    with np.errstate(invalid='ignore'):  # a missing band gives NaN, as wanted
        return np.where(r <= R_BREAK, a0_low + cr_low * r, a0_high + cr_high * r) + c_gr * gr + c_ri * ri + c_iz * iz


def _vis_inputs(lsst):
    """``(r, g-r, r-i, i-z)`` of LSST catalog rows, float64."""
    g = lsst['g_cModel_mag'].values.astype(float)
    r = lsst['r_cModel_mag'].values.astype(float)
    i = lsst['i_cModel_mag'].values.astype(float)
    z = lsst['z_cModel_mag'].values.astype(float)
    return np.vstack([r, g - r, r - i, i - z])


def pairs_11(groups):
    """True and LSST row indices of the 1-1 groups."""
    true_rows = []
    lsst_rows = []
    for g in groups:
        if len(g[0]) == 1 and len(g[1]) == 1:
            true_rows.append(g[0][0])
            lsst_rows.append(g[1][0])
    return np.array(true_rows, int), np.array(lsst_rows, int)


def _scatter(resid):
    """Standard deviation and normalised median absolute deviation of fit residuals."""
    std = np.std(resid)
    mad = np.median(np.abs(resid - np.median(resid)))
    nmad = 1.4826 * mad
    return float(std), float(nmad)


def fit_vis(lsst, euclid, groups):
    """Fit the piecewise VIS model on the 1-1 groups.

    Returns
    -------
    dict
        ``params`` (``a0_low``, ``a0_high``, ``cr_low``, ``cr_high``, ``c_gr``, ``c_ri``, ``c_iz``), ``errors`` (1
        sigma; ``a0_high`` is derived from the continuity), ``r_break``, ``n_pairs``, ``resid_std``, ``resid_nmad``.
    """
    ti, li = pairs_11(groups)
    y = euclid['vis_mag'].values[ti].astype(float)
    X = _vis_inputs(lsst.iloc[li])
    ok = np.isfinite(y) & np.isfinite(X).all(axis=0)
    p, cov = curve_fit(vis_model, X[:, ok], y[ok])
    a0_low, cr_low, cr_high, c_gr, c_ri, c_iz = p
    e = np.sqrt(np.diag(cov))

    # a0_high is not fitted: it comes from the continuity at the break
    a0_high = a0_low + (cr_low - cr_high) * R_BREAK
    jac = np.array([1, R_BREAK, -R_BREAK, 0, 0, 0])  # d a0_high / d (a0_low, cr_low, cr_high, c_gr, c_ri, c_iz)
    a0_high_err = np.sqrt(jac @ cov @ jac)

    params = {'a0_low': float(a0_low),
              'a0_high': float(a0_high),
              'cr_low': float(cr_low),
              'cr_high': float(cr_high),
              'c_gr': float(c_gr),
              'c_ri': float(c_ri),
              'c_iz': float(c_iz)}
    errors = {'a0_low': float(e[0]),
              'a0_high': float(a0_high_err),
              'cr_low': float(e[1]),
              'cr_high': float(e[2]),
              'c_gr': float(e[3]),
              'c_ri': float(e[4]),
              'c_iz': float(e[5])}

    resid = y[ok] - vis_model(X[:, ok], *p)
    std, nmad = _scatter(resid)
    return {'model': 'vis_piecewise_r_griz',
            'r_break': R_BREAK,
            'params': params,
            'errors': errors,
            'n_pairs': int(ok.sum()),
            'resid_std': std,
            'resid_nmad': nmad}


def fit_f814(lsst, hst, groups, max_err=1.0):
    """Fit ``F814W = a i + b`` on the 1-1 groups, weighted by 1 / F814W error (errors below `max_err`).

    Returns
    -------
    dict
        ``params`` (``a``, ``b``), ``errors``, ``n_pairs``, ``resid_std``, ``resid_nmad``.
    """
    ti, li = pairs_11(groups)
    y = hst['F814_MAG'].values[ti].astype(float)
    err = hst['F814_MAGERR'].values[ti].astype(float)
    x = lsst['i_cModel_mag'].values[li].astype(float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(err) & (err > 0) & (err < max_err)
    p, cov = np.polyfit(x[ok], y[ok], deg=1, w=1 / err[ok], cov='unscaled')
    e = np.sqrt(np.diag(cov))
    resid = y[ok] - np.polyval(p, x[ok])
    std, nmad = _scatter(resid)
    return {'model': 'f814_linear_i',
            'params': {'a': float(p[0]), 'b': float(p[1])},
            'errors': {'a': float(e[0]), 'b': float(e[1])},
            'n_pairs': int(ok.sum()),
            'resid_std': std,
            'resid_nmad': nmad}


def predict(fit, lsst):
    """Survey magnitude predicted by a stored fit (`fit_vis` or `fit_f814` output) for LSST catalog rows."""
    p = fit['params']
    if fit['model'] == 'vis_piecewise_r_griz':
        return vis_model(_vis_inputs(lsst), p['a0_low'], p['cr_low'], p['cr_high'], p['c_gr'], p['c_ri'], p['c_iz'])
    return p['a'] * lsst['i_cModel_mag'].values.astype(float) + p['b']


def data_label(case, tel):
    """Which data a fit comes from, e.g. ``{'survey': 'Euclid DR1 VIS', 'lsst': 'LSST DP2'}``."""
    from .catalogs import euclid_release
    lsst = 'LSST'
    text = f'{case.collections} {case.repo}'.upper()
    for release in ('DP2', 'DP1'):
        if release in text:
            lsst = f'LSST {release}'
            break
    if tel == 'Euclid':
        release = euclid_release(case)
        survey = f'Euclid {release} VIS' if release else 'Euclid VIS'
    else:
        survey = {'hlf_goodss': 'HST HLF GOODS-S F814W', 'cosmos_acs': 'HST COSMOS ACS F814W'}[case.hst_catalog]
    return {'survey': survey, 'lsst': lsst, 'linking_length': case.linking_length}


def fit_case(case, tel, lsst, true, groups):
    """Fit the magnitude of survey `tel` for a case and label it with the data it comes from."""
    if tel == 'Euclid':
        fit = fit_vis(lsst, true, groups)
    else:
        fit = fit_f814(lsst, true, groups)
    result = data_label(case, tel)
    result.update(fit)
    return result


def load_color_fit(case, tel, refit=True):
    """Stored fit of survey `tel` for a case (``color_fit`` of ``fit_params.json``), refitted if missing.

    Parameters
    ----------
    case : Case
    tel : {'Euclid', 'HST'}
    refit : bool, default True
        When nothing is stored, fit from the saved catalogs and 1-1 groups (not written back; run the `colorfit`
        stage to store it).

    Returns
    -------
    dict or None
    """
    path = case.path('catalogs', 'fit_params.json', makedirs=False)
    if os.path.exists(path):
        with open(path) as f:
            stored = json.load(f).get('color_fit', {})
        if tel in stored:
            return stored[tel]
    if not refit:
        return None
    from . import stages
    lsst, trues = stages.load_cats(case)
    return fit_case(case, tel, lsst, trues[tel], stages.load_groups(case, tel))


def format_fit(fit):
    """Text of a fit in the notebook style (one parameter per line)."""
    p = fit['params']
    if fit['model'] == 'vis_piecewise_r_griz':
        rows = [('a_0^{low}', p['a0_low']), ('a_0^{high}', p['a0_high']), ('c_r^{low}', p['cr_low']),
                ('c_r^{high}', p['cr_high']), ('c_{g-r}', p['c_gr']), ('c_{r-i}', p['c_ri']), ('c_{i-z}', p['c_iz'])]
    else:
        rows = [('a', p['a']), ('b', p['b'])]
    lines = [f"{fit['survey']} fit ({fit['lsst']}, {fit['n_pairs']:,} 1-1 pairs, residual NMAD {fit['resid_nmad']:.3f}):"]
    for name, value in rows:
        lines.append(f'- ${name} = {value:.4g}$')
    return '\n'.join(lines)
