"""Pipeline stages. Each stage reads/writes files under case.out_dir, so stages can be run as separate slurm jobs.

Layout:
  catalogs/{lsst,euclid,hst}.parquet, meta.json       <- prep
  catalogs/lsst_fit.parquet, fit_params.json          <- magfit
  groups/{E,H}_fof.pkl                                <- fof   (FoF groups with >=1 true and >=1 obs)
  groups/{E,H}_eot_part_XXXX.pkl                      <- eot   (array job)
  groups/{E,H}_no_overlap.pkl                         <- eot_merge
  sb/{E,H}_part_XXXX.parquet                          <- sb    (array job)
  lsst_sb.parquet                                     <- sb_merge
  plots/                                              <- plots
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import catalogs, color_fit, deps
from .config import TRUE_MAG_COL, TRUE_MAG_COL_IN_LSST, TRUE_MAGERR_COL, initial


# ----------------------------------------------------------------------------- io helpers
def read_cat(case, name):
    """Read one catalog parquet file of the case.

    Parameters
    ----------
    case : Case
    name : str
        File stem in ``catalogs/``, e.g. ``'lsst'``, ``'euclid'``, ``'hst'``, ``'lsst_fit'``.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_parquet(case.path('catalogs', f'{name}.parquet'))


def load_cats(case, lsst_name='lsst'):
    """Read the LSST catalog and the two true catalogs of the case.

    Parameters
    ----------
    case : Case
    lsst_name : str, default 'lsst'
        LSST catalog to read: ``'lsst'`` (output of `prep`) or ``'lsst_fit'`` (also has the predicted VIS / F814
        magnitudes, output of `magfit`).

    Returns
    -------
    lsst : pandas.DataFrame
    trues : dict of pandas.DataFrame
        ``{survey: catalog}`` for the surveys of `case.telescopes`.
    """
    lsst = read_cat(case, lsst_name)
    return lsst, {t: read_cat(case, t.lower()) for t in case.telescopes}


def dump(obj, path):
    """Pickle an object atomically.

    The data are written to a temporary file that replaces `path` only when complete, so a killed job never leaves
    a half-written file.

    Parameters
    ----------
    obj : object
    path : str
    """
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)  # atomic: a killed job never leaves a half-written file


def load(path):
    """Load a pickled object.

    Parameters
    ----------
    path : str

    Returns
    -------
    object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def chunk(seq, task_id, n_tasks):
    """Take one strided slice of a sequence, for one task of a slurm array.

    A strided split balances the work: groups are stored in catalog order, so contiguous slices would be unbalanced.

    Parameters
    ----------
    seq : sequence
    task_id : int
        Slice index, ``0 <= task_id < n_tasks``.
    n_tasks : int
        Number of slices.

    Returns
    -------
    sequence
        ``seq[task_id::n_tasks]``.
    """
    return seq[task_id::n_tasks]


def _quiet():
    """Silence the RuntimeWarnings (NaN moments, divisions by zero) that would otherwise be printed for many groups."""
    warnings.filterwarnings('ignore', category=RuntimeWarning)


# ----------------------------------------------------------------------------- prep
def prep(case):
    """Stage `prep`: build the three catalogs and write them to disk.

    Reads the LSST object table (butler), the Euclid tiles and the HST catalog (see `catalogs.assemble`).

    Parameters
    ----------
    case : Case

    Notes
    -----
    Writes ``catalogs/lsst.parquet`` and ``catalogs/<survey>.parquet`` for each survey of `telescopes`, ``catalogs/meta.json`` and ``catalogs/lsst_all_radec.parquet`` (LSST
    positions before the sky cuts, used by the sky maps). Group indices in every later stage are row
    positions in these files.
    """
    lsst, trues, meta, lsst_all = catalogs.assemble(case)
    lsst_all.to_parquet(case.path('catalogs', 'lsst_all_radec.parquet'))  # whole footprint, for the sky maps
    lsst.to_parquet(case.path('catalogs', 'lsst.parquet'))
    for tel, df in trues.items():
        df.to_parquet(case.path('catalogs', f'{tel.lower()}.parquet'))
    catalogs.save_meta(case.path('catalogs', 'meta.json'), meta)
    print(json.dumps(meta, indent=2))


def lsst_footprint(case):
    """Stage `footprint`: write ``catalogs/lsst_all_radec.parquet`` (all LSST galaxy positions, before the sky cuts).

    `prep` already writes it; this stage makes it for runs prepared before, reading only ra, dec and i-extendedness
    from the butler (needs the full LSST stack).

    Parameters
    ----------
    case : Case
    """
    df = catalogs.lsst_all_radec(case)
    df.to_parquet(case.path('catalogs', 'lsst_all_radec.parquet'))
    print(f'{len(df):,} LSST positions written')


# ----------------------------------------------------------------------------- FoF
def fof(case, tel):
    """Stage `fof`: friends-of-friends matching of a true catalog with the LSST catalog.

    Parameters
    ----------
    case : Case
        Uses `linking_length`.
    tel : {'Euclid', 'HST'}

    Notes
    -----
    Writes ``groups/<E|H>_fof.pkl``, a list of ``[true_indices, lsst_indices]`` that keeps only groups with at least one
    object of each catalog, and ``groups/<E|H>_fof_stats.json`` (number of groups and how many span several tracts).
    """
    deps.add_friendly_to_path()
    from friendly.matching import FoF_matching

    lsst, trues = load_cats(case)
    groups = FoF_matching(trues[tel], lsst, case.linking_length)
    groups = [g for g in groups if len(g[0]) > 0 and len(g[1]) > 0]  # no true-only / obs-only groups
    dump(groups, case.path('groups', f'{initial(tel)}_fof.pkl'))

    # how many groups straddle a tract boundary? (matters if x, y are per-tract, see Case.common_frame_xy)
    tract = lsst['tract_id'].values
    n_multi = sum(len(set(tract[g[1]])) > 1 for g in groups)
    stats = {'n_groups': len(groups), 'n_multi_tract_groups': int(n_multi), 'common_frame_xy': case.common_frame_xy}
    with open(case.path('groups', f'{initial(tel)}_fof_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'{tel}: {stats}')


# ----------------------------------------------------------------------------- ellipse overlap test
def eot(case, tel, task_id, n_tasks):
    """Stage `eot`: ellipse overlap test on one slice of the FoF groups.

    Each FoF group is split into the connected components of the graph whose edges join intersecting ellipses (LSST
    ellipses from the second moments, true-catalog ellipses from their sizes broadened by the PSF). Components that
    lack a true or an LSST object are dropped.

    Parameters
    ----------
    case : Case
        Uses `eot_dense_max`.
    tel : {'Euclid', 'HST'}
    task_id : int
        Slice index (``SLURM_ARRAY_TASK_ID`` in an array job).
    n_tasks : int
        Number of slices.

    Notes
    -----
    Writes ``groups/<E|H>_eot_part_<task_id>.pkl``.
    """
    from .eot_fast import EllipseCache, split_group

    _quiet()
    lsst, trues = load_cats(case)
    groups = chunk(load(case.path('groups', f'{initial(tel)}_fof.pkl')), task_id, n_tasks)

    cache = EllipseCache(lsst, trues[tel], pixel_scale=case.lsst_pixel_scale)  # LSST ellipses, computed once
    out = []
    for grp in tqdm(groups, desc=f'EOT {tel} {task_id}/{n_tasks}', file=sys.stderr):
        for g in split_group(cache, grp, dense_max=case.eot_dense_max):
            if len(g[0]) > 0 and len(g[1]) > 0:  # keep groups with at least one true and one LSST object
                out.append(g)
    dump(out, case.path('groups', f'{initial(tel)}_eot_part_{task_id:04d}.pkl'))


def eot_merge(case, tel, n_tasks):
    """Stage `eot_merge`: concatenate the parts written by `eot`.

    Parameters
    ----------
    case : Case
    tel : {'Euclid', 'HST'}
    n_tasks : int
        Number of parts to expect.

    Raises
    ------
    FileNotFoundError
        If a part is missing.

    Notes
    -----
    Writes ``groups/<E|H>_no_overlap.pkl``: ``{'data': list of [true_indices, lsst_indices], 'metadata': dict}``.
    """
    parts = [case.path('groups', f'{initial(tel)}_eot_part_{i:04d}.pkl') for i in range(n_tasks)]
    missing = [p for p in parts if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f'{len(missing)} EOT parts missing, e.g. {missing[0]}')
    merged = []
    for p in parts:
        merged.extend(load(p))
    meta = {'linking_length': case.linking_length, 'tel_name': tel, 'case': case.name}
    # same format as the pickles saved from the notebook: {"data", "metadata"}
    dump({'data': merged, 'metadata': meta}, case.path('groups', f'{initial(tel)}_no_overlap.pkl'))
    print(f'{tel}: {len(merged):,} no-overlap groups')


def load_groups(case, tel):
    """Load the merged groups of the ellipse overlap test.

    Parameters
    ----------
    case : Case
    tel : {'Euclid', 'HST'}

    Returns
    -------
    list
        Groups ``[true_indices, lsst_indices]``, indices are row positions in the `prep` catalogs.
    """
    return load(case.path('groups', f'{initial(tel)}_no_overlap.pkl'))['data']


# ----------------------------------------------------------------------------- magnitude fits
def _f64(a):
    """Convert to a float64 array.

    The fits must run in float64: the Euclid columns are float32 and ``numpy.polyfit`` on float32 gave a wrong line
    (slope 0.47 instead of 1.03 for the VIS to i fit).

    Parameters
    ----------
    a : array_like

    Returns
    -------
    numpy.ndarray
    """
    return np.asarray(a, dtype=np.float64)


def _pairs_11(groups):
    """Select the groups with exactly one true and one LSST object.

    Parameters
    ----------
    groups : list
        Groups ``[true_indices, lsst_indices]``.

    Returns
    -------
    true_idx, lsst_idx : numpy.ndarray of int
        Catalog rows of the true and of the LSST object of each 1-1 group, sorted by LSST row. The order matters:
        the magnitude trees hold out a random part of their training set, so sorting makes `magfit` independent of
        the order of the groups.
    """
    true_idx = []
    lsst_idx = []
    for g in groups:
        if len(g[0]) == 1 and len(g[1]) == 1:
            true_idx.append(g[0][0])
            lsst_idx.append(g[1][0])
    true_idx = np.array(true_idx, dtype=int)
    lsst_idx = np.array(lsst_idx, dtype=int)

    order = np.argsort(lsst_idx)
    return true_idx[order], lsst_idx[order]


BANDS = 'ugrizy'
TREE_PARAMS = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True, random_state=0)


def _tree_features(lsst, pixel_scale):
    """Features of the VIS / F814 tree models.

    Parameters
    ----------
    lsst : pandas.DataFrame
        LSST catalog with ``{b}_cModel_mag`` for u, g, r, i, z, y, the i-band moments, ``psf_fwhm_i`` and ``ellipticity``.
    pixel_scale : float
        LSST pixel scale [arcsec/pixel].

    Returns
    -------
    X : numpy.ndarray
        Shape (N, 10). Missing or non-finite bands give NaN, which the trees handle natively.
    names : list of str
        ``['i_mag', 'u-i', 'g-i', 'r-i', 'z-i', 'y-i', 'size_arcsec', 'psf_sigma_arcsec', 'size_over_psf',
        'ellipticity']``. The size is the observed i-band sigma from the second moments.
    """
    M = {}
    for b in BANDS:
        m = _f64(lsst[f'{b}_cModel_mag'].values).copy()
        m[~np.isfinite(m)] = np.nan
        M[b] = m
    trace = _f64(lsst['Ixx_pixel_i'].values) + _f64(lsst['Iyy_pixel_i'].values)
    trace[~(trace > 0)] = np.nan
    size = np.sqrt(trace / 2) * pixel_scale  # sigma, arcsec
    psf = _f64(lsst['psf_fwhm_i'].values) / 2.3548  # sigma, arcsec
    names = ['i_mag'] + [f'{b}-i' for b in 'ugrzy'] + ['size_arcsec', 'psf_sigma_arcsec', 'size_over_psf', 'ellipticity']
    X = np.column_stack([M['i']] + [M[b] - M['i'] for b in 'ugrzy'] + [size, psf, size / psf, _f64(lsst['ellipticity'].values)])
    return X, names


def _usable_columns(X):
    """Select the feature columns a tree can use.

    A band absent from the whole catalog (the worst-seeing rerun has no u band) or a constant column cannot be binned
    by scikit-learn and carries no information.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix, shape (N, k).

    Returns
    -------
    list of int
        Indices of the columns with at least two distinct finite values.
    """
    return [j for j in range(X.shape[1]) if len(np.unique(X[np.isfinite(X[:, j]), j])) >= 2]


def _fit_tree(X, y, ra, dec, block):
    """Train a gradient-boosted tree, with held-out metrics.

    Metrics are measured on a model trained without 20% of the sky blocks, so that neighbouring objects do not leak
    between training and test sets. The returned model is then refit on all the pairs.

    Parameters
    ----------
    X : numpy.ndarray
        Features, shape (n, k).
    y : numpy.ndarray
        Target magnitude, shape (n,).
    ra, dec : numpy.ndarray
        Positions [deg], used to define the sky blocks.
    block : float
        Size of a sky block [deg].

    Returns
    -------
    model : sklearn.ensemble.HistGradientBoostingRegressor
    metrics : dict
        ``n_train``, ``n_test``, ``bias`` (median of prediction minus truth), ``mad_scatter`` (1.4826 * MAD), ``std``,
        ``frac_abs_gt_1`` (fraction of errors above 1 mag). Empty if fewer than 200 pairs fall in the training or
        in the test set.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    held = (np.floor(ra / block).astype(int) * 1000 + np.floor(dec / block).astype(int)) % 5 == 0
    metrics = {}
    if held.sum() >= 200 and (~held).sum() >= 200:
        m = HistGradientBoostingRegressor(**TREE_PARAMS).fit(X[~held], y[~held])
        r = m.predict(X[held]) - y[held]
        metrics = {'n_train': int((~held).sum()), 'n_test': int(held.sum()), 'bias': float(np.median(r)),
                   'mad_scatter': float(1.4826 * np.median(np.abs(r - np.median(r)))), 'std': float(r.std()),
                   'frac_abs_gt_1': float(np.mean(np.abs(r) > 1))}
    return HistGradientBoostingRegressor(**TREE_PARAMS).fit(X, y), metrics


def magfit(case):
    """Stage `magfit`: magnitudes of all LSST objects in the Euclid VIS and HST F814 bands (surveys of the case).

    For each survey a gradient-boosted tree is trained on the 1-1 groups to predict the survey magnitude from the LSST
    ugrizy colours and size information (see `_tree_features`); it is then applied to every LSST object. Objects
    without an i-band magnitude get NaN.

    Parameters
    ----------
    case : Case

    Notes
    -----
    Writes ``catalogs/lsst_fit.parquet`` (the LSST catalog plus ``vis_mag`` and ``F814_MAG``),
    ``catalogs/{euclid,hst}_tree.pkl`` (model and feature names) and ``catalogs/fit_params.json`` (features used,
    held-out metrics, and the linear ``euclid_to_i`` / ``hst_to_i`` relations used only for magnitude histograms in
    the plots).
    """
    os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    lsst, trues = load_cats(case)
    X_all, names = _tree_features(lsst, case.lsst_pixel_scale)
    params = {'tree_features': names, 'tree_params': TREE_PARAMS, 'tree_metrics': {}}

    for tel in case.telescopes:
        ti, li = _pairs_11(load_groups(case, tel))
        y = _f64(trues[tel][TRUE_MAG_COL[tel]].values[ti])
        ok = np.isfinite(y)
        li, y = li[ok], y[ok]
        block = 0.5 if len(y) > 200_000 else 0.03  # HST sits in one small field
        cols = _usable_columns(X_all[li])
        if len(cols) < len(names):
            dropped = []
            for j in range(len(names)):
                if j not in cols:
                    dropped.append(names[j])
            print(f'{tel}: features dropped (all-NaN or constant in the training pairs): {dropped}')
        X_train = X_all[li][:, cols]
        ra_train = _f64(lsst['ra'].values[li])
        dec_train = _f64(lsst['dec'].values[li])
        model, metrics = _fit_tree(X_train, y, ra_train, dec_train, block)
        col = TRUE_MAG_COL_IN_LSST[tel]
        pred = model.predict(X_all[:, cols])
        pred[~np.isfinite(X_all[:, 0])] = np.nan  # no i-band magnitude, no prediction
        lsst[col] = pred
        with open(case.path('catalogs', f'{tel.lower()}_tree.pkl'), 'wb') as f:
            pickle.dump({'model': model, 'features': [names[j] for j in cols]}, f)
        params['tree_metrics'][tel] = metrics
        params.setdefault('tree_features_used', {})[tel] = [names[j] for j in cols]
        print(tel, metrics, flush=True)

    # --- true-mag -> LSST i linear relations (used only for the magnitude histograms in the plots)
    for tel in case.telescopes:
        ti, li = _pairs_11(load_groups(case, tel))
        x = _f64(trues[tel][TRUE_MAG_COL[tel]].values[ti])
        y = _f64(lsst['i_cModel_mag'].values[li])
        ok = np.isfinite(x) & np.isfinite(y)
        params[f'{tel.lower()}_to_i'] = np.polyfit(x[ok], y[ok], deg=1).tolist()  # [slope, intercept]

    # --- analytic fits of the survey magnitudes from the LSST bands (stored, see color_fit.load_color_fit)
    params['color_fit'] = {tel: color_fit.fit_case(case, tel, lsst, trues[tel], load_groups(case, tel))
                           for tel in case.telescopes}

    lsst.to_parquet(case.path('catalogs', 'lsst_fit.parquet'))
    with open(case.path('catalogs', 'fit_params.json'), 'w') as f:
        json.dump(params, f, indent=2)


def colorfit(case):
    """Stage `colorfit`: fit the survey magnitudes from the LSST bands and store them in ``fit_params.json``.

    `magfit` already does it; this stage adds (or refreshes) the ``color_fit`` entry of runs made before, from the
    saved catalogs and 1-1 groups only. The fits are printed in the notebook style.

    Parameters
    ----------
    case : Case
    """
    path = case.path('catalogs', 'fit_params.json')
    params = {}
    if os.path.exists(path):
        with open(path) as f:
            params = json.load(f)
    lsst, trues = load_cats(case)
    params['color_fit'] = {tel: color_fit.fit_case(case, tel, lsst, trues[tel], load_groups(case, tel))
                           for tel in case.telescopes}
    with open(path, 'w') as f:
        json.dump(params, f, indent=2)
    for fit in params['color_fit'].values():
        print(color_fit.format_fit(fit), end='\n\n')


# ----------------------------------------------------------------------------- blending entropy
def _groups_without_11(groups):
    """Remove the 1-1 groups.

    Parameters
    ----------
    groups : list
        Groups ``[true_indices, lsst_indices]``.

    Returns
    -------
    list
        The groups that do not have exactly one true and one LSST object.
    """
    return [g for g in groups if not (len(g[0]) == 1 and len(g[1]) == 1)]


def sb(case, tel, task_id, n_tasks):
    """Stage `sb`: blending entropy of the LSST objects of one slice of the groups.

    1-1 groups are skipped (their entropy is fixed in `sb_merge`).

    Parameters
    ----------
    case : Case
    tel : {'Euclid', 'HST'}
    task_id : int
        Slice index (``SLURM_ARRAY_TASK_ID`` in an array job).
    n_tasks : int
        Number of slices.

    Notes
    -----
    Writes ``sb/<E|H>_part_<task_id>.parquet``, indexed by LSST row, with columns ``blending_entropy_<tel>``,
    ``purity_<tel>`` and ``sb_flag_<tel>`` (see `sb_merge`), for every LSST object of the slice's groups.
    The blending entropy is described, used and tested in arXiv:2609.12158 (https://arxiv.org/abs/2609.12158).
    """
    from .sb_core import compute_Sb

    _quiet()
    lsst, trues = load_cats(case, 'lsst_fit')
    lsst['mag_i_cModel'] = lsst[TRUE_MAG_COL_IN_LSST[tel]]  # friendly compares magnitudes in the band of the true catalog
    col, fcol = f'blending_entropy_{tel}', f'sb_flag_{tel}'
    lsst[col] = np.nan
    lsst['purity'] = np.nan
    lsst[fcol] = None

    groups = chunk(_groups_without_11(load_groups(case, tel)), task_id, n_tasks)
    # fills `col`, `purity` and `fcol` in place
    compute_Sb(groups, trues[tel], lsst, Sb_col_name_in_obs_cat=col, flag_col_name_in_obs_cat=fcol)

    # keep every object of the groups (the flag is set for all of them, the entropy can be NaN)
    part = lsst.loc[lsst[fcol].notna(), [col, 'purity', fcol]].rename(columns={'purity': f'purity_{tel}'})
    part.to_parquet(case.path('sb', f'{initial(tel)}_part_{task_id:04d}.parquet'))


def sb_merge(case, n_tasks):
    """Stage `sb_merge`: assemble the entropy and purity of every LSST object.

    ``blending_entropy_<tel>`` is a number only where the entropy is defined, NaN elsewhere. The column
    ``sb_flag_<tel>`` says which case each object is in:

    ==========================  =========  ==========================================================================
    ``sb_flag_<tel>``           S_b        meaning
    ==========================  =========  ==========================================================================
    ``'computed'``              ``>= 0``   entropy computed by `sb` (group other than 1-1)
    ``'one_to_one'``            ``0``      1-1 group after the overlap test, a single candidate (purity 1)
    ``'out_footprint'``         NaN        outside the footprint of the survey (``in_<survey>_footprint`` False)
    ``'no_fof_match'``          NaN        in the footprint, no galaxy of the survey within the FoF linking length
    ``'no_ellipse_overlap'``    NaN        in a FoF group, but its ellipse overlaps no galaxy (dropped by `eot`)
    ``'obj_only_neighbours'``   NaN        in a group, but overlaps only other LSST objects, no galaxy
    ``'zero_overlap'``          NaN        in a group, touches galaxies but every overlap area is 0
    ``'failed'``                NaN        in a group, computation failed (e.g. missing magnitude); reported as a warning
    ==========================  =========  ==========================================================================

    ``purity_<tel>`` is NaN for the objects in no group (the first three NaN cases).

    Parts written before the flag existed are read too: their -1 values become NaN with ``'obj_only_neighbours'``.

    Parameters
    ----------
    case : Case
    n_tasks : int
        Number of `sb` parts to expect for each survey.

    Raises
    ------
    FileNotFoundError
        If a part is missing.
    RuntimeError
        If an LSST object appears in several groups.

    Notes
    -----
    Writes ``lsst_sb.parquet`` (the LSST catalog with the entropy and purity columns) and ``sb/sb_merge_stats.json``.
    The blending entropy is described, used and tested in arXiv:2609.12158 (https://arxiv.org/abs/2609.12158).
    """
    lsst = read_cat(case, 'lsst_fit')
    N = len(lsst)
    stats = {}
    for tel in case.telescopes:
        files = [case.path('sb', f'{initial(tel)}_part_{i:04d}.parquet') for i in range(n_tasks)]
        missing = [p for p in files if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f'{tel}: {len(missing)} Sb parts missing, e.g. {missing[0]}')
        parts = pd.concat([pd.read_parquet(p) for p in files])
        if parts.index.duplicated().any():  # an obs object belongs to exactly one group
            raise RuntimeError(f'{tel}: LSST object present in several groups')
        col, pcol = f'blending_entropy_{tel}', f'purity_{tel}'

        # which LSST objects are in a 1-1 group, which in a larger group
        in_one = np.zeros(N, bool)
        in_other = np.zeros(N, bool)
        for g in load_groups(case, tel):
            if len(g[0]) == 1 and len(g[1]) == 1:
                in_one[g[1]] = True
            else:
                in_other[g[1]] = True
        if (in_one & in_other).any():
            raise RuntimeError(f'{tel}: LSST object in both a 1-1 and a larger group')

        fcol = f'sb_flag_{tel}'
        part_sb = parts[col].to_numpy(float, copy=True)
        if fcol in parts:
            part_flag = parts[fcol].to_numpy(object)
        else:  # parts written before the flag existed: friendly's -1 = only LSST neighbours
            part_flag = np.where(part_sb == -1, 'obj_only_neighbours',
                                 np.where(np.isfinite(part_sb), 'computed', 'failed')).astype(object)
            part_sb[part_sb == -1] = np.nan

        sb = np.full(N, np.nan)
        pur = np.full(N, np.nan)
        flag = np.full(N, 'failed', dtype=object)  # objects of larger groups missing from the parts stay 'failed'
        sb[parts.index.values] = part_sb
        pur[parts.index.values] = parts[pcol].values
        flag[parts.index.values] = part_flag
        sb[in_one], pur[in_one], flag[in_one] = 0.0, 1.0, 'one_to_one'

        # objects in no group: why?
        none = ~(in_one | in_other)
        flag[none] = 'no_ellipse_overlap'
        fof_path = case.path('groups', f'{initial(tel)}_fof.pkl')
        if os.path.exists(fof_path):
            in_fof = np.zeros(N, bool)
            for g in load(fof_path):
                in_fof[g[1]] = True
            flag[none & ~in_fof] = 'no_fof_match'
        foot = f'in_{tel.lower()}_footprint'
        if foot in lsst:
            flag[none & ~lsst[foot].to_numpy(bool)] = 'out_footprint'

        n_failed = int((flag == 'failed').sum())
        stats[tel] = {'n_objects': N}
        for name in np.unique(flag.astype(str)):
            stats[tel]['n_' + name] = int((flag == name).sum())
        if n_failed:
            print(f'WARNING {tel}: {n_failed:,} objects in groups with more than one member have a NaN entropy '
                  '(missing magnitude or a failed computation); they are flagged as failed')
        lsst[col], lsst[pcol], lsst[fcol] = sb, pur, flag
    lsst.to_parquet(case.path('lsst_sb.parquet'))
    with open(case.path('sb', 'sb_merge_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
