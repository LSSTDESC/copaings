"""Blending entropy S_b of the LSST objects, group by group, computed with `friendly`.

The blending entropy is described, used and tested in arXiv:2609.12158 (https://arxiv.org/abs/2609.12158).
For one group, friendly builds a graph whose nodes are the true galaxies and the LSST objects and whose edges join
overlapping ellipses. The entropy of an LSST object is the entropy of its probabilities to match each galaxy it
overlaps (from the overlap areas and the magnitude differences).
"""
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import deps

deps.add_friendly_to_path()
import friendly.ellipses as ellipses  # noqa: E402
import friendly.entropy as entropy  # noqa: E402
import friendly.graphs as graphs  # noqa: E402


def group_sb(group, true_cat, obs_cat, link_type='ellipse'):
    """Blending entropy and purity of the LSST objects of one group.

    Parameters
    ----------
    group : sequence
        ``[true_rows, lsst_rows]``: rows of the group members in the true and in the LSST catalog.
    true_cat : pandas.DataFrame
        True catalog with ``ra``, ``dec``, ``size_true``, ``size_minor_true``, ``position_angle_true`` and ``mag_i``.
    obs_cat : pandas.DataFrame
        LSST catalog with ``ra``, ``dec``, ``x``, ``y``, the i-band moments, ``psf_fwhm_i``, ``mag_i_cModel`` and
        ``blendedness``.
    link_type : str, default 'ellipse'
        ``'ellipse'`` (overlap of ellipses) or anything else for friendly's Gaussian overlap.

    Returns
    -------
    dict
        ``{lsst_row: (entropy, purity, flag)}`` for the LSST objects of the group. friendly gives -1 when the entropy
        cannot be computed; here it becomes NaN and `flag` says why:

        * ``'computed'``: entropy >= 0;
        * ``'obj_only_neighbours'``: the object overlaps only LSST objects of the group, no galaxy;
        * ``'zero_overlap'``: the object touches galaxies, but every overlap area is 0;
        * ``'failed'``: friendly gave NaN (e.g. undefined ellipse or missing magnitude).
    """
    gal_infos = ellipses.ellipse_infos(group, true_cat, obs_cat, dc2_type='galaxy')
    obj_infos = ellipses.ellipse_infos(group, true_cat, obs_cat, dc2_type='object')
    gal_params = ellipses.ellipse_parameters(gal_infos)
    obj_params = ellipses.ellipse_parameters(obj_infos)

    # friendly wants one column per graph node (x, y, a, b, theta and A..F in rows)
    infos = pd.concat([gal_infos, obj_infos]).T
    params = pd.concat([gal_params, obj_params]).T

    G = graphs.NetworkX_graph(group, true_cat, obs_cat, infos, params, link_type)
    entropy.blending_entropy(G)

    result = {}
    for node, data in G.nodes(data=True):
        if data['galaxy']:
            continue
        # an LSST object
        s = data['blending_entropy']
        if s == -1:
            touches_galaxy = any(G.nodes[c]['galaxy'] for c in G.adj[node])
            flag = 'zero_overlap' if touches_galaxy else 'obj_only_neighbours'
            s = np.nan
        elif np.isfinite(s):
            flag = 'computed'
        else:
            flag = 'failed'
        result[node] = (s, data['purity'], flag)
    return result


def group_sb_renumbered(group, true_cat, obs_cat, link_type='ellipse'):
    """Same as `group_sb`, for a group where a true galaxy and an LSST object have the same row number.

    friendly uses the row number of a source as its node name in the graph, for both catalogs. If galaxy 3 and LSST
    object 3 are in the same group, they would become one node. So the group is copied into two small catalogs where
    the row numbers cannot collide:

    * LSST sub-catalog: the n_obs LSST objects of the group, rows ``0 .. n_obs-1``;
    * true sub-catalog: n_obs filler rows (never used), then the galaxies of the group, rows ``n_obs ..``.

    The results are then given back with the real LSST row numbers.

    Parameters
    ----------
    group, true_cat, obs_cat, link_type
        See `group_sb`.

    Returns
    -------
    dict
        ``{lsst_row: (entropy, purity, flag)}``, with the rows of the full LSST catalog.
    """
    true_rows, lsst_rows = list(group[0]), list(group[1])
    n_obs = len(lsst_rows)

    obs_sub = obs_cat.iloc[lsst_rows].reset_index(drop=True)
    filler = true_cat.iloc[[0] * n_obs]
    true_sub = pd.concat([filler, true_cat.iloc[true_rows]], ignore_index=True)

    new_group = [list(range(n_obs, n_obs + len(true_rows))), list(range(n_obs))]
    result = group_sb(new_group, true_sub, obs_sub, link_type)
    return {lsst_rows[k]: value for k, value in result.items()}


def compute_Sb(groups, true_cat, obs_cat, link_type='ellipse', Sb_col_name_in_obs_cat='blending_entropy',
               flag_col_name_in_obs_cat='sb_flag'):
    """Blending entropy and purity of the LSST objects of a list of groups.

    Parameters
    ----------
    groups : list
        Groups ``[true_rows, lsst_rows]``.
    true_cat : pandas.DataFrame
        True catalog (see `group_sb`).
    obs_cat : pandas.DataFrame
        LSST catalog (see `group_sb`). Modified in place: the entropy goes to the column `Sb_col_name_in_obs_cat`,
        the purity to the column ``purity`` and the flag (see `group_sb`) to `flag_col_name_in_obs_cat`.
    link_type : str, default 'ellipse'
        Overlap definition, see `group_sb`.
    Sb_col_name_in_obs_cat : str, default 'blending_entropy'
        Column of `obs_cat` that receives the entropy.
    flag_col_name_in_obs_cat : str, default 'sb_flag'
        Column of `obs_cat` that receives the flag.
    """
    rows, sb, purity, flags = [], [], [], []
    with np.errstate(invalid='ignore', divide='ignore'):  # undefined ellipses (NaN moments) give NaN, not an error
        for group in tqdm(groups, file=sys.stderr):
            if set(group[0]) & set(group[1]):
                result = group_sb_renumbered(group, true_cat, obs_cat, link_type)
            else:
                result = group_sb(group, true_cat, obs_cat, link_type)
            for row, (s, p, flag) in result.items():
                rows.append(row)
                sb.append(s)
                purity.append(p)
                flags.append(flag)

    if flag_col_name_in_obs_cat not in obs_cat:
        obs_cat[flag_col_name_in_obs_cat] = None
    obs_cat[flag_col_name_in_obs_cat] = obs_cat[flag_col_name_in_obs_cat].astype(object)
    obs_cat.loc[rows, Sb_col_name_in_obs_cat] = sb
    obs_cat.loc[rows, 'purity'] = purity
    obs_cat.loc[rows, flag_col_name_in_obs_cat] = flags
