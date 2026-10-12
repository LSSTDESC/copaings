"""Blending entropy per group, using friendly only (no LSST stack).

Same logic as ``my_blending_lib.compute_Sb``, plus a guard against node-id collisions (see `_group_sb_remapped`).
"""
import sys
import warnings

import pandas as pd
from tqdm import tqdm

import friendly.ellipses as ellipses  # deps.py puts friendly on sys.path
import friendly.entropy as entropy
import friendly.graphs as graphs


def study_ellipses(grp, true_cat, obj_cat):
    """Ellipse information and conic parameters of the objects of one group.

    Parameters
    ----------
    grp : sequence
        ``[true_indices, lsst_indices]``.
    true_cat, obj_cat : pandas.DataFrame
        True and LSST catalogs (full catalogs: friendly indexes them by row position).

    Returns
    -------
    obj_infos, obj_param : pandas.DataFrame
        Centre, axes, angle and conic parameters of the LSST objects.
    gal_infos, gal_param : pandas.DataFrame
        Same for the true objects.
    """
    obj_infos = ellipses.ellipse_infos(grp, true_cat, obj_cat, dc2_type='object')
    obj_param = ellipses.ellipse_parameters(obj_infos)
    gal_infos = ellipses.ellipse_infos(grp, true_cat, obj_cat, dc2_type='galaxy')
    gal_param = ellipses.ellipse_parameters(gal_infos)
    return obj_infos, obj_param, gal_infos, gal_param


def _group_sb(group, true_cat, obs_cat, link_type):
    """Blending entropy and purity of the LSST objects of one group.

    Parameters
    ----------
    group : sequence
        ``[true_indices, lsst_indices]``.
    true_cat, obs_cat : pandas.DataFrame
        True and LSST catalogs.
    link_type : str
        ``'ellipse'`` (overlap of ellipses) or the Gaussian overlap definition of friendly.

    Returns
    -------
    dict
        ``{lsst_row: (entropy, purity)}`` for the LSST objects of the group.
    """
    obj_infos, obj_param, gal_infos, gal_param = study_ellipses(group, true_cat, obs_cat)
    grp_infos = pd.merge(gal_infos.T, obj_infos.T, how='outer', left_index=True, right_index=True)
    grp_infos = grp_infos.T[['x', 'y', 'a', 'b', 'theta']].T
    grp_param = pd.merge(gal_param.T, obj_param.T, how='outer', left_index=True, right_index=True)
    G = graphs.NetworkX_graph(group, true_cat, obs_cat, grp_infos, grp_param, link_type)
    entropy.blending_entropy(G)
    return {idx: (data['blending_entropy'], data['purity']) for idx, data in G.nodes(data=True)
            if data.get('galaxy') is False}  # belongs to obs_cat


def _group_sb_remapped(group, true_cat, obs_cat, link_type):
    """Blending entropy and purity of a group where a true and an LSST object have the same row number.

    friendly uses the row number of an object both as its graph node id and as its position in its catalog, so the
    two collide and ``param[i]`` raises KeyError. The group is evaluated on small sub-catalogs where the LSST objects
    are numbered 0..n_o-1 and the true objects n_o.. (dummy rows pad the start of the true catalog), and the results
    are mapped back to the real LSST rows.

    Parameters
    ----------
    group : sequence
        ``[true_indices, lsst_indices]``.
    true_cat, obs_cat : pandas.DataFrame
    link_type : str

    Returns
    -------
    dict
        ``{lsst_row: (entropy, purity)}``.
    """
    idx_t, idx_o = list(group[0]), list(group[1])
    n_o = len(idx_o)
    obs_sub = obs_cat.iloc[idx_o].reset_index(drop=True)
    true_sub = pd.concat([true_cat.iloc[[0] * n_o], true_cat.iloc[idx_t]], ignore_index=True)
    res = _group_sb([list(range(n_o, n_o + len(idx_t))), list(range(n_o))], true_sub, obs_sub, link_type)
    return {idx_o[k]: v for k, v in res.items()}


def compute_Sb(groups, true_cat, obs_cat, link_type='ellipse', Sb_col_name_in_obs_cat='blending_entropy'):
    """Blending entropy and purity of the LSST objects of a list of groups.

    Needs only friendly (no LSST stack); same logic as ``my_blending_lib.compute_Sb`` plus a guard for node id
    collisions.

    Parameters
    ----------
    groups : list
        Groups ``[true_indices, lsst_indices]``.
    true_cat : pandas.DataFrame
        True catalog with ``mag_i``.
    obs_cat : pandas.DataFrame
        LSST catalog with ``mag_i_cModel`` (magnitude in the band of the true catalog) and ``blendedness``. Modified in
        place.
    link_type : str, default 'ellipse'
        Overlap definition: ``'ellipse'`` or friendly's Gaussian overlap.
    Sb_col_name_in_obs_cat : str, default 'blending_entropy'
        Column of `obs_cat` that receives the entropy; the purity goes to the column ``purity``.
    """
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    for group in tqdm(groups, file=sys.stderr):
        collide = bool(set(group[0]) & set(group[1]))
        res = (_group_sb_remapped if collide else _group_sb)(group, true_cat, obs_cat, link_type)
        for idx, (ent, pur) in res.items():
            obs_cat.loc[idx, Sb_col_name_in_obs_cat] = ent
            obs_cat.loc[idx, 'purity'] = pur
