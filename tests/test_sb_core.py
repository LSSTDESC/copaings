"""compute_Sb must also work when a true object and an LSST object of a group share the same row number."""
import numpy as np
import pandas as pd
import pytest

from copaings import deps

deps.add_friendly_to_path()
from copaings.sb_core import compute_Sb  # noqa: E402

PS = 0.2


def _cats():
    rng = np.random.default_rng(4)
    n = 12
    ra = 53.1 + rng.uniform(0, 4, n) / 3600
    dec = -28.0 + rng.uniform(0, 4, n) / 3600
    lsst = pd.DataFrame({'ra': ra, 'dec': dec, 'x': (53.1 - ra) * np.cos(np.radians(dec)) * 3600 / PS,
                         'y': (dec + 28.0) * 3600 / PS, 'Ixx_pixel_i': rng.uniform(3, 6, n),
                         'Iyy_pixel_i': rng.uniform(3, 6, n), 'Ixy_pixel_i': rng.uniform(-1, 1, n), 'psf_fwhm_i': 0.8,
                         'mag_i_cModel': rng.uniform(22, 25, n), 'blendedness': 0.1})
    true = pd.DataFrame({'ra': 53.1 + rng.uniform(0, 4, n) / 3600, 'dec': -28.0 + rng.uniform(0, 4, n) / 3600,
                         'size_true': rng.uniform(0.2, 0.6, n), 'size_minor_true': rng.uniform(0.1, 0.3, n),
                         'position_angle_true': rng.uniform(0, 180, n), 'mag_i': rng.uniform(22, 25, n)})
    return lsst, true


def _check_written(lsst, rows):
    # every object of the group gets a flag; S_b is a number exactly when the flag is 'computed'
    flag = lsst.loc[rows, 'sb_flag']
    assert flag.notna().all()
    assert (np.isfinite(lsst.loc[rows, 'sb']) == (flag == 'computed')).all()
    assert (flag == 'computed').any()


def test_group_with_colliding_ids():
    lsst, true = _cats()
    lsst['sb'] = np.nan
    # true object 3 and LSST object 3 have the same row number but are different objects
    compute_Sb([[[3, 4, 5], [3, 6]]], true, lsst, Sb_col_name_in_obs_cat='sb')
    _check_written(lsst, [3, 6])
    assert lsst['sb_flag'].notna().sum() == 2  # only the group's LSST objects are written


def test_collision_free_group_unchanged():
    lsst, true = _cats()
    lsst['sb'] = np.nan
    compute_Sb([[[0, 1], [7, 8]]], true, lsst, Sb_col_name_in_obs_cat='sb')
    _check_written(lsst, [7, 8])


def _graph(gal_mag, obj_mag, edges):
    """Small friendly graph: galaxies 'g*', objects 'o*', edges (a, b, overlap_fraction)."""
    import networkx as nx
    G = nx.Graph()
    for n, m in gal_mag.items():
        G.add_node(n, galaxy=True, magnitude=m)
    for n, m in obj_mag.items():
        G.add_node(n, galaxy=False, magnitude=m)
    for a, b, frac in edges:
        G.add_edge(a, b, overlap_fraction=frac)
    return G


def test_friendly_zero_overlap_is_not_a_perfect_match():
    import friendly.entropy as entropy
    # o1 touches g1 but the overlap area is 0, o2 overlaps g1 normally, o3 touches only o2
    G = _graph({'g1': 22.0}, {'o1': 22.0, 'o2': 22.0, 'o3': 23.0},
               [('o1', 'g1', 0.0), ('o2', 'g1', 0.5), ('o3', 'o2', 0.2)])
    entropy.blending_entropy(G)
    assert G.nodes['o1']['blending_entropy'] == -1  # was 0 (empty probability list)
    assert G.nodes['o2']['blending_entropy'] == 0
    assert G.nodes['o3']['blending_entropy'] == -1


def test_group_sb_flags(monkeypatch):
    from copaings import sb_core
    G = _graph({'g1': 22.0}, {'o1': 22.0, 'o2': 22.0, 'o3': 23.0},
               [('o1', 'g1', 0.0), ('o2', 'g1', 0.5), ('o3', 'o2', 0.2)])
    for n in G:
        G.nodes[n]['purity'] = 1.0

    monkeypatch.setattr(sb_core.ellipses, 'ellipse_infos', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(sb_core.ellipses, 'ellipse_parameters', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(sb_core.graphs, 'NetworkX_graph', lambda *a, **k: G)
    res = sb_core.group_sb(None, None, None)
    assert np.isnan(res['o1'][0]) and res['o1'][2] == 'zero_overlap'
    assert res['o2'][0] == 0 and res['o2'][2] == 'computed'
    assert np.isnan(res['o3'][0]) and res['o3'][2] == 'obj_only_neighbours'
