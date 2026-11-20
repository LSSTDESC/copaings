"""sb_merge conventions: 0 for 1-1 groups, NaN + a flag when the entropy is not defined."""
import json
import pickle

import numpy as np
import pandas as pd

from copaings import stages
from copaings.config import Case


def test_conventions(tmp_path):
    case = Case(name='t', out_root=str(tmp_path), telescopes=['Euclid'], n_sb_tasks=1)
    N = 10
    pd.DataFrame({'i_cModel_mag': np.linspace(20, 25, N)}).to_parquet(case.path('catalogs', 'lsst_fit.parquet'))
    groups = [[[5], [0]],              # 1-1 group  -> S_b = 0, purity 1
              [[1, 2], [1]],           # computed by the sb stage
              [[3, 4], [2, 3, 7]]]     # object 3 has no computed value -> NaN (a failure)
    with open(case.path('groups', 'E_no_overlap.pkl'), 'wb') as f:
        pickle.dump({'data': groups, 'metadata': {}}, f)
    pd.DataFrame({'blending_entropy_Euclid': [0.4, 0.1, np.nan], 'purity_Euclid': [0.9, 0.8, 0.7],
                  'sb_flag_Euclid': ['computed', 'computed', 'obj_only_neighbours']}, index=[1, 2, 7]).to_parquet(
        case.path('sb', 'E_part_0000.parquet'))

    stages.sb_merge(case, n_tasks=1)
    out = pd.read_parquet(case.path('lsst_sb.parquet'))
    s, p, f = out['blending_entropy_Euclid'].values, out['purity_Euclid'].values, out['sb_flag_Euclid'].values
    assert s[0] == 0 and p[0] == 1 and f[0] == 'one_to_one'
    assert s[1] == 0.4 and s[2] == 0.1 and f[1] == f[2] == 'computed'
    assert np.isnan(s[3]) and f[3] == 'failed'
    assert np.isnan(s[7]) and f[7] == 'obj_only_neighbours'
    rest = [4, 5, 6, 8, 9]  # in no group
    assert np.all(np.isnan(s[rest])) and np.all(np.isnan(p[rest]))
    assert np.all(f[rest] == 'no_ellipse_overlap')  # no FoF file and no footprint column here
    stats = json.load(open(case.path('sb', 'sb_merge_stats.json')))['Euclid']
    assert stats['n_failed'] == 1 and stats['n_one_to_one'] == 1 and stats['n_no_ellipse_overlap'] == 5


def test_no_group_reasons(tmp_path):
    case = Case(name='t', out_root=str(tmp_path), telescopes=['Euclid'], n_sb_tasks=1)
    N = 6
    pd.DataFrame({'i_cModel_mag': np.linspace(20, 25, N),
                  'in_euclid_footprint': [True, True, True, True, False, False]}).to_parquet(
        case.path('catalogs', 'lsst_fit.parquet'))
    with open(case.path('groups', 'E_no_overlap.pkl'), 'wb') as f:
        pickle.dump({'data': [[[0], [0]]], 'metadata': {}}, f)
    with open(case.path('groups', 'E_fof.pkl'), 'wb') as f:
        pickle.dump([[[0, 1], [0, 1, 4]]], f)  # 1 and 4 were in FoF but dropped by the overlap test
    pd.DataFrame({'blending_entropy_Euclid': [], 'purity_Euclid': [], 'sb_flag_Euclid': []}).to_parquet(
        case.path('sb', 'E_part_0000.parquet'))

    stages.sb_merge(case, n_tasks=1)
    f = pd.read_parquet(case.path('lsst_sb.parquet'))['sb_flag_Euclid'].tolist()
    assert f == ['one_to_one', 'no_ellipse_overlap', 'no_fof_match', 'no_fof_match', 'out_footprint', 'out_footprint']


def test_old_parts_without_flag(tmp_path):
    case = Case(name='t', out_root=str(tmp_path), telescopes=['Euclid'], n_sb_tasks=1)
    pd.DataFrame({'i_cModel_mag': np.linspace(20, 25, 3)}).to_parquet(case.path('catalogs', 'lsst_fit.parquet'))
    with open(case.path('groups', 'E_no_overlap.pkl'), 'wb') as f:
        pickle.dump({'data': [[[0, 1], [0, 1]]], 'metadata': {}}, f)
    pd.DataFrame({'blending_entropy_Euclid': [0.3, -1.0], 'purity_Euclid': [0.9, 0.5]}, index=[0, 1]).to_parquet(
        case.path('sb', 'E_part_0000.parquet'))

    stages.sb_merge(case, n_tasks=1)
    out = pd.read_parquet(case.path('lsst_sb.parquet'))
    assert out['blending_entropy_Euclid'][0] == 0.3 and np.isnan(out['blending_entropy_Euclid'][1])
    assert out['sb_flag_Euclid'].tolist()[:2] == ['computed', 'obj_only_neighbours']
