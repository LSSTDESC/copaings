"""sb_merge conventions: 0 for 1-1 groups, -1 for objects with no overlapping galaxy, NaN only for failures."""
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
              [[3, 4], [2, 3]]]        # object 3 has no computed value -> NaN (a failure)
    with open(case.path('groups', 'E_no_overlap.pkl'), 'wb') as f:
        pickle.dump({'data': groups, 'metadata': {}}, f)
    pd.DataFrame({'blending_entropy_Euclid': [0.4, 0.1], 'purity_Euclid': [0.9, 0.8]}, index=[1, 2]).to_parquet(
        case.path('sb', 'E_part_0000.parquet'))

    stages.sb_merge(case, n_tasks=1)
    out = pd.read_parquet(case.path('lsst_sb.parquet'))
    s, p = out['blending_entropy_Euclid'].values, out['purity_Euclid'].values
    assert s[0] == 0 and p[0] == 1
    assert s[1] == 0.4 and s[2] == 0.1
    assert np.isnan(s[3])
    assert np.all(s[4:] == -1) and np.all(p[4:] == -1)
    stats = json.load(open(case.path('sb', 'sb_merge_stats.json')))['Euclid']
    assert stats['n_nan_unexpected'] == 1 and stats['n_one_to_one_zero'] == 1 and stats['n_no_overlap_minus1'] == 6
