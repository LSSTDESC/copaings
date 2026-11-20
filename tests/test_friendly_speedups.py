"""The faster friendly functions must give the same results as the straightforward versions."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from copaings import deps

deps.add_friendly_to_path()
import friendly.ellipses as ellipses  # noqa: E402
from friendly.matching import FoF_matching, radec_to_unit_vectors  # noqa: E402


def _random_pair(rng):
    x1, y1 = rng.uniform(-3, 3, 2)
    x2, y2 = x1 + rng.uniform(-8, 8), y1 + rng.uniform(-8, 8)
    p1 = ellipses.ab2AB(x1, y1, rng.uniform(1, 6), rng.uniform(0.5, 3), rng.uniform(0, 180))
    p2 = ellipses.ab2AB(x2, y2, rng.uniform(1, 6), rng.uniform(0.5, 3), rng.uniform(0, 180))
    return p1, p2, 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def test_overlap_area_same_as_full_grid():
    rng = np.random.default_rng(3)
    for _ in range(15):
        p1, p2, xc, yc = _random_pair(rng)
        xlim, ylim = [xc - 100, xc + 100], [yc - 100, yc + 100]
        # straightforward version: test every point of the 1000 x 1000 grid
        xg, yg = np.meshgrid(np.linspace(*xlim, 1000), np.linspace(*ylim, 1000))
        in1 = ellipses.ellipse_equation(xg, yg, p1) < 0
        in2 = ellipses.ellipse_equation(xg, yg, p2) < 0
        n1, n2, n_overlap = in1.sum(), in2.sum(), (in1 & in2).sum()
        area, n_points = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0]), 1000 * 1000
        expected = ((n1 + n2 - n_overlap) * area / n_points, n_overlap * area / n_points)
        assert ellipses.overlap_area_MC(p1, p2, xlim, ylim) == expected


def test_is_overlapping_many_pairs_same_as_one_by_one():
    rng = np.random.default_rng(5)
    pairs = [_random_pair(rng)[:2] for _ in range(200)]
    p1 = np.array([p[0] for p in pairs])
    p2 = np.array([p[1] for p in pairs])
    one_by_one = [ellipses.is_overlapping(a, b) for a, b in pairs]
    assert ellipses.is_overlapping(p1, p2).tolist() == one_by_one
    assert 0 < sum(one_by_one) < len(pairs)  # both cases are tested


def test_fof_groups():
    rng = np.random.default_rng(6)
    truth = pd.DataFrame({'ra': 53 + rng.uniform(0, 30, 80) / 3600, 'dec': -28 + rng.uniform(0, 30, 80) / 3600})
    obj = pd.DataFrame({'ra': 53 + rng.uniform(0, 30, 60) / 3600, 'dec': -28 + rng.uniform(0, 30, 60) / 3600})
    groups = FoF_matching(truth, obj, linking_length=2.0)

    # every source is in exactly one group
    assert sorted(t for g in groups for t in g[0]) == list(range(80))
    assert sorted(o for g in groups for o in g[1]) == list(range(60))

    # brute force: two sources closer than 2" must be in the same group
    points = np.vstack([radec_to_unit_vectors(truth['ra'], truth['dec']), radec_to_unit_vectors(obj['ra'], obj['dec'])])
    group_of = np.zeros(140, int)
    for k, (t, o) in enumerate(groups):
        group_of[t] = k
        group_of[[80 + i for i in o]] = k
    close = cdist(points, points) < 2 * np.sin(np.radians(2.0 / 3600) / 2)
    i, j = np.nonzero(close)
    assert (group_of[i] == group_of[j]).all()
