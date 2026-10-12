"""mc_fast.overlap_area_MC must reproduce friendly's Monte Carlo overlap area exactly."""
import numpy as np
import pytest

from copaings import deps  # noqa: F401

pytest.importorskip('friendly')
import friendly.ellipses as ellipses  # noqa: E402
from copaings import mc_fast  # noqa: E402


def test_identical_to_friendly():
    rng = np.random.default_rng(3)
    for _ in range(15):
        x1, y1 = rng.uniform(-3, 3, 2)
        x2, y2 = x1 + rng.uniform(-8, 8), y1 + rng.uniform(-8, 8)
        p1 = ellipses.ab2AB(x1, y1, rng.uniform(1, 6), rng.uniform(0.5, 3), rng.uniform(0, 180))
        p2 = ellipses.ab2AB(x2, y2, rng.uniform(1, 6), rng.uniform(0.5, 3), rng.uniform(0, 180))
        xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        lim = dict(xlim=[xc - 100, xc + 100], ylim=[yc - 100, yc + 100])
        assert mc_fast.overlap_area_MC(p1, p2, **lim) == mc_fast._original(p1, p2, **lim)
