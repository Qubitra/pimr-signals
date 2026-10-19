"""Tests for AR coefficient fitting and oscillatory-mode extraction.

The tests check that ar2 and ar4 recover coefficients from known series.
They also check that oscillatory_modes identifies a stable complex-root mode.
"""

import numpy as np
from numpy.testing import assert_allclose

from pimr.physics.ar_models import ar2, ar4, oscillatory_modes


def test_ar2_recovers_coefficients():
    rng = np.random.default_rng(42)
    x = np.zeros(300)

    for t in range(2, len(x)):
        x[t] = 0.25 + 0.6 * x[t - 1] - 0.2 * x[t - 2] + rng.normal(0, 0.1)

    assert_allclose(AR2(x), [0.25, 0.6, -0.2], atol=0.05)


def test_ar4_recovers_coefficients():
    rng = np.random.default_rng(42)
    x = np.zeros(500)

    for t in range(4, len(x)):
        x[t] = (
            0.2
            + 0.4 * x[t - 1]
            - 0.2 * x[t - 2]
            + 0.1 * x[t - 3]
            + 0.05 * x[t - 4]
            + rng.normal(0, 0.1)
        )

    assert_allclose(AR4(x), [0.2, 0.4, -0.2, 0.1, 0.05], atol=0.05)


def test_oscillatory_modes_finds_complex_root_pair():
    modes = oscillatory_modes([0.0, 0.70710678, -0.5])

    assert len(modes) == 1
    assert modes[0]["radius"] < 1
    assert modes[0]["omega"] > 0