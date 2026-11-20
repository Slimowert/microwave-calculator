import numpy as np
import pytest
from scipy.special import eval_chebyt

from microwave_calculator.extended import lowpass, multi_divider


F0 = 2.4e9
GRID = (0.01 * F0, 2 * F0, 101)


@pytest.mark.parametrize("outputs", [2, 3, 8])
def test_multiport_wilkinson(outputs):
    result = multi_divider("wilkinson_n", 50, 0.0008, 3.55, F0, *GRID, outputs)
    s = result.center_s
    assert s.shape == (outputs + 1, outputs + 1)
    np.testing.assert_allclose(s[1:, 1:], 0, atol=1e-10)
    np.testing.assert_allclose(abs(s[1:, 0])**2, 1 / outputs, atol=1e-10)
    assert abs(s[0, 0]) < 1e-10


def test_resistive_divider_matches_and_dissipates_half():
    result = multi_divider("resistive", 50, None, None, F0, *GRID)
    np.testing.assert_allclose(result.center_s, (np.ones((3, 3)) - np.eye(3)) / 2, atol=1e-10)
    assert np.sum(abs(result.center_s[:, 0])**2) == pytest.approx(0.5)


@pytest.mark.parametrize("number", [1, 2, 7, 8])
def test_cascaded_wilkinson(number):
    from microwave_calculator.core import divider
    impedances = np.linspace(55, 95, number) if number > 1 else [50*np.sqrt(2)]
    result = multi_divider("wilkinson_sections", 50, .0008, 3.55, F0, *GRID,
                           impedances=impedances, resistances=[100]*number)
    assert len(result.sections) == number
    assert result.scattering.shape == (101, 3, 3)
    np.testing.assert_allclose(result.scattering[:, 1, 0], result.scattering[:, 2, 0], atol=1e-12)
    if number == 1:
        reference = divider("wilkinson_equal", 50, 1, .0008, 3.55, F0, *GRID)
        np.testing.assert_allclose(result.scattering, reference.scattering, atol=1e-12)


@pytest.mark.parametrize("kind,order", [
    ("butterworth", 3), ("butterworth", 10), ("chebyshev", 3), ("chebyshev", 4),
])
def test_lowpass_power_response(kind, order):
    ripple = 0.5
    result = lowpass(kind, 50, F0, order, ripple, *GRID)
    x = result.frequency / F0
    expected = (1 / (1 + x**(2 * order)) if kind == "butterworth" else
                1 / (1 + (10**(ripple / 10) - 1) * eval_chebyt(order, x)**2))
    np.testing.assert_allclose(abs(result.scattering[:, 1, 0])**2, expected, atol=1e-10)
    assert len(result.sections) == order
