import numpy as np
import pytest

from microwave_calculator.core import InputError, divider, microstrip, transformer


F0 = 2.4e9
SUBSTRATE = (0.0008, 3.55)
GRID = (F0 / 2, 2 * F0, 7)


def calculate_divider(kind, ratio):
    return divider(kind, 50, ratio, *SUBSTRATE, F0, *GRID)


def test_microstrip_reference_and_inverse():
    line = microstrip(0.001, 4, F0, width=0.002)
    # Hand substitution at W/h=2 in Pozar (3.195–3.196).
    assert line.summary["Эффективная проницаемость"] == pytest.approx(3.066946709513841)
    assert line.summary["Волновое сопротивление, Ом"] == pytest.approx(51.03739018, rel=1e-6)
    target = microstrip(*SUBSTRATE, F0, impedance=50)
    analysis = microstrip(*SUBSTRATE, F0, width=target.summary["Ширина, мм"] / 1000)
    assert analysis.summary["Волновое сопротивление, Ом"] == pytest.approx(50, rel=0.001)


def test_transformer_at_design_and_half_wave():
    result = transformer(50, 100, F0, 4, *GRID)
    gamma, vswr = (trace.values for trace in result.traces)
    assert result.summary["Длина участка, мм"] == pytest.approx(15.61419052)
    assert result.summary["Сопротивление участка, Ом"] == pytest.approx(70.71067812)
    assert gamma[2] < 1e-6
    assert vswr[2] == pytest.approx(1)
    assert gamma[-1] == pytest.approx(1 / 3)
    matched = transformer(50, 50, F0, 4, *GRID)
    assert np.max(matched.traces[0].values) == 0


@pytest.mark.parametrize("ratio", [1, 2, 0.5])
def test_tee_matches_input_and_conserves_power(ratio):
    result = calculate_divider("tee", ratio)
    s = result.center_s
    assert abs(s[0, 0]) < 1e-6
    assert abs(s[2, 0] / s[1, 0]) ** 2 == pytest.approx(ratio, rel=1e-6)
    assert abs(s[1, 2]) > 0.1
    np.testing.assert_allclose(np.sum(abs(result.scattering[:, :, 0])**2, axis=1), 1, atol=1e-6)


@pytest.mark.parametrize("kind,ratio", [
    ("wilkinson_equal", 1), ("wilkinson_unequal", 1),
    ("wilkinson_unequal", 2), ("wilkinson_unequal", 0.5),
])
def test_wilkinson_design_frequency(kind, ratio):
    result = calculate_divider(kind, ratio)
    s = result.center_s
    assert np.max(abs(np.diag(s))) < 1e-6
    assert abs(s[1, 2]) < 1e-6
    assert abs(s[1, 0]) ** 2 == pytest.approx(1 / (1 + ratio), rel=1e-6)
    assert abs(s[2, 0]) ** 2 == pytest.approx(ratio / (1 + ratio), rel=1e-6)
    assert abs(np.angle(s[2, 0] / s[1, 0])) < 1e-6
    if ratio == 1:
        assert result.summary["Развязывающий резистор, Ом"] == 100
        for section in result.sections.values():
            assert section["z_ohm"] == pytest.approx(50 * np.sqrt(2))


@pytest.mark.parametrize("kind", ["tee", "wilkinson_unequal", "wilkinson_equal"])
def test_network_is_reciprocal_and_passive(kind):
    s = calculate_divider(kind, 1 if kind == "wilkinson_equal" else 2).scattering
    np.testing.assert_allclose(s, s.swapaxes(1, 2), atol=1e-6)
    power = s.conj().swapaxes(1, 2) @ s
    assert np.max(np.linalg.eigvalsh(power)) <= 1 + 1e-6


def test_inverted_ratio_swaps_outputs_across_band():
    direct = calculate_divider("wilkinson_unequal", 2).scattering
    inverse = calculate_divider("wilkinson_unequal", 0.5).scattering
    indices = [0, 2, 1]
    np.testing.assert_allclose(direct, inverse[:, indices][:, :, indices], atol=1e-6)


@pytest.mark.parametrize("run", [
    lambda: microstrip(0, 4, F0, width=0.002),
    lambda: microstrip(*SUBSTRATE, F0, impedance=1000),
    lambda: transformer(50, 100, -F0, 4, *GRID),
    lambda: transformer(50, 100, F0, 4, 2 * F0, F0, 100),
    lambda: transformer(50, 100, F0, 4, F0, 2 * F0, 1),
    lambda: calculate_divider("wilkinson_unequal", 0),
])
def test_invalid_inputs(run):
    with pytest.raises(InputError):
        run()
