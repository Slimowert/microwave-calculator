import csv
import json

import numpy as np
import pytest
from matplotlib.figure import Figure

from microwave_calculator.core import InputError, network_s
from microwave_calculator.distribution import distributed_divider, power_fractions
from microwave_calculator.export import write_csv, write_json
from microwave_calculator.hfss import supports_hfss, write_hfss
from microwave_calculator.plots import plot_dashboard_trace, plot_power, plot_schematic


def calculate(kind, percentages, *, stages=1):
    return distributed_divider(kind, 50, .0008, 3.55, 2.4e9,
                               1.2e9, 3.6e9, 17, len(percentages), percentages, stages=stages)


DISTRIBUTIONS = ([50, 50], [25, 75], [20, 30, 50], [25] * 4,
                 [10] * 10, [n + .5 for n in range(5, 15)])


@pytest.mark.parametrize("kind", ["tee", "wilkinson_equal", "resistive"])
@pytest.mark.parametrize("percentages", DISTRIBUTIONS)
def test_explicit_power_distribution_and_passivity(kind, percentages):
    result = calculate(kind, percentages)
    n = len(percentages)
    assert result.scattering.shape == (17, n + 1, n + 1)
    power = abs(result.center_s[:, 0]) ** 2
    np.testing.assert_allclose(power[1:] / sum(power[1:]), np.array(percentages) / 100, atol=1e-10)
    np.testing.assert_allclose(result.scattering, result.scattering.swapaxes(1, 2), atol=1e-10)
    assert abs(result.center_s[0, 0]) < 1e-10
    energy = result.scattering.conj().swapaxes(1, 2) @ result.scattering
    assert np.max(np.linalg.eigvalsh(energy)) <= 1 + 1e-9
    if kind == "tee":
        np.testing.assert_allclose(energy, np.broadcast_to(np.eye(n + 1), energy.shape), atol=1e-10)
        assert np.max(abs(result.center_s[1:, 1:])) > .1
    elif kind == "wilkinson_equal":
        np.testing.assert_allclose(result.center_s[1:, 1:], 0, atol=1e-10)
        assert sum(power[1:]) == pytest.approx(1)
    else:
        np.testing.assert_allclose(result.scattering.diagonal(axis1=1, axis2=2), 0, atol=1e-10)
        efficiency = 1 / (n ** 2 * max(percentages) / 100)
        assert sum(power[1:]) == pytest.approx(efficiency)
        assert result.summary["Потери, %"] == pytest.approx(100 * (1 - efficiency))


@pytest.mark.parametrize("percentages", [[50, 50], [20, 30, 50], [10] * 10])
@pytest.mark.parametrize("stages", [2, 8])
def test_multisection_wilkinson_preserves_output_distribution(percentages, stages):
    result = calculate("wilkinson_equal", percentages, stages=stages)
    power = abs(result.center_s[1:, 0]) ** 2
    np.testing.assert_allclose(power / sum(power), np.array(percentages) / 100, atol=1e-10)
    assert len(result.topology["branches"]) == len(percentages)
    assert all(len(branch["isolation"]) == stages for branch in result.topology["branches"])
    energy = result.scattering.conj().swapaxes(1, 2) @ result.scattering
    assert np.max(np.linalg.eigvalsh(energy)) <= 1 + 1e-9


@pytest.mark.parametrize("outputs,percentages", [
    (1, [100]), (11, [100 / 11] * 11), (True, [50, 50]), (2.0, [50, 50]),
    (3, [50, 50]), (2, [0, 100]), (2, [-1, 101]), (2, [40, 40]),
    (3, [33.33] * 3), (2, [float("nan"), 50]), (2, [float("inf"), 50]),
])
def test_invalid_distribution_is_rejected(outputs, percentages):
    with pytest.raises(InputError):
        power_fractions(outputs, percentages)


def test_grounded_resistor_network_is_matched_t_attenuator():
    amplitude, z0 = .5, 50
    series = z0 * (1 - amplitude) / (1 + amplitude)
    shunt = 2 * z0 * amplitude / (1 - amplitude ** 2)
    s = network_s(np.array([1e9]), 1e9, z0, [], ports=2,
                  resistors=[(0, 2, series), (2, 1, series)], shunts=[(2, shunt)])[0]
    np.testing.assert_allclose(s, [[0, amplitude], [amplitude, 0]], atol=1e-12)


def test_ten_output_exports_have_unique_port_names_and_actual_components(tmp_path):
    result = calculate("wilkinson_equal", [10] * 10)
    path = tmp_path / "ports.csv"
    write_csv(path, result)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert len(reader.fieldnames) == len(set(reader.fieldnames))
        assert "Re(S[1,11]), 1" in reader.fieldnames
        assert "Re(S[11,1]), 1" in reader.fieldnames
        assert "S[11,1], дБ" in reader.fieldnames
        assert len(list(reader)) == 17
    path = tmp_path / "ports.json"
    write_json(path, result)
    data = json.loads(path.read_text())
    assert data["inputs"]["power_percentages"] == [10] * 10
    assert len(data["s_at_f0"]) == 11
    assert len(data["topology"]["branches"]) == 10


def test_hfss_rejects_multioutput_layout_before_writing(tmp_path):
    result = calculate("tee", [20, 30, 50])
    path = tmp_path / "model.py"
    path.write_text("existing")
    assert not supports_hfss(result)
    with pytest.raises(ValueError, match="разводки"):
        write_hfss(path, result)
    assert path.read_text() == "existing"


@pytest.mark.parametrize("kind", ["tee", "wilkinson_equal", "resistive"])
def test_all_output_ports_are_drawn_and_plotted(kind):
    result = calculate(kind, [10] * 10)
    figure = Figure(figsize=(10, 6))
    plot_schematic(figure, result)
    texts = [text.get_text() for text in figure.axes[0].texts]
    assert all(any(text.startswith(f"P{port} ·") for text in texts) for port in range(2, 12))
    plot_dashboard_trace(figure, result)
    assert len(figure.axes[0].get_legend().get_texts()) == 10
    plot_dashboard_trace(figure, result, reflection=True)
    assert len(figure.axes[0].get_legend().get_texts()) == 11
    plot_power(figure, result)
    texts = [text.get_text() for ax in figure.axes for text in ax.texts]
    assert any("P11" in text for text in texts)
    assert any("Потери" in text for text in texts) == (kind == "resistive")


def test_desktop_explicit_percentages_and_port_limit(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from microwave_calculator.ui import QApplication, CalculatorWindow

    app = QApplication.instance() or QApplication([])
    window = CalculatorWindow()
    window.show()
    try:
        for kind in ("tee", "wilkinson_equal", "resistive"):
            window._open_module(kind)
            assert "ratio" not in window.fields
            assert window.output_count.minimum() == 2
            assert window.output_count.maximum() == 10
            window.output_count.setValue(3)
            assert len(window.power_entries) == 3
            assert window.result is None
            assert any("P4" in text.get_text() for text in window.mini_scheme.figure.axes[0].texts)
            for entry, value in zip(window.power_entries, [20, 30, 50]):
                entry.setText(str(value))
            window.calculate_button.click()
            assert window.result is not None, window.feedback.text()
            assert window.result.center_s.shape == (4, 4)
            assert not window.exports["HFSS"].isEnabled()
            window.power_entries[0].setText("19")
            window.calculate_button.click()
            assert window.result is None
            assert "100%" in window.feedback.text()
            assert not window.scheme_export.isEnabled()
            window.power_entries[0].setText("nan")
            window.calculate_button.click()
            assert window.result is None
            window.equal_power_button.click()
            window.calculate_button.click()
            assert window.result is not None, window.feedback.text()
            power = abs(window.result.center_s[1:, 0]) ** 2
            np.testing.assert_allclose(power / sum(power), [1 / 3] * 3, atol=1e-10)
            window.output_count.setValue(10)
            app.processEvents()
            assert len(window.power_entries) == 10
            assert all(entry.height() >= 28 for entry in window.power_entries)
            window.calculate_button.click()
            assert window.result is not None, window.feedback.text()
            assert window.result.center_s.shape == (11, 11)
            window.output_count.setValue(2)
            window.calculate_button.click()
            assert window.result.center_s.shape == (3, 3)
    finally:
        window.close()
