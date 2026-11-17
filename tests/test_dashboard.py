import numpy as np
import pytest
from matplotlib.figure import Figure

from microwave_calculator.core import divider
from microwave_calculator.plots import DB_FLOOR, plot_dashboard_trace


def test_frequency_marker_uses_design_frequency_outside_sample_grid():
    result = divider("wilkinson_equal", 50, 1, 0.0008, 3.55, 2.4e9, 1.2e9, 3.6e9, 2)
    figure = Figure()
    plot_dashboard_trace(figure, result)
    marker = figure.axes[0].lines[-1]
    assert marker.get_xdata()[0] == pytest.approx(2.4)
    assert marker.get_ydata()[0] == pytest.approx(10 * np.log10(0.5))
    plot_dashboard_trace(figure, result, reflection=True)
    marker = figure.axes[0].lines[-1]
    assert marker.get_ydata()[0] == DB_FLOOR
    assert "≤ -100" in figure.axes[0].texts[0].get_text()


def test_dashboard_navigation_display_controls_and_resizing(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from microwave_calculator.ui import QApplication, CalculatorWindow

    app = QApplication.instance() or QApplication([])
    window = CalculatorWindow()
    window.show()
    try:
        for width, height in ((1600, 940), (1360, 850)):
            window.resize(width, height)
            app.processEvents()
            scroll = window.tabs.widget(0)
            assert scroll.widget().height() == scroll.viewport().height()
        result = window.result
        window.show_matrix.click()
        assert window.matrix_card.isHidden()
        window.show_matrix.click()
        assert not window.matrix_card.isHidden()
        window.show_sweep.click()
        assert window.preview_card.isHidden()
        assert window.reflection_card.isHidden()
        window.show_sweep.click()
        assert not window.preview_card.isHidden()
        assert window.result is result
        window.advanced_button.click()
        assert not window.advanced_container.isHidden()
        for index in (3, 1, 4, 5, 6, 0):
            window.navigation[index][0].click()
            assert window.tabs.currentIndex() == index
        window.fields["f0"][0].setText("3")
        assert window.result is None
        assert all(not button.isEnabled() for _, button in window.export_actions)
        assert not window.navigation[3][0].isEnabled()
        window.calculate_button.click()
        assert window.result.f0 == 3e9
        window._open_module("microstrip_synthesis")
        assert window.result.kind == "microstrip"
        assert window.matrix_card.isHidden()
        assert window.preview_card.isHidden()
        assert not window.exports["CSV"].isEnabled()
    finally:
        window.close()
