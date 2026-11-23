import numpy as np
import pytest
from matplotlib.figure import Figure

from microwave_calculator.core import divider
from microwave_calculator.plots import plot_dashboard_trace


def test_frequency_marker_uses_design_frequency_outside_sample_grid():
    result = divider("wilkinson_equal", 50, 1, 0.0008, 3.55, 2.4e9, 1.2e9, 3.6e9, 2)
    figure = Figure()
    plot_dashboard_trace(figure, result)
    marker = figure.axes[0].lines[-1]
    assert marker.get_xdata()[0] == pytest.approx(2.4)
    assert marker.get_ydata()[0] == pytest.approx(10 * np.log10(0.5))
    plot_dashboard_trace(figure, result, reflection=True)
    marker = figure.axes[0].lines[-1]
    assert marker.get_ydata()[0] == pytest.approx(1)
    port_lines = figure.axes[0].lines[:3]
    assert [line.get_label() for line in port_lines] == ["Порт 1", "Порт 2", "Порт 3"]
    gamma = np.abs(result.scattering.diagonal(axis1=1, axis2=2))
    np.testing.assert_allclose(
        [line.get_ydata() for line in port_lines], ((1 + gamma) / (1 - gamma)).T
    )


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
        assert window.tabs.count() == 6
        assert window.scheme_card.title.text() == "Мини-схема"
        assert window.mini_scheme.figure.axes
        assert window.preview_card.title.text() == "Передача на выходы"
        assert window.reflection_card.title.text() == "КСВ по портам"
        window.show_sweep.click()
        assert window.preview_card.isHidden()
        assert window.reflection_card.isHidden()
        window.show_sweep.click()
        assert not window.preview_card.isHidden()
        assert window.result is result
        window.advanced_button.click()
        assert not window.advanced_container.isHidden()
        window.advanced_button.click()
        window.fields["count"][0].setText("x")
        window.calculate_button.click()
        assert not window.advanced_container.isHidden()
        assert window.fields["count"][0].hasFocus()
        window.fields["count"][0].setText("101")
        window.calculate_button.click()
        for index in (2, 1, 3, 4, 5, 0):
            window.navigation[index][0].click()
            assert window.tabs.currentIndex() == index
        window.fields["f0"][0].setText("3")
        assert window.result is None
        assert all(not button.isEnabled() for _, button in window.export_actions)
        assert window.navigation[2][0].isEnabled()
        assert not window.preview_card.isHidden()
        assert not window.preview.figure.axes[0].lines
        window.navigation[2][0].click()
        assert window.tabs.currentIndex() == 2
        assert len(window.response.figure.axes) == 2
        assert all(not axis.lines for axis in window.response.figure.axes)
        window.navigation[0][0].click()
        window.calculate_button.click()
        assert window.result.f0 == 3e9
        assert window.device.findData("wilkinson_sections") == -1
        window._open_module("wilkinson_equal")
        single_section = window.result
        assert "impedances" not in window.fields and "resistances" not in window.fields
        for sections in (2, 8):
            window.section_count.setValue(sections)
            assert window.result is None
            window.calculate_button.click()
            result = window.result
            assert result is not None, window.feedback.text()
            assert result.kind == "wilkinson_sections"
            assert len(result.sections) == sections
            assert all(section["z_ohm"] > 0 and section["r_ohm"] > 0
                       for section in result.sections.values())
            np.testing.assert_allclose(result.scattering[:, 1, 0], result.scattering[:, 2, 0])
            for row, column in ((0, 0), (1, 1), (1, 2)):
                assert np.max(abs(result.scattering[:, row, column])) < np.max(
                    abs(single_section.scattering[:, row, column])
                )
        window.section_count.setValue(1)
        window.calculate_button.click()
        assert window.result.kind == "wilkinson_equal"
        window._open_module("microstrip_synthesis")
        assert window.result.kind == "microstrip"
        assert window.preview_card.isHidden()
        assert not window.exports["CSV"].isEnabled()
    finally:
        window.close()
