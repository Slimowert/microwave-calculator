import csv
import json

import pytest

from microwave_calculator.core import divider
from microwave_calculator.export import json_value, write_csv, write_json


def test_export(tmp_path):
    result = divider("wilkinson_unequal", 50, 2, 0.0008, 3.55, 2.4e9, 1.2e9, 3.6e9, 11)
    json_path, csv_path = tmp_path / "run.json", tmp_path / "run.csv"
    write_json(json_path, result)
    write_csv(csv_path, result)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["format_version"] == 1
    assert data["sections"] == result.sections
    assert data["s_at_f0"][2][0]["real"] == pytest.approx(-((2 / 3) ** 0.5))
    assert json_value(float("-inf")) == {"value": None, "status": "negative_infinity"}
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 11
    assert float(rows[0]["f, Hz"]) == 1.2e9
    assert float(rows[-1]["Re(S31), 1"]) == pytest.approx(
        result.scattering[-1, 2, 0].real
    )


def test_desktop_workflow(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from microwave_calculator.ui import QApplication, CalculatorWindow, QFileDialog

    app = QApplication.instance() or QApplication([])
    window = CalculatorWindow()
    window.show()
    try:
        for index in range(window.device.count()):
            window.device.setCurrentIndex(index)
            window.calculate_button.click()
            app.processEvents()
            assert window.result is not None, window.feedback.text()
            assert window.table.rowCount() > 0
        window.device.setCurrentIndex(window.device.findData("wilkinson_unequal"))
        window.calculate_button.click()
        for extension in ("JSON", "CSV", "PNG"):
            target = tmp_path / ("measurement." + extension.lower())
            monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
            window.exports[extension].click()
            assert target.stat().st_size > 0
        assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        window.navigation[4][0].click()
        target = tmp_path / "design"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
        hfss = next(button for name, button in window.export_actions if name == "HFSS")
        hfss.click()
        assert target.with_suffix(".py").exists()
        assert "design.py" in window.statusBar().currentMessage()
        window.navigation[0][0].click()
        window.fields["f0"][0].setText("-2")
        assert not window.exports["JSON"].isEnabled()
        window.calculate_button.click()
        assert window.result is None
        assert "частота" in window.feedback.text().lower()
        window.fields["f0"][0].setText("2.4")
        window.calculate_button.click()
        unavailable = tmp_path / "missing" / "run.json"
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *args: (str(unavailable), "")
        )
        window.exports["JSON"].click()
        assert "Не удалось сохранить" in window.feedback.text()
    finally:
        window.close()
