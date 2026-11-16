"""Desktop interface for Microwave Calculator."""

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from matplotlib import get_data_path
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QPointF, Qt, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QHeaderView,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .core import InputError, Result, divider, microstrip, transformer
from .export import write_csv, write_json
from .extended import lowpass, multi_divider
from .plots import book_figure, plot_overview, plot_power, plot_response, plot_schematic


DEVICES = {
    "microstrip_analysis": "Микрополосковая линия · анализ",
    "microstrip_synthesis": "Микрополосковая линия · синтез",
    "transformer": "Четвертьволновой трансформатор",
    "tee": "Т-образный делитель",
    "wilkinson_equal": "Уилкинсон · равное деление",
    "wilkinson_unequal": "Уилкинсон · неравное деление",
    "resistive": "Резистивный делитель",
    "wilkinson_n": "Уилкинсон · несколько выходов",
    "butterworth": "ФНЧ Баттерворта",
    "chebyshev": "ФНЧ Чебышёва",
}
FREQUENCY = ("Гц", "МГц", "ГГц")
LENGTH = ("м", "мм")
SCALE = {"Гц": 1, "МГц": 1e6, "ГГц": 1e9, "м": 1, "мм": 1e-3}
FIELDS = {
    "f0": ("Рабочая частота f₀", "2,4", FREQUENCY),
    "h": ("Толщина подложки h", "0,8", LENGTH),
    "er": ("Проницаемость εr", "3,55", ("1",)),
    "width": ("Ширина проводника W", "1,8", LENGTH),
    "z0": ("Сопротивление Z₀", "50", ("Ом",)),
    "load": ("Нагрузка R", "100", ("Ом",)),
    "effective": ("Эффективная проницаемость εэфф", "2,7", ("1",)),
    "ratio": ("Отношение r = P₃/P₂", "1", ("1",)),
    "start": ("Начальная частота fmin", "1,2", FREQUENCY),
    "stop": ("Конечная частота fmax", "3,6", FREQUENCY),
    "count": ("Число точек N", "1001", ("1",)),
    "outputs": ("Число выходов N", "3", ("1",)),
    "order": ("Порядок фильтра n", "3", ("1",)),
    "ripple": ("Пульсации Ap", "0,5", ("дБ",)),
}
INTEGERS = {"count", "outputs", "order"}
FILTERS = {"butterworth", "chebyshev"}

STYLE = """
QMainWindow, QWidget#workspace { background: #eef3f9; color: #142b46; }
QWidget#parameterPanel { background: white; }
QWidget#hero { background: #142e50; border-radius: 13px; }
QLabel { color: #213950; }
QLabel#headline { font-size: 23px; font-weight: bold; color: #ffffff; }
QLabel#heroSubtitle { color: #b9d2ec; }
QLabel#muted { color: #4d6780; font-size: 11px; }
QWidget#card { background: white; border: 1px solid #dae5f0; border-radius: 12px; }
QLabel#cardTitle { color: #183957; font-weight: bold; font-size: 14px; }
QWidget#metric { background: #eff6fd; border-radius: 8px; }
QLabel#metricName { color: #42607b; font-size: 11px; }
QLabel#metricValue { color: #0756a4; font-size: 17px; font-weight: bold; }
QPushButton { padding: 8px 12px; border: 1px solid #ceddeb; border-radius: 8px;
              background: white; color: #173d63; font-weight: bold; }
QPushButton:hover { background: #e6f2fd; }
QWidget#hero QPushButton { background: #244569; border-color: #406185; color: #f0f7ff; }
QWidget#hero QPushButton:hover { background: #335a81; }
QWidget#hero QPushButton:disabled { color: #9db1c8; background: #1a3555; }
QWidget#hero QPushButton#primary { background: #2375cf; color: white; border-color: #4d97e5; }
QWidget#hero QPushButton#primary:hover { background: #3687df; }
QLineEdit, QComboBox { border: 1px solid #d1deeb; border-radius: 7px; padding: 5px 7px;
                       background: #f8fbfe; color: #17334f; }
QComboBox { padding-right: 13px; }
QComboBox::drop-down { width: 20px; border: none; background: transparent; }
QComboBox::down-arrow { image: none; width: 0; height: 0; }
QComboBox QAbstractItemView { background: white; color: #17334f;
                            selection-background-color: #dcecff; selection-color: #17334f; }
QLabel#fixedUnit { color: #526a83; }
QLineEdit:focus, QComboBox:focus, QPushButton:focus { border-color: #2375cf; }
QTabWidget::pane { border: 0; }
QTabBar::tab { color: #35546f; padding: 8px 15px; margin-right: 4px;
               border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #064c94; border-bottom-color: #2375cf; font-weight: bold; }
QTabBar::tab:disabled { color: #73869c; }
QTableWidget { color: #17334f; background: white; alternate-background-color: #edf5fc;
               border: 1px solid #e3ebf3; border-radius: 6px; }
QHeaderView::section { background: #eaf2fa; color: #244868; padding: 5px; border: 0; }
"""


def card(title, widget):
    box = QWidget()
    box.setObjectName("card")
    layout = QVBoxLayout(box)
    layout.setContentsMargins(13, 10, 13, 10)
    layout.setSpacing(6)
    heading = QLabel(title)
    heading.setObjectName("cardTitle")
    layout.addWidget(heading)
    layout.addWidget(widget, 1)
    return box


def canvas():
    return FigureCanvasQTAgg(Figure(layout="constrained", facecolor="white"))


class UnitComboBox(QComboBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#526d89"), 1.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        x, y = self.width() - 13, self.height() / 2
        painter.drawLine(QPointF(x - 3.5, y - 1.5), QPointF(x, y + 2))
        painter.drawLine(QPointF(x, y + 2), QPointF(x + 3.5, y - 1.5))


class CalculatorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        font_path = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id >= 0:
            family = QFontDatabase.applicationFontFamilies(font_id)[0]
            QApplication.instance().setFont(QFont(family, 9))
        self.setWindowTitle("Microwave Calculator — расчёты СВЧ")
        self.resize(1360, 850)
        self.setStyleSheet(STYLE)
        self.result = None
        central = QWidget()
        central.setObjectName("workspace")
        self.setCentralWidget(central)
        page = QVBoxLayout(central)
        page.setContentsMargins(16, 14, 16, 14)
        page.setSpacing(9)
        hero = QWidget()
        hero.setObjectName("hero")
        page.addWidget(hero)
        toolbar = QHBoxLayout(hero)
        toolbar.setContentsMargins(17, 12, 14, 12)
        toolbar.setSpacing(8)
        brand = QVBoxLayout()
        brand.setSpacing(1)
        title = QLabel("Microwave Calculator")
        title.setObjectName("headline")
        subtitle = QLabel("Линии передачи, делители мощности и LC-фильтры")
        subtitle.setObjectName("heroSubtitle")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        toolbar.addLayout(brand)
        toolbar.addStretch()
        self.calculate_button = QPushButton("Рассчитать")
        self.calculate_button.setObjectName("primary")
        self.calculate_button.clicked.connect(self.calculate)
        toolbar.addWidget(self.calculate_button)
        reset = QPushButton("Сброс")
        reset.clicked.connect(self.rebuild_form)
        toolbar.addWidget(reset)
        self.exports = {}
        for extension in ("JSON", "CSV", "PNG"):
            button = QPushButton(f"Экспорт {extension}")
            button.clicked.connect(lambda checked=False, ext=extension: self.save(ext))
            toolbar.addWidget(button)
            self.exports[extension] = button
        self.tabs = QTabWidget()
        page.addWidget(self.tabs, 1)
        dashboard = QWidget()
        layout = QGridLayout(dashboard)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(12)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 5)
        layout.setRowStretch(0, 5)
        layout.setRowStretch(1, 6)
        self.tabs.addTab(dashboard, "Обзор")
        left = QWidget()
        left.setObjectName("parameterPanel")
        left.setMinimumWidth(320)
        panel = QVBoxLayout(left)
        panel.setContentsMargins(0, 0, 4, 0)
        panel.setSpacing(7)
        panel.addWidget(QLabel("Расчётный модуль"))
        self.device = UnitComboBox()
        for key, label in DEVICES.items():
            self.device.addItem(label, key)
        panel.addWidget(self.device)
        self.form = QFormLayout()
        self.form.setVerticalSpacing(5)
        self.form.setHorizontalSpacing(11)
        panel.addLayout(self.form)
        self.feedback = QLabel()
        self.feedback.setObjectName("muted")
        self.feedback.setWordWrap(True)
        panel.addWidget(self.feedback)
        panel.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(left)
        layout.addWidget(card("Параметры", scroll), 0, 0)
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        self.metrics = QGridLayout()
        self.metrics.setSpacing(6)
        overview_layout.addLayout(self.metrics)
        self.matrix = QTableWidget()
        self.matrix.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.matrix.setAlternatingRowColors(True)
        self.matrix.setWordWrap(False)
        self.matrix.setShowGrid(False)
        self.matrix.verticalHeader().setDefaultSectionSize(25)
        overview_layout.addWidget(self.matrix, 1)
        self.model = QLabel()
        self.model.setWordWrap(True)
        self.model.setObjectName("muted")
        overview_layout.addWidget(self.model)
        layout.addWidget(card("Результаты на центральной частоте", overview), 0, 1)
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(12)
        self.preview = canvas()
        self.preview_card = card("Частотные характеристики", self.preview)
        bottom_layout.addWidget(self.preview_card, 3)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(12)
        self.mini_scheme = canvas()
        self.balance = canvas()
        reference = QWidget()
        reference_layout = QVBoxLayout(reference)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.setSpacing(3)
        reference_layout.addWidget(self.mini_scheme, 1)
        self.reference_source = QLabel()
        self.reference_source.setObjectName("muted")
        self.reference_source.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reference_layout.addWidget(self.reference_source)
        side_layout.addWidget(card("Схема из учебника", reference), 1)
        self.balance_card = card("Баланс мощности на f₀", self.balance)
        side_layout.addWidget(self.balance_card, 1)
        bottom_layout.addWidget(side, 1)
        layout.addWidget(bottom, 1, 0, 1, 2)
        results = QWidget()
        results_layout = QVBoxLayout(results)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.table)
        self.tabs.addTab(results, "Подробности")
        self.schematic = canvas()
        self.response = canvas()
        scheme = QWidget()
        scheme_layout = QVBoxLayout(scheme)
        controls = QHBoxLayout()
        self.scheme_view = UnitComboBox()
        self.scheme_view.addItem("Из учебника", "book")
        self.scheme_view.addItem("Расчётная схема", "calculated")
        self.scheme_view.currentIndexChanged.connect(self.update_schematic)
        controls.addWidget(self.scheme_view)
        controls.addStretch()
        scheme_layout.addLayout(controls)
        self.scheme_note = QLabel()
        self.scheme_note.setWordWrap(True)
        self.scheme_note.setObjectName("muted")
        scheme_layout.addWidget(self.scheme_note)
        scheme_layout.addWidget(self.schematic, 1)
        self.tabs.addTab(scheme, "Схема")
        self.tabs.addTab(self.response, "Графики")
        self.device.currentIndexChanged.connect(self.rebuild_form)
        self.device.setCurrentIndex(self.device.findData("wilkinson_equal"))
        self.rebuild_form()
        self.calculate()

    @Slot()
    def invalidate(self) -> None:
        self.result = None
        self.table.setRowCount(0)
        self.matrix.setRowCount(0)
        self.matrix.setColumnCount(0)
        self.set_metrics(())
        self.matrix.hide()
        self.balance_card.hide()
        self.preview_card.hide()
        self.reference_source.clear()
        self.model.clear()
        for view in (
            self.schematic,
            self.response,
            self.preview,
            self.mini_scheme,
            self.balance,
        ):
            view.figure.clear()
            view.draw_idle()
        for button in self.exports.values():
            button.setEnabled(False)
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)
        self.feedback.setText("Введите исходные данные и выполните расчёт.")

    def set_metrics(self, metrics: Iterable[tuple[str, str]]) -> None:
        while self.metrics.count():
            item = self.metrics.takeAt(0)
            item.widget().deleteLater()
        for index, (name, value) in enumerate(metrics):
            tile = QWidget()
            tile.setObjectName("metric")
            layout = QVBoxLayout(tile)
            layout.setContentsMargins(10, 6, 10, 6)
            layout.setSpacing(2)
            label = QLabel(name)
            label.setObjectName("metricName")
            number = QLabel(value)
            number.setObjectName("metricValue")
            number.setWordWrap(True)
            layout.addWidget(label)
            layout.addWidget(number)
            self.metrics.addWidget(tile, index // 4, index % 4)

    @Slot()
    def update_schematic(self) -> None:
        if self.result is not None:
            self.render_schematic(self.result)

    def render_schematic(self, result: Result) -> None:
        if self.scheme_view.currentData() == "calculated":
            self.scheme_note.clear()
            plot_schematic(self.schematic.figure, result)
        else:
            book_figure(self.schematic.figure, result)
            if result.kind == "wilkinson_unequal":
                note = "На рисунке показаны внутренние нагрузки. Выходные трансформаторы программы видны в расчётной схеме."
            elif result.kind == "tee":
                note = "На рисунке — модель Т-соединения. В расчёте используются идеальное соединение и четвертьволновые ветви."
            elif result.kind in FILTERS:
                note = "На рисунке — пример ФНЧ второго порядка. Выбранный порядок виден в расчётной схеме."
            else:
                note = "Параметры на рисунке условные. Номиналы проекта видны в расчётной схеме и на вкладке «Подробности»."
            self.scheme_note.setText(note)
        self.schematic.draw_idle()

    @Slot()
    def rebuild_form(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        kind = self.device.currentData()
        if kind.startswith("microstrip"):
            keys = ["f0", "h", "er", "width" if kind.endswith("analysis") else "z0"]
        elif kind == "transformer":
            keys = ["f0", "z0", "load", "effective", "start", "stop", "count"]
        elif kind in FILTERS:
            keys = ["f0", "z0", "order"]
            if kind == "chebyshev":
                keys.append("ripple")
            keys.extend(("start", "stop", "count"))
        elif kind == "resistive":
            keys = ["f0", "z0", "start", "stop", "count"]
        else:
            keys = ["f0", "z0", "h", "er"]
            if kind == "wilkinson_n":
                keys.append("outputs")
            elif kind != "wilkinson_equal":
                keys.append("ratio")
            keys.extend(("start", "stop", "count"))
        self.fields = {}
        for key in keys:
            label, default, units = FIELDS[key]
            if kind in FILTERS and key == "f0":
                label = "Граница полосы f₀"
            entry = QLineEdit(
                "2" if key == "ratio" and kind == "wilkinson_unequal" else default
            )
            if len(units) > 1:
                unit = UnitComboBox()
                unit.addItems(units)
                unit.setCurrentIndex(len(units) - 1)
                unit.currentIndexChanged.connect(self.invalidate)
            else:
                unit = QLabel("" if units[0] == "1" else units[0])
                unit.setObjectName("fixedUnit")
                unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            unit.setFixedWidth(82)
            unit.setAccessibleName(f"Единица измерения · {label}")
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(entry)
            row_layout.addWidget(unit)
            self.form.addRow(label, row)
            entry.textChanged.connect(self.invalidate)
            self.fields[key] = (entry, unit)
        self.invalidate()

    def read_fields(self) -> dict[str, int | float]:
        data = {}
        for key, (entry, unit) in self.fields.items():
            raw = entry.text().strip().replace(",", ".")
            try:
                unit_name = (
                    unit.currentText() if isinstance(unit, QComboBox) else unit.text()
                )
                data[key] = (
                    int(raw)
                    if key in INTEGERS
                    else float(raw) * SCALE.get(unit_name, 1)
                )
            except ValueError as error:
                entry.setFocus()
                raise InputError(
                    f"{FIELDS[key][0]}: введите {'целое число' if key in INTEGERS else 'число'}."
                ) from error
        return data

    @Slot()
    def calculate(self) -> None:
        self.invalidate()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            p = self.read_fields()
            kind = self.device.currentData()
            if kind == "microstrip_analysis":
                result = microstrip(p["h"], p["er"], p["f0"], width=p["width"])
            elif kind == "microstrip_synthesis":
                result = microstrip(p["h"], p["er"], p["f0"], impedance=p["z0"])
            elif kind == "transformer":
                result = transformer(
                    p["z0"],
                    p["load"],
                    p["f0"],
                    p["effective"],
                    p["start"],
                    p["stop"],
                    p["count"],
                )
            elif kind in FILTERS:
                result = lowpass(
                    kind,
                    p["z0"],
                    p["f0"],
                    p["order"],
                    p.get("ripple"),
                    p["start"],
                    p["stop"],
                    p["count"],
                )
            elif kind in ("resistive", "wilkinson_n"):
                result = multi_divider(
                    kind,
                    p["z0"],
                    p.get("h"),
                    p.get("er"),
                    p["f0"],
                    p["start"],
                    p["stop"],
                    p["count"],
                    p.get("outputs", 2),
                )
            else:
                result = divider(
                    kind,
                    p["z0"],
                    p.get("ratio", 1),
                    p["h"],
                    p["er"],
                    p["f0"],
                    p["start"],
                    p["stop"],
                    p["count"],
                )
            self.show_result(result)
            self.result = result
            self.feedback.setText("Расчёт выполнен.")
            self.exports["JSON"].setEnabled(True)
            self.exports["CSV"].setEnabled(result.frequency is not None)
            self.exports["PNG"].setEnabled(True)
        except (InputError, OverflowError, np.linalg.LinAlgError) as error:
            self.feedback.setText(f"Ошибка расчёта: {error}")
        except Exception:
            logging.exception("Calculation or rendering failed")
            self.feedback.setText(
                "Внутренняя ошибка расчёта. Проверьте исходные данные."
            )
        finally:
            QApplication.restoreOverrideCursor()

    def show_result(self, result: Result) -> None:
        """Populate the result tabs and overview for a calculation."""
        rows = self._detail_rows(result)
        if result.center_s is not None:
            rows.extend(self._render_matrix(result.center_s))
        self._render_details(rows)
        self._render_metrics(result)
        self.model.setText(result.model)
        self.render_schematic(result)
        self.reference_source.setText(
            book_figure(self.mini_scheme.figure, result, compact=True)
        )
        if result.frequency is not None:
            self.preview_card.show()
            plot_overview(self.preview.figure, result)
        if result.center_s is not None and result.center_s.shape[0] >= 3:
            self.balance_card.show()
            plot_power(self.balance.figure, result)
        for view in (self.mini_scheme, self.preview, self.balance):
            view.draw_idle()
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, True)
        if result.frequency is not None:
            plot_response(self.response.figure, result)
            self.response.draw_idle()
            self.tabs.setTabEnabled(3, True)

    @staticmethod
    def _detail_rows(result: Result) -> list[tuple[str, float | str]]:
        rows: list[tuple[str, float | str]] = list(result.summary.items())
        if result.kind == "microstrip":
            rows.extend(
                (
                    ("Частота, ГГц", result.f0 / 1e9),
                    ("Толщина подложки, мм", result.inputs["h_m"] * 1e3),
                    ("Проницаемость подложки", result.inputs["er"]),
                )
            )
            if "target_ohm" in result.inputs:
                rows.append(
                    ("Требуемое сопротивление, Ом", result.inputs["target_ohm"])
                )
        for name, section in result.sections.items():
            if "inductance_h" in section:
                rows.append((f"{name}, нГн", section["inductance_h"] * 1e9))
                continue
            if "capacitance_f" in section:
                rows.append((f"{name}, пФ", section["capacitance_f"] * 1e12))
                continue
            rows.extend(
                (
                    (f"{name} · Z, Ом", section["z_ohm"]),
                    (f"{name} · W, мм", section["width_m"] * 1e3),
                    (f"{name} · εэфф", section["effective_er"]),
                    (f"{name} · длина, мм", section["length_m"] * 1e3),
                )
            )
        return rows

    def _render_matrix(self, scattering: np.ndarray) -> list[tuple[str, str]]:
        rows = []
        self.matrix.show()
        n = scattering.shape[0]
        self.matrix.setRowCount(n)
        self.matrix.setColumnCount(n)
        self.matrix.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
            if n <= 4
            else QHeaderView.ResizeMode.ResizeToContents
        )
        headers = [f"Порт {port + 1}" for port in range(n)]
        self.matrix.setHorizontalHeaderLabels(headers)
        self.matrix.setVerticalHeaderLabels(headers)
        for i in range(n):
            for j in range(n):
                value = scattering[i, j]
                real = 0.0 if abs(value.real) < 0.000005 else value.real
                imag = 0.0 if abs(value.imag) < 0.000005 else value.imag
                label = f"{real:.5f} {imag:+.5f}j"
                cell = QTableWidgetItem(label)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.matrix.setItem(i, j, cell)
                rows.append((f"S{i + 1}{j + 1} на f₀", label))
        return rows

    def _render_details(self, rows: list[tuple[str, float | str]]) -> None:
        self.table.setRowCount(len(rows))
        for row, (label, value) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(label))
            self.table.setItem(
                row,
                1,
                QTableWidgetItem(value if isinstance(value, str) else f"{value:.8g}"),
            )
        self.table.resizeColumnToContents(0)

    def _render_metrics(self, result: Result) -> None:
        metrics = []
        if result.sections and "z_ohm" in next(iter(result.sections.values())):
            impedances = list(
                dict.fromkeys(section["z_ohm"] for section in result.sections.values())
            )
            metrics.append(
                (
                    "Сопротивление линий",
                    " / ".join(f"{z:.4g}" for z in impedances) + " Ом",
                )
            )
        for name, value in result.summary.items():
            if name.startswith("Доля мощности"):
                if result.inputs.get("outputs", 2) <= 3:
                    metrics.append((f"Мощность · порт {name[-1]}", f"{value:.1%}"))
            elif name == "P3/P2 на f₀":
                continue
            else:
                label, separator, unit = name.rpartition(", ")
                metrics.append(
                    (label if separator else name, f"{value:.5g} {unit}".rstrip())
                )
        if result.center_s is not None and result.center_s.shape[0] >= 3:
            for label, amplitude in (
                ("Возвратные потери", abs(result.center_s[0, 0])),
                ("Развязка P₂/P₃", abs(result.center_s[1, 2])),
            ):
                metrics.append(
                    (
                        label,
                        "≥ 100 дБ"
                        if amplitude < 1e-5
                        else f"{(-20 * np.log10(amplitude)):.2f} дБ",
                    )
                )
        self.set_metrics(metrics[:8])

    def save(self, extension: str) -> None:
        if self.result is None:
            return
        suffix = extension.lower()
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Экспорт {extension}",
            f"{self.result.kind}.{suffix}",
            f"{extension} (*.{suffix})",
        )
        if not path:
            return
        if Path(path).suffix.lower() != f".{suffix}":
            path += f".{suffix}"
        try:
            if extension == "JSON":
                write_json(path, self.result)
            elif extension == "CSV":
                write_csv(path, self.result)
            else:
                canvas = (
                    self.schematic
                    if self.tabs.currentIndex() == 2 or self.result.frequency is None
                    else self.response
                )
                canvas.figure.savefig(path, dpi=170)
            self.feedback.setText(f"Файл сохранён: {path}")
        except (OSError, ValueError) as error:
            self.feedback.setText(f"Не удалось сохранить файл: {error}")


def main() -> int:
    app = QApplication([])
    window = CalculatorWindow()
    window.show()
    return app.exec()
