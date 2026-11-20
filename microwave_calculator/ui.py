"""Desktop interface for Microwave Calculator."""

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from matplotlib import get_data_path
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QPointF, QSize, Qt, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QSizePolicy,
    QStyle,
    QStyleOptionButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .core import InputError, Result, divider, microstrip, transformer
from .desktop_style import STYLE, Sidebar, card, icon
from .export import write_csv, write_json
from .extended import lowpass, multi_divider
from .hfss import HFSS_KINDS, write_hfss
from .plots import (
    book_figure,
    plot_dashboard_trace,
    plot_power,
    plot_response,
    plot_schematic,
    plot_topology_preview,
)


DEVICES = {
    "microstrip_analysis": "Микрополосковая линия · анализ",
    "microstrip_synthesis": "Микрополосковая линия · синтез",
    "transformer": "Четвертьволновой трансформатор",
    "tee": "Т-делитель",
    "wilkinson_equal": "Wilkinson 1:1",
    "wilkinson_unequal": "Wilkinson · неравное деление",
    "resistive": "Резистивный делитель",
    "wilkinson_sections": "Wilkinson · 1–8 секций",
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
    "impedances": ("Z секций, Ом", "50.2835214;52.26305344;58.57082316;70.71067812;85.36673603;95.66987903;99.43615444", ("Ом",)),
    "resistances": ("R секций, Ом", "8867.676398;652.6363658;424.8722393;208.042083;120.5338472;70.66287939;96.22736412", ("Ом",)),
    "order": ("Порядок фильтра n", "3", ("1",)),
    "ripple": ("Пульсации Ap", "0,5", ("дБ",)),
}
INTEGERS = {"count", "outputs", "order"}
FILTERS = {"butterworth", "chebyshev"}


class PlotCanvas(FigureCanvasQTAgg):
    def sizeHint(self):
        # Matplotlib's default hint follows the last rendered size. A stable
        # hint prevents a scrollable dashboard from growing after every draw.
        return QSize(300, 180)


def canvas():
    view = PlotCanvas(Figure(layout="constrained", facecolor="white"))
    view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    return view


class CheckBox(QCheckBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.isChecked():
            option = QStyleOptionButton()
            self.initStyleOption(option)
            rect = self.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, self)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor("white"), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            x, y = rect.x(), rect.y()
            painter.drawLine(QPointF(x + 3, y + 8), QPointF(x + 6, y + 11))
            painter.drawLine(QPointF(x + 6, y + 11), QPointF(x + 12, y + 4))


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
        family = "Segoe UI" if "Segoe UI" in QFontDatabase.families() else "DejaVu Sans"
        if font_id >= 0 and family == "DejaVu Sans":
            family = QFontDatabase.applicationFontFamilies(font_id)[0]
        font = QFont(family)
        font.setPixelSize(14)
        QApplication.instance().setFont(font)
        self.setWindowTitle("RF Power Divider Calculator — расчёты СВЧ")
        self.setWindowIcon(icon("wave", "#0757c9", 32))
        self.resize(1600, 940)
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(STYLE)
        self.result = None
        self.exports = {}
        self.export_actions = []
        central = QWidget()
        central.setObjectName("workspace")
        self.setCentralWidget(central)
        shell = QHBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(self._build_sidebar())
        workspace = QWidget()
        page = QVBoxLayout(workspace)
        page.setContentsMargins(16, 15, 16, 12)
        page.setSpacing(14)
        shell.addWidget(workspace, 1)
        page.addWidget(self._build_toolbar())
        self.tabs = QTabWidget()
        self.tabs.setObjectName("pages")
        self.tabs.tabBar().hide()
        self.tabs.currentChanged.connect(self._sync_navigation)
        page.addWidget(self.tabs, 1)
        self._build_dashboard()
        self._build_result_pages()
        self._build_export_page()
        self._build_help_page()
        self._build_home_page()
        self.device.setCurrentIndex(self.device.findData("wilkinson_equal"))
        self.device.currentIndexChanged.connect(self.rebuild_form)
        self.rebuild_form()
        self.calculate()
        self._sync_navigation(0)
        self.calculate_button.setShortcut("Ctrl+Return")
        self.calculate_button.setToolTip("Рассчитать (Ctrl+Enter)")

    def _build_sidebar(self) -> QWidget:
        sidebar = Sidebar()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(240)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 27, 10, 18)
        layout.setSpacing(6)
        brand = QHBoxLayout()
        brand.setContentsMargins(5, 0, 3, 0)
        brand.setSpacing(11)
        logo = QLabel()
        logo.setPixmap(icon("wave", "#ffffff", 34).pixmap(34, 34))
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setFixedSize(44, 46)
        logo.setStyleSheet("background: #084a9f; border-radius: 9px;")
        brand.addWidget(logo, 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(6)
        title = QLabel("RF Power Divider\nCalculator")
        title.setObjectName("brandTitle")
        text.addWidget(title)
        subtitle = QLabel("Калькулятор СВЧ\nделителей мощности")
        subtitle.setObjectName("brandSubtitle")
        text.addWidget(subtitle)
        brand.addLayout(text)
        layout.addLayout(brand)
        layout.addSpacing(40)
        self.navigation = {}
        for title, symbol, index in (
            ("Главная", "home", 6),
            ("Расчёт", "calculator", 0),
            ("Частотный анализ", "wave", 3),
            ("S-параметры", "matrix", 1),
            ("Экспорт", "download", 4),
            ("Справка", "help", 5),
        ):
            button = QPushButton(title)
            button.setObjectName("navigation")
            button.setCheckable(True)
            button.setIcon(icon(symbol, "#19233d"))
            button.setIconSize(QSize(21, 21))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda checked=False, target=index: self.tabs.setCurrentIndex(target))
            layout.addWidget(button)
            self.navigation[index] = (button, symbol)
        layout.addStretch()
        version = QLabel("RF ENGINEERING  /  V1.0")
        version.setObjectName("sectionCaption")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)
        return sidebar

    def _build_toolbar(self) -> QWidget:
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(2, 0, 0, 0)
        layout.setSpacing(9)
        accent = QLabel()
        accent.setFixedSize(5, 26)
        accent.setStyleSheet("background: #06bdff; border-radius: 2px;")
        layout.addWidget(accent)
        layout.addSpacing(4)
        self.headline = QLabel("Калькулятор делителей мощности")
        self.headline.setObjectName("headline")
        layout.addWidget(self.headline, 1)
        self.calculate_button = QPushButton("  Рассчитать")
        self.calculate_button.setObjectName("primary")
        self.calculate_button.setIcon(icon("play", "white", 18))
        self.calculate_button.clicked.connect(self.calculate)
        layout.addWidget(self.calculate_button)
        reset = QPushButton("  Сброс")
        reset.setIcon(icon("reset", "#19233d", 19))
        reset.clicked.connect(self.rebuild_form)
        layout.addWidget(reset)
        for extension in ("JSON", "CSV", "PNG"):
            button = self._export_button(extension)
            self.exports[extension] = button
            if extension != "PNG":
                layout.addWidget(button)
        return toolbar

    def _export_button(self, extension: str) -> QPushButton:
        button = QPushButton(f"  Экспорт {extension}")
        button.setIcon(icon("download", "#19233d", 19))
        button.clicked.connect(lambda checked=False, ext=extension: self.save(ext))
        self.export_actions.append((extension, button))
        return button

    def _build_dashboard(self) -> None:
        dashboard = QWidget()
        dashboard.setMinimumWidth(1010)
        layout = QVBoxLayout(dashboard)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        top = QWidget()
        top.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(12)
        self.parameter_card = self._build_parameters()
        top_layout.addWidget(self.parameter_card, 46)
        right = QWidget()
        results = QVBoxLayout(right)
        results.setContentsMargins(0, 0, 0, 0)
        results.setSpacing(10)
        overview = QWidget()
        self.metrics = QGridLayout(overview)
        self.metrics.setContentsMargins(0, 0, 0, 0)
        self.metrics.setSpacing(7)
        self.results_card = card("Результаты расчёта", overview)
        results.addWidget(self.results_card)
        self.matrix = QTableWidget()
        self.matrix.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.matrix.setAlternatingRowColors(True)
        self.matrix.setWordWrap(False)
        self.matrix.setShowGrid(True)
        self.matrix.verticalHeader().setDefaultSectionSize(27)
        self.matrix.verticalHeader().setMinimumSectionSize(25)
        self.matrix.setMinimumHeight(118)
        self.matrix.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.matrix_card = card("S-матрица", self.matrix, "matrix")
        results.addWidget(self.matrix_card, 1)
        self.model = QLabel()
        self.model.setWordWrap(True)
        self.model.setObjectName("notice")
        self.model.setMinimumHeight(35)
        self.warning_card = card("Предупреждения", self.model, "warning")
        results.addWidget(self.warning_card)
        top_layout.addWidget(right, 54)
        top.setMinimumHeight(395)
        layout.addWidget(top, 11)
        bottom = QWidget()
        bottom.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(12)
        self.preview = canvas()
        self.reflection = canvas()
        self.preview_card = card("|S₂₁|, |S₃₁| по частоте", self.preview, "wave")
        self.reflection_card = card("Отражение и связь выходов", self.reflection, "wave")
        bottom_layout.addWidget(self.preview_card, 1)
        bottom_layout.addWidget(self.reflection_card, 1)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)
        self.mini_scheme = canvas()
        self.scheme_card = card("Схема (Wilkinson 1:1)", self.mini_scheme, "scheme")
        open_scheme = QPushButton("Открыть")
        open_scheme.setObjectName("textButton")
        open_scheme.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        self.scheme_card.header_layout.addWidget(open_scheme)
        side_layout.addWidget(self.scheme_card, 1)
        self.balance = canvas()
        self.balance_card = card("Баланс мощности", self.balance, "pie")
        side_layout.addWidget(self.balance_card, 1)
        bottom_layout.addWidget(side, 1)
        bottom.setMinimumHeight(340)
        layout.addWidget(bottom, 10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(dashboard)
        self.tabs.addTab(scroll, "Расчёт")

    def _build_parameters(self) -> QWidget:
        content = QWidget()
        panel = QHBoxLayout(content)
        panel.setContentsMargins(2, 2, 2, 2)
        panel.setSpacing(13)
        controls = QWidget()
        controls.setMinimumWidth(280)
        fields_layout = QVBoxLayout(controls)
        fields_layout.setContentsMargins(0, 0, 0, 0)
        fields_layout.setSpacing(8)
        self.form = QFormLayout()
        self.form.setVerticalSpacing(10)
        self.form.setHorizontalSpacing(12)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        fields_layout.addLayout(self.form)
        self.device = UnitComboBox()
        self.device.setAccessibleName("Тип устройства")
        for key, label in DEVICES.items():
            self.device.addItem(label, key)
        self.show_matrix = CheckBox("Показать S-матрицу")
        self.show_matrix.setChecked(True)
        self.show_matrix.toggled.connect(self._update_view_options)
        fields_layout.addWidget(self.show_matrix)
        self.show_sweep = CheckBox("Частотные графики")
        self.show_sweep.setChecked(True)
        self.show_sweep.toggled.connect(self._update_view_options)
        fields_layout.addWidget(self.show_sweep)
        self.sweep_container = QWidget()
        self.sweep_layout = QHBoxLayout(self.sweep_container)
        self.sweep_layout.setContentsMargins(0, 0, 0, 0)
        self.sweep_layout.setSpacing(5)
        fields_layout.addWidget(self.sweep_container)
        self.advanced_button = QPushButton("▸ Подложка и точность")
        self.advanced_button.setObjectName("disclosure")
        self.advanced_button.setCheckable(True)
        self.advanced_button.toggled.connect(self._toggle_advanced)
        fields_layout.addWidget(self.advanced_button)
        self.advanced_container = QWidget()
        self.advanced_form = QFormLayout(self.advanced_container)
        self.advanced_form.setContentsMargins(0, 0, 0, 0)
        self.advanced_form.setVerticalSpacing(7)
        fields_layout.addWidget(self.advanced_container)
        self.advanced_container.hide()
        self.feedback = QLabel()
        self.feedback.setObjectName("status")
        self.feedback.setWordWrap(True)
        fields_layout.addStretch()
        fields_layout.addWidget(self.feedback)
        panel.addWidget(controls, 3)
        miniature = QWidget()
        miniature.setObjectName("miniPreview")
        miniature.setMinimumWidth(135)
        miniature.setMaximumWidth(210)
        mini_layout = QVBoxLayout(miniature)
        mini_layout.setContentsMargins(8, 10, 8, 10)
        mini_layout.setSpacing(5)
        mini_title = QLabel("Мини-схема")
        mini_title.setObjectName("miniTitle")
        mini_title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        mini_layout.addWidget(mini_title)
        self.parameter_scheme = canvas()
        self.parameter_scheme.setMaximumHeight(220)
        mini_layout.addWidget(self.parameter_scheme, 1)
        self.device_name = QLabel()
        self.device_name.setObjectName("deviceName")
        self.device_name.setWordWrap(True)
        self.device_name.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        mini_layout.addWidget(self.device_name)
        self.device_description = QLabel()
        self.device_description.setObjectName("deviceDescription")
        self.device_description.setWordWrap(True)
        self.device_description.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.device_description.setMinimumHeight(50)
        mini_layout.addWidget(self.device_description)
        panel.addWidget(miniature, 2)
        return card("Параметры", content, "settings")

    def _build_result_pages(self) -> None:
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(card("S-параметры и подробные результаты", self.table, "matrix"), "Подробности")
        self.schematic = canvas()
        self.response = canvas()
        scheme = QWidget()
        scheme_layout = QVBoxLayout(scheme)
        scheme_layout.setContentsMargins(5, 0, 5, 0)
        controls = QHBoxLayout()
        self.scheme_view = UnitComboBox()
        self.scheme_view.addItem("Расчётная схема", "calculated")
        self.scheme_view.addItem("Из учебника", "book")
        self.scheme_view.currentIndexChanged.connect(self.update_schematic)
        controls.addWidget(self.scheme_view)
        controls.addStretch()
        controls.addWidget(self.exports["PNG"])
        scheme_layout.addLayout(controls)
        self.scheme_note = QLabel()
        self.scheme_note.setWordWrap(True)
        self.scheme_note.setObjectName("muted")
        scheme_layout.addWidget(self.scheme_note)
        scheme_layout.addWidget(card("Схема устройства", self.schematic, "scheme"), 1)
        self.tabs.addTab(scheme, "Схема")
        self.tabs.addTab(card("Частотный анализ", self.response, "wave"), "Графики")

    def _build_export_page(self) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 24, 20, 24)
        layout.setSpacing(18)
        note = QLabel("Сохраните текущий расчёт в удобном формате.")
        note.setObjectName("cardTitle")
        layout.addWidget(note)
        for extension, title, description in (
            ("JSON", "Исходные данные и результаты", "Параметры устройства, номиналы элементов и комплексная S-матрица на f₀."),
            ("CSV", "Частотные характеристики", "Все точки частотной сетки для анализа в Excel, Python и других приложениях."),
            ("PNG", "Графики и схемы", "Частотные графики. Для сохранения схемы откройте её и нажмите «Экспорт PNG»."),
            ("HFSS", "3D-модель для Ansys HFSS", "Скрипт .py для линии, Т-делителя или двухвыходного Wilkinson. "
             "В Electronics Desktop: Tools → Run Script. Затем Analyze All для построения графиков."),
        ):
            row = QWidget()
            line = QHBoxLayout(row)
            text = QVBoxLayout()
            heading = QLabel(title)
            heading.setObjectName("cardTitle")
            text.addWidget(heading)
            body = QLabel(description)
            body.setWordWrap(True)
            body.setObjectName("muted")
            text.addWidget(body)
            line.addLayout(text, 1)
            line.addWidget(self._export_button(extension))
            layout.addWidget(row)
        layout.addStretch()
        self.tabs.addTab(card("Экспорт расчёта", content, "download"), "Экспорт")

    def _build_help_page(self) -> None:
        text = QLabel(
            "<h2>Расчёты СВЧ в одном окне</h2>"
            "<p>Выберите устройство, задайте параметры и нажмите <b>Рассчитать</b> "
            "или <b>Ctrl+Enter</b>. Числа можно вводить с точкой или запятой.</p>"
            "<h3>Параметры и единицы</h3>"
            "<p>Z₀ — опорное сопротивление портов, f₀ — рабочая частота, "
            "r = P₃/P₂ — отношение выходных мощностей. Параметры подложки и число "
            "точек частотной сетки доступны в раскрывающемся блоке.</p>"
            "<h3>Результаты</h3>"
            "<p>S-матрица показана на рабочей частоте. Графики описывают поведение "
            "в заданном диапазоне; оранжевая линия отмечает f₀. Return Loss — "
            "возвратные потери, Isolation — развязка выходов. Для идеального "
            "согласования значения ограничены порогом отображения 100 дБ.</p>"
            "<h3>Модель</h3>"
            "<p>Используются аналитические модели с идеальными соединениями. "
            "Потери, дисперсия и паразитные эффекты не учитываются. Ограничения "
            "текущего расчёта приведены в блоке «Предупреждения».</p>"
            "<p>На странице схемы можно переключиться между расчётной топологией "
            "и иллюстрацией из Microwave Engineering, D. M. Pozar, 4-е издание.</p>"
        )
        text.setWordWrap(True)
        text.setMargin(20)
        text.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(text)
        self.tabs.addTab(card("Справка", scroll, "help"), "Справка")

    def _build_home_page(self) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(25, 30, 25, 25)
        heading = QLabel("От линии передачи до готового делителя")
        heading.setObjectName("headline")
        layout.addWidget(heading)
        description = QLabel("Выберите расчётный модуль. Параметры, схема и характеристики появятся в рабочем окне.")
        description.setWordWrap(True)
        description.setObjectName("muted")
        layout.addWidget(description)
        layout.addSpacing(25)
        grid = QGridLayout()
        grid.setSpacing(14)
        for position, (kind, title) in enumerate((
            ("wilkinson_equal", "Wilkinson 1:1"),
            ("wilkinson_unequal", "Неравный делитель"),
            ("wilkinson_sections", "Многосекционный Wilkinson"),
            ("microstrip_synthesis", "Микрополосковая линия"),
            ("transformer", "Трансформатор λ/4"),
            ("butterworth", "LC-фильтр нижних частот"),
        )):
            button = QPushButton(title)
            button.setObjectName("moduleButton")
            button.setIcon(icon("scheme", "#0757c9", 28))
            button.setIconSize(QSize(28, 28))
            button.clicked.connect(lambda checked=False, key=kind: self._open_module(key))
            grid.addWidget(button, position // 2, position % 2)
        layout.addLayout(grid)
        layout.addStretch()
        self.tabs.addTab(card("Расчётные модули", content, "home"), "Главная")

    def _open_module(self, kind: str) -> None:
        self.device.setCurrentIndex(self.device.findData(kind))
        self.tabs.setCurrentIndex(0)
        self.calculate()

    def _sync_navigation(self, index: int) -> None:
        active = 0 if index == 2 else index
        titles = {
            0: "Калькулятор делителей мощности", 1: "S-параметры и результаты",
            2: "Схема устройства", 3: "Частотный анализ", 4: "Экспорт расчёта",
            5: "Справка", 6: "Расчётные модули",
        }
        title = titles.get(index, titles[0])
        self.headline.setToolTip(title)
        if self.width() < 1320:
            title = {0: "Калькулятор СВЧ", 1: "S-параметры"}.get(index, title)
        self.headline.setText(title)
        for target, (button, symbol) in self.navigation.items():
            button.setChecked(active == target)
            button.setIcon(icon(symbol, "white" if active == target else "#19233d"))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "headline") and hasattr(self, "tabs"):
            self._sync_navigation(self.tabs.currentIndex())

    def _toggle_advanced(self, expanded: bool) -> None:
        self.advanced_container.setVisible(expanded)
        self.advanced_button.setText(("▾ " if expanded else "▸ ") + "Подложка и точность")

    def _update_view_options(self) -> None:
        result = self.result
        has_s = result is not None and result.center_s is not None
        self.matrix_card.setVisible(has_s and self.show_matrix.isChecked())
        has_sweep = result is not None and result.frequency is not None
        self.preview_card.setVisible(has_sweep and self.show_sweep.isChecked())
        self.reflection_card.setVisible(has_sweep and self.show_sweep.isChecked())

    def _set_feedback(self, message: str, state: str = "") -> None:
        self.feedback.setText(message)
        if self.tabs.currentIndex() != 0:
            self.statusBar().showMessage(message)
        elif self.statusBar().currentMessage():
            self.statusBar().clearMessage()
        self.feedback.setProperty("state", state)
        self.feedback.style().unpolish(self.feedback)
        self.feedback.style().polish(self.feedback)

    @Slot()
    def invalidate(self) -> None:
        self.result = None
        self.table.setRowCount(0)
        self.matrix.setRowCount(0)
        self.matrix.setColumnCount(0)
        self.set_metrics((("Результат", "—"), ("Частота f₀", "—")))
        self.matrix_card.hide()
        self.balance_card.hide()
        self.preview_card.hide()
        self.reflection_card.hide()
        self.model.setText("Задайте параметры устройства и выполните расчёт.")
        for view in (
            self.schematic, self.response, self.preview, self.reflection,
            self.mini_scheme, self.balance,
        ):
            view.figure.clear()
            view.draw_idle()
        for _, button in self.export_actions:
            button.setEnabled(False)
        for index in (1, 2, 3):
            self.tabs.setTabEnabled(index, False)
        for index in (1, 3):
            self.navigation[index][0].setEnabled(False)
        self._set_feedback("Изменены параметры. Нажмите «Рассчитать».")

    def set_metrics(self, metrics: Iterable[tuple[str, str]]) -> None:
        while self.metrics.count():
            item = self.metrics.takeAt(0)
            item.widget().hide()
            item.widget().deleteLater()
        for index in range(self.metrics.columnCount()):
            self.metrics.setColumnStretch(index, 0)
        values = list(metrics)
        columns = len(values) if len(values) <= 6 else 4
        for index, (name, value) in enumerate(values):
            tile = QWidget()
            tile.setObjectName("metric")
            layout = QVBoxLayout(tile)
            layout.setContentsMargins(6, 7, 6, 7)
            layout.setSpacing(5)
            label = QLabel(name)
            label.setObjectName("metricName")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            number = QLabel(value)
            number.setObjectName("metricValue")
            number.setAlignment(Qt.AlignmentFlag.AlignCenter)
            number.setWordWrap(len(value) > 12)
            layout.addWidget(label)
            layout.addWidget(number)
            self.metrics.addWidget(tile, index // columns, index % columns)
            self.metrics.setColumnStretch(index % columns, 1)

    @Slot()
    def update_schematic(self) -> None:
        if self.result is not None:
            self.render_schematic(self.result)

    def render_schematic(self, result: Result) -> None:
        self.scheme_view.setEnabled(result.kind != "wilkinson_sections")
        if result.kind == "wilkinson_sections" or self.scheme_view.currentData() == "calculated":
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
        # Keep the device selector alive when rebuilding the dynamic form.
        for form in (self.form, self.advanced_form):
            while form.rowCount():
                row = form.takeRow(0)
                for item in (row.labelItem, row.fieldItem):
                    if item is not None and item.widget() is not None:
                        widget = item.widget()
                        if widget is self.device:
                            widget.setParent(None)
                        else:
                            widget.hide()
                            widget.deleteLater()
        while self.sweep_layout.count():
            item = self.sweep_layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        kind = self.device.currentData()
        device_label = QLabel("Устройство", self.form.parentWidget())
        device_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        device_label.ensurePolished()
        device_label.setMinimumWidth(device_label.sizeHint().width())
        self.form.addRow(device_label, self.device)
        if kind.startswith("microstrip"):
            keys = ["f0", "h", "er", "width" if kind.endswith("analysis") else "z0"]
        elif kind == "transformer":
            keys = ["z0", "f0", "load", "effective", "start", "stop", "count"]
        elif kind in FILTERS:
            keys = ["z0", "f0", "order"]
            if kind == "chebyshev":
                keys.append("ripple")
            keys.extend(("start", "stop", "count"))
        elif kind == "resistive":
            keys = ["z0", "f0", "start", "stop", "count"]
        else:
            keys = ["z0", "f0", "h", "er"]
            if kind == "wilkinson_sections":
                keys.extend(("impedances", "resistances"))
            else:
                keys.append("ratio")
            keys.extend(("start", "stop", "count"))
        self.fields = {}
        for key in keys:
            label, default, units = FIELDS[key]
            label = {
                "z0": "Z₀, Ом", "f0": "Частота f₀", "ratio": "Деление P₃/P₂",
                "outputs": "Выходы N", "h": "Подложка h", "er": "Проницаемость εr",
                "width": "Ширина W", "load": "Нагрузка R", "effective": "ε эффективная",
                "order": "Порядок n", "ripple": "Пульсации Aₚ", "count": "Точек N",
            }.get(key, label)
            if kind in FILTERS and key == "f0":
                label = "Граница f₀"
            entry = QLineEdit("2" if key == "ratio" and kind == "wilkinson_unequal" else default)
            if kind == "wilkinson_sections":
                entry.setText({"f0": "5,5", "start": "4,9", "stop": "6", "count": "201"}.get(key, default))
            entry.setAccessibleName(label)
            entry.setMinimumWidth(44)
            if len(units) > 1:
                unit = UnitComboBox()
                unit.addItems(units)
                unit.setCurrentIndex(len(units) - 1)
                unit.setFixedWidth(80)
                unit.currentIndexChanged.connect(self.invalidate)
            else:
                unit = QLabel("" if units[0] in ("1", "Ом") else units[0])
                unit.setObjectName("fixedUnit")
                unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
                unit.setFixedWidth(30 if units[0] == "дБ" else 0)
            unit.setAccessibleName(f"Единица измерения · {label}")
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setSpacing(5)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(entry, 1)
            row_layout.addWidget(unit)
            if key in ("start", "stop"):
                self.sweep_layout.addWidget(QLabel("от" if key == "start" else "до"))
                self.sweep_layout.addWidget(row, 1)
            elif key in ("h", "er", "count") and not kind.startswith("microstrip"):
                self.advanced_form.addRow(label, row)
            else:
                field_label = QLabel(label, self.form.parentWidget())
                field_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
                field_label.ensurePolished()
                field_label.setMinimumWidth(field_label.sizeHint().width())
                self.form.addRow(field_label, row)
            if key == "ratio" and kind == "wilkinson_equal":
                entry.setEnabled(False)
                entry.setToolTip("Wilkinson 1:1 делит мощность поровну.")
            entry.textChanged.connect(self.invalidate)
            self.fields[key] = (entry, unit)
        has_sweep = "start" in self.fields
        self.show_sweep.setVisible(has_sweep)
        self.sweep_container.setVisible(has_sweep)
        self.show_matrix.setVisible(not kind.startswith("microstrip") and kind != "transformer")
        self.advanced_button.setVisible(self.advanced_form.rowCount() > 0)
        self.advanced_container.setVisible(self.advanced_button.isChecked() and self.advanced_form.rowCount() > 0)
        self.device_name.setText(DEVICES[kind])
        self.device.setToolTip(DEVICES[kind])
        descriptions = {
            "wilkinson_equal": "Два равных выхода с согласованием и развязкой портов.",
            "wilkinson_unequal": "Два выхода с заданным отношением мощностей.",
            "wilkinson_sections": "Два выхода. Введите Z и R через «;», начиная с выходов. Число пар = число секций.",
            "tee": "Четвертьволновые ветви без развязывающего резистора.",
            "resistive": "Широкополосный резистивный делитель на два выхода.",
            "transformer": "Согласование нагрузки участком линии длиной λ/4.",
            "butterworth": "Максимально плоская характеристика в полосе пропускания.",
            "chebyshev": "Заданная пульсация и крутой спад за полосой пропускания.",
        }
        self.device_description.setText(descriptions.get(kind, "Анализ и синтез геометрии микрополосковой линии."))
        reference_kind = "microstrip" if kind.startswith("microstrip") else kind
        reference = Result(reference_kind, {}, {}, "", 1.0)
        plot_topology_preview(self.parameter_scheme.figure, reference)
        self.parameter_scheme.draw_idle()
        self.invalidate()

    def read_fields(self) -> dict[str, int | float]:
        data = {}
        for key, (entry, unit) in self.fields.items():
            raw = entry.text().strip().replace(",", ".")
            try:
                if key in ("impedances", "resistances"):
                    data[key] = tuple(float(item) for item in raw.split(";"))
                    continue
                unit_name = (
                    unit.currentText() if isinstance(unit, QComboBox) else unit.text()
                )
                data[key] = (
                    int(raw)
                    if key in INTEGERS
                    else float(raw) * SCALE.get(unit_name, 1)
                )
            except ValueError as error:
                self.tabs.setCurrentIndex(0)
                if self.advanced_container.isAncestorOf(entry):
                    self.advanced_button.setChecked(True)
                entry.setFocus()
                entry.selectAll()
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
            elif kind in ("resistive", "wilkinson_sections"):
                result = multi_divider(
                    kind,
                    p["z0"],
                    p.get("h"),
                    p.get("er"),
                    p["f0"],
                    p["start"],
                    p["stop"],
                    p["count"],
                    impedances=p.get("impedances", ()), resistances=p.get("resistances", ()),
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
            self._update_view_options()
            self._set_feedback("Расчёт выполнен · f₀ = " + f"{result.f0 / 1e9:.3f} ГГц", "success")
            for extension, button in self.export_actions:
                available = (extension != "CSV" or result.frequency is not None) and (
                    extension != "HFSS" or result.kind in HFSS_KINDS
                )
                button.setEnabled(available)
                if extension == "HFSS":
                    button.setToolTip("Сохранить скрипт построения модели" if available
                                      else "Для этого устройства физическая разводка пока не задана.")
        except (InputError, OverflowError, np.linalg.LinAlgError) as error:
            self._set_feedback(f"Ошибка расчёта: {error}", "error")
        except Exception:
            logging.exception("Calculation or rendering failed")
            self._set_feedback("Внутренняя ошибка расчёта. Проверьте исходные данные.", "error")
        finally:
            QApplication.restoreOverrideCursor()

    def show_result(self, result: Result) -> None:
        """Populate the result tabs and overview for a calculation."""
        rows = self._detail_rows(result)
        if result.center_s is not None:
            rows.extend(self._render_matrix(result.center_s))
        self._render_details(rows)
        self._render_metrics(result)
        notice = (
            "Идеализированная модель. Не учитываются потери линий и паразитные эффекты."
            if result.kind not in ("microstrip", "resistive")
            else result.model.split("\n")[0]
        )
        if result.kind == "tee":
            notice += " Т-соединение не обеспечивает развязку выходов."
        self.model.setText(notice)
        self.model.setToolTip(result.model)
        self.render_schematic(result)
        plot_schematic(self.mini_scheme.figure, result, compact=True)
        self.scheme_card.title.setText(f"Схема ({DEVICES[self.device.currentData()]})")
        self.matrix_card.title.setText(f"S-матрица (на f₀ = {result.f0 / 1e9:.3f} ГГц)")
        if result.frequency is not None:
            plot_dashboard_trace(self.preview.figure, result)
            plot_dashboard_trace(self.reflection.figure, result, reflection=True)
            if result.center_s is None:
                titles = ("Коэффициент отражения |Γ|", "КСВН по частоте")
            elif result.center_s.shape[0] == 2:
                titles = ("Передача |S₂₁| по частоте", "Отражение |S₁₁| по частоте")
            else:
                titles = ("Передача на выходы по частоте", "Отражение и связь выходов")
            self.preview_card.title.setText(titles[0])
            self.reflection_card.title.setText(titles[1])
        if result.center_s is not None and result.center_s.shape[0] >= 3:
            self.balance_card.show()
            plot_power(self.balance.figure, result)
        for view in (self.mini_scheme, self.preview, self.reflection, self.balance):
            view.draw_idle()
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, True)
        self.navigation[1][0].setEnabled(True)
        if result.frequency is not None:
            plot_response(self.response.figure, result)
            self.response.draw_idle()
            self.tabs.setTabEnabled(3, True)
            self.navigation[3][0].setEnabled(True)

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
            if "r_ohm" in section:
                rows.append((f"{name} · R, Ом", section["r_ohm"]))
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
        n = scattering.shape[0]
        self.matrix.setRowCount(n)
        self.matrix.setColumnCount(n + 1)
        self.matrix.verticalHeader().hide()
        self.matrix.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch if n <= 4 else QHeaderView.ResizeMode.ResizeToContents
        )
        subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        headers = ["S" + str(port + 1).translate(subscript) for port in range(n)]
        self.matrix.setHorizontalHeaderLabels(["Порт"] + headers)
        self.matrix.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for i in range(n):
            port = QTableWidgetItem(headers[i])
            port.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.matrix.setItem(i, 0, port)
            for j in range(n):
                value = scattering[i, j]
                real = 0.0 if abs(value.real) < 0.00005 else value.real
                imag = 0.0 if abs(value.imag) < 0.00005 else value.imag
                label = f"{real:.4f} {'−' if imag < 0 else '+'} j{abs(imag):.4f}"
                cell = QTableWidgetItem(label)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.matrix.setItem(i, j + 1, cell)
                rows.append((f"S{i + 1}{j + 1} на f₀", label))
        self.matrix.show()
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
        is_divider = result.center_s is not None and result.center_s.shape[0] >= 3
        if is_divider:
            sections = list(result.sections.values())
            if result.kind == "wilkinson_sections":
                metrics.append(("Секций", str(len(sections))))
            elif sections and "z_ohm" in sections[0]:
                impedances = list(dict.fromkeys(section["z_ohm"] for section in sections[:2]))
                metrics.append(("Z₂ / Z₃" if len(impedances) > 1 else "Z ветви", " / ".join(f"{z:.2f}" for z in impedances) + " Ω"))
            resistor = result.summary.get("Развязывающий резистор, Ом", result.summary.get("Каждый резистор звезды, Ом"))
            if resistor is not None:
                metrics.append(("R", f"{resistor:.4g} Ω"))
            outputs = result.center_s.shape[0] - 1
            if outputs == 2:
                for port in (1, 2):
                    metrics.append((("P₂", "P₃")[port - 1], f"{abs(result.center_s[port, 0]) ** 2:.0%}"))
            else:
                metrics.append(("Выходы N", str(outputs)))
                metrics.append(("На каждый выход", f"{abs(result.center_s[1, 0]) ** 2:.1%}"))
            for label, amplitude in (("Return Loss", abs(result.center_s[0, 0])), ("Isolation", abs(result.center_s[1, 2]))):
                metrics.append((label, "≥ 100 дБ" if amplitude < 1e-5 else f"{-20 * np.log10(amplitude):.2f} дБ"))
        else:
            for name, value in result.summary.items():
                label, separator, unit = name.rpartition(", ")
                metrics.append((label if separator else name, f"{value:.4g} {unit}".rstrip()))
        self.set_metrics(metrics[:8])

    def save(self, extension: str) -> None:
        if self.result is None:
            return
        suffix = "py" if extension == "HFSS" else extension.lower()
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
            elif extension == "HFSS":
                write_hfss(path, self.result)
            else:
                canvas = (
                    self.schematic
                    if self.tabs.currentIndex() == 2 or self.result.frequency is None
                    else self.response
                )
                canvas.figure.savefig(path, dpi=170)
            self._set_feedback(f"Файл сохранён: {path}", "success")
        except (OSError, ValueError) as error:
            self._set_feedback(f"Не удалось сохранить файл: {error}", "error")


def main() -> int:
    app = QApplication([])
    window = CalculatorWindow()
    window.show()
    return app.exec()
