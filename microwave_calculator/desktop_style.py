"""Shared visual components for the desktop calculator."""

from math import sin

from PySide6.QtCore import QByteArray, QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


STYLE = """
QMainWindow, QWidget#workspace, QWidget#pages { background: #f5f8fd; color: #10182f; }
QWidget { font-size: 14px; color: #10182f; }
QWidget#sidebar { border-right: 1px solid #dce4f0; }
QLabel { background: transparent; }
QLabel#brandTitle { font-size: 14px; font-weight: 700; }
QLabel#brandSubtitle { color: #596982; font-size: 11px; }
QLabel#headline { font-size: 23px; font-weight: 700; letter-spacing: -0.6px; }
QLabel#sectionCaption { color: #8390a5; font-size: 10px; font-weight: 600; letter-spacing: 1px; }
QLabel#muted, QLabel#fixedUnit { color: #61718a; font-size: 11px; }
QLabel#status { color: #5d718b; font-size: 11px; }
QLabel#status[state="success"] { color: #087d68; }
QLabel#status[state="error"] { color: #c63b38; }
QWidget#card { background: white; border: 1px solid #dce4f0; border-radius: 10px; }
QWidget#cardHeader { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #ffffff,stop:1 #f5f8fd);
    border: 0; border-bottom: 1px solid #e3e9f2; border-top-left-radius: 10px; border-top-right-radius: 10px; }
QLabel#cardTitle { font-size: 15px; font-weight: 700; }
QWidget#metric { background: #ffffff; border: 1px solid #dce4f0; border-radius: 7px; }
QLabel#metricName { color: #34425d; font-size: 11px; }
QLabel#metricValue { color: #ff740a; font-size: 18px; font-weight: 700; }
QPushButton { padding: 8px 13px; min-height: 18px; border: 1px solid #d3ddec;
    border-radius: 7px; background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #ffffff,stop:1 #f7f9fd);
    color: #152039; font-weight: 500; }
QPushButton:hover { background: #edf5ff; border-color: #a4bde0; }
QPushButton:pressed { background: #dceaff; }
QPushButton:disabled { color: #9ba8bc; background: #f6f8fb; border-color: #e3e9f2; }
QPushButton#primary { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #0757c9,stop:1 #034298);
    color: white; border-color: #064bb0; font-weight: 600; }
QPushButton#primary:hover { background: #1268df; }
QPushButton#navigation { text-align: left; padding: 12px 18px; border: 0; border-radius: 7px;
    background: transparent; font-size: 14px; font-weight: 500; }
QPushButton#navigation:hover { background: #e7effb; }
QPushButton#navigation:checked { color: white; font-weight: 600;
    border-left: 5px solid #00d3dc; padding-left: 13px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0655bd,stop:1 #093d84); }
QPushButton#navigation:disabled { color: #a4afbf; }
QPushButton#disclosure, QPushButton#textButton { border: 0; background: transparent;
    color: #45658e; text-align: left; padding: 4px 0; font-size: 11px; min-height: 15px; }
QPushButton#textButton { color: #0757c9; }
QPushButton#moduleButton { text-align: left; padding: 20px; font-size: 14px; }
QLineEdit, QComboBox { background: #ffffff; border: 1px solid #d2ddec; border-radius: 5px;
    padding: 5px 8px; min-height: 18px; selection-background-color: #d2e8ff; color: #14203a; }
QLineEdit:hover, QComboBox:hover { border-color: #96b5e0; }
QLineEdit:focus, QComboBox:focus { border-color: #1763e8; background: #fcfdff; }
QLineEdit:disabled, QComboBox:disabled { background: #f1f4f9; color: #a0adc0; }
QComboBox { padding-right: 24px; }
QComboBox::drop-down { width: 23px; border: none; background: transparent; }
QComboBox::down-arrow { image: none; }
QComboBox QAbstractItemView { background: white; color: #14203a; outline: 0;
    border: 1px solid #cedbed; padding: 4px; selection-background-color: #d6eaff; selection-color: #0a4595; }
QCheckBox { spacing: 8px; font-size: 12px; }
QCheckBox::indicator { width: 15px; height: 15px; }
QCheckBox::indicator:unchecked { border: 1px solid #b9cbe1; background: #f9fbfe; border-radius: 3px; }
QCheckBox::indicator:checked { border-radius: 3px; background: #0757d3; border: 1px solid #0757d3; }
QCheckBox:disabled { color: #8997ad; }
QWidget#miniPreview { background: #f3f6fb; border-radius: 7px; }
QLabel#miniTitle { color: #566580; font-size: 11px; font-weight: 600; }
QLabel#deviceName { font-size: 11px; font-weight: 700; }
QLabel#deviceDescription { color: #65748c; font-size: 10px; }
QLabel#notice { background: #e7f5ff; color: #23435e; border: 1px solid #caeaff;
    border-radius: 5px; padding: 8px 10px; font-size: 11px; }
QTabWidget::pane { border: 0; }
QTableWidget { background: white; alternate-background-color: #f7faff;
    color: #14203a; border: 1px solid #dce4ef; border-radius: 5px; gridline-color: #dce4ef;
    selection-background-color: #e3eeff; selection-color: #14203a; font-size: 12px; }
QHeaderView::section { background: #e5eaf3; color: #172039; font-weight: 600;
    padding: 4px 6px; border: 0; border-right: 1px solid #d3dce9; border-bottom: 1px solid #d3dce9; }
QScrollArea { background: transparent; border: 0; }
QScrollBar:vertical { background: transparent; width: 7px; margin: 1px; }
QScrollBar::handle:vertical { background: #cbd7e7; border-radius: 3px; min-height: 25px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { color: #14203a; background: white; border: 1px solid #d4dfed; padding: 6px; }
"""


ICON_PATHS = {
    "home": '<path d="m3 10 9-7 9 7M5 9v12h5v-7h4v7h5V9"/>',
    "calculator": '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M8 7h8M8 11h1m3 0h1m3 0h0M8 15h1m3 0h1m3 0h0M8 18h1m3 0h1m3 0h0"/>',
    "wave": '<path d="M2 12h4l3-8 4 16 3-10 2 2h4"/>',
    "matrix": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M15 7h-4a2.5 2.5 0 0 0 0 5h2a2.5 2.5 0 0 1 0 5H9"/>',
    "download": '<path d="M12 3v12m-5-5 5 5 5-5M4 15v6h16v-6"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9 9a3 3 0 1 1 5 2c-1 1-2 1-2 3m0 3h.01"/>',
    "settings": '<path d="m9 3-1 3-3 1 1 3-2 2 2 2-1 3 3 1 1 3h5l1-3 3-1-1-3 2-2-2-2 1-3-3-1-1-3Z"/><circle cx="11.5" cy="12" r="3"/>',
    "layers": '<path d="m3 8 9-5 9 5-9 5Zm0 5 9 5 9-5M3 18l9 5 9-5"/>',
    "play": '<path d="m8 4 12 8-12 8Z" fill="currentColor" stroke-linejoin="round"/>',
    "reset": '<path d="M4 9a8 8 0 1 1 0 6M4 4v5h5M12 7v5l-3 2"/>',
    "warning": '<path d="m12 3 10 18H2ZM12 9v5m0 3h.01"/>',
    "scheme": '<circle cx="12" cy="4" r="2"/><circle cx="4" cy="20" r="2"/><circle cx="20" cy="20" r="2"/><circle cx="12" cy="12" r="2"/><path d="M12 6v4m-1 4-6 4m8-4 6 4"/>',
    "pie": '<path d="M11 3a9 9 0 1 0 10 10H11Zm3-1v8h8a9 9 0 0 0-8-8Z"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10h.01"/>',
}


def icon(name: str, color: str = "#40628f", size: int = 22) -> QIcon:
    """Render crisp, bundled vector icons without platform-dependent glyphs."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" '
        f'fill="none" color="{color}" stroke="{color}" stroke-width="1.7" '
        f'stroke-linecap="round" stroke-linejoin="round">{ICON_PATHS[name]}</svg>'
    )
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.fill(Qt.GlobalColor.transparent)
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QIcon(pixmap)


class Card(QWidget):
    def __init__(self, title: str, widget: QWidget, symbol: str = "layers") -> None:
        super().__init__()
        self.setObjectName("card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        header = QWidget()
        header.setObjectName("cardHeader")
        self.header_layout = QHBoxLayout(header)
        self.header_layout.setContentsMargins(13, 8, 13, 8)
        self.header_layout.setSpacing(9)
        symbol_label = QLabel()
        symbol_label.setPixmap(icon(symbol, "#ff8b0a" if symbol == "warning" else "#40689f").pixmap(21, 21))
        self.header_layout.addWidget(symbol_label)
        self.title = QLabel(title)
        self.title.setObjectName("cardTitle")
        self.header_layout.addWidget(self.title)
        self.header_layout.addStretch()
        layout.addWidget(header)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(11, 9, 11, 9)
        body_layout.addWidget(widget)
        layout.addWidget(body, 1)


class Sidebar(QWidget):
    """A subtle wave mesh painted natively, so the sidebar scales cleanly."""

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor("#f4f7fc"))
        gradient.setColorAt(1, QColor("#edf3fb"))
        painter.fillRect(self.rect(), gradient)
        painter.setPen(QPen(QColor("#dce5f1"), 1))
        painter.drawLine(self.width() - 1, 0, self.width() - 1, self.height())
        painter.setPen(QPen(QColor(83, 184, 244, 42), 0.7))
        height, width = self.height(), self.width()
        for line in range(25):
            path = QPainterPath()
            for step in range(61):
                x = step * width / 60
                y = height - 42 - line * 3 - (x / width) ** 2 * (135 + line * 2)
                y += 22 * sin(x / width * 5 + line * 0.12)
                point = QPointF(x, y)
                if step == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            painter.drawPath(path)
        painter.end()


def card(title: str, widget: QWidget, symbol: str = "layers") -> Card:
    return Card(title, widget, symbol)
