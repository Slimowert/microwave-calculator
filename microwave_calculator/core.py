"""Analytical models. All dimensional parameters use SI units."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite, log, pi, sqrt

import numpy as np
from scipy.constants import c
from scipy.optimize import brentq


class InputError(ValueError):
    """Indicate that a calculation input is outside the supported range."""


@dataclass(frozen=True)
class Trace:
    """Store named samples of a frequency-dependent quantity."""

    name: str
    unit: str
    values: np.ndarray


@dataclass(frozen=True)
class Result:
    """Store calculation inputs, design values, and optional sweep data."""

    kind: str
    inputs: dict[str, float | np.integer | None | list[float]]
    summary: dict[str, float]
    model: str
    f0: float
    frequency: np.ndarray | None = None
    traces: tuple[Trace, ...] = ()
    scattering: np.ndarray | None = None
    center_s: np.ndarray | None = None
    sections: dict[str, dict[str, float]] = field(default_factory=dict)
    topology: dict[str, object] = field(default_factory=dict)


MICRO = (
    "Квазистатическая микрополосковая модель Позара. Толщина металла, "
    "потери и дисперсия не учитываются. Расчётный диапазон 0,01 ≤ W/h ≤ 100, "
    "1 ≤ εr ≤ 20. При h > λ₀/20 размеры требуют особой осторожности."
)
IDEAL = (
    "Линии без потерь, постоянные параметры по частоте, идеальные соединения "
    "и резистор. Изгибы, излучение, взаимная связь и паразитные параметры "
    "не учитываются. Опорные сопротивления внешних портов одинаковы."
)


def positive(value: float, label: str, minimum: float = 0) -> float:
    """Validate a positive finite value and return it unchanged."""
    if not isfinite(value) or value <= 0 or value < minimum:
        bound = f"не менее {minimum}" if minimum else "больше нуля"
        raise InputError(f"{label}: укажите конечное число {bound}.")
    return value


def substrate(h: float, er: float) -> None:
    """Validate the substrate thickness and relative permittivity."""
    positive(h, "Толщина подложки")
    positive(er, "Проницаемость подложки", 1)
    if er > 20:
        raise InputError("Проницаемость подложки: допустимо от 1 до 20.")


def microstrip_model(u: float, er: float) -> tuple[float, float]:
    """Return characteristic impedance and effective permittivity at W/h=u."""
    effective = (er + 1) / 2 + (er - 1) / (2 * sqrt(1 + 12 / u))
    if u <= 1:
        impedance = 60 / sqrt(effective) * log(8 / u + u / 4)
    else:
        impedance = 120 * pi / (sqrt(effective) * (u + 1.393 + 0.667 * log(u + 1.444)))
    return impedance, effective


def synthesize_width(z: float, h: float, er: float) -> float:
    """Find a conductor width within the impedance tolerance of the model."""
    positive(z, "Волновое сопротивление")
    substrate(h, er)
    # The textbook approximation has two branches and a small jump near W/h=1.
    intervals = ((0.01, 1.0), (np.nextafter(1.0, 2.0), 100.0))

    def residual(u: float) -> float:
        return microstrip_model(u, er)[0] - z

    candidates = []
    for lo, hi in intervals:
        candidates.extend((lo, hi))
        if residual(lo) * residual(hi) < 0:
            candidates.append(brentq(residual, lo, hi))
    u = min(candidates, key=lambda value: abs(residual(value)))
    if abs(microstrip_model(u, er)[0] / z - 1) > 0.001:
        raise InputError(
            f"Волновое сопротивление {z:.6g} Ом: синтез с допуском 0,1% "
            "невозможен в диапазоне модели."
        )
    return positive(u * h, "Вычисленная ширина проводника")


def wavelength(frequency: float, effective: float) -> float:
    """Return the guided wavelength in metres."""
    length = (c / sqrt(effective)) / frequency
    if not isfinite(length) or not isfinite(length * 1e3):
        raise InputError("Частота: длина волны выходит за численный диапазон.")
    return length


def microstrip(
    h: float,
    er: float,
    f0: float,
    *,
    width: float | None = None,
    impedance: float | None = None,
) -> Result:
    """Analyze a given width or synthesize one for a target impedance."""
    substrate(h, er)
    positive(f0, "Частота")
    data: dict[str, float | np.integer | None] = {"h_m": h, "er": er, "f0_hz": f0}
    if width is None and impedance is not None:
        data["target_ohm"] = impedance
        width = synthesize_width(impedance, h, er)
    elif width is not None and impedance is None:
        data["width_m"] = positive(width, "Ширина проводника")
    else:
        raise InputError("Укажите ширину или требуемое сопротивление.")
    width_ratio = width / h
    if not 0.01 - 1e-15 <= width_ratio <= 100 + 1e-12:
        raise InputError("Ширина проводника: отношение W/h должно быть от 0,01 до 100.")
    z, effective = microstrip_model(width_ratio, er)
    return Result(
        "microstrip",
        data,
        {
            "Ширина, мм": width * 1e3,
            "Волновое сопротивление, Ом": z,
            "Эффективная проницаемость": effective,
            "Длина волны, мм": wavelength(f0, effective) * 1e3,
        },
        MICRO,
        f0,
    )


def frequency_grid(start: float, stop: float, count: int | np.integer) -> np.ndarray:
    """Build a strictly increasing, uniformly spaced frequency sweep."""
    positive(start, "Начальная частота")
    positive(stop, "Конечная частота")
    if start >= stop:
        raise InputError("Конечная частота должна быть выше начальной.")
    if not isinstance(count, (int, np.integer)) or not 2 <= count <= 10001:
        raise InputError("Число точек: введите целое число от 2 до 10001.")
    frequency = np.linspace(start, stop, count)
    if np.any(np.diff(frequency) <= 0):
        raise InputError("Частотная сетка: увеличьте расстояние между границами.")
    return frequency


def transformer(
    z0: float,
    load: float,
    f0: float,
    effective: float,
    start: float,
    stop: float,
    count: int | np.integer,
) -> Result:
    """Calculate a quarter-wave transformer and its reflection response."""
    positive(z0, "Сопротивление питающей линии")
    positive(load, "Сопротивление нагрузки")
    positive(f0, "Центральная частота")
    positive(effective, "Эффективная проницаемость", 1)
    frequency = frequency_grid(start, stop, count)
    z = sqrt(z0) * sqrt(load)
    length = wavelength(f0, effective) / 4
    theta = electrical_length(frequency, f0)
    cosine, sine = np.cos(theta), np.sin(theta)
    # Reflection expressed without dividing by the impedance denominator.
    scale = max(z0, load)
    src, dst = z0 / scale, load / scale
    gamma = abs(
        (dst - src)
        * cosine
        / ((dst + src) * cosine + 2j * sqrt(src) * sqrt(dst) * sine)
    )
    gamma = np.minimum(gamma, 1.0)
    with np.errstate(divide="ignore"):
        vswr = (1 + gamma) / (1 - gamma)
    return Result(
        "transformer",
        {
            "z0_ohm": z0,
            "load_ohm": load,
            "f0_hz": f0,
            "effective_er": effective,
            "start_hz": start,
            "stop_hz": stop,
            "count": count,
        },
        {
            "Сопротивление участка, Ом": z,
            "Длина участка, мм": length * 1e3,
            "Входное сопротивление на f₀, Ом": z0,
            "КСВН на f₀": 1.0,
        },
        "Линия без потерь. Нагрузка, сопротивление линии и εэфф постоянны по частоте.",
        f0,
        frequency,
        (Trace("|Γ|", "1", gamma), Trace("КСВН", "1", vswr)),
    )


def electrical_length(frequency: np.ndarray, f0: float) -> np.ndarray:
    """Return the electrical length in radians for a quarter-wave line."""
    with np.errstate(over="ignore"):
        theta = (frequency / f0) * (pi / 2)
    if not np.all(np.isfinite(theta)):
        raise InputError(
            "Частотный диапазон слишком велик для заданной центральной частоты."
        )
    return theta


def network_s(
    frequency: np.ndarray,
    f0: float,
    z0: float,
    lines: Sequence[tuple[int, int, float]],
    resistor: tuple[int, int, float] | None = None,
    *,
    ports: int = 3,
    resistors: Sequence[tuple[int, int, float]] = (),
    shunts: Sequence[tuple[int, float]] = (),
) -> np.ndarray:
    """Solve node voltages and line-end currents using chain equations.

    Voltages and currents are normalized to the common port impedance.
    Keeping both line currents avoids singular cotangents at half wavelengths.
    """
    resistors = (*resistors, resistor) if resistor is not None else resistors
    nodes = 1 + max(ports - 1, *(max(a, b) for a, b, _ in (*lines, *resistors)),
                    *(node for node, _ in shunts))
    size = nodes + 2 * len(lines)
    matrix = np.zeros((len(frequency), size, size), dtype=complex)
    rhs = np.zeros((len(frequency), size, ports), dtype=complex)
    for port in range(ports):
        matrix[:, port, port] = 1
        rhs[:, port, port] = 2
    theta = electrical_length(frequency, f0)
    cosine, sine = np.cos(theta), np.sin(theta)
    for index, (a, b, impedance) in enumerate(lines):
        ia = nodes + 2 * index
        ib = ia + 1
        z = impedance / z0
        matrix[:, a, ia] += 1
        matrix[:, b, ib] += 1
        matrix[:, ia, a] = 1
        matrix[:, ia, b] = -cosine
        matrix[:, ia, ib] = 1j * z * sine
        matrix[:, ib, ia] = 1
        matrix[:, ib, b] = -1j * sine / z
        matrix[:, ib, ib] = cosine
    for a, b, resistance in resistors:
        conductance = z0 / resistance
        for row, other in ((a, b), (b, a)):
            matrix[:, row, row] += conductance
            matrix[:, row, other] -= conductance
    for node, resistance in shunts:
        matrix[:, node, node] += z0 / resistance
    return np.linalg.solve(matrix, rhs)[:, :ports, :] - np.eye(ports)


def s_name(row: int, column: int) -> str:
    """Keep legacy names and disambiguate port indices 10 and 11."""
    return f"S{row}{column}" if max(row, column) < 10 else f"S[{row},{column}]"


def _divider_topology(
    kind: str, z0: float, ratio: float
) -> tuple[
    list[str],
    list[tuple[int, int, float]],
    tuple[int, int, float] | None,
    dict[str, float],
]:
    """Choose the divider branches and optional isolation resistor."""
    k = sqrt(ratio)
    labels = ["Ветвь к порту 2", "Ветвь к порту 3"]
    summary = {}
    resistor = None
    if kind == "tee":
        lines = [(0, 1, z0 * sqrt(1 + ratio)), (0, 2, z0 * sqrt(1 + 1 / ratio))]
    elif kind == "wilkinson_equal" or (kind == "wilkinson_unequal" and ratio == 1):
        if ratio != 1:
            raise InputError("Для равного деления отношение P3/P2 должно быть равно 1.")
        lines = [(0, 1, z0 * sqrt(2)), (0, 2, z0 * sqrt(2))]
        resistor = (1, 2, 2 * z0)
        if kind == "wilkinson_unequal":
            summary.update(
                {
                    "Внутреннее сопротивление 2, Ом": z0,
                    "Внутреннее сопротивление 3, Ом": z0,
                }
            )
    elif kind == "wilkinson_unequal":
        lines = [
            (0, 3, z0 * sqrt(k) * sqrt(1 + ratio)),
            (0, 4, z0 / sqrt(k) * sqrt(1 + 1 / ratio)),
            (3, 1, z0 * sqrt(k)),
            (4, 2, z0 / sqrt(k)),
        ]
        resistor = (3, 4, z0 * (k + 1 / k))
        summary.update(
            {
                "Внутреннее сопротивление 2, Ом": z0 * k,
                "Внутреннее сопротивление 3, Ом": z0 / k,
            }
        )
        labels.extend(("Трансформатор порта 2", "Трансформатор порта 3"))
    else:
        raise InputError("Неизвестный тип делителя.")
    return labels, lines, resistor, summary


def divider(
    kind: str,
    z0: float,
    ratio: float,
    h: float,
    er: float,
    f0: float,
    start: float,
    stop: float,
    count: int | np.integer,
) -> Result:
    """Synthesize a two-output divider and calculate its scattering response."""
    positive(z0, "Опорное сопротивление")
    positive(ratio, "Отношение P3/P2")
    positive(f0, "Центральная частота")
    substrate(h, er)
    frequency = frequency_grid(start, stop, count)
    labels, lines, resistor, summary = _divider_topology(kind, z0, ratio)
    sections = {}
    for label, (_, _, impedance) in zip(labels, lines, strict=True):
        try:
            width = synthesize_width(impedance, h, er)
        except InputError as error:
            raise InputError(f"{label}. {error}") from error
        _, effective = microstrip_model(width / h, er)
        sections[label] = {
            "z_ohm": impedance,
            "width_m": width,
            "effective_er": effective,
            "length_m": wavelength(f0, effective) / 4,
        }
    if resistor is not None:
        summary["Развязывающий резистор, Ом"] = resistor[2]
    s = network_s(np.r_[frequency, f0], f0, z0, lines, resistor)
    center = s[-1]
    s = s[:-1]
    summary["Доля мощности на порту 2"] = abs(center[1, 0]) ** 2
    summary["Доля мощности на порту 3"] = abs(center[2, 0]) ** 2
    summary["P3/P2 на f₀"] = abs(center[2, 0] / center[1, 0]) ** 2
    p2, p3 = abs(s[:, 1, 0]) ** 2, abs(s[:, 2, 0]) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        traces = [
            Trace(f"S{i + 1}{j + 1}", "дБ", 20 * np.log10(abs(s[:, i, j])))
            for i, j in ((0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (1, 2))
        ]
        phase = np.angle(s[:, 2, 0] / s[:, 1, 0], deg=True)
        phase[(p2 == 0) | (p3 == 0)] = np.nan
        traces.extend((Trace("P3/P2", "1", p3 / p2), Trace("Разность фаз", "°", phase)))
    model = IDEAL + "\n" + MICRO
    if kind == "tee":
        model += "\nТ-схема не согласует все три порта одновременно и не обеспечивает развязку выходов."
    return Result(
        kind,
        {
            "z0_ohm": z0,
            "r_p3_over_p2": ratio,
            "h_m": h,
            "er": er,
            "f0_hz": f0,
            "start_hz": start,
            "stop_hz": stop,
            "count": count,
        },
        summary,
        model,
        f0,
        frequency,
        tuple(traces),
        s,
        center,
        sections,
    )
