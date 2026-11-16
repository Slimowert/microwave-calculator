"""Additional divider topologies and lumped low-pass prototypes."""

from math import asinh, pi, sin, sinh, sqrt, tanh

import numpy as np

from .core import (
    IDEAL,
    MICRO,
    InputError,
    Result,
    Trace,
    frequency_grid,
    microstrip,
    network_s,
    positive,
)


def db(values: np.ndarray | complex) -> np.ndarray | float:
    """Convert an amplitude or array of amplitudes to decibels."""
    with np.errstate(divide="ignore"):
        return 20 * np.log10(abs(values))


def multi_divider(
    kind: str,
    z0: float,
    h: float | None,
    er: float | None,
    f0: float,
    start: float,
    stop: float,
    count: int | np.integer,
    outputs: int = 3,
) -> Result:
    """Calculate a resistive or multi-output Wilkinson divider."""
    positive(z0, "Опорное сопротивление")
    positive(f0, "Центральная частота")
    frequency = frequency_grid(start, stop, count)
    sections = {}
    if kind == "resistive":
        outputs = 2
        star = [(port, 3, z0 / 3) for port in range(3)]
        lines = []
        summary = {"Каждый резистор звезды, Ом": z0 / 3}
        model = "Три идеальных резистора. Половина входной мощности рассеивается при согласованных портах."
    elif kind == "wilkinson_n":
        if not isinstance(outputs, int) or not 2 <= outputs <= 8:
            raise InputError("Число выходов: введите целое число от 2 до 8.")
        impedance = z0 * sqrt(outputs)
        line = microstrip(h, er, f0, impedance=impedance).summary
        lines = [(0, port, impedance) for port in range(1, outputs + 1)]
        star = [(port, outputs + 1, z0) for port in range(1, outputs + 1)]
        sections["Каждая из ветвей"] = {
            "z_ohm": impedance,
            "width_m": line["Ширина, мм"] * 1e-3,
            "effective_er": line["Эффективная проницаемость"],
            "length_m": line["Длина волны, мм"] * 0.25e-3,
        }
        summary = {"Каждый резистор звезды, Ом": z0, "Число выходов": outputs}
        model = IDEAL + "\nРезисторы соединены в плавающую звезду.\n" + MICRO
    else:
        raise InputError("Неизвестный тип делителя.")
    full = network_s(
        np.r_[frequency, f0], f0, z0, lines, ports=outputs + 1, resistors=star
    )
    s, center = full[:-1], full[-1]
    output_ports = range(1, outputs + 1)
    traces = [
        Trace(f"S{i + 1}{i + 1}", "дБ", db(s[:, i, i])) for i in range(outputs + 1)
    ]
    traces += [Trace(f"S{i + 1}1", "дБ", db(s[:, i, 0])) for i in output_ports]
    traces.append(Trace("S23", "дБ", db(s[:, 1, 2])))
    for port in output_ports:
        summary[f"Доля мощности на порту {port + 1}"] = abs(center[port, 0]) ** 2
    return Result(
        kind,
        {
            "z0_ohm": z0,
            "h_m": h,
            "er": er,
            "f0_hz": f0,
            "start_hz": start,
            "stop_hz": stop,
            "count": count,
            "outputs": outputs,
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


def _lowpass_prototype(
    kind: str, order: int, ripple: float | None
) -> tuple[list[float], float, str]:
    """Return normalized elements, load, and model description."""
    indices = range(1, order + 1)
    a = [sin((2 * i - 1) * pi / (2 * order)) for i in indices]
    termination = 1
    if kind == "butterworth":
        g = [2 * value for value in a]
        model = "ФНЧ Баттерворта. Частота f₀ — уровень −3,01 дБ."
    elif kind == "chebyshev":
        positive(ripple, "Пульсации")
        if not 0.01 <= ripple <= 3:
            raise InputError("Пульсации: допустимо от 0,01 до 3 дБ.")
        beta = asinh(1 / sqrt(10 ** (ripple / 10) - 1))
        gamma = sinh(beta / order)
        g = [2 * a[0] / gamma]
        for i in range(1, order):
            b = gamma**2 + sin(i * pi / order) ** 2
            g.append(4 * a[i - 1] * a[i] / (b * g[-1]))
        if order % 2 == 0:
            termination = 1 / tanh(beta / 2) ** 2
        model = "ФНЧ Чебышёва I рода. Частота f₀ — граница полосы пульсаций."
    else:
        raise InputError("Неизвестный тип фильтра.")
    return g, termination, model


def _lowpass_response(
    coefficients: list[float],
    termination: float,
    z0: float,
    omega: float,
    ratio: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Scale the ladder elements and compute its scattering matrices."""
    chain = np.tile(np.eye(2, dtype=complex), (len(ratio), 1, 1))
    sections = {}
    for i, value in enumerate(coefficients, 1):
        element = np.tile(np.eye(2, dtype=complex), (len(ratio), 1, 1))
        if i % 2:
            dimensioned = positive(z0 / omega * value, f"Индуктивность L{i}")
            sections[f"L{i}"] = {"inductance_h": dimensioned}
            element[:, 0, 1] = 1j * ratio * value
        else:
            dimensioned = positive(value / omega / z0, f"Ёмкость C{i}")
            sections[f"C{i}"] = {"capacitance_f": dimensioned}
            element[:, 1, 0] = 1j * ratio * value
        chain = chain @ element
    a = chain[:, 0, 0]
    b = chain[:, 0, 1]
    c = chain[:, 1, 0]
    d = chain[:, 1, 1]
    denominator = a * termination + b + c * termination + d
    s = np.empty_like(chain)
    s[:, 0, 0] = (a * termination + b - c * termination - d) / denominator
    s[:, 1, 1] = (-a * termination + b - c * termination + d) / denominator
    s[:, 1, 0] = s[:, 0, 1] = 2 * sqrt(termination) / denominator
    return s, sections


def lowpass(
    kind: str,
    z0: float,
    f0: float,
    order: int,
    ripple: float | None,
    start: float,
    stop: float,
    count: int | np.integer,
) -> Result:
    """Calculate a low-pass ladder and its response over a frequency sweep."""
    positive(z0, "Сопротивление источника")
    positive(f0, "Граница полосы")
    if not isinstance(order, int) or not 1 <= order <= 10:
        raise InputError("Порядок фильтра: введите целое число от 1 до 10.")
    frequency = frequency_grid(start, stop, count)
    g, termination, model = _lowpass_prototype(kind, order, ripple)
    load = positive(z0 * termination, "Расчётная нагрузка")
    omega = 2 * pi * f0
    positive(omega, "Угловая частота")
    ratio = np.r_[frequency, f0] / f0
    if not np.all(np.isfinite(ratio)) or np.max(ratio) > 1e12:
        raise InputError(
            "Частотный диапазон: слишком велик относительно границы полосы."
        )
    s, sections = _lowpass_response(g, termination, z0, omega, ratio)
    traces = (
        Trace("S11", "дБ", db(s[:-1, 0, 0])),
        Trace("S21", "дБ", db(s[:-1, 1, 0])),
    )
    return Result(
        kind,
        {
            "z0_ohm": z0,
            "f0_hz": f0,
            "order": order,
            "ripple_db": ripple if kind == "chebyshev" else None,
            "start_hz": start,
            "stop_hz": stop,
            "count": count,
        },
        {"Нагрузка, Ом": load, "Передача на f₀, дБ": db(s[-1, 1, 0])},
        model
        + " Идеальные сосредоточенные L и C. S-параметры отнесены к сопротивлениям источника и нагрузки.",
        f0,
        frequency,
        traces,
        s[:-1],
        s[-1],
        sections,
    )
