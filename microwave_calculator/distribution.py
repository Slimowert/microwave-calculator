"""Power dividers with explicit percentages for two to ten output ports.

The unequal Wilkinson uses the generalized floating-star network of Ahn,
Lee and Myung (IEEE MTT-S, 2004), followed by output impedance transformers.
Every response is solved from the actual transmission lines and resistors.
"""

from math import sqrt

import numpy as np
from scipy.optimize import least_squares

from .core import (
    IDEAL, MICRO, InputError, Result, Trace, divider, frequency_grid,
    microstrip, microstrip_model, network_s, positive, s_name, substrate,
)
from .extended import synthesize_wilkinson


def power_fractions(outputs: int, percentages) -> np.ndarray:
    """Validate explicit percentages without silently correcting user input."""
    if isinstance(outputs, bool) or not isinstance(outputs, (int, np.integer)) or not 2 <= outputs <= 10:
        raise InputError("Число выходных портов: укажите целое число от 2 до 10.")
    try:
        values = np.asarray(percentages, dtype=float)
    except (ValueError, TypeError) as error:
        raise InputError("Распределение мощности: введите проценты для каждого выхода.") from error
    if values.shape != (outputs,):
        raise InputError(f"Распределение мощности: требуется {outputs} значений.")
    for port, value in enumerate(values, 2):
        if not np.isfinite(value) or not 0 < value < 100:
            raise InputError(f"Порт {port}: процент мощности должен быть больше 0 и меньше 100.")
    total = float(values.sum())
    if abs(total - 100) > 1e-6:
        raise InputError(f"Сумма мощностей должна быть 100%; сейчас {total:.8g}%.")
    return values / total


def distributed_divider(
    kind: str, z0: float, h: float | None, er: float | None, f0: float,
    start: float, stop: float, count: int, outputs: int, percentages,
    *, stages: int = 1,
) -> Result:
    """Synthesize a physical divider and solve its full (N+1)-port network."""
    fractions = power_fractions(outputs, percentages)
    if kind not in ("tee", "wilkinson_equal", "resistive"):
        raise InputError("Неизвестный тип делителя.")
    positive(z0, "Опорное сопротивление")
    positive(f0, "Центральная частота")
    frequency = frequency_grid(start, stop, count)
    if isinstance(stages, bool) or not isinstance(stages, int) or not 1 <= stages <= 8:
        raise InputError("Количество секций: укажите целое число от 1 до 8.")
    if kind != "wilkinson_equal" and stages != 1:
        raise InputError("Несколько секций доступны для делителя Уилкинсона.")
    if kind != "resistive":
        substrate(h, er)

    sections = {}
    lines, resistors, shunts = [], [], []
    topology = {"kind": kind, "outputs": int(outputs), "stages": stages, "branches": []}
    next_node = outputs + 1

    def node():
        nonlocal next_node
        value = next_node
        next_node += 1
        return value

    def line(name, a, b, impedance):
        try:
            dimensions = microstrip(h, er, f0, impedance=impedance).summary
        except InputError as error:
            raise InputError(f"{name}. {error}") from error
        sections[name] = {
            "z_ohm": float(impedance),
            "width_m": dimensions["Ширина, мм"] / 1000,
            "effective_er": dimensions["Эффективная проницаемость"],
            "length_m": dimensions["Длина волны, мм"] / 4000,
        }
        lines.append((a, b, float(impedance)))

    def resistor(name, a, b, resistance):
        positive(float(resistance), name)
        sections[name] = {"r_ohm": float(resistance)}
        if b is None:
            shunts.append((a, float(resistance)))
        else:
            resistors.append((a, b, float(resistance)))

    legacy = None
    if kind == "resistive":
        # A matched (N+1)-arm star, with matched T-pads on the smaller outputs.
        # The percentages describe the delivered power, not the incident power.
        star = node()
        base_r = z0 * (outputs - 1) / (outputs + 1)
        resistor("R входа", 0, star, base_r)
        topology["input_resistor"] = "R входа"
        for port, fraction in enumerate(fractions, 1):
            branch = {"port": port + 1, "lines": [], "resistors": [], "pads": []}
            attenuation = sqrt(float(fraction / max(fractions)))
            connection = port if abs(attenuation - 1) < 1e-12 else node()
            name = f"R звезды · порт {port + 1}"
            resistor(name, star, connection, base_r)
            branch["resistors"].append(name)
            if connection != port:
                center = node()
                series = z0 * (1 - attenuation) / (1 + attenuation)
                parallel = 2 * z0 * attenuation / (1 - attenuation ** 2)
                for suffix, a, b, resistance in (
                    ("a", connection, center, series), ("b", center, port, series),
                    ("g", center, None, parallel),
                ):
                    name = f"R{port + 1}{suffix}"
                    resistor(name, a, b, resistance)
                    branch["pads"].append(name)
            topology["branches"].append(branch)
        model = ("Согласованная резистивная звезда с выходными Т-аттенюаторами. "
                 "Проценты заданы относительно суммы выходных мощностей. "
                 "Мощность, рассеиваемая резисторами, показана отдельно.")
        result_kind = "resistive"
        summary = {"Каждый резистор звезды, Ом": base_r}
    elif kind == "tee":
        for port, fraction in enumerate(fractions, 1):
            name = f"Ветвь к порту {port + 1}"
            line(name, 0, port, z0 / sqrt(float(fraction)))
            topology["branches"].append({"port": port + 1, "lines": [name], "isolation": []})
        model = IDEAL + "\nТ-соединение не обеспечивает развязку выходов.\n" + MICRO
        result_kind, summary = "tee", {}
        if outputs == 2:
            legacy = divider("tee", z0, fractions[1] / fractions[0], h, er, f0, start, stop, count)
    else:
        equal = bool(np.ptp(fractions) < 1e-12)
        # A fixes the internal terminations Ri=A/pi. Choose it within the
        # synthesizable range of BOTH the branches and output transformers.
        amplitude = z0 * sqrt(float(np.exp(np.mean(np.log(fractions)))))
        original_amplitude = amplitude
        if not equal:
            lower_z = microstrip_model(100, er)[0]
            upper_z = microstrip_model(0.01, er)[0]
            lower = lower_z * sqrt(float(max(fractions))) * 1.001
            upper = upper_z * float(min(fractions)) / 1.001
            if lower >= upper:
                raise InputError("Заданное распределение требует сопротивлений линий вне диапазона подложки.")
            amplitude = float(np.clip(amplitude, lower, upper))
        a_design = amplitude ** 2 / z0
        references = a_design / fractions
        if outputs == 2 and equal and stages > 1:
            legacy = synthesize_wilkinson(stages, z0, h, er, f0, start, stop, count)
            normalized_z = np.array([value["z_ohm"] / z0 for value in legacy.sections.values()])
            normalized_r = np.array([value["r_ohm"] / (2 * z0) for value in legacy.sections.values()])
        elif stages == 1:
            normalized_z = np.array([sqrt(z0 / a_design)])
            normalized_r = np.array([1.0])
        else:
            normalized_z, normalized_r = _fit_modes(
                stages, z0, references, a_design, er, f0, start, stop,
            )
        stars = [node() for _ in range(stages)]
        for port, reference in enumerate(references, 1):
            branch = {"port": port + 1, "lines": [], "isolation": [], "transformer": None}
            # Values are indexed from the outputs back toward the input,
            # matching the pre-existing multi-section synthesis convention.
            internal = port if equal else node()
            downstream = internal
            for index, (nz, nr) in enumerate(zip(normalized_z, normalized_r), 1):
                upstream = 0 if index == stages else node()
                name = f"Ветвь к порту {port + 1}" if stages == 1 else f"Порт {port + 1} · секция {index}"
                line(name, upstream, downstream, reference * nz)
                branch["lines"].append(name)
                r_name = f"R{port + 1}.{index}"
                resistor(r_name, downstream, stars[index - 1], reference * nr)
                branch["isolation"].append(r_name)
                downstream = upstream
            if not equal:
                name = f"Трансформатор порта {port + 1}"
                line(name, internal, port, sqrt(z0 * reference))
                branch["transformer"] = name
            topology["branches"].append(branch)
        summary = {"Секций в плече": stages}
        model = (IDEAL + "\nОбобщённая схема Уилкинсона: плавающие резистивные звёзды "
                 "и выходные четвертьволновые трансформаторы при неравном делении.\n" + MICRO)
        result_kind = "wilkinson_n" if outputs > 2 else "wilkinson_equal" if equal else "wilkinson_unequal"
        if stages > 1:
            result_kind = "wilkinson_sections"
        # Preserve existing two-output synthesis and HFSS layout contracts.
        if outputs == 2 and stages == 1 and np.isclose(amplitude, original_amplitude, rtol=1e-12):
            legacy = divider(result_kind, z0, fractions[1] / fractions[0], h, er, f0, start, stop, count)
        elif outputs == 2 and stages == 1:
            summary["Развязывающий резистор, Ом"] = float(sum(references))

    if legacy is None:
        grid = np.r_[frequency, f0]
        # Bound matrix memory for up to 10 outputs and eight sections.
        block = max(1, min(128, 1_000_000 // max(1, (next_node + 2 * len(lines)) ** 2)))
        full = np.concatenate([
            network_s(chunk, f0, z0, lines, ports=outputs + 1, resistors=resistors, shunts=shunts)
            for chunk in np.array_split(grid, (len(grid) + block - 1) // block)
        ])
        scattering, center = full[:-1], full[-1]
    else:
        scattering, center = legacy.scattering, legacy.center_s
        summary.update(legacy.summary)
    power = abs(center[:, 0]) ** 2
    delivered = float(power[1:].sum())
    summary["Число выходов"] = int(outputs)
    for port, fraction in enumerate(fractions, 2):
        summary[f"Заданная доля выхода {port}, %"] = float(fraction * 100)
        summary[f"Доля мощности на порту {port}"] = float(power[port - 1])
        summary[f"Распределение на выходе {port}, %"] = float(power[port - 1] / delivered * 100)
    summary["Передано на выходы, %"] = delivered * 100
    summary["Потери, %"] = max(0.0, 1 - float(power.sum())) * 100
    with np.errstate(divide="ignore"):
        traces = tuple(
            Trace(s_name(i + 1, j + 1), "дБ", 20 * np.log10(abs(scattering[:, i, j])))
            for i in range(outputs + 1) for j in range(outputs + 1)
            if i == j or j == 0 or (i > 0 and j > i)
        )
    inputs = {
        "z0_ohm": z0, "h_m": h, "er": er, "f0_hz": f0,
        "start_hz": start, "stop_hz": stop, "count": count,
        "outputs": int(outputs), "power_percentages": (fractions * 100).tolist(),
        "stages": stages,
    }
    if legacy is not None:
        inputs = legacy.inputs | inputs
    topology["components"] = sections.copy()
    # Keep legacy dimensions in their original order for the two-output HFSS template.
    visible_sections = legacy.sections if legacy is not None else sections
    if legacy is None and outputs == 2 and kind == "wilkinson_equal" and stages == 1:
        names = [branch["lines"][0] for branch in topology["branches"]]
        names += [branch["transformer"] for branch in topology["branches"] if branch.get("transformer")]
        visible_sections = {name: sections[name] for name in names}
    return Result(result_kind, inputs, summary, model, f0, frequency, traces,
                  scattering, center, visible_sections, topology)


def _fit_modes(stages, z0, references, a_design, er, f0, start, stop):
    """Optimize the common mode and degenerate odd modes in internal references."""
    frequency = frequency_grid(start, stop, 129)
    theta = frequency / f0 * np.pi / 2
    cosine, sine = np.cos(theta), 1j * np.sin(theta)
    minimum = microstrip_model(100, er)[0] / min(references) * 1.001
    maximum = microstrip_model(.01, er)[0] / max(references) / 1.001
    if minimum >= maximum:
        raise InputError("Распределение не допускает реализуемые многосекционные ветви на этой подложке.")
    lower = np.log([minimum] * stages + [.025] * stages)
    upper = np.log([maximum] * stages + [100] * stages)
    initial = np.log(np.r_[np.geomspace(1, z0 / a_design, stages), np.geomspace(1, 5, stages)])
    initial = np.clip(initial, lower, upper)

    def residual(parameters):
        impedances, resistances = np.split(np.exp(parameters), 2)
        voltage, current = np.array([[z0 / a_design], [0.0]]), np.ones((2, 1))
        for impedance, resistance in zip(impedances[::-1], resistances[::-1]):
            voltage, current = (cosine * voltage + sine * impedance * current,
                                cosine * current + sine * voltage / impedance)
            current[1] += voltage[1] / resistance
        reflection = (voltage - current) / (voltage + current)
        return np.r_[reflection.real.ravel(), reflection.imag.ravel()]

    solution = least_squares(residual, initial, bounds=(lower, upper), max_nfev=200, ftol=1e-7)
    return np.split(np.exp(solution.x), 2)
