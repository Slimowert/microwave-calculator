"""Matplotlib views shared by the interface and PNG export."""

from importlib.resources import files

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.patches import Rectangle


from .core import Result, s_name

DB_FLOOR = -100
SERIES = ("#0752d9", "#00b9c9", "#8755c8", "#e88832", "#288f57",
          "#d64f84", "#716638", "#339dcc", "#b45137", "#6873a0", "#657482")

BOOK_FIGURES = {
    "microstrip": ("microstrip", "3.25(a)", 147),
    "transformer": ("transformer", "5.10", 247),
    "tee": ("tee", "7.6", 325),
    "resistive": ("resistive", "7.7", 327),
    "wilkinson_equal": ("wilkinson_equal", "7.8(b)", 328),
    "wilkinson_unequal": ("wilkinson_unequal", "7.13", 332),
    "wilkinson_n": ("wilkinson_n", "7.14", 333),
    "butterworth": ("lowpass", "8.24", 402),
    "chebyshev": ("lowpass", "8.24", 402),
}


def book_figure(figure: Figure, result: Result, *, compact: bool = False) -> str:
    """Draw the reference figure and return its short citation."""
    if result.kind == "wilkinson_sections":
        plot_schematic(figure, result, compact=compact)
        return "Расчётная схема"
    name, number, page = BOOK_FIGURES[result.kind]
    resource = files("microwave_calculator").joinpath("figures", f"{name}.png")
    with resource.open("rb") as stream:
        pixels = imread(stream, format="png")
    figure.clear()
    ax = figure.subplots()
    ax.imshow(pixels, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    if not compact:
        ax.set_title(
            f"D. M. Pozar · Microwave Engineering · рис. {number}, с. {page}",
            color="#183957",
            fontsize=11,
            pad=12,
        )
    return f"Pozar · рис. {number} · с. {page}"


def style_axes(ax: Axes) -> None:
    """Apply the common style to a frequency-response axis."""
    ax.set_axisbelow(True)
    ax.set_facecolor("#ffffff")
    ax.spines[["top", "right"]].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color("#53627a")
    ax.tick_params(colors="#26334b", labelsize=9, length=3)
    ax.xaxis.label.set_color("#26334b")
    ax.yaxis.label.set_color("#26334b")
    ax.title.set_color("#1c3e62")
    ax.title.set_fontsize(10)
    ax.grid(color="#e0e6f0", linewidth=0.7)


def plot_topology_preview(figure: Figure, result: Result) -> None:
    """Show a symbolic topology that stays valid while inputs are edited."""
    if result.kind not in ("wilkinson_equal", "tee"):
        book_figure(figure, result, compact=True)
        return
    figure.clear()
    ax = figure.subplots()
    ax.axis("off")
    color = "#202b3d"
    ax.plot([0.08, 0.22], [0.5, 0.5], color=color, lw=1.1)
    ax.plot([0.22, 0.22], [0.3, 0.7], color=color, lw=1.1)
    ax.plot(0.08, 0.5, "o", mfc="white", mec=color, ms=4)
    ax.text(0.08, 0.57, "P₁", fontsize=9, ha="center")
    for index, y in enumerate((0.7, 0.3), 2):
        ax.plot([0.22, 0.93], [y, y], color=color, lw=1.1)
        ax.add_patch(Rectangle((0.34, y - 0.025), 0.32, 0.05,
                               facecolor="white", edgecolor=color, lw=1.1, zorder=3))
        ax.text(0.50, y + (0.065 if index == 2 else -0.10), "Z ветви", fontsize=8, ha="center")
        ax.plot(0.93, y, "o", mfc="white", mec=color, ms=4)
        ax.text(0.93, y + 0.065, f"P{index}", fontsize=9, ha="center")
    if result.kind == "wilkinson_equal":
        ax.plot([0.76, 0.76], [0.3, 0.7], color=color, lw=1.1, marker="o", ms=2.5)
        ax.add_patch(Rectangle((0.735, 0.44), 0.05, 0.12,
                               facecolor="white", edgecolor=color, lw=1.1, zorder=3))
        ax.text(0.82, 0.5, "R", va="center", fontsize=9)
    ax.set(xlim=(0, 1.02), ylim=(0.12, 0.9))


def plot_schematic(figure: Figure, result: Result, *, compact: bool = False) -> None:
    """Draw the calculated topology for the selected device."""
    figure.clear()
    ax = figure.subplots()
    ax.axis("off")
    if result.topology:
        _plot_distribution_schematic(ax, result, compact=compact)
        return
    if result.kind in ("butterworth", "chebyshev"):
        _plot_filter_schematic(ax, result)
        return
    if result.kind == "wilkinson_sections":
        number = len(result.sections) or 3
        ax.plot([-1, 0], [0, 0], color="#245875")
        ax.plot([0, 0], [-1, 1], color="#245875")
        ax.text(-1, 0.1, "P1")
        for level, port in ((1, "P2"), (-1, "P3")):
            ax.plot([0, number+0.4], [level, level], color="#245875")
            ax.text(number+0.45, level, port)
        for index in range(number):
            position = number-index
            ax.plot([position, position], [-1, 1], color="#245875")
            ax.text(position, 0, f"R{index+1}", ha="center", bbox=dict(facecolor="white"))
            ax.text(position-0.5, 1.15, f"Z{index+1}", ha="center")
        ax.set(xlim=(-1.2, number+1.4), ylim=(-1.5, 1.8))
        return
    if result.kind in ("resistive", "wilkinson_n"):
        _plot_multi_divider_schematic(ax, result)
        return
    if result.kind == "microstrip":
        _plot_microstrip_schematic(ax, result)
        return
    if result.kind == "transformer":
        _plot_transformer_schematic(ax, result)
        return
    _plot_two_output_schematic(ax, result, compact=compact)


def _plot_filter_schematic(ax: Axes, result: Result) -> None:
    ax.plot([0, 1], [0.65, 0.65], color="#245875")
    ax.plot([0, 1], [0.25, 0.25], color="#8799aa")
    for x, (name, values) in zip(
        np.linspace(0.10, 0.90, len(result.sections)), result.sections.items()
    ):
        if "inductance_h" in values:
            ax.add_patch(
                Rectangle(
                    (x - 0.035, 0.61), 0.07, 0.08, fc="white", ec="#245875", zorder=3
                )
            )
            ax.text(x, 0.76, name, ha="center")
        else:
            ax.plot([x, x], [0.25, 0.65], color="#245875")
            ax.add_patch(
                Rectangle(
                    (x - 0.025, 0.42), 0.05, 0.06, fc="white", ec="#245875", zorder=3
                )
            )
            ax.text(x, 0.35, name, ha="right")
    ax.set(xlim=(-0.05, 1.05), ylim=(0, 1))


def plot_distribution_preview(figure: Figure, kind: str, outputs: int) -> None:
    """Redraw all selected ports while a new calculation is pending."""
    branches = []
    for port in range(2, outputs + 2):
        branches.append({"port": port, "lines": [f"Z{port}"],
                         "isolation": [f"R{port}"] if kind == "wilkinson_equal" else [],
                         "resistors": [f"R{port}"], "pads": []})
    result = Result(kind, {}, {}, "", 0, topology={
        "kind": kind, "outputs": outputs, "stages": 1, "branches": branches,
        "components": {}, "input_resistor": "R входа",
    })
    figure.clear()
    ax = figure.subplots()
    ax.axis("off")
    _plot_distribution_schematic(ax, result, compact=True)


def _plot_distribution_schematic(ax: Axes, result: Result, *, compact: bool) -> None:
    """Draw actual branches and isolation stars using explicit common net labels."""
    topology = result.topology
    n = topology["outputs"]
    stages = topology["stages"]
    kind = topology["kind"]
    components = topology["components"]
    color = "#26354c"
    font = (7 if n <= 5 else 6) if compact else 9
    if stages > 3:
        font -= .8
    center = (n - 1) / 2
    line_end = .14 + .59 * stages
    output_x = line_end + .27

    def wire(xs, ys):
        ax.plot(xs, ys, color=color, lw=1.0)

    def element(x1, x2, y, label):
        wire([x1, x2], [y, y])
        mid = (x1 + x2) / 2
        width = (x2 - x1) * .65
        ax.add_patch(Rectangle((mid - width / 2, y - .075), width, .15,
                               facecolor="white", edgecolor=color, lw=1, zorder=3))
        ax.text(mid, y + .16, label, fontsize=font, ha="center", va="bottom")

    def nominal(name, key, symbol):
        value = components.get(name, {}).get(key)
        return f"{value:.3g} Ω" if value is not None else symbol

    wire([-.04, .14], [center, center])
    ax.plot(-.04, center, "o", mfc="white", mec=color, ms=4)
    ax.text(-.07, center + .2, "P₁", ha="center", fontsize=font + 1)
    wire([.14, .14], [0, n - 1])
    if kind == "resistive":
        element(-.025, .13, center,
                nominal(topology["input_resistor"], "r_ohm", "R₁"))

    for index, branch in enumerate(topology["branches"]):
        port, y = branch["port"], n - 1 - index
        if kind == "resistive":
            element(.14, .4, y, nominal(branch["resistors"][0], "r_ohm", f"R{port}"))
            wire([.4, 1.0], [y, y])
            if branch["pads"]:
                first, second, ground = branch["pads"]
                element(.43, .66, y, nominal(first, "r_ohm", first))
                element(.70, .93, y, nominal(second, "r_ohm", second))
                wire([.68, .68], [y, y - .37])
                ax.add_patch(Rectangle((.665, y - .29), .03, .16,
                                       facecolor="white", edgecolor=color, zorder=3))
                ax.text(.71, y - .23, nominal(ground, "r_ohm", ground), fontsize=font - .5, va="center")
                for offset, width in ((.38, .07), (.42, .045), (.46, .02)):
                    wire([.68 - width / 2, .68 + width / 2], [y - offset] * 2)
        else:
            names = list(reversed(branch["lines"]))
            boundaries = np.linspace(.14, line_end, len(names) + 1)
            for i, name in enumerate(names):
                stage = stages - i
                label = nominal(name, "z_ohm", f"Z{port}") if stages == 1 else f"Z{port}.{stage}"
                element(boundaries[i], boundaries[i + 1], y, label)
                if kind == "wilkinson_equal":
                    # Matching labels below and in the resistor panel denote
                    # one electrical node; wires do not cross output ports.
                    ax.plot(boundaries[i + 1], y, ".", color=color, ms=3)
                    ax.text(boundaries[i + 1], y - .16, f"u{port}.{stage}", fontsize=font - 1,
                            ha="center", va="top")
            wire([line_end, output_x], [y, y])
            if branch.get("transformer"):
                name = branch["transformer"]
                element(line_end + .06, output_x - .04, y,
                        nominal(name, "z_ohm", "Zт") if stages == 1 else f"T{port}")
        ax.plot(output_x, y, "o", mfc="white", mec=color, ms=4)
        percentages = result.inputs.get("power_percentages")
        label = f"P{port}" + (f" · {percentages[index]:.4g}%" if percentages else "")
        ax.text(output_x + .035, y, label, fontsize=font, va="center")

    right = output_x + .28
    if kind == "wilkinson_equal":
        for stage in range(stages):
            left = output_x + (.57 if stages == 1 else .9) + stage * (.38 if stages == 1 else .55)
            common = left + .25
            wire([common, common], [0, n - 1])
            for index, branch in enumerate(topology["branches"]):
                y, port = n - 1 - index, branch["port"]
                name = branch["isolation"][stage]
                element(left, common, y, nominal(name, "r_ohm", name) if stages == 1 else name)
                ax.text(left - .01, y - .18, f"u{port}.{stage + 1}", fontsize=font - 1,
                        ha="left", va="top")
                ax.plot(common, y, ".", color=color, ms=3)
            heading = f"Звезда {stage + 1}" if stages == 1 else f"Зв. {stage + 1}"
            ax.text((left + common) / 2, n - .2, heading, fontsize=font, ha="center")
            right = common + .12
        note = "λ/4 на f₀ · u: одноимённые соединённые узлы · звёзды плавающие"
        if stages > 1:
            note += "\nZ — линии, T — выходные трансформаторы. Номиналы — в разделе «Подробности»."
    elif kind == "tee":
        note = "Все участки λ/4 на f₀ · выходы без развязки"
    else:
        note = "Резистивная звезда; Т-аттенюаторы на выходах с меньшей долей мощности"
    if not result.f0:
        note = "Предварительная топология · нажмите «Рассчитать»"
    ax.text(.0, -.65, note, fontsize=font, color="#52657f", va="top")
    ax.set(xlim=(-.14, right), ylim=(-1.15 if stages > 1 else -.95, n + .15))


def _plot_multi_divider_schematic(ax: Axes, result: Result) -> None:
    n = result.inputs["outputs"]
    ys = np.linspace(0.85, 0.18, n)
    resistor = result.summary["Каждый резистор звезды, Ом"]
    ax.plot([0.03, 0.19], [0.52, 0.52], color="#245875")
    ax.text(0.03, 0.59, "P1", color="#245875")
    x_star = 0.54 if result.kind == "resistive" else 0.86
    ax.plot([x_star, x_star], [ys[-1], ys[0]], color="#245875")
    if result.kind == "resistive":
        ax.plot([0.19, x_star], [0.52, 0.52], color="#245875")
        ax.add_patch(
            Rectangle((0.31, 0.505), 0.09, 0.03, fc="white", ec="#245875", zorder=3)
        )
    else:
        ax.plot([0.19, 0.19], [ys[-1], ys[0]], color="#245875")
    for i, y in enumerate(ys, 2):
        if result.kind == "wilkinson_n":
            ax.plot([0.19, x_star], [y, y], color="#245875")
            ax.add_patch(
                Rectangle(
                    (0.28, y - 0.015), 0.11, 0.03, fc="white", ec="#245875", zorder=3
                )
            )
            ax.plot(0.53, y, "o", color="#245875")
            ax.text(0.53, y + 0.032, f"P{i}", ha="center")
            resistor_x = 0.65
        else:
            ax.plot([x_star, 0.91], [y, y], color="#245875")
            ax.text(0.93, y, f"P{i}", va="center")
            resistor_x = 0.69
        ax.add_patch(
            Rectangle(
                (resistor_x, y - 0.014), 0.08, 0.028, fc="white", ec="#245875", zorder=3
            )
        )
    ax.text(
        0.53,
        0.05,
        f"R = {resistor:.4g} Ом"
        + (
            f" · {result.sections['Каждая из ветвей']['z_ohm']:.4g} Ом, λ/4"
            if result.sections
            else ""
        ),
        ha="center",
    )
    ax.set(xlim=(0, 1), ylim=(0, 1))


def _plot_microstrip_schematic(ax: Axes, result: Result) -> None:
    ax.add_patch(Rectangle((0.12, 0.3), 0.76, 0.42, facecolor="#c9e2f8"))
    ax.add_patch(Rectangle((0.32, 0.72), 0.36, 0.025, facecolor="#bd6a28"))
    ax.add_patch(Rectangle((0.12, 0.275), 0.76, 0.025, facecolor="#bd6a28"))
    ax.text(0.5, 0.80, f"W = {result.summary['Ширина, мм']:.4g} мм", ha="center")
    ax.text(
        0.5,
        0.49,
        f"h = {result.inputs['h_m'] * 1e3:.4g} мм, εr = {result.inputs['er']:.4g}",
        ha="center",
    )
    ax.text(0.5, 0.18, "Заземление", ha="center")
    ax.set(xlim=(0, 1), ylim=(0, 1))


def _plot_transformer_schematic(ax: Axes, result: Result) -> None:
    ax.plot([0, 1], [0.5, 0.5], color="#245875")
    ax.text(
        0.5,
        0.58,
        f"{result.summary['Сопротивление участка, Ом']:.4g} Ом, λ/4",
        ha="center",
    )
    ax.text(0, 0.39, f"Вход {result.inputs['z0_ohm']:.4g} Ом")
    ax.text(1, 0.39, f"Нагрузка {result.inputs['load_ohm']:.4g} Ом", ha="right")
    ax.set(xlim=(-0.07, 1.07), ylim=(0, 1))


def _plot_two_output_schematic(ax: Axes, result: Result, *, compact: bool) -> None:
    color = "#202b3d"
    sections = list(result.sections.values())
    unequal = len(sections) == 4
    x_internal = 0.53 if unequal else 0.73
    x_output = 0.89
    ys = (0.70, 0.30)
    label_size = 8 if compact else 11
    ax.plot([0.09, 0.20], [0.5, 0.5], color=color)
    ax.plot(0.09, 0.5, "o", color=color, mfc="white", ms=4)
    ax.text(0.09, 0.61, "P₁", ha="center", fontsize=label_size, color=color)
    for index, y in enumerate(ys):
        offset = 0.1 if index == 0 else -0.1
        alignment = "bottom" if index == 0 else "top"
        ax.plot([0.20, 0.20, x_output], [0.5, y, y], color=color)
        ax.plot(x_output, y, "o", color=color, mfc="white", ms=4)
        x_branch = 0.34 if unequal else 0.43
        ax.add_patch(Rectangle((x_branch - 0.105, y - 0.027), 0.21, 0.054,
                               facecolor="white", edgecolor=color, linewidth=1.2, zorder=3))
        ax.text(
            x_branch,
            y + offset,
            f"{sections[index]['z_ohm']:.4g} Ом · λ/4",
            ha="center",
            va=alignment,
            fontsize=label_size,
        )
        ax.text(
            x_output,
            y + offset,
            ("P₂", "P₃")[index],
            ha="center",
            va=alignment,
            fontsize=label_size,
            color=color,
        )
        if unequal:
            ax.add_patch(Rectangle((0.63, y - 0.027), 0.12, 0.054,
                                   facecolor="white", edgecolor=color, linewidth=1.2, zorder=3))
            ax.text(
                0.69,
                y + offset,
                f"{sections[index + 2]['z_ohm']:.4g} Ом · λ/4",
                ha="center",
                va=alignment,
                fontsize=label_size,
            )
            internal = result.summary[f"Внутреннее сопротивление {index + 2}, Ом"]
            if not compact:
                ax.text(
                    x_internal + 0.025,
                    y - offset * 0.7,
                    f"Внутр. {internal:.4g} Ом",
                    ha="left",
                    va="center",
                    fontsize=label_size,
                )
    if "Развязывающий резистор, Ом" in result.summary:
        ax.plot([x_internal] * 2, ys, color=color, marker="o", ms=3.5)
        ax.add_patch(
            Rectangle(
                (x_internal - 0.025, 0.42),
                0.05,
                0.16,
                facecolor="white",
                edgecolor=color,
                zorder=3,
            )
        )
        ax.text(
            x_internal - 0.045 if not unequal else x_internal + 0.045,
            0.5,
            f"R = {result.summary['Развязывающий резистор, Ом']:.4g} Ом",
            va="center",
            ha="left" if unequal else "right",
            fontsize=label_size,
        )
    ax.set(xlim=(0, 1), ylim=(0, 1))


def plot_response(figure: Figure, result: Result | None, *, transformer: bool = False) -> None:
    """Plot the frequency response or its axes while a calculation is pending."""
    figure.clear()
    if result is not None and result.frequency is None:
        plot_schematic(figure, result)
        return

    transformer = result.scattering is None if result is not None else transformer
    transmission, vswr = figure.subplots(1, 2)
    _plot_dashboard_trace(transmission, result, reflection=False, transformer=transformer)
    _plot_dashboard_trace(vswr, result, reflection=True, transformer=transformer)
    transmission.set_title("Отражение |Γ|" if transformer else "Передача на выходы")
    vswr.set_title("КСВ на входе" if transformer else "КСВ по портам")


def plot_overview(figure: Figure, result: Result) -> None:
    """Plot the compact frequency response used in the overview."""
    if result.center_s is None or result.center_s.shape[0] < 3:
        plot_response(figure, result)
        return
    figure.clear()
    ax1, ax2 = figure.subplots(1, 2)
    traces = {trace.name: trace.values for trace in result.traces}
    x = result.frequency / 1e9
    for ax, title, names in (
        (
            ax1,
            "Передача на выходы",
            [s_name(i, 1) for i in range(2, result.center_s.shape[0] + 1)],
        ),
        (ax2, "Отражение и связь", ["S11", "S23"]),
    ):
        for index, name in enumerate(names):
            if name in traces:
                ax.plot(
                    x,
                    np.maximum(traces[name], DB_FLOOR),
                    label=name,
                    lw=1.7,
                    color=SERIES[index % len(SERIES)],
                    ls="-" if index % 2 == 0 else "--",
                )
        ax.axvline(result.f0 / 1e9, color="#b76b30", ls=":", lw=1)
        ax.set(
            xlim=(x[0], x[-1]),
            title=title,
            xlabel="Частота, ГГц",
            ylabel=f"дБ · от {DB_FLOOR}",
        )
        style_axes(ax)
        ax.legend(fontsize=8, frameon=False)


def plot_dashboard_trace(
    figure: Figure, result: Result | None, *, reflection: bool = False, transformer: bool = False,
) -> None:
    """Draw one dashboard chart, including the actual value at f₀ when available."""
    figure.clear()
    _plot_dashboard_trace(figure.subplots(), result, reflection=reflection, transformer=transformer)


def _plot_dashboard_trace(
    ax: Axes, result: Result | None, *, reflection: bool, transformer: bool,
) -> None:
    if result is None:
        ylabel = "КСВ" if reflection else "|Γ|" if transformer else "Уровень, дБ"
        ax.set(xlabel="Частота, ГГц", ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        ax.set_xticks(np.linspace(0, 1, 6), labels=[])
        ax.set_yticks(np.linspace(0, 1, 6), labels=[])
        style_axes(ax)
        ax.text(
            0.5, 0.5, "Нажмите «Рассчитать»", transform=ax.transAxes,
            ha="center", va="center", color="#61718a", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=4),
        )
        return

    frequency_ghz = result.frequency / 1e9
    center_ghz = result.f0 / 1e9
    is_db = result.scattering is not None and not reflection

    if result.scattering is None:
        name = "КСВН" if reflection else "|Γ|"
        trace = next(trace for trace in result.traces if trace.name == name)
        values = trace.values
        center_vswr = result.summary["КСВН на f₀"]
        center = center_vswr if reflection else (center_vswr - 1) / (center_vswr + 1)
        series = [(name, values, center)]
    elif reflection:
        gamma = np.abs(result.scattering.diagonal(axis1=1, axis2=2))
        center_gamma = np.abs(result.center_s.diagonal())
        series = [
            (f"Порт {port}", _vswr(values), _vswr(center))
            for port, (values, center) in enumerate(zip(gamma.T, center_gamma), start=1)
        ]
    else:
        subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        series = [
            (s_name(port + 1, 1).translate(subscript),
             _display_db(np.abs(result.scattering[:, port, 0])),
             _display_db(abs(result.center_s[port, 0])))
            for port in range(1, result.center_s.shape[0])
        ]

    for index, (label, values, _) in enumerate(series):
        ax.plot(
            frequency_ghz, np.where(np.isfinite(values), values, np.nan),
            label=label, color=SERIES[index % len(SERIES)], lw=1.9,
            ls=("-", "--", "-.", ":")[index % 4],
        )

    if frequency_ghz[0] <= center_ghz <= frequency_ghz[-1]:
        ax.axvline(center_ghz, color="#ff8527", ls="--", lw=0.9, alpha=0.9)
        name, _, center = series[0]
        if np.isfinite(center):
            ax.plot(center_ghz, center, "o", ms=6, mfc="white", mec="#ff7c12",
                    mew=1.3, clip_on=False, zorder=5)
            label = f"f₀ = {center_ghz:.3f} ГГц"
            if len(series) == 1:
                value = f"≤ {DB_FLOOR}" if is_db and center == DB_FLOOR else f"{center:.3g}"
                label += f"\n{name} = {value}" + (" дБ" if is_db else "")
            ax.annotate(
                label, xy=(center_ghz, center), xytext=(0.48, 0.22 if reflection else 0.58),
                textcoords="axes fraction", fontsize=8, color="#17233a",
                bbox=dict(boxstyle="round,pad=0.6", fc="#fffdf9", ec="#ff8527", lw=0.8),
                arrowprops=dict(arrowstyle="-", color="#ff8527", lw=0.8),
            )

    ax.set(xlim=(frequency_ghz[0], frequency_ghz[-1]), xlabel="Частота, ГГц")
    if reflection:
        ax.set(ylabel="КСВ", ylim=(0.98, max(1.05, ax.get_ylim()[1])))
    elif is_db:
        minimum = min(float(np.min(values)) for _, values, _ in series)
        ax.set(ylabel="Уровень, дБ", ylim=(max(DB_FLOOR - 2, min(-20, minimum - 2)), 0))
    else:
        ax.set(ylabel="|Γ|")
    style_axes(ax)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    ax.legend(fontsize=8, loc="best", ncols=2 if len(series) > 4 else 1,
              frameon=True, facecolor="white", edgecolor="#cbd8e9", framealpha=0.95)


def _display_db(amplitude):
    return 20 * np.log10(np.maximum(amplitude, 10 ** (DB_FLOOR / 20)))


def _vswr(gamma):
    # Values at and above unity represent complete reflection for these passive models.
    return np.divide(1 + gamma, 1 - gamma, out=np.full_like(gamma, np.inf), where=gamma < 1)


def plot_power(figure: Figure, result: Result) -> None:
    """Show a compact donut and a readable accounting of the incident power."""
    figure.clear()
    if result.center_s is None or result.center_s.shape[0] < 3:
        return
    power = abs(result.center_s[:, 0]) ** 2
    reflected = float(power[0])
    loss = max(0, 1 - float(power.sum()))
    values = list(power[1:]) + [reflected, loss]
    labels = [f"P{i}" for i in range(2, len(power) + 1)] + ["Отражено", "Потери"]
    colors = list(SERIES[:len(power) - 1]) + ["#ed923b", "#9ba8ba"]
    selected = [(label, value, color) for label, value, color in zip(labels, values, colors) if value > 1e-10]
    ax, text_ax = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.25]})
    ax.pie([value for _, value, _ in selected], colors=[color for _, _, color in selected],
           startangle=90, counterclock=False,
           wedgeprops={"width": 0.31, "edgecolor": "white", "linewidth": 1.4})
    ax.text(0, 0, "P₁\n100%", ha="center", va="center", weight="bold", fontsize=11, color="#111b32")
    ax.set_aspect("equal")
    text_ax.axis("off")
    text_ax.text(0.02, 0.91, "Входная мощность: 100%", fontsize=8, color="#18253e", va="top")
    spacing = min(0.26, 0.72 / max(1, len(selected)))
    for index, (label, value, color) in enumerate(selected):
        y = 0.65 - index * spacing
        text_ax.plot(0.05, y, "s", ms=6, color=color, clip_on=False)
        db = 10 * np.log10(value)
        text_ax.text(0.16, y, f"{label}  {value:.1%}  ({db:.2f} дБ)",
                     fontsize=7.5 if len(selected) <= 6 else 6.5, va="center", color="#18253e")
    text_ax.set(xlim=(0, 1.1), ylim=(-0.15, 1))
