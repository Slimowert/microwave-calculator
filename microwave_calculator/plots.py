"""Matplotlib views shared by the interface and PNG export."""

from importlib.resources import files

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.patches import Rectangle


from .core import Result

DB_FLOOR = -100
SERIES = ("#1467bc", "#0b969e", "#7357bb", "#be642a", "#556e9a")

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
    ax.set_facecolor("#f8fbff")
    ax.spines[["top", "right"]].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color("#cbd9e7")
    ax.tick_params(colors="#4b657e", labelsize=8, length=3)
    ax.xaxis.label.set_color("#415e79")
    ax.yaxis.label.set_color("#415e79")
    ax.title.set_color("#1c3e62")
    ax.title.set_fontsize(10)
    ax.grid(color="#dce8f4", linewidth=0.7)


def plot_schematic(figure: Figure, result: Result, *, compact: bool = False) -> None:
    """Draw the calculated topology for the selected device."""
    figure.clear()
    ax = figure.subplots()
    ax.axis("off")
    if result.kind in ("butterworth", "chebyshev"):
        _plot_filter_schematic(ax, result)
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
    color = "#245875"
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


def plot_response(figure: Figure, result: Result) -> None:
    """Plot the detailed frequency response or the schematic for a static result."""
    figure.clear()
    traces = {trace.name: trace for trace in result.traces}
    if not traces:
        plot_schematic(figure, result)
        return
    if result.scattering is None:
        groups = [("|Γ|",), ("КСВН",)]
    elif "P3/P2" in traces:
        groups = [
            ("S11", "S22", "S33"),
            ("S21", "S31", "S23"),
            ("P3/P2",),
            ("Разность фаз",),
        ]
    elif result.center_s.shape[0] == 2:
        groups = [("S21",), ("S11",)]
    else:
        groups = [
            tuple(name for name in traces if name.endswith("1") and name != "S11"),
            tuple(name for name in traces if not name.endswith("1") or name == "S11"),
        ]
    frequency_ghz = result.frequency / 1e9
    columns = 2 if len(groups) == 4 else 1
    axes = figure.subplots(2, columns, squeeze=False)
    for ax, group in zip(axes.flat, groups, strict=True):
        for index, name in enumerate(group):
            trace = traces[name]
            if trace.unit == "дБ":
                values = np.maximum(trace.values, DB_FLOOR)
            else:
                values = np.where(np.isfinite(trace.values), trace.values, np.nan)
            ax.plot(
                frequency_ghz,
                values,
                label=name,
                lw=1.7,
                color=SERIES[index % len(SERIES)],
            )
        if result.frequency[0] <= result.f0 <= result.frequency[-1]:
            ax.axvline(result.f0 / 1e9, color="#777777", ls=":", label="f₀")
        ax.set(
            xlabel=f"Частота, ГГц (f₀ = {result.f0 / 1e9:.4g} ГГц)",
            ylabel=f"дБ (порог {DB_FLOOR})" if trace.unit == "дБ" else trace.unit,
        )
        style_axes(ax)
        ax.legend(fontsize=8, loc="best", frameon=False)


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
            [f"S{i}1" for i in range(2, result.center_s.shape[0] + 1)],
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


def plot_power(figure: Figure, result: Result) -> None:
    """Show the divider power balance at the design frequency."""
    figure.clear()
    ax = figure.subplots()
    ax.axis("off")
    if result.center_s is None or result.center_s.shape[0] < 3:
        ax.text(0.5, 0.5, "Баланс доступен для делителей", ha="center", va="center")
        return
    power = abs(result.center_s[:, 0]) ** 2
    reflected = float(power[0])
    loss = max(0, 1 - float(power.sum()))
    values = list(power[1:]) + [reflected, loss]
    labels = [f"P{i}" for i in range(2, len(power) + 1)] + ["Отражено", "Потери"]
    selected = [(label, value) for label, value in zip(labels, values) if value > 1e-10]
    wedges, _ = ax.pie(
        [value for _, value in selected],
        colors=["#0757ad", "#0cbabe", "#6479de", "#84abdc", "#d7803e"],
        startangle=90,
        wedgeprops={"width": 0.38, "edgecolor": "white"},
    )
    ax.text(0, 0, "P₁\n100%", ha="center", va="center", weight="bold")
    ax.legend(
        wedges,
        [f"{label}  {value:.1%}" for label, value in selected],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
        frameon=False,
    )
