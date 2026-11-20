"""Export a calculated planar layout as a standalone AEDT script."""

from importlib.resources import files
from pathlib import Path

from .core import Result, synthesize_width


HFSS_KINDS = {"microstrip", "tee", "wilkinson_equal", "wilkinson_unequal"}


def _quantity(value: float, unit: str) -> str:
    return f"{value:.15g}{unit}"


def hfss_script(result: Result) -> str:
    if result.kind not in HFSS_KINDS:
        raise ValueError("Для этого режима пока нет физической разводки HFSS.")
    inputs = result.inputs
    height, er = inputs["h_m"], inputs["er"]
    variables = {"$eps_r": _quantity(er, ""), "sub_h": _quantity(height * 1000, "mm"),
                 "freq0": _quantity(result.f0 / 1e9, "GHz")}
    if result.kind == "microstrip":
        width = result.summary["Ширина, мм"]
        impedance = result.summary["Волновое сопротивление, Ом"]
        variables["line_len"] = _quantity(result.summary["Длина волны, мм"] / 4, "mm")
        variables.update(sweep_lo="freq0/2", sweep_hi="freq0*1.5")
        count = 201
    else:
        impedance = inputs["z0_ohm"]
        width = 1000 * synthesize_width(impedance, height, er)
        sections = list(result.sections.values())
        widths = [section["width_m"] * 1000 for section in sections]
        lengths = [section["length_m"] * 1000 for section in sections]
        pitch = max(width, *widths) + height * 1000
        if min(lengths[:2]) <= pitch + max(widths):
            raise ValueError("Ветви слишком коротки для этой разводки HFSS. Проверьте частоту и подложку.")
        for index, (section_width, length) in enumerate(zip(widths, lengths), 2):
            variables["w%d" % index] = _quantity(section_width, "mm")
            variables["len%d" % index] = _quantity(length, "mm")
        variables.update(pitch=_quantity(pitch, "mm"), lead="4*sub_h+w_feed",
                         joint_x="(len2+len3-abs(len2-len3))/2-pitch",
                         top_y="len2-joint_x", bot_y="joint_x-len3",
                         sweep_lo=_quantity(inputs["start_hz"] / 1e9, "GHz"),
                         sweep_hi=_quantity(inputs["stop_hz"] / 1e9, "GHz"))
        if result.kind != "tee":
            variables["r_iso"] = _quantity(result.summary["Развязывающий резистор, Ом"], "ohm")
        count = int(inputs["count"])
    base = {"w_feed": _quantity(width, "mm"), "z_ref": _quantity(impedance, "ohm")}
    parameters = repr((result.kind, list((base | variables).items()), count))
    template = files("microwave_calculator").joinpath("hfss_template.py.txt").read_text(encoding="utf-8")
    return template.replace("__CALCULATOR_PARAMETERS__", parameters)


def write_hfss(path: str | Path, result: Result) -> None:
    Path(path).write_text(hfss_script(result), encoding="utf-8")
