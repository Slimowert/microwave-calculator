"""Portable, explicitly versioned result exports."""

from __future__ import annotations

import csv
import json
import math
from os import PathLike
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from .core import Result


def status(value: float) -> str:
    """Return the export status for a real numeric value."""
    if math.isnan(value):
        return "undefined"
    if math.isinf(value):
        return "positive_infinity" if value > 0 else "negative_infinity"
    return "ok"


def json_value(value: object) -> object:
    """Convert NumPy, complex and nonfinite values to JSON representations."""
    if isinstance(value, complex):
        return {"real": json_value(value.real), "imag": json_value(value.imag)}
    if isinstance(value, (float, np.floating)):
        return (
            float(value)
            if math.isfinite(value)
            else {"value": None, "status": status(value)}
        )
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    return value


def write_json(
    path: str | bytes | PathLike[str] | PathLike[bytes] | int, result: Result
) -> None:
    """Write the versioned calculation summary as UTF-8 JSON."""
    data = {
        "format_version": 1,
        "device": result.kind,
        "model": result.model,
        "inputs": result.inputs,
        "sections": result.sections,
        "summary_f0": result.summary,
        "s_at_f0": result.center_s,
        "s_unit": "1",
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(json_value(data), file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")


def write_csv(
    path: str | bytes | PathLike[str] | PathLike[bytes] | int, result: Result
) -> None:
    """Write the frequency sweep and its trace statuses as CSV."""
    if result.frequency is None:
        raise ValueError("У этого расчёта нет частотной характеристики.")
    headers = ["f, Hz"]
    columns = [result.frequency]
    for trace in result.traces:
        headers.extend((f"{trace.name}, {trace.unit}", f"{trace.name}, status"))
        columns.extend(
            (
                [
                    float(value) if math.isfinite(value) else ""
                    for value in trace.values
                ],
                [status(value) for value in trace.values],
            )
        )
    if result.scattering is not None:
        for i in range(result.scattering.shape[1]):
            for j in range(result.scattering.shape[2]):
                name = f"S{i + 1}{j + 1}"
                headers.extend((f"Re({name}), 1", f"Im({name}), 1"))
                values = result.scattering[:, i, j]
                columns.extend((values.real, values.imag))
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(zip(*columns, strict=True))
