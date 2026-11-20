import ast
import os
import subprocess
from pathlib import Path

import pytest

from microwave_calculator.core import divider, microstrip, transformer
from microwave_calculator.hfss import hfss_script, write_hfss


def sample(kind):
    if kind == "microstrip":
        return microstrip(0.0008, 3.55, 2.4e9, impedance=50)
    return divider(kind, 50, 2 if kind == "wilkinson_unequal" else 1,
                   0.0008, 3.55, 2.4e9, 1.2e9, 3.6e9, 11)


@pytest.mark.parametrize("kind", ["microstrip", "tee", "wilkinson_equal", "wilkinson_unequal"])
def test_script_parameters(kind):
    result = sample(kind)
    tree = ast.parse(hfss_script(result))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign))
    name, items, count = ast.literal_eval(assignment.value)
    variables = dict(items)
    assert name == kind
    assert variables["sub_h"] == "0.8mm"
    assert variables["freq0"] == "2.4GHz"
    assert count == (201 if kind == "microstrip" else 11)
    assert ("r_iso" in variables) == kind.startswith("wilkinson")
    assert ("w4" in variables) == (kind == "wilkinson_unequal")
    for index, section in enumerate(result.sections.values(), 2):
        assert float(variables[f"w{index}"].removesuffix("mm")) == pytest.approx(section["width_m"] * 1000)


def test_unsupported_layout_preserves_existing_file(tmp_path):
    target = tmp_path / "existing.py"
    target.write_text("original")
    with pytest.raises(ValueError, match="разводки"):
        write_hfss(target, transformer(50, 100, 2.4e9, 3, 1e9, 3e9, 11))
    assert target.read_text() == "original"


@pytest.mark.skipif(not os.getenv("HFSS_TEST_EXE"), reason="Requires licensed Electronics Desktop")
def test_electronics_desktop_builds_and_validates(tmp_path):
    checks = []
    for kind in ("microstrip", "tee", "wilkinson_equal", "wilkinson_unequal"):
        script = tmp_path / (kind + ".py")
        write_hfss(script, sample(kind))
        checks.extend([f"__file__ = {str(script)!r}", "execfile(__file__)",
                       "assert design.ValidateDesign() == 1",
                       "assert list(analysis.GetSweeps('Setup1')) == ['Sweep']",
                       f"assert len(reports.GetAllReportNames()) == {3 if kind == 'microstrip' else 6}",
                       "oDesktop.CloseProject(project.GetName())"])
    runner = tmp_path / "verify.py"
    marker = tmp_path / "validated"
    runner.write_text("\n".join(checks + [f"open({str(marker)!r}, 'w').close()"]), encoding="utf-8")
    subprocess.run([os.environ["HFSS_TEST_EXE"], "-ng", "-RunScriptAndExit", str(runner)],
                   cwd=tmp_path, timeout=300, check=False)
    assert marker.exists(), runner.with_suffix(".log").read_text(errors="replace")
    projects = list(Path(tmp_path).glob("*.aedt"))
    assert len(projects) == 4
    for project in projects:
        assert "Frequency='2.4GHz'" in project.read_text(errors="replace")
