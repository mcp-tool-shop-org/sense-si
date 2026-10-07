import importlib.util
from pathlib import Path

BAR = Path(__file__).resolve().parents[3] / "tools" / "coverage_bar.py"


def _load():
    spec = importlib.util.spec_from_file_location("coverage_bar", BAR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


XML = """<?xml version="1.0" ?>
<coverage><packages><package><classes><class><lines>
  <line number="1" hits="1"/>
  <line number="2" hits="0"/>
  <line number="3" hits="1" branch="true" condition-coverage="50% (1/2)"/>
  <line number="4" hits="1" branch="true" condition-coverage="100% (2/2)"/>
</lines></class></classes></package></packages></coverage>
"""


def test_a_partial_line_counts_as_uncovered(tmp_path):
    bar = _load()
    assert bar.line_counts(XML) == (2, 1, 1)
    assert bar.percent(2, 1, 1) == 50.0
    assert bar.percent(0, 0, 0) == 100.0
    path = tmp_path / "coverage.xml"
    path.write_text(XML, encoding="utf-8")
    assert bar.main(["coverage_bar.py", str(path), "50"]) == 0
    assert bar.main(["coverage_bar.py", str(path), "90"]) == 1
    assert bar.main(["coverage_bar.py"]) == 2
