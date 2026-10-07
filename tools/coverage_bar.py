"""Count a partial line as uncovered, the way Codecov does.

coverage.py's own total credits every branch that ran. A line that ran,
but not all of whose branches ran, is a partial, and Codecov leaves it
out of the numerator. This is the number ``--cov-fail-under`` does not
compute, so the pytest job runs this after the coverage.py bar.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def line_counts(xml_text: str) -> tuple[int, int, int]:
    root = ET.fromstring(xml_text)
    hits = misses = partials = 0
    for line in root.iter("line"):
        taken = int(line.get("hits") or 0)
        branched = line.get("branch") == "true"
        complete = (line.get("condition-coverage") or "").startswith("100%")
        if taken <= 0:
            misses += 1
        elif branched and not complete:
            partials += 1
        else:
            hits += 1
    return hits, misses, partials


def percent(hits: int, misses: int, partials: int) -> float:
    total = hits + misses + partials
    if total == 0:
        return 100.0
    return 100.0 * hits / total


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: coverage_bar.py coverage.xml PERCENT", file=sys.stderr)
        return 2
    threshold = float(argv[2])
    hits, misses, partials = line_counts(Path(argv[1]).read_text(encoding="utf-8"))
    measured = percent(hits, misses, partials)
    print(
        f"partials-as-uncovered {measured:.2f}%"
        f"  hits={hits} misses={misses} partials={partials}"
        f"  bar={threshold:g}"
    )
    if measured < threshold:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
