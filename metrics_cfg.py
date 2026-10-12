"""
Loader for metrics.JSON, which allows full-line "//" comments (e.g. to switch
a metric off with a note) on top of plain JSON. A comma left dangling before a
closing brace by a commented-out last entry is tolerated.
"""
import json
import re

METRICS_FILE = "metrics.JSON"


def load_metrics_cfg(path: str = METRICS_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if not line.lstrip().startswith("//")]
    text = re.sub(r",(\s*[}\]])", r"\1", "".join(lines))
    return json.loads(text)
