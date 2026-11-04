"""Summarize the separate CPU-only observation/logging diagnostic."""

import json
import statistics
from pathlib import Path

base = Path(__file__).resolve().parent
report = {}
for arm in ("off", "on"):
    report[arm] = {}
    for rank in (0, 1):
        rows = json.loads(
            (base / f"cpu-profile-{arm}/audit/cpu-profile-rank{rank}.json").read_text()
        )
        rows = [x for x in rows if x["iteration"] is not None and x["iteration"] >= 12]
        result = {}
        for name in sorted(set(x["function"] for x in rows)):
            values = [x["microseconds"] for x in rows if x["function"] == name]
            result[name] = {
                "calls": len(values),
                "mean_us": statistics.mean(values),
                "median_us": statistics.median(values),
                "max_us": max(values),
                "sum_us": sum(values),
            }
        report[arm][str(rank)] = result
(base / "cpu-profile-summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
