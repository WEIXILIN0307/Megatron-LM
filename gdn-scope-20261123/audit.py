"""Check the archived results without importing the GPU runtime."""

import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split("  ", 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name


def read(path):
    path = root / path
    if not path.exists():
        path = path.with_name(path.name + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


heads = read("heads.json")
positive = {
    "shared-acceptance": (1, 3, "shared"),
    "feature-acceptance-fixtures": (1, 47, "feature"),
    "feature-acceptance-two-rank": (2, 4, "feature"),
    "gdn2-acceptance-fixtures": (1, 12, "gdn2"),
    "graph-acceptance-single": (1, 5, "graph"),
    "graph-acceptance-two-rank": (2, 2, "graph"),
    "precision-acceptance-complete": (1, 21, "precision"),
    "mrope-acceptance": (1, 1, "mrope"),
    "alignment-release": (2, 35, "alignment"),
}
for suite, (ranks, count, key) in positive.items():
    directory = root / "results" / suite / "reports"
    reports = sorted(directory.glob("rank-*.json"))
    if not reports:
        reports = sorted(directory.glob("rank-*.json.gz"))
    assert len(reports) == ranks, suite
    assert read(f"results/{suite}/exit.json")["exit_code"] == 0, suite
    source = read(f"results/{suite}/source.json")
    assert source["status"] == "", (suite, source)
    assert source["commit"] == heads[key].get("validation_head", heads[key]["head"]), suite
    for path in reports:
        report = read(path.relative_to(root))
        assert report["collected"] == count and report["exit_code"] == 0, path
        assert len(report["outcomes"]) == count, path
        assert all(item["outcome"] == "passed" for item in report["outcomes"]), path
    print(suite, f"{count} passed per rank, {ranks} rank(s)")

for suite, ranks, count, error in (
    ("mrope-main-control-fixtures", 1, 1, "Tensor-likes are not close"),
    ("alignment-control-release", 2, 2, "CUBLAS_STATUS_NOT_SUPPORTED"),
):
    directory = root / "results" / suite / "reports"
    reports = sorted(directory.glob("rank-*.json")) or sorted(directory.glob("rank-*.json.gz"))
    assert len(reports) == ranks, suite
    assert read(f"results/{suite}/exit.json")["exit_code"] == 1, suite
    for path in reports:
        report = read(path.relative_to(root))
        assert report["collected"] == count and report["exit_code"] == 1, path
        assert len(report["outcomes"]) == count, path
        assert all(item["outcome"] == "failed" and error in item["detail"]
                   for item in report["outcomes"]), path
    print(suite, "expected old-code failure verified")

for name in ("precision-acceptance", "alignment"):
    result = read(f"source/{name}-remote-source-check.json")
    assert result["checked"] > 3000 and not result["missing"] and not result["mismatches"]

archive = root / "reproduction" / "81b5e39259bbebbcc951d1e5e269ed3ce190bde5"
for name, expected in read("source/archived-tool-blobs.json").items():
    data = (archive / name).read_bytes()
    blob = b"blob " + str(len(data)).encode() + b"\0" + data
    assert hashlib.sha1(blob).hexdigest() == expected, name
print("Checksums, source identities, acceptance results and historical tool blobs verified.")
