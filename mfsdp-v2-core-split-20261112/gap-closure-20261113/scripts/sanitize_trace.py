"""Export only tables required for event/collective attribution, without system metadata."""

import gzip
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:3])
destination.mkdir(parents=True, exist_ok=True)
manifest = []
tables = ["StringIds", "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME",
          "CUPTI_ACTIVITY_KIND_CUDA_EVENT", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
          "CUPTI_ACTIVITY_KIND_KERNEL"]
for path in sorted(source.glob("trace-rank-*.sqlite")):
    src = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    dst = sqlite3.connect(":memory:")
    strings = dict(src.execute("select id,value from StringIds"))
    used_strings = set()
    counts = {}
    for name in tables:
        if name == "StringIds":
            continue
        schema = src.execute("select sql from sqlite_master where name=?", (name,)).fetchone()[0]
        dst.execute(schema)
        columns = [r[1] for r in src.execute(f"pragma table_info({name})")]
        rows = []
        # Retain row order, including irrelevant NVTX ranges: indices identify invocations.
        for raw in src.execute(f"select * from {name} order by rowid"):
            row = dict(zip(columns, raw))
            if name == "NVTX_EVENTS":
                text = row["text"] or strings.get(row["textId"], "")
                row["text"] = text if text.startswith(("proof|", "MFSDP ")) else "[omitted]"
                row["textId"] = None
                for key in ("uint64Value", "int64Value", "doubleValue", "uint32Value", "int32Value",
                            "floatValue", "jsonTextId", "jsonText", "binaryData"):
                    row[key] = None
            for key in ("nameId", "demangledName", "shortName", "mangledName"):
                if row.get(key) is not None:
                    used_strings.add(row[key])
            rows.append(tuple(row[c] for c in columns))
        dst.executemany(f"insert into {name} values ({','.join('?' for _ in columns)})", rows)
        counts[name] = len(rows)
    dst.execute(src.execute("select sql from sqlite_master where name='StringIds'").fetchone()[0])
    dst.executemany("insert into StringIds values (?,?)", [(i, strings[i]) for i in sorted(used_strings)])
    counts["StringIds"] = len(used_strings)
    dst.commit()
    payload = gzip.compress(dst.serialize(), mtime=0)
    output = destination / (path.name + ".gz")
    output.write_bytes(payload)
    manifest.append({"file": output.name, "bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest(), "rows": counts,
                     "original_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    src.close()
    dst.close()
for path in sorted(source.glob("ownership-rank-*.json")):
    data = json.loads(path.read_text())
    for group in data["groups"].values():
        group.pop("parameter_allreduce", None)  # DeviceMesh dimensions are authoritative.
    (destination / path.name).write_text(json.dumps(data, separators=(",", ":")) + "\n")
(destination / "exit.json").write_text((source / "exit.json").read_text())
(destination / "trace-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(source.name, sum(r["bytes"] for r in manifest), "compressed bytes")
