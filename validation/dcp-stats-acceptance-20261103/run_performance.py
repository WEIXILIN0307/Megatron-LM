"""Repeat the existing pretrain timers; no per-operator instrumentation."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

base = Path(__file__).resolve().parent
prefix = os.environ.get("DCP_PERF_PREFIX", "perf")
manifest = (
    "performance-runs.json" if prefix == "perf" else prefix + "-performance-runs.json"
)
telemetry = (base / (prefix + "-gpu.csv")).open("w")
monitor = subprocess.Popen(
    [
        "nvidia-smi",
        "-i",
        os.environ.get("DCP_GPU_INDICES", "2,4"),
        "--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw",
        "--format=csv",
        "--loop=1",
    ],
    stdout=telemetry,
    stderr=subprocess.STDOUT,
)
rows = []
try:
    for repeat, order in enumerate(
        [["off", "on", "fixed"], ["on", "fixed", "off"], ["fixed", "off", "on"]]
    ):
        for arm in order:
            name = f"{prefix}-{repeat}-{arm}"
            cmd = [
                sys.executable,
                "-S",
                str(base / "launch_train.py"),
                "--container",
                os.environ.get("DCP_CONTAINER", "dcp-stats-revalidation"),
                name,
                "--measure",
                "--iters",
                os.environ.get("DCP_PERF_ITERS", "41"),
                "--load",
                os.environ.get("DCP_PERF_CHECKPOINT", "initial-aligned"),
            ]
            if arm == "on":
                cmd += ["--on"]
            if arm == "fixed":
                cmd += ["--fixed"]
            start = time.time()
            subprocess.run(cmd, check=True)
            end = time.time()
            rows.append(
                dict(name=name, repeat=repeat, arm=arm, start_unix=start, end_unix=end)
            )
            (base / manifest).write_text(json.dumps(rows, indent=2) + "\n")
finally:
    monitor.terminate()
    monitor.wait()
    telemetry.close()
