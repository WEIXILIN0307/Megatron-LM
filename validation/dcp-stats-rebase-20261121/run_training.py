"""Replay the requested checkpoint-matched TP2xCP2 pair in a fresh output directory."""

import os
import subprocess
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
container = os.environ.get("DCP_CONTAINER", "dcp-stats-rebase-20261121")
env = dict(os.environ, DCP_GPU_INDICES=os.environ.get("DCP_GPU_INDICES", "0,2,5,6"))
runs = [
    ["initial-tp", "--iters", "1", "--save"],
    [
        "tp-off",
        "--iters",
        "6",
        "--audit",
        "--load",
        "initial-tp",
        "--save",
        "--eval",
        "--log-interval",
        "2",
    ],
    [
        "tp-on",
        "--iters",
        "6",
        "--audit",
        "--load",
        "initial-tp",
        "--save",
        "--eval",
        "--log-interval",
        "2",
        "--on",
    ],
]
for args in runs:
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(base / "launch_train.py"),
            *args,
            "--gpus",
            "4",
            "--tp",
            "2",
            "--container",
            container,
        ],
        env=env,
        check=True,
    )
