"""Real DP2 x CP2 checkpoint replay with varying sample lengths and log windows."""

import os
import subprocess
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
runs = [
    [
        "initial-dp2",
        "--gpus",
        "4",
        "--cp",
        "2",
        "--iters",
        "1",
        "--save",
        "--lengths-file",
        "dp2-lengths.csv",
    ],
    [
        "dp2-off",
        "--gpus",
        "4",
        "--cp",
        "2",
        "--iters",
        "10",
        "--load",
        "initial-dp2",
        "--audit",
        "--eval",
        "--save",
        "--log-interval",
        "3",
        "--lengths-file",
        "dp2-lengths.csv",
    ],
    [
        "dp2-on",
        "--gpus",
        "4",
        "--cp",
        "2",
        "--iters",
        "10",
        "--load",
        "initial-dp2",
        "--audit",
        "--eval",
        "--save",
        "--log-interval",
        "3",
        "--lengths-file",
        "dp2-lengths.csv",
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
            "--container",
            "dcp-stats-acceptance-20261103",
        ],
        env=dict(
            os.environ,
            DCP_REPO_ROOT="/data/songbinbin/Proj/Proj_LWX/Megatron-LM-dcp-stats",
            DCP_GPU_INDICES="1,4,5,6",
        ),
        check=True,
    )
