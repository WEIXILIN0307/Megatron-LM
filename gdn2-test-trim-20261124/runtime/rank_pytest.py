"""Run the existing isolated GPU tests and retain each rank's actual outcomes."""

import json
import os
import sys
from pathlib import Path

import pytest
import torch


class RankReport:
    """Record failures in any phase and the call/skip outcome of each test."""

    def __init__(self) -> None:
        self.outcomes = []

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" or report.failed or report.skipped:
            self.outcomes.append(
                {
                    "nodeid": report.nodeid,
                    "when": report.when,
                    "outcome": report.outcome,
                    "detail": str(report.longrepr) if not report.passed else "",
                }
            )

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        directory = Path(sys.argv[1])
        directory.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ["RANK"])
        result = {
            "rank": rank,
            "collected": len(session.items),
            "exit_code": int(exitstatus),
            "outcomes": self.outcomes,
        }
        (directory / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    status = pytest.main(sys.argv[2:], plugins=[RankReport()])
    # No unconditional barrier: a failed peer must be able to exit for torchrun.
    if status == 0 and torch.distributed.is_initialized():
        torch.cuda.synchronize()
        torch.distributed.destroy_process_group()
    sys.exit(status)
