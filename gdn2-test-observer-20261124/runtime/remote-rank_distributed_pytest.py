"""Use a test-only store timeout that accommodates asymmetric cold kernel compilation."""

from datetime import timedelta
import os
import sys

import pytest
import torch

from rank_pytest import RankReport
from tests.unit_tests.test_utilities import Utils

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
Utils.initialize_distributed()
Utils.store.set_timeout(timedelta(minutes=10))
print("Test store timeout: 10 minutes for cold compilation before lazy PP initialization", flush=True)
status = pytest.main(sys.argv[2:], plugins=[RankReport()])
if status == 0 and torch.distributed.is_initialized():
    torch.cuda.synchronize()
    torch.distributed.destroy_process_group()
sys.exit(status)
