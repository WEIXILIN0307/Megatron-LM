"""Verify update, eval, save and real teacher transfer records."""
import hashlib
import json
import re
from pathlib import Path

here = Path(__file__).resolve().parent
assert "30 passed" in (here / "pytest-final.log").read_text()
assert (here / "launch-live.log").read_text().splitlines() == ["TP1 exit=0", "TP2 exit=0"]
producer = (here / "producer.log").read_text()
assert 'Backend UCX was instantiated' in producer
assert len(re.findall(r'POST /v1/completions[^\n]*200 OK', producer)) == 10
rows = []
for tp in (1, 2):
    log = (here / f'live-tp{tp}.log').read_text()
    assert 'successfully loaded checkpoint' in log
    assert re.findall(r'iteration\s+(\d+)/\s+2 .*?number of skipped iterations:\s+0', log) == ['1', '2']
    assert len(re.findall(r'validation loss at iteration', log)) == 3
    assert re.search('successfully saved checkpoint from iteration +2', log)
    requests = len(re.findall(r'POST .*?/v1/completions .*?200 OK', log))
    transfers = len(re.findall(r'GET .*?/done\?req_id=.*?200 OK', log))
    assert requests == transfers == 5
    assert (here / f'tiny-tp{tp}-saved/latest_checkpointed_iteration.txt').read_text().strip() == '2'
    rows.append(dict(tp=tp, updates=2, validation_microbatches=3, transfer_acks=transfers, checkpoint_iteration=2))
timeout = (here / 'tp-failure-final.log').read_text()
for pattern in ['rank=1 entering real TP broadcast', 'ReadTimeout', 'closing signal SIGTERM', 'torchrun exit=1']:
    assert pattern in timeout
refused = (here / 'launcher-refused-final.log').read_text()
for pattern in ['successfully loaded checkpoint', 'Connection refused', 'no fetchable sample', 'closing signal SIGTERM']:
    assert pattern in refused
assert (here / 'fault-checks.log').read_text().strip() == 'TP2 finetune exit=1'
report = {
    'runtime_commit': '1b369864d4e77a54961dd1a82013b8498853224c',
    'pr_head': 'f319f96c099e7c0a4c1106ffdf4a077aa46d9a8e',
    'modelopt_commit': '0e6a792cfccf0bda1ebaecfec8f9b8d294858d62',
    'cpu_tests': 30,
    'live_service': rows,
    'teacher_weights_sha256': hashlib.sha256((here / 'tiny-qwen3/model.safetensors').read_bytes()).hexdigest(),
    'tp2_launcher_fault_checks': ['http_timeout_at_batch_entry', 'connection_refused_at_finetune_entry'],
}
(here / 'correctness.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
