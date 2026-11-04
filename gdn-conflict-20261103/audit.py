"""Check the archived GPU results and full Qwen training-state fingerprints."""

import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).parent
for line in (root / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name

expected_counts = {
    'baseline-qknorm': 3,
    'shared-final': 19,
    'scope-final': 71,
    'training-final': 12,
    'combined-recheck': 62,
    'combined-full': 73,
    'graph': 7,
    'precision': 2,
    'precision-config': 19,
    'feature-two-rank': 4,
}
for suite, count in expected_counts.items():
    reports = sorted((root / suite / 'reports').glob('rank-*.json'))
    assert len(reports) == (2 if suite == 'feature-two-rank' else 1), suite
    for path in reports:
        report = json.loads(path.read_text())
        assert report['exit_code'] == 0, path
        assert report['outcomes'], path
        assert all(item['outcome'] == 'passed' for item in report['outcomes']), path
        if count is not None:
            assert len(report['outcomes']) == count, path

results = []
for arm in ('off', 'zero', 'on'):
    with gzip.open(root / f'qwen-feature-{arm}/result.rank0.json.gz', 'rt') as f:
        results.append(json.load(f))
assert all(len(result['states']) == 3 for result in results)
for states in zip(*(result['states'] for result in results)):
    assert states[0] == states[1] == states[2]
assert all(result['pool_verified_before_reset'] for result in results)
assert results[0]['offload_status']['selected_transfer_bytes'] == 0
assert results[1]['offload_status']['selected_transfer_bytes'] == 0
assert results[2]['offload_status']['selected_transfer_bytes'] > 0
print('Checksums, per-rank tests and exact Qwen training-state comparisons passed.')
