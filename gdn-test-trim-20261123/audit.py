import hashlib
import json
from pathlib import Path

root = Path(__file__).parent
for line in (root / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
heads = json.loads((root / 'heads.json').read_text())
runs = {'feature-single': ('feature', 1, 29), 'feature-two-rank': ('feature', 2, 4), 'gdn2-single': ('gdn2', 1, 10), 'graph-single': ('graph', 1, 5), 'graph-two-rank': ('graph', 2, 2), 'precision': ('precision', 1, 21)}
for name, (source, ranks, count) in runs.items():
    directory = root / 'results' / name
    assert json.loads((directory / 'exit.json').read_text())['exit_code'] == 0
    identity = json.loads((directory / 'source.json').read_text())
    assert identity == {'commit': heads[source]['validation_head'], 'status': ''}
    reports = list((directory / 'reports').glob('rank-*.json'))
    assert len(reports) == ranks
    for path in reports:
        report = json.loads(path.read_text())
        assert report['exit_code'] == 0 and report['collected'] == count
        assert len(report['outcomes']) == count
        assert all(item['outcome'] == 'passed' for item in report['outcomes'])
    print(name, count, 'passed per rank;', ranks, 'rank(s)')
print('Checksums, source identities and execution outcomes verified.')
