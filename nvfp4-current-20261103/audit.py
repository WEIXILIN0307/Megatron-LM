"""Verify the archived NVFP4 training results and regression controls."""

import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).parent
for line in (root / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name

for suite, ranks, count in (
    ('native', 1, 18),
    ('graph', 1, 9),
    ('gdn2-two-rank', 2, 2),
    ('gdn2-pp2', 2, 2),
    ('alignment-fixed', 2, 2),
    ('alignment-control', 2, 2),
):
    reports = sorted((root / suite / 'reports').glob('rank-*.json'))
    assert len(reports) == ranks, suite
    for path in reports:
        report = json.loads(path.read_text())
        assert len(report['outcomes']) == count, path
        if suite == 'alignment-control':
            assert report['exit_code'] == 1, path
            assert all(item['outcome'] == 'failed' and 'CUBLAS_STATUS_NOT_SUPPORTED' in
                       item['detail'] for item in report['outcomes']), path
        else:
            assert report['exit_code'] == 0, path
            assert all(item['outcome'] == 'passed' for item in report['outcomes']), path

results = []
for arm in ('eager', 'graph-off', 'graph-repeat', 'graph-on'):
    with gzip.open(root / f'qwen-{arm}/result.json.gz', 'rt') as f:
        results.append(json.load(f))
assert [len(result['graph_counts']) for result in results] == [0, 24, 24, 24]
assert all(len(result['records']) == 6 for result in results)
for records in zip(*(result['records'] for result in results)):
    for key in ('input_sha256', 'loss', 'grad_norm', 'gradients', 'weights', 'optimizer'):
        assert all(records[0][key] == record[key] for record in records[1:]), key
    assert records[-1]['pool']['current_in_use'] == 0
assert all(record['selected_bytes'] > 0 for record in results[-1]['records'][3:])
env = json.loads((root / 'environment-default.json').read_text())
assert env['nvfp4_check'][0] is True
assert env['disable_rht'] is False and env['disable_stochastic_rounding'] is False
assert env['backward_override'] is None
print('Checksums, NVFP4 training/graph tests, alignment controls and Qwen equality passed.')

with gzip.open(root / 'shape-repro/result.json.gz', 'rt') as f:
    shape_results = json.load(f)
assert [(item['width'], item['outcome']) for item in shape_results] == [
    (4096, 'passed'), (4112, 'failed'), (4128, 'passed')]
assert 'CUBLAS_STATUS_NOT_SUPPORTED' in shape_results[1]['error']
for suite in ('qwen-tp2-eager', 'qwen-tp2-shapes'):
    errors = sorted((root / suite).rglob('stderr.log.gz'))
    assert len(errors) == 2, suite
    for path in errors:
        with gzip.open(path, 'rt') as f:
            assert 'CUBLAS_STATUS_NOT_SUPPORTED' in f.read(), path
for path in (root / 'qwen-tp2-shapes').rglob('stdout.log.gz'):
    with gzip.open(path, 'rt') as f:
        record = next(line for line in f if line.startswith('FAILED_GEMM '))
    gemm = json.loads(record[len('FAILED_GEMM '):])
    assert [item['shape'] for item in gemm['operands']] == [[4112, 1024], [128, 4112]]
    assert all(item['type'] == 'NVFP4TensorStorage' for item in gemm['operands'])
    assert gemm['layout'] == 'NN' and gemm['grad'] is True and gemm['out_is_none'] is True
print('Qwen TP2 baseline failure and isolated TE shape controls also verified.')
