"""Summarize a predeclared paired comparison, including all successful repeats."""
import csv, datetime, json, math, re, statistics
from zoneinfo import ZoneInfo
from pathlib import Path
root=Path(__file__).resolve().parent
plan=json.loads((root/'plan.json').read_text())
steady_first, steady_last=plan['steady_steps']; count=steady_last-steady_first+1
reference=json.loads((root/'clock-reference.json').read_text())
clock_offset=reference['unix_seconds']-reference['perf_counter_seconds']
telemetry=[]
with (root/'gpu-samples.csv').open() as handle:
    for row in csv.reader(handle):
        timestamp=datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=ZoneInfo(reference['telemetry_timezone'])).timestamp()
        telemetry.append((timestamp,int(row[1]),float(row[2])))
runs=[]; histories={}

for name in plan['order']:
    directory=root/name
    assert json.loads((directory/'exit_code.json').read_text())['exit_code']==0,name
    log_path=directory/'train.log'
    if not log_path.exists(): log_path=directory/'iterations.txt'
    logs=log_path.read_text()
    status=re.findall(r'number of skipped iterations:\s*(\d+)\s*\| number of nan iterations:\s*(\d+)',logs)
    assert len(status)==60 and all(a==b=='0' for a,b in status),name
    ranks=[json.loads((directory/f'measurements-rank-{i}.json').read_text()) for i in range(2)]
    for rank in ranks:
        assert [x['step'] for x in rank['steps']]==list(range(1,61)),name
        assert all(x['seconds']>0 and math.isfinite(x['seconds']) and x['skipped']==0 for x in rank['steps'])
        assert all(math.isfinite(v) for row in rank['steps'] for v in row['loss'].values())
    windows=[[row for row in r['steps'] if steady_first<=row['step']<=steady_last] for r in ranks]
    wall=max(w[-1]['end_counter'] for w in windows)-min(w[0]['start_counter'] for w in windows)
    step_max=[max(windows[r][i]['seconds'] for r in range(2)) for i in range(count)]
    samples=[m for t,i,m in telemetry if min(w[0]['start_counter'] for w in windows)+clock_offset<=t<=max(w[-1]['end_counter'] for w in windows)+clock_offset]
    assert samples,name
    row={'name':name,'sampled_steady_peak_gpu_memory_mib':max(samples),'mode':'partial' if name.startswith('partial') else 'eager',
         'steady_steps':count,'steady_wall_seconds':wall,'steady_wall_seconds_per_step':wall/count,
         'global_tokens_per_second':count*plan['global_batch_size']*plan['sequence_length']/wall,
         'max_rank_step_median_seconds':statistics.median(step_max),
         'max_rank_step_p90_seconds':sorted(step_max)[math.ceil(.9*len(step_max))-1],
         'peak_allocated_bytes':max(x['peak_allocated_bytes'] for w in windows for x in w),
         'peak_reserved_bytes':max(x['peak_reserved_bytes'] for w in windows for x in w),
         'lifecycle_peak_allocated_bytes':max(x['peak_allocated_bytes'] for r in ranks for x in r['steps']),
         'lifecycle_peak_reserved_bytes':max(x['peak_reserved_bytes'] for r in ranks for x in r['steps']),
         'capture_seconds':max((c['seconds'] for r in ranks for c in r['capture']),default=0)}
    assert all(len(r['capture'])==(1 if row['mode']=='partial' else 0) for r in ranks)
    histories[name]=[[x['loss'] for x in r['steps']] for r in ranks]
    runs.append(row)
reference=histories['eager-1']
loss_equal=all(history==reference for history in histories.values())
assert loss_equal,'Loss histories differ; investigate before claiming performance'
summary={}
for mode in ('eager','partial'):
    rows=[r for r in runs if r['mode']==mode]
    summary[mode]={
        'mean_steady_seconds_per_step':statistics.mean(r['steady_wall_seconds_per_step'] for r in rows),
        'stdev_steady_seconds_per_step':statistics.stdev(r['steady_wall_seconds_per_step'] for r in rows),
        'aggregate_global_tokens_per_second':3*count*4096/sum(r['steady_wall_seconds'] for r in rows),
        'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows),
        'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in rows),
        'lifecycle_peak_allocated_bytes':max(r['lifecycle_peak_allocated_bytes'] for r in rows),
        'lifecycle_peak_reserved_bytes':max(r['lifecycle_peak_reserved_bytes'] for r in rows),
        'sampled_steady_peak_gpu_memory_mib':max(r['sampled_steady_peak_gpu_memory_mib'] for r in rows),
        'mean_capture_seconds':statistics.mean(r['capture_seconds'] for r in rows),
    }
pairs=[]
for i in (1,2,3):
    eager=next(r for r in runs if r['name']==f'eager-{i}')
    partial=next(r for r in runs if r['name']==f'partial-{i}')
    pairs.append(eager['steady_wall_seconds']/partial['steady_wall_seconds'])
speedup=summary['eager']['mean_steady_seconds_per_step']/summary['partial']['mean_steady_seconds_per_step']
savings=summary['eager']['mean_steady_seconds_per_step']-summary['partial']['mean_steady_seconds_per_step']
result={'plan':plan,'runs':runs,'summary':summary,'mean_speedup_ratio':speedup,
        'paired_speedup_ratios':pairs,'loss_histories_equal':loss_equal,
        'capture_amortization_steps':math.ceil(summary['partial']['mean_capture_seconds']/savings) if savings>0 else None,
        'timing_scope':'CUDA-synchronized stock train_step boundaries. Steady wall window includes inter-step logging/measurement overhead. Maximum-rank step times computed offline, no per-step distributed timing collective.',
        'limits':['Proxy has 16 experts, not the full 64-expert OLMoE; no full-model speedup inference.',
                  'Two GPUs with EP2 have expert-DP size 1; this timing run does not benchmark expert-DP collective overlap.',
                  'Shared host; balanced order and GPU telemetry reduce but cannot eliminate interference.',
                  'Capture amortization is a steady-rate estimate, not measured full-job time-to-solution.']}
(root/'performance-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('plan','runs')},indent=2))
