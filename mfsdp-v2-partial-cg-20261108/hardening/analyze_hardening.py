"""Audit real regression, full-model dropout training and native checkpoint evidence."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root=Path('/artifacts/hardening');production=Path('/artifacts/production')
report={'scope':'FSDP v2 partial CUDA Graph integration and end-to-end MoE validation only','complete':False,'regressions':{},'full_size_runs':{}}
for name,count,ranks in [('all-partial-4ranks-final',74,4),('fsdp-regression',22,4),('transformer-regression-final',33,4)]:
 out=root/name
 assert json.loads((out/'exit_code.json').read_text())['exit_code']==0,name
 log=(out/'pytest.log').read_text()
 found=re.findall(r'\b(\d+) passed\b',log)
 assert len(found)==ranks and all(int(v)==count for v in found),(name,found)
 report['regressions'][name]={'per_rank_passed':count,'ranks':ranks}
source=Path('/workspace/Megatron-LM')
for name in report['regressions']:
 for path,digest in json.loads((root/name/'source-sha256.json').read_text()).items():
  assert hashlib.sha256((source/path).read_bytes()).hexdigest()==digest,(name,path)

def scalars(name):
 a=EventAccumulator(str(production/name/'tensorboard'),size_guidance={'scalars':0});a.Reload()
 return {tag:[(v.step,v.value) for v in a.Scalars(tag)] for tag in a.Tags()['scalars']}
for name,first in [('hardening-eager-dropout',1),('hardening-partial-dropout',1),('hardening-eager-dropout-resume',7),('hardening-partial-dropout-resume',7)]:
 out=production/name
 assert json.loads((out/'exit_code.json').read_text())['exit_code']==0,name
 cmd=json.loads((out/'command.json').read_text())
 for option in ('--attention-dropout','--hidden-dropout'):assert float(cmd[cmd.index(option)+1])==0.1
 assert '--te-rng-tracker' in cmd and '--no-load-rng' not in cmd
 text=(out/'train.log').read_text()
 rows=re.findall(r'iteration\s+(\d+)/\s*\d+\s*\| consumed samples:\s*(\d+).*?number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)',text)
 assert [int(r[0]) for r in rows]==list(range(first,13)),(name,rows)
 assert all(int(samples)==int(step)*8 and int(skip)==int(nan)==0 for step,samples,skip,nan in rows)
 values=scalars(name)
 for tag in ('lm loss','grad-norm','params-norm','load_balancing_loss'):
  assert [v[0] for v in values[tag]]==list(range(first,13)),(name,tag)
  assert all(math.isfinite(v) for _,v in values[tag]),(name,tag)
 assert any('validation' in tag for tag in values)
 for path,digest in json.loads((out/'source-sha256.json').read_text()).items():
  if path.startswith('/workspace/Megatron-LM/'):
   assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,(name,path)
 report['full_size_runs'][name]={'steps':[first,12],'attention_dropout':0.1,'hidden_dropout':0.1,'nan_or_skips':0,'final_lm_loss':dict(values['lm loss'])[12]}
for left,right in [('hardening-eager-dropout','hardening-partial-dropout'),('hardening-eager-dropout-resume','hardening-partial-dropout-resume')]:
 p=production/f'compare-{left}-vs-{right}-step12.json';x=json.loads(p.read_text())
 assert x['passed'] and x['tensor_count']==6489 and x['tensor_elements']==20_757_485_568
 assert x['checked_non_tensor_entries']==30282
 report.setdefault('full_size_parity',[]).append({'report':p.name,'tensors':x['tensor_count'],'elements':x['tensor_elements'],'actual_non_tensor_entries':x['checked_non_tensor_entries'],'bit_exact':True})
trace=production/'hardening-partial-dropout/torch_profile/rank-0.json.gz'
events=json.loads(gzip.decompress(trace.read_bytes()))['traceEvents']
steps=[e for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep#')]
launches=[e for e in events if e.get('cat')=='cuda_runtime' and e.get('name','').startswith('cudaGraphLaunch')]
assert len(steps)==1 and len(launches)==256,(len(steps),len(launches))
assert all(steps[0]['ts']<=e['ts']<=steps[0]['ts']+steps[0]['dur'] for e in launches)
report['profile']={'recorded_optimizer_iterations':1,'cudaGraphLaunch':256,'expected':'32 units x GA4 x forward/backward2','trace':str(trace.relative_to(production)),'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'timing_used_as_performance':False}
report['continuation_metrics']={}
for mode in ('eager','partial'):
 a=scalars('hardening-'+mode+'-dropout');b=scalars('hardening-'+mode+'-dropout-resume')
 report['continuation_metrics'][mode]={tag:max(abs(dict(a[tag])[step]-dict(b[tag])[step]) for step in range(7,13)) for tag in ('lm loss','grad-norm','params-norm','load_balancing_loss')}
report['acceptance_evidence_passed']=True
(root/'acceptance-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
