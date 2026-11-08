"""Read real DCP RNG payloads and check paired runs plus CUDA RNG continuation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

root=Path('/artifacts/production')

def canonical(value):
 if isinstance(value,torch.Generator):return {'generator_state':canonical(value.get_state())}
 if isinstance(value,torch.Tensor):
  array=value.detach().cpu().contiguous().numpy()
  return {'shape':list(array.shape),'dtype':str(array.dtype),'sha256':hashlib.sha256(array.tobytes()).hexdigest()}
 if isinstance(value,np.ndarray):return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {str(k):canonical(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [canonical(v) for v in value]
 assert isinstance(value,(str,int,float,bool,type(None))),type(value)
 return value

def read(name):
 r=dcp.FileSystemReader(root/name/'checkpoints/iter_0000012');m=r.read_metadata().state_dict_metadata
 keys=[k for k in m if k.startswith('rng_state.')]
 assert keys,name
 state={k:torch.empty(m[k].size,dtype=m[k].properties.dtype) if isinstance(m[k],TensorStorageMetadata) else None for k in keys}
 planner=dcp.DefaultLoadPlanner(flatten_state_dict=False,allow_partial_load=True)
 dcp.load(state,storage_reader=r,planner=planner,no_dist=True)
 return {k:canonical(v) for k,v in planner.state_dict.items()}
names=['hardening-eager-dropout','hardening-partial-dropout','hardening-eager-dropout-resume','hardening-partial-dropout-resume']
values={n:read(n) for n in names}
for a,b in [(names[0],names[1]),(names[2],names[3])]:assert values[a]==values[b],(a,b)
checks={}
for mode in ('eager','partial'):
 a=values['hardening-'+mode+'-dropout'];b=values['hardening-'+mode+'-dropout-resume']
 cuda_keys=[k for k in a if '.cuda_rng_state' in k or '.rng_tracker_states.' in k]
 checks[mode]={k:a[k]==b[k] for k in cuda_keys}
 assert all(checks[mode].values()),checks[mode]
report={'passed':True,'paired_all_RNG_payloads_equal':True,'CUDA_RNG_continuation_equal':checks,'checkpoint_step':12,'states':values}
Path('/artifacts/hardening/rng-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='states'},indent=2))
