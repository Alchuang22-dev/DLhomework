"""Provided loaders, immutable sampling plans and feature/loss inputs."""
from pathlib import Path
import hashlib,json,os,platform,random
import numpy as np
import torch
from .kernels import pair_features,canonical_prior

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'


def configure(device='cpu',threads=4,seed=416):
 if not 1<=threads<=4:raise ValueError('Use 1 to 4 CPU compute threads')
 torch.set_num_threads(threads)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 d=torch.device(device)
 if d.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu')
 os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.mplconfig'))
 return d


def metadata(device,seed=416):
 manifest=DATA/'manifest.json'
 return dict(device=str(device),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),manifest_id=json.loads(manifest.read_text())['dataset_id'],manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),seed=seed,dtype='float32',release_revision='r9',selection_metric='full_chain_macro_nll')


def load_chains(split,with_pairs=False,with_prior=False,device='cpu',prior_dir=None):
 manifest=json.loads((DATA/'manifest.json').read_text());result=[]
 projection=torch.from_numpy(np.load(DATA/'assets/projection.npy')).to(device)
 for entry in manifest['entries']:
  if entry['split']!=split:continue
  with np.load(DATA/entry['file'],allow_pickle=False) as f:
   a=torch.from_numpy(f['node_features'].copy()).float().to(device)
   idx=torch.from_numpy(f['residue_index'].copy()).long().to(device)
   y=torch.from_numpy(f['target_bin'].copy()).long().to(device)
   mask=torch.from_numpy(f['coord_mask'].copy()).bool().to(device)
  n=len(a);gap=(idx[:,None]-idx[None,:]).abs()
  valid=mask[:,None]&mask[None,:]&(gap>=6)
  queries=(valid&torch.ones(n,n,dtype=torch.bool,device=device).triu(1)).nonzero()
  item=dict(id=entry['id'],group=entry['group'],a=a,idx=idx,y=y,mask=mask,valid=valid,queries=queries,entry=entry)
  if with_pairs:item['features']=pair_features(a,idx,projection).permute(2,0,1).contiguous()
  if with_prior:item['prior']=canonical_prior(torch.from_numpy(np.load((Path(prior_dir) if prior_dir else DATA/'priors')/f"{entry['id']}.npy").copy()).to(device))
  result.append(item)
 if not result:raise ValueError('No chains for split '+split)
 return result


def load_plans():
 return torch.load(DATA/'assets/plans.pt',map_location='cpu',weights_only=True)


def materialize_windows(chains,plan):
 lookup={x['id']:x for x in chains};batches=[]
 for batch in plan:
  xs=[];ys=[];vs=[]
  for cid,r,c in batch:
   item=lookup[cid]
   xs.append(item['features'][:,r:r+32,c:c+32]);ys.append(item['y'][r:r+32,c:c+32]);vs.append(item['valid'][r:r+32,c:c+32])
  batches.append((torch.stack(xs),torch.stack(ys),torch.stack(vs)))
 return batches


def materialize_queries(chains,plan):
 lookup={x['id']:x for x in chains}
 return [(lookup[cid],pairs.to(lookup[cid]['a'].device)) for cid,pairs in plan]


def sample_windows(chains,updates,rng,batch_size=4):
 plan=[]
 for _ in range(updates):
  batch=[]
  for _ in range(batch_size):
   item=chains[int(rng.integers(len(chains)))];n=len(item['a'])
   for attempt in range(10000):
    r,c=map(int,rng.integers(0,n-31,size=2))
    if bool(item['valid'][r+6:r+26,c+6:c+26].any()):break
   else:raise ValueError('No eligible window for '+item['id'])
   batch.append((item['id'],r,c))
  plan.append(batch)
 return plan


def sample_queries(chains,updates,rng):
 plan=[]
 for _ in range(updates):
  item=chains[int(rng.integers(len(chains)))];pairs=item['queries']
  ids=rng.choice(len(pairs),size=min(128,len(pairs)),replace=False)
  plan.append((item['id'],pairs[ids].cpu()))
 return plan
