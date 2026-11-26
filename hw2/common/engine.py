"""Provided fixed-budget CPU training and chain-level evaluation."""
from pathlib import Path
import copy,csv,hashlib,json,sys,time
try:
 import resource
except ImportError:
 resource=None
import psutil
import numpy as np
import torch
from torch.nn import functional as F
from .data import DATA,ROOT,configure,metadata,load_chains,load_plans,materialize_windows,materialize_queries
from .kernels import window_loss,chain_metrics,edge_features


def rss_mib():
 if resource is not None:
  value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return value/(1024**2 if sys.platform=='darwin' else 1024)
 memory=psutil.Process().memory_info()
 return getattr(memory,'peak_wset',memory.rss)/2**20


def graph_cache(chains,builder,mode):
 cache={}
 for item in chains:
  edge=builder(item['idx'],item['prior'],mode=mode)
  cache[item['id']]=(edge,edge_features(item['idx'],item['prior'],edge))
 return cache


@torch.no_grad()
def predict(model,item,kind,graph=None):
 if kind=='cnn':return model(item['features'][None])[0].permute(1,2,0)
 edge,ef=graph;h=model.encode(item['a'],edge,ef)
 output=item['prior'].clone()
 for pairs in item['queries'].split(128):
  i,j=pairs.T;value=model.decode(h,pairs,item['prior'][i,j]);output[i,j]=value;output[j,i]=value
 return output


@torch.no_grad()
def evaluate(model,chains,kind,graphs=None,baseline=None):
 rows=[]
 for item in chains:
  if baseline=='frozen_prior':scores=item['prior']
  elif baseline=='sequence':
   table=torch.from_numpy(np.load(DATA/'assets/sequence_baseline.npy')).to(item['a'].device)
   gap=(item['idx'][:,None]-item['idx'][None,:]).abs()
   bucket=torch.bucketize(gap,torch.tensor([12,24,48],device=gap.device),right=True)
   scores=table[bucket]
  else:scores=predict(model,item,kind,None if graphs is None else graphs[item['id']])
  values=chain_metrics(scores,item['y'],item['mask'],item['idx'])
  rows.append({'chain_id':item['id'],'length':len(item['a']),**values})
 result={}
 for key in ['nll','precision_at_l']:
  values=[row[key] for row in rows if row[key] is not None]
  result[key]=float(np.mean(values)) if values else None;result[key+'_chains']=len(values)
 result['chains']=len(rows);result['valid_pairs']=sum(row['valid_pairs'] for row in rows)
 return result,rows



@torch.no_grad()
def validation_nll(model,chains,kind,graphs=None):
 """Full validation set; equal weight per eligible chain, without contact sorting."""
 values=[]
 for item in chains:
  if not len(item['queries']):continue
  scores=predict(model,item,kind,None if graphs is None else graphs[item['id']])
  i,j=item['queries'].T
  values.append(float(F.cross_entropy(scores[i,j],item['y'][i,j])))
 if not values:raise ValueError('No eligible validation chains')
 return float(np.mean(values))


def write_csv(path,rows):
 if not rows:return
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plot_curves(results,path):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
 for result in results:
  rows=result['history'];axes[0].plot([r['step'] for r in rows],[r['train_loss'] for r in rows],label=result['configuration'],linewidth=1)
  checks=[r for r in rows if r['validation_nll'] is not None]
  axes[1].plot([r['step'] for r in checks],[r['validation_nll'] for r in checks],marker='o',label=result['configuration'])
 for ax,title in zip(axes,['Training objective (window/query mean)','Full validation NLL (chain macro mean)']):
  ax.set(xlabel='Parameter updates',ylabel='Cross-entropy',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.savefig(path,dpi=160);plt.close(fig)


def train_one(name,model,optimizer,train,validation,test,plans,out,kind='cnn',builder=None,updates=300,evaluate_test=True,seed=416,graph_mode=None):
 start=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if kind=='cnn':
  batches=materialize_windows(train,plans['cnn_train'][:updates])
  graphs=None
  def objective(batch):return window_loss(model(batch[0]),batch[1],batch[2])
 else:
  graphs=graph_cache(train+validation,builder,graph_mode or name)
  batches=materialize_queries(train,plans['gnn_train'][:updates])
  def objective(batch):
   item,queries=batch;i,j=queries.T;edge,ef=graphs[item['id']]
   return F.cross_entropy(model(item['a'],edge,ef,queries,item['prior'][i,j]),item['y'][i,j])
 # Warm one forward/backward without updating weights; no stochastic layers or
 # running statistics are present. This excludes first-use compilation from time.
 model.train();warm=objective(batches[0]);warm.backward();optimizer.zero_grad()
 best=float('inf');selected=None;saved=None;history=[];update_seconds=0.;validation_seconds=0.;validation_evaluations=0;test_evaluations=0
 for step,batch in enumerate(batches,1):
  tick=time.perf_counter();model.train();optimizer.zero_grad();loss=objective(batch)
  if not torch.isfinite(loss):raise FloatingPointError(f'{name}: nonfinite loss at {step}')
  loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  if not torch.isfinite(norm):raise FloatingPointError(f'{name}: nonfinite gradient')
  optimizer.step();update_seconds+=time.perf_counter()-tick
  row={'step':step,'train_loss':float(loss.detach()),'validation_nll':None,'gradient_norm_before_clip':float(norm)}
  if step%50==0:
   model.eval();tick=time.perf_counter()
   score=validation_nll(model,validation,kind,graphs)
   validation_seconds+=time.perf_counter()-tick;validation_evaluations+=1
   if not np.isfinite(score):raise FloatingPointError('nonfinite validation NLL')
   row['validation_nll']=score
   if score<best:best=score;selected=step;saved=copy.deepcopy(model.state_dict())
  history.append(row)
 assert selected is not None
 model.load_state_dict(saved);model.eval();tick=time.perf_counter()
 val_metrics,val_rows=evaluate(model,validation,kind,graphs)
 test_metrics=test_rows=None
 if evaluate_test:
  test_graphs=graph_cache(test,builder,graph_mode or name) if kind=='gnn' else None
  test_metrics,test_rows=evaluate(model,test,kind,test_graphs);test_evaluations+=1
 evaluation_seconds=time.perf_counter()-tick
 checkpoint=out/(name+'.pt');torch.save(model.state_dict(),checkpoint)
 result=dict(configuration=name,updates=updates,parameters=sum(p.numel() for p in model.parameters()),selected_step=selected,selected_validation_nll=best,validation=val_metrics,test=test_metrics,update_seconds=update_seconds,validation_seconds=validation_seconds,validation_evaluations=validation_evaluations,test_evaluations=test_evaluations,selection_metric='full_chain_macro_nll',checkpoint_candidates=list(range(50,updates+1,50)),evaluation_seconds=evaluation_seconds,run_seconds=time.perf_counter()-start,process_peak_rss_mib=rss_mib(),history=history,metadata=metadata(next(model.parameters()).device,seed=seed),checkpoint=checkpoint.name)
 result['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 (out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n');write_csv(out/(name+'_history.csv'),history)
 write_csv(out/(name+'_validation.csv'),val_rows)
 if test_rows is not None:write_csv(out/(name+'_test.csv'),test_rows)
 print(name,'selected',selected,'val NLL',round(val_metrics['nll'],5),'test NLL',round(test_metrics['nll'],5) if test_metrics else None,'update seconds',round(update_seconds,2),flush=True)
 return result


def run_unit(unit,args):
 device=configure(args.device,args.threads);kind='gnn' if unit=='GNN' or (unit=='Optimization' and args.task=='gnn') else 'cnn';need_prior=kind=='gnn'
 prior_dir=None;prior_info=None
 if need_prior:
  from .priors import validate_prior_cache
  prior_dir=DATA/'priors' if args.use_reference_prior else Path(args.prior_dir or ROOT/'results/CNN/plain_priors')
  prior_info=validate_prior_cache(prior_dir)
  print('GNN CNN-source:',prior_info['source'],'checkpoint',prior_info['cnn_checkpoint_sha256'],flush=True)
 train=load_chains('train',with_pairs=kind=='cnn',with_prior=need_prior,device=device,prior_dir=prior_dir)
 val=load_chains('validation',with_pairs=kind=='cnn',with_prior=need_prior,device=device,prior_dir=prior_dir)
 test=load_chains('test',with_pairs=kind=='cnn',with_prior=need_prior,device=device,prior_dir=prior_dir)
 plans=load_plans();output=Path(args.output) if args.output else ROOT/'results'/unit;output.mkdir(parents=True,exist_ok=True)
 initial=torch.load(DATA/'assets'/('gnn_initial.pt' if kind=='gnn' else 'cnn_initial.pt'),map_location=device,weights_only=True)
 if unit=='CNN':
  from CNN.model import DistanceCNN
  configs=['plain','dilated'] if args.mode=='all' else [args.mode]
 elif unit=='GNN':
  from GNN.model import DistanceGNN,build_edges
  configs=['sequence','predicted'] if args.mode=='all' else [args.mode]
 else:
  from Optimization.model import load_model,load_graph_builder
  from Optimization.optimizers import make_optimizer
  configs=['sgd','momentum','adam'] if args.mode=='all' else [args.mode]
  if kind=='gnn':build_edges=load_graph_builder(device)
 results=[]
 for name in configs:
  if unit=='CNN':model=DistanceCNN(name).to(device)
  elif unit=='GNN':model=DistanceGNN().to(device)
  else:model=load_model(device,args.task)
  model.load_state_dict(initial)
  if unit=='Optimization':optimizer=make_optimizer(model.parameters(),name,lr={'sgd':.05,'momentum':.01,'adam':.0001 if kind=='gnn' else .001}[name])
  else:optimizer=torch.optim.Adam(model.parameters(),lr=.0001 if unit=='GNN' else .001,betas=(.9,.999),eps=1e-8)
  result=train_one(name,model,optimizer,train,val,test,plans,output,kind,build_edges if need_prior else None,200 if unit=='Optimization' else 300,graph_mode='predicted' if unit=='Optimization' and need_prior else None)
  result['task']=kind
  result['learning_rate']=optimizer.lr if unit=='Optimization' else optimizer.param_groups[0]['lr']
  (output/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
  results.append(result)
 baseline_name='part2_plain_cnn' if need_prior else 'sequence'
 baseline={}
 for split,chains in [('validation',val),('test',test)]:
  metrics,baseline_rows=evaluate(None,chains,kind,baseline='frozen_prior' if need_prior else 'sequence')
  baseline[split]=metrics
  if need_prior and split=='validation':
   write_csv(output/'cnn_baseline_validation.csv',baseline_rows)
 if need_prior:
  for result in results:
   result['prior_source']=prior_info
   (output/(result['configuration']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 summary=dict(unit=unit,metadata=metadata(device),baseline={'name':baseline_name,**baseline},runs=[{k:v for k,v in result.items() if k!='history'} for result in results])
 summary['task']=kind
 if need_prior:summary['cnn_baseline_validation_csv']='cnn_baseline_validation.csv'
 if need_prior:summary['prior_source']=prior_info
 if need_prior:
  if 'cnn_metrics' in prior_info:
   summary['cnn_baseline_check']={split:{key:abs(baseline[split][key]-prior_info['cnn_metrics'][split][key]) for key in ['nll','precision_at_l']} for split in ['validation','test']}
   if any(x['nll']>1e-5 for x in summary['cnn_baseline_check'].values()):
    raise ValueError('Cached CNN baseline disagrees with its source checkpoint metrics')
  else:
   summary['cnn_baseline_check']=dict(status='passed',method='cache_integrity',verified_chains=prior_info['verified_chains'],source_metrics_compared=False,
    note='Supplied fallback: provenance and all prediction hashes verified. Baseline metrics were evaluated in this run; no source score table is distributed.')
 if unit=='CNN' and any(r['configuration']=='plain' for r in results):
  from .priors import export_priors
  info=export_priors(output/'plain.pt',output/'plain_priors',device)
  summary['exported_prior_cache']={k:v for k,v in info.items() if k!='files'}
  print('Part Three inputs exported to',output/'plain_priors',flush=True)
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');plot_curves(results,output/'curves.png')
 rows=[]
 for result in results:
  rows.append(dict(configuration=result['configuration'],selected_step=result['selected_step'],validation_nll=result['validation']['nll'],validation_precision=result['validation']['precision_at_l'],test_nll=result['test']['nll'],test_precision=result['test']['precision_at_l'],update_seconds=result['update_seconds']))
 write_csv(output/'summary.csv',rows)
 return summary
