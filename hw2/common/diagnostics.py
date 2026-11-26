"""Provided public diagnostics. Expected CNN/GNN tensors are serialized fixtures.

Only optimizer diagnostics use torch.optim, as an independent allowed oracle.
No assessed CNN/GNN forward implementation is provided by these diagnostics.
"""
import json,traceback
from pathlib import Path
import torch
from .data import DATA,ROOT,configure,metadata
from .kernels import edge_features


def maxerr(actual,expected):
 if actual.shape!=expected.shape:raise AssertionError(f'shape {tuple(actual.shape)} != {tuple(expected.shape)}')
 if actual.numel()==0:return 0.
 if not torch.isfinite(actual).all():raise AssertionError('nonfinite output')
 return float((actual-expected).abs().max().detach())


def close(a,b):
 error=maxerr(a,b);torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10);return error


def cnn_checks(check):
 from CNN.model import DistanceCNN
 fixtures=torch.load(DATA/'assets/public_cnn.pt',weights_only=True)
 for f in fixtures:
  def run():
   model=DistanceCNN(f['mode']).double();model.load_state_dict(f['state'])
   x=f['x'].clone().requires_grad_();y=model(x);error=close(y,f['expected'])
   transpose_error=close(model(x.transpose(-1,-2)).transpose(-1,-2),y)
   (y*f['upstream']).sum().backward();gradient_error=close(x.grad,f['dx'])
   for name,p in model.named_parameters():gradient_error=max(gradient_error,close(p.grad,f['parameter_gradients'][name]))
   r,c=f['r'],f['c'];window=f['full'][...,r:r+32,c:c+32]
   context_error=close(model(window)[...,6:26,6:26],f['central'])
   count=sum(p.numel() for p in model.parameters());assert count==9888
   return dict(parameters=count,output_error=error,transpose_error=transpose_error,gradient_error=gradient_error,central_context_error=context_error)
  check(f"cnn_{f['mode']}",run)


def gnn_checks(check):
 from GNN.model import build_edges,MessagePassingLayer,PairHead,DistanceGNN
 fixtures=torch.load(DATA/'assets/public_gnn.pt',weights_only=True)
 for number,f in enumerate(fixtures['edge_cases']):
  def run(f=f):
   edges=build_edges(f['idx'],f['prior'],f['mode'],f['k'])
   actual={tuple(e) for e in edges.T.tolist()};expected={tuple(e) for e in f['expected'].T.tolist()}
   assert edges.shape[0]==2 and len(actual)==edges.shape[1] and actual==expected
   assert all(i!=j and (j,i) in actual for i,j in actual)
   return dict(nodes=len(f['idx']),directed_edges=edges.shape[1],max_in_degree=int(torch.bincount(edges[1],minlength=len(f['idx'])).max()))
  check('edges_'+str(number),run)
 for number,f in enumerate(fixtures['layer_cases']):
  def run(f=f):
   layer=MessagePassingLayer().double();layer.load_state_dict(f['state']);h=f['h'].clone().requires_grad_()
   y=layer(h,f['edges'],f['ef']);error=close(y,f['expected']);(y*f['upstream']).sum().backward();gradient_error=close(h.grad,f['dh'])
   for name,p in layer.named_parameters():gradient_error=max(gradient_error,close(p.grad,f['parameter_gradients'][name]))
   return dict(output_error=error,gradient_error=gradient_error,isolated=f['edges'].shape[1]==0)
  check('message_layer_'+str(number),run)
 def head_run():
  f=fixtures['head_case'];head=PairHead().double();head.load_state_dict(f['state']);y=head(f['hi'],f['hj'],f['prior'])
  return dict(output_error=close(y,f['expected']),symmetry_error=close(y,head(f['hj'],f['hi'],f['prior'])))
 check('pair_head',head_run)
 def model_run():
  f=fixtures['model_case'];model=DistanceGNN().double();model.load_state_dict(f['state']);pairs=f['pairs'];prior=f['prior'];idx=f['idx'];a=f['a']
  edges=build_edges(idx,prior);ef=edge_features(idx,prior,edges)
  y=model(a,edges,ef,pairs,prior[pairs[:,0],pairs[:,1]])
  output_error=close(y,f['expected']);y.sum().backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
  errors=[]
  for prior in [prior,torch.zeros_like(prior)]:
   edges=build_edges(idx,prior);ef=edge_features(idx,prior,edges);original=model(a,edges,ef,pairs,prior[pairs[:,0],pairs[:,1]])
   perm=f['perm'];inverse=torch.argsort(perm);q=inverse[pairs];pnew=prior[perm][:,perm]
   enew=build_edges(idx[perm],pnew);efnew=edge_features(idx[perm],pnew,enew)
   transformed=model(a[perm],enew,efnew,q,pnew[q[:,0],q[:,1]])
   errors.append(close(transformed,original))
   restored={tuple(map(int,perm[e].tolist())) for e in enew.T};assert restored=={tuple(e) for e in edges.T.tolist()}
  return dict(parameters=sum(p.numel() for p in model.parameters()),output_error=output_error,permutation_error=max(errors),tied_scores_included=True)
 check('model_and_rebuilt_permutation',model_run)


def optimizer_pair(kind,params,reference,lr,betas=(.8,.95),eps=1e-7):
 from Optimization.optimizers import SGD,Momentum,Adam
 if kind=='sgd':return SGD(params,lr),torch.optim.SGD(reference,lr=lr)
 if kind=='momentum':return Momentum(params,lr,momentum=.9),torch.optim.SGD(reference,lr=lr,momentum=.9)
 return Adam(params,lr,betas=betas,eps=eps),torch.optim.Adam(reference,lr=lr,betas=betas,eps=eps)


def optimizer_trajectory(kind,steps=5,seed=71):
 generator=torch.Generator().manual_seed(seed)
 params=[torch.randn(shape,generator=generator,dtype=torch.float64,requires_grad=True) for shape in [(2,3),(4,)]]
 reference=[p.detach().clone().requires_grad_() for p in params];ours,oracle=optimizer_pair(kind,params,reference,.013)
 error=0.
 for _ in range(steps):
  for p,q in zip(params,reference):p.grad=torch.randn(p.shape,generator=generator,dtype=p.dtype);q.grad=p.grad.clone()
  ours.step();oracle.step()
  for p,q in zip(params,reference):error=max(error,close(p,q))
 return dict(max_parameter_error=error,steps=steps,tensors=2,state_persistence_and_bias_correction='checked by multi-step parameter trajectories')


def optimizer_checks(check):
 for kind in ['sgd','momentum','adam']:
  check(kind+'_multi_tensor',lambda kind=kind:optimizer_trajectory(kind))



def run_diagnostics(unit,args):
 configure(args.device,args.threads);checks=[]
 def check(name,fn):
  try:values=fn() or {};checks.append(dict(name=name,passed=True,**values));print('PASS',name,flush=True)
  except Exception as e:
   checks.append(dict(name=name,passed=False,error_type=type(e).__name__,error=str(e)[:3000]));print('FAIL',name,type(e).__name__,str(e)[:200],flush=True)
 {'CNN':cnn_checks,'GNN':gnn_checks,'Optimization':optimizer_checks}[unit](check)
 report=dict(unit=unit,passed=all(x['passed'] for x in checks),checks=checks,metadata=metadata('cpu'))
 output=Path(args.output) if args.output else ROOT/'results'/unit/'diagnostics.json';output.parent.mkdir(parents=True,exist_ok=True)
 output.write_text(json.dumps(report,indent=2)+'\n');print('Report:',output,flush=True)
 return report
