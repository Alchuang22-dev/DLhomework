"""Run the prescribed GNN experiments; default device is CPU."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.engine import run_unit

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--mode',choices=['all', 'sequence', 'predicted'],default='all')
 p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
 p.add_argument('--output',default=None)
 group=p.add_mutually_exclusive_group()
 group.add_argument('--prior-dir',default=None,help='Part Two plain CNN prediction cache; default results/CNN/plain_priors')
 group.add_argument('--use-reference-prior',action='store_true',help='Explicit staff fallback for independent Part Three work')
 args=p.parse_args();run_unit('GNN',args)
