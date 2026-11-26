from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.diagnostics import run_diagnostics

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true')
 p.add_argument('--device',choices=['cpu'],default='cpu');p.add_argument('--threads',type=int,default=4);p.add_argument('--output',default=None)
 args=p.parse_args();report=run_diagnostics('Optimization',args)
 raise SystemExit(0 if report['passed'] else 1)
