"""Provided export utility; CNN/train.py --mode all invokes it automatically."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data import configure
from common.priors import export_priors

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='results/CNN/plain.pt')
    p.add_argument('--output',default='results/CNN/plain_priors')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();configure(args.device,args.threads)
    info=export_priors(args.checkpoint,args.output,args.device)
    print(json.dumps({k:v for k,v in info.items() if k!='files'},indent=2))
