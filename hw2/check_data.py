"""Verify the immutable course data before filling the assessed methods."""
from pathlib import Path
import hashlib,json,platform
import numpy as np
import torch
D=Path(__file__).resolve().parent/'data'
checks=json.loads((D/'checksums.json').read_text())
for name,expected in checks.items():
 p=D/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
  raise RuntimeError('Missing or modified course data: '+name)
m=json.loads((D/'manifest.json').read_text())
print(json.dumps(dict(passed=True,files=len(checks),dataset_id=m['dataset_id'],counts=m['counts'],python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),indent=2))
