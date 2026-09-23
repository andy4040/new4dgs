"""Sequential, identical splits/resolution/count limits for four required variants."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--config',default='configs/smoke.json');p.add_argument('--out',default='runs/matrix');a=p.parse_args()
out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True);data=Path(a.data).resolve();rows=[]
for variant in ['rgb','fm_rgb','fm_persistent','canonical_4dgs']:
 dest=out/variant
 subprocess.run([sys.executable,'run.py','--config',a.config,'--data',str(data),'--variant',variant,'--out',str(dest)],check=True)
 m=json.loads((dest/'metrics.json').read_text())
 for split,metrics in m['image'].items():
  rows.append({'variant':variant,'split':split,**metrics,'endpoint_chamfer_squared':m['distribution']['endpoint_symmetric_chamfer_squared'],'trajectory_3d_error':'unavailable','seconds_total':m['total_seconds_including_preprocessing']})
with (out/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
