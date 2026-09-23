"""Download the original coffee_martini release into an external data directory."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile

URL='https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip'
SIZE=1186324684
p=argparse.ArgumentParser();p.add_argument('--root',default='/workspace/datasets');p.add_argument('--keep-archive',action='store_true');a=p.parse_args()
root=Path(a.root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
scene=root/'coffee_martini';archive=root/'coffee_martini.zip';manifest=root/'coffee_martini.download.json'
if scene.exists():
 if manifest.exists() and (scene/'poses_bounds.npy').exists():
  print(f'Already extracted: {scene}');raise SystemExit(0)
 raise SystemExit(f'Refusing to overwrite existing directory: {scene}')
start=time.perf_counter()
if not archive.exists() or archive.stat().st_size!=SIZE:
 subprocess.run(['curl','-fL','--retry','3','-C','-',URL,'-o',str(archive)],check=True)
if archive.stat().st_size!=SIZE:raise RuntimeError('Release asset size mismatch')
sha=hashlib.sha256()
with archive.open('rb') as f:
 for block in iter(lambda:f.read(1024*1024),b''):sha.update(block)
with zipfile.ZipFile(archive) as z:
 for info in z.infolist():
  if not (root/info.filename).resolve().is_relative_to(root):raise ValueError('Unsafe zip path')
  if not info.filename.startswith('coffee_martini/'):raise ValueError(f'Unexpected archive structure: {info.filename}')
 bad=z.testzip()
 if bad:raise RuntimeError(f'ZIP CRC failed: {bad}')
 z.extractall(root)
if not (scene/'poses_bounds.npy').exists():raise RuntimeError('Missing calibration after extraction')
report={'source_url':URL,'source_repository':'https://github.com/facebookresearch/Neural_3D_Video',
 'license':'CC-BY-NC-4.0','archive_size':SIZE,'archive_sha256_local':sha.hexdigest(),
 'integrity':'ZIP member CRC verified; SHA256 measured locally, no upstream digest supplied',
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'data_root':str(scene),
 'download_or_resume_and_extraction_seconds':time.perf_counter()-start,'archive_retained':a.keep_archive}
manifest.write_text(json.dumps(report,indent=2))
if not a.keep_archive:archive.unlink()
print(json.dumps(report,indent=2))
