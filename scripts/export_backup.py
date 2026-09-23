"""Create a downloadable artifact; this is NOT off-instance backup by itself."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

p=argparse.ArgumentParser();p.add_argument('--out',default='/workspace/new4dgs-backup.tar.gz');a=p.parse_args()
root=Path(__file__).resolve().parents[1];dest=Path(a.out).resolve();dest.parent.mkdir(parents=True,exist_ok=True)
status=subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)
if status.strip():raise SystemExit('Commit all source/report changes before exporting; refusing stale source archive.')
commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
selected=['synthetic','fixture_fm_rgb_v2','fixture_cuda','fixture_baseline','fixture_reloaded','fixture_tracks','fixture_appearance','n3dv_attempt']
manifest={'commit':commit,'remote_push_succeeded':False,'external_artifact_backup_succeeded':False,
          'note':'Download or copy off this nonpersistent Vast instance before deletion. No raw data included.','files':[]}
with tempfile.TemporaryDirectory(prefix='new4dgs_export_') as tmp:
 tmp=Path(tmp);source=tmp/'source.tar';bundle=tmp/'new4dgs.git.bundle'
 subprocess.run(['git','-C',str(root),'archive','--format=tar','--prefix=new4dgs/','-o',str(source),'HEAD'],check=True)
 subprocess.run(['git','-C',str(root),'bundle','create',str(bundle),'--all'],check=True)
 with tarfile.open(dest,'w:gz') as tar:
  tar.add(source,arcname='source.tar');tar.add(bundle,arcname=bundle.name)
  for name in selected:
   folder=root/'runs'/name
   if not folder.exists():continue
   for f in sorted(folder.rglob('*')):
    if not f.is_file():continue
    rel='results/'+str(f.relative_to(root/'runs'))
    manifest['files'].append({'path':rel,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    tar.add(f,arcname=rel)
  m=tmp/'MANIFEST.json';m.write_text(json.dumps(manifest,indent=2));tar.add(m,arcname='MANIFEST.json')
sha=hashlib.sha256(dest.read_bytes()).hexdigest();Path(str(dest)+'.sha256').write_text(f'{sha}  {dest.name}\n')
print(json.dumps({'archive':str(dest),'bytes':dest.stat().st_size,'sha256':sha,'source_commit':commit,'remote_push_succeeded':False,'external_backup_succeeded':False},indent=2))
