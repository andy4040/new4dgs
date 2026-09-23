"""Build resized derived PNG cache, selecting explicit split keys only."""
import argparse,concurrent.futures,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from data import N3DV
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--splits',nargs='+',default=['train']);a=p.parse_args()
cfg=json.loads(Path(a.config).read_text());d=N3DV(cfg);cv2.setNumThreads(1);start=time.perf_counter()
keys=set(k for split in a.splits for k in d.keys(split));root=Path(cfg['image_cache'])/f"w{cfg['image_width']}";root.mkdir(parents=True,exist_ok=True)
def camera(name):
    frames={f for c,f in keys if c==name};folder=root/name;folder.mkdir(exist_ok=True)
    missing={f for f in frames if not (folder/f'{f:04d}.png').exists()}
    if not missing:return name,0
    cap=cv2.VideoCapture(str(d.root/(name+'.mp4')));count=0
    for f in range(max(missing)+1):
        if not cap.grab():raise RuntimeError(f'{name}/{f}: decode failure')
        if f not in missing:continue
        ok,img=cap.retrieve()
        if not ok:raise RuntimeError(f'{name}/{f}: retrieve failure')
        c=d.cameras[name];img=cv2.resize(img,(c['width'],c['height']),interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(folder/f'{f:04d}.png'),img,[cv2.IMWRITE_PNG_COMPRESSION,1]):raise IOError(name)
        count+=1
    cap.release();print(name,count,flush=True);return name,count
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(camera,sorted({c for c,f in keys})))
manifest=dict(source_root=str(d.root),splits=a.splits,keys=sorted(keys),new_images=results,seconds=time.perf_counter()-start)
(root/('manifest_'+'_'.join(a.splits)+'.json')).write_text(json.dumps(manifest,indent=2));print(json.dumps({'seconds':manifest['seconds'],'images':len(keys)}),flush=True)
