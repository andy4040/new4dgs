"""Generate a synthetic calibrated planar fixture, NEVER coffee_martini data."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--out',default='runs/generated_fixture');a=p.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(10)
texture=rng.integers(0,256,(512,512,3),dtype=np.uint8)
texture=cv2.GaussianBlur(texture,(5,5),0)
for _ in range(600):
 x,y=rng.integers(5,506,2);r=int(rng.integers(2,8));color=tuple(int(x) for x in rng.integers(0,256,3));cv2.circle(texture,(x,y),r,color,-1)
poses=[]
for ci,center in enumerate([0.,-.3,.3]):
 pose=np.zeros((3,5));pose[:,:3]=[[0,1,0],[1,0,0],[0,0,-1]];pose[:,3]=[center,0,0];pose[:,4]=[240,320,280];poses.append(pose)
 folder=root/f'cam{ci:02d}'/'images';folder.mkdir(parents=True,exist_ok=True)
 yy,xx=np.mgrid[:240,:320].astype('float32')
 for frame in range(31):
  t=(frame-10)/20;dx=.08*np.sin(np.pi*t);dy=.04*t
  worldx=(xx-160)*2/280+center-dx;worldy=(yy-120)*2/280-dy
  img=cv2.remap(texture,(worldx*130+256).astype("float32"),(worldy*130+256).astype("float32"),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
  cv2.imwrite(str(folder/f'{frame:04d}.png'),img)
np.save(root/'poses_bounds.npy',np.concatenate([np.array(poses).reshape(3,15),np.tile([1.,4.],(3,1))],axis=1))
(root/'SYNTHETIC_NOT_N3DV.json').write_text(json.dumps({'source':'procedurally generated plane texture','is_n3dv':False}))
print(root.resolve())
