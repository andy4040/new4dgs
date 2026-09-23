"""Overlay persistent IDs on an evaluation image; inspect displacement separately."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import N3DV
from covariance_renderer import project

p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run)
cfg=json.loads((root/'config.json').read_text());data=N3DV(cfg)
paths=np.load(root/'trajectories.npz');pos=paths['positions'];ids=paths['ids'];colors=paths['colors']
cam=data.camera('cam00','cpu');background=data.image('cam00',10,'view').numpy()
uv=[];depth=[]
for pp in pos:
 u,q=project(torch.tensor(pp,dtype=torch.float32),cam);uv.append(u.numpy());depth.append(q[:,2].numpy())
uv=np.array(uv);depth=np.array(depth)
valid=np.where((depth[0]>0)&(uv[0,:,0]>=0)&(uv[0,:,0]<cam['width'])&(uv[0,:,1]>=0)&(uv[0,:,1]<cam['height']))[0]
selection=valid[np.linspace(0,len(valid)-1,min(48,len(valid))).astype(int)] if len(valid) else []
fig,ax=plt.subplots(figsize=(10,7.5));ax.imshow(background,extent=(0,cam['width'],cam['height'],0))
for i in selection:
 ax.plot(*uv[:,i].T,color=colors[i],linewidth=1)
 ax.scatter(*uv[0,i],s=12,color=colors[i]);ax.text(*uv[0,i],str(ids[i]),fontsize=5,color=colors[i])
ax.set(xlim=(0,cam['width']),ylim=(cam['height'],0),title='cam00 evaluation frame 10 + fixed-ID projected paths (t=0..1)\nID persistence does not establish material correspondence')
fig.tight_layout();fig.savefig(root/'projection_overlay_cam00.png',dpi=150);plt.close(fig)
fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(projection='3d')
for i in selection:
 delta=pos[:,i]-pos[0,i];ax.plot(*delta.T,color=colors[i],linewidth=1)
ax.set(xlabel='dx (world units)',ylabel='dy (world units)',zlabel='dz (world units)',title='Displacement from each ID reference position; common origin')
fig.tight_layout();fig.savefig(root/'displacements_3d.png',dpi=150);plt.close(fig)
data.save_audit(root/'overlay_evaluation_access.json')
