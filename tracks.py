"""Optional conservative training-only LK tracks. No downloaded model assumed.
Requires manually supplied reliable-region masks (exclude liquid/specular areas).
Associate ID at t=0 via nearest projection, visible surface and mask. Retain only
forward/backward-consistent tracks triangulatable in >=2 training cameras.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from covariance_renderer import project,render


def build_tracks(data,model,mask_dir):
    if not mask_dir: raise ValueError('Tracking requires --track-masks with reliable region masks')
    frames=data.cfg['train_frames']; tracks={}
    with torch.no_grad():
        for name in data.train_cameras:
            mask=cv2.imread(str(Path(mask_dir)/(name+'.png')),cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            camera=data.camera(name,model.xyz.device)
            uv,q=project(model.xyz,camera); out=render(model.xyz,model.covariance(),model.color(),model.opacity(),camera)
            h,w=out['depth'].shape
            mask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
            start=(data.image(name,10).numpy()*255).astype('uint8')
            previous=cv2.cvtColor(start,cv2.COLOR_RGB2GRAY)
            corners=cv2.goodFeaturesToTrack(previous,2000,.02,5,mask=mask)
            if corners is None: continue
            corners=corners[:,0]; uv=uv.cpu().numpy(); q=q.cpu().numpy(); dep=out['depth'].cpu().numpy(); alpha=out['alpha'].cpu().numpy()
            assigned=[]; points=[]; used=set()
            for p in corners:
                gid=int(np.linalg.norm(uv-p,axis=1).argmin()); u,v=np.round(p).astype(int)
                if gid in used or np.linalg.norm(uv[gid]-p)>3 or q[gid,2]<=0 or alpha[v,u]<.1: continue
                if abs(q[gid,2]-dep[v,u])>max(.02*abs(dep[v,u]),data.extent*.01):continue
                assigned.append(gid); points.append(p); used.add(gid)
            if not points: continue
            p=np.float32(points).reshape(-1,1,2); alive=np.ones(len(p),dtype=bool)
            for frame in frames[1:]:
                current=cv2.cvtColor((data.image(name,frame).numpy()*255).astype('uint8'),cv2.COLOR_RGB2GRAY)
                nxt,s,_=cv2.calcOpticalFlowPyrLK(previous,current,p,None)
                back,sb,_=cv2.calcOpticalFlowPyrLK(current,previous,nxt,None)
                fb=np.linalg.norm(back[:,0]-p[:,0],axis=1)
                alive &= s[:,0].astype(bool)&sb[:,0].astype(bool)&(fb<.75)
                for idx,gid in enumerate(assigned):
                    u,v=nxt[idx,0]; iu,iv=int(round(u)),int(round(v))
                    good=alive[idx] and 0<=iu<w and 0<=iv<h and mask[iv,iu]>0
                    if good: tracks.setdefault((gid,frame),[]).append((name,[float(u),float(v)],float(np.exp(-fb[idx]))))
                    else: alive[idx]=False
                p=nxt; previous=current
    accepted=[]
    for (gid,frame),observations in tracks.items():
        if len(observations)<2: continue
        rows=[]
        for name,uv,conf in observations:
            c=data.cameras[name]; P=(c['K']@torch.cat([c['R'],c['t'][:,None]],1)).numpy()
            rows.extend([uv[0]*P[2]-P[0],uv[1]*P[2]-P[1]])
        _,_,vh=np.linalg.svd(np.array(rows)); X=vh[-1,:3]/vh[-1,3]
        ok=True
        for name,uv,conf in observations:
            c=data.cameras[name]; q=c['R'].numpy()@X+c['t'].numpy(); pred=c['K'].numpy()@q
            if q[2]<=0 or np.linalg.norm(pred[:2]/pred[2]-uv)>2:ok=False
        if ok:
            for name,uv,conf in observations: accepted.append({'id':gid,'frame':frame,'camera':name,'uv':uv,'confidence':conf})
    return accepted


def track_loss(x,frame,observations,data):
    losses=[]
    for o in observations:
        if o['frame']!=frame:continue
        uv,_=project(x[o['id']:o['id']+1],data.camera(o['camera'],x.device))
        target=x.new_tensor(o['uv'])
        # Smooth L1 in image-width units; confidence weight retained.
        losses.append(torch.nn.functional.smooth_l1_loss(uv[0]/data.cfg['image_width'],target/data.cfg['image_width'])*o['confidence'])
    return torch.stack(losses).mean() if losses else x.sum()*0
