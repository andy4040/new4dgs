"""Training-only SIFT triangulation and endpoint RGB optimization."""
import math
import itertools
import cv2
import numpy as np
import torch
from torch import nn
from covariance_renderer import render


class Gaussians(nn.Module):
    def __init__(self, xyz, rgb, scale):
        super().__init__()
        self.xyz=nn.Parameter(xyz.clone())
        self.log_scale=nn.Parameter(torch.full_like(xyz, math.log(scale)))
        self.lower=nn.Parameter(torch.zeros_like(xyz))
        self.color_logits=nn.Parameter(torch.logit(rgb.clamp(.01,.99)))
        self.opacity_logits=nn.Parameter(torch.full((len(xyz),),-1.,device=xyz.device))
        self.register_buffer('ids',torch.arange(len(xyz),device=xyz.device))
        self.register_buffer('parent_ids',torch.full((len(xyz),),-1,device=xyz.device,dtype=torch.long))

    def factor(self):
        L=torch.diag_embed(self.log_scale.clamp(-12,4).exp())
        off=torch.zeros_like(L)
        off[:,1,0]=self.lower[:,0]; off[:,2,0]=self.lower[:,1]; off[:,2,1]=self.lower[:,2]
        return L+off

    def covariance(self):
        L=self.factor(); return L@L.transpose(-1,-2)+torch.eye(3,device=L.device)*1e-8

    def color(self): return self.color_logits.sigmoid()
    def opacity(self): return self.opacity_logits.sigmoid()


def triangulate(data, frame, limit):
    if frame not in (10,30) or frame not in data.cfg['train_frames']: raise PermissionError(frame)
    sift=cv2.SIFT_create(nfeatures=4000)
    features={}; imgs={}; projections={}
    for name in data.train_cameras:
        img=(data.image(name,frame,'train').numpy()*255).astype(np.uint8)
        imgs[name]=img; features[name]=sift.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY),None)
        c=data.cameras[name]; projections[name]=c['K'].numpy()@np.column_stack([c['R'].numpy(),c['t'].numpy()])
    xyz=[]; colors=[]; observations=[]
    for ca,cb in itertools.combinations(data.train_cameras,2):
        ka,da=features[ca]; kb,db=features[cb]
        if da is None or db is None or len(db)<2 or len(da)<2: continue
        matcher=cv2.BFMatcher()
        forward=matcher.knnMatch(da,db,k=2); reverse=matcher.knnMatch(db,da,k=2)
        rev={m.queryIdx:m.trainIdx for m,n in reverse if m.distance<.7*n.distance}
        matches=[m for m,n in forward if m.distance<.7*n.distance and rev.get(m.trainIdx)==m.queryIdx]
        if len(matches)<8: continue
        a=np.float64([ka[m.queryIdx].pt for m in matches]); b=np.float64([kb[m.trainIdx].pt for m in matches])
        # Geometry verification uses calibrated reprojection, never held-out cameras.
        X=cv2.triangulatePoints(projections[ca],projections[cb],a.T,b.T)
        X=(X[:3]/X[3:]).T
        good=np.isfinite(X).all(1)
        for name,uv in [(ca,a),(cb,b)]:
            c=data.cameras[name]; q=X@c['R'].numpy().T+c['t'].numpy()
            z=q[:,2]; pred=q@c['K'].numpy().T
            pred=pred[:,:2]/pred[:,2:]
            good &= (z>0)&(np.linalg.norm(pred-uv,axis=1)<1.5)
        center_a=np.array(data.cameras[ca]['center']); center_b=np.array(data.cameras[cb]['center'])
        ra=X-center_a; rb=X-center_b
        cosine=(ra*rb).sum(1)/(np.linalg.norm(ra,axis=1)*np.linalg.norm(rb,axis=1)+1e-12)
        good &= (cosine<np.cos(np.deg2rad(1))) & (np.linalg.norm(ra,axis=1)<100*data.extent)
        for point,pix in zip(X[good],a[good]):
            u,v=np.round(pix).astype(int)
            xyz.append(point); colors.append(imgs[ca][v,u]/255.); observations.append([ca,cb])
    if len(xyz)<16:
        raise RuntimeError(f'Only {len(xyz)} valid triangulations at frame {frame}; increase initialization resolution or improve calibration. No random geometry fallback.')
    xyz=np.array(xyz); colors=np.array(colors)
    # Remove repeated pair observations using a small spatial voxel.
    _,idx=np.unique(np.round(xyz/(data.extent*.002)),axis=0,return_index=True)
    rng=np.random.default_rng(data.cfg['seed']+frame); rng.shuffle(idx); idx=idx[:limit]
    return torch.tensor(xyz[idx],dtype=torch.float32),torch.tensor(colors[idx],dtype=torch.float32),{'frame':frame,'raw_points':len(xyz),'selected':len(idx),'source':'SIFT mutual ratio / calibrated reprojection / positive depth / parallax'}


def position_lr(cfg,step,extent):
    # Upstream exponential schedule times scene.cameras_extent (spatial_lr_scale).
    ratio=min(step/cfg['endpoint_position_lr_max_steps'],1)
    return extent*math.exp((1-ratio)*math.log(cfg['endpoint_position_lr_init'])+ratio*math.log(cfg['endpoint_position_lr_final']))


def optimize_endpoint(data,frame,model,cfg):
    opt=torch.optim.Adam([{'params':[model.xyz],'lr':position_lr(cfg,0,data.extent)},
        {'params':[model.log_scale,model.lower,model.color_logits,model.opacity_logits],'lr':cfg['endpoint_other_lr']}])
    history=[]
    for i in range(cfg['endpoint_steps']):
        opt.param_groups[0]['lr']=position_lr(cfg,i,data.extent)
        name=data.train_cameras[i%len(data.train_cameras)]
        target=data.image(name,frame).to(model.xyz.device)
        out=render(model.xyz,model.covariance(),model.color(),model.opacity(),data.camera(name,model.xyz.device))
        loss=(out['rgb']-target).abs().mean()
        opt.zero_grad();loss.backward();opt.step();history.append(float(loss.detach()))
    with torch.no_grad():
        errors=[]; coverages=[]
        for name in data.train_cameras:
            out=render(model.xyz,model.covariance(),model.color(),model.opacity(),data.camera(name,model.xyz.device))
            errors.append(float((out['rgb']-data.image(name,frame).to(model.xyz.device)).abs().mean()))
            coverages.append(float((out['alpha']>.1).float().mean()))
    return {'train_l1':float(np.mean(errors)),'coverage_gt_0_1':float(np.mean(coverages)),
            'loss_history':history,'geometry_verified':False,
            'passes_rgb_gate':float(np.mean(errors))<=cfg['endpoint_max_l1'],
            'note':'RGB gate is necessary, not proof of accurate depth or material identity.'}
