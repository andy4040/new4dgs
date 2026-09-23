"""Canonical hustvl 4DGS deformation with the SAME strict data and reference renderer.
Controlled fixed-count/SH0 baseline, not a claim to reproduce published full recipe.
Upstream source is imported from a pinned sibling checkout, never modified.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from data import normalized_time
from covariance_renderer import render
from evaluation import image_metrics,visualize


def initialize(reference,cfg,extent):
    path=Path(cfg.get('upstream_path','../4DGaussians')).resolve()
    if not (path/'scene/gaussian_model.py').exists():raise FileNotFoundError('Missing pinned 4DGaussians checkout')
    sys.path.insert(0,str(path))
    from arguments import ModelHiddenParams,OptimizationParams
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud
    from utils.sh_utils import RGB2SH
    parser=argparse.ArgumentParser();hp=ModelHiddenParams(parser);op=OptimizationParams(parser)
    ns=parser.parse_args([]);hidden=hp.extract(ns);opt=op.extract(ns)
    if cfg['smoke_only']:
        hidden.multires=[1,2];hidden.kplanes_config=dict(hidden.kplanes_config,resolution=[16,16,16,8])
    hidden.no_do=True;hidden.no_dshs=True
    model=GaussianModel(0,hidden)
    xyz=reference.xyz.detach().cpu().numpy();rgb=reference.color().detach().cpu().numpy()
    model.create_from_pcd(BasicPointCloud(xyz,rgb,np.zeros_like(xyz)),extent,1.)
    model._deformation.deformation_net.set_aabb(xyz.max(0)+extent*.1,xyz.min(0)-extent*.1)
    model._deformation.cuda()
    cov=reference.covariance().detach().cpu().numpy();vals,vecs=np.linalg.eigh(cov)
    vecs[:,:,0]*=np.linalg.det(vecs)[:,None]
    q=Rotation.from_matrix(vecs).as_quat()[:,[3,0,1,2]] # scipy xyzw -> upstream wxyz
    with torch.no_grad():
        model._scaling.copy_(torch.tensor(np.sqrt(vals).clip(1e-8).copy(),device='cuda').log())
        model._rotation.copy_(torch.tensor(q.copy(),device='cuda'))
        model._opacity.copy_(reference.opacity_logits[:,None])
        model._features_dc.copy_(RGB2SH(reference.color())[:,None])
    model.training_setup(opt)
    return model,hidden


def state(model,t):
    from utils.general_utils import build_scaling_rotation
    sh=model.get_features
    x,s,r,o,c=model._deformation(model._xyz,model._scaling,model._rotation,model._opacity,sh,
                               model._xyz.new_full((len(model._xyz),1),t))
    L=build_scaling_rotation(s.exp(),r)
    cov=L@L.transpose(-1,-2)+torch.eye(3,device=x.device)*1e-8
    color=(.28209479177387814*c[:,0]+.5).clamp(0,1)
    return x,cov,color,o.sigmoid().squeeze(-1)


def run_baseline(data,reference,endpoint,cfg,out,timings,start):
    if cfg['appearance_time_rank'] or cfg['track_weight']:raise ValueError('Baseline first comparison uses fixed appearance, no tracking')
    tick=time.perf_counter();model,hidden=initialize(reference,cfg,data.extent)
    history=[]
    for i in range(cfg['trajectory_steps']):
        model.update_learning_rate(i)
        frames=[f for f in cfg['train_frames'] if f not in (10,30)]+[10,30]
        frame=frames[i%len(frames)];x,cov,color,opacity=state(model,normalized_time(frame))
        names=random.sample(data.train_cameras,min(cfg['views_per_step'],len(data.train_cameras)))
        rgb=torch.stack([(render(x,cov,color,opacity,data.camera(n,'cuda'))['rgb']-data.image(n,frame).cuda()).abs().mean() for n in names]).mean()
        reg=model.compute_regulation(hidden.time_smoothness_weight,hidden.l1_time_planes,hidden.plane_tv_weight)
        loss=rgb+reg;model.optimizer.zero_grad();loss.backward();model.optimizer.step()
        history.append({'step':i,'frame':frame,'rgb_l1':float(rgb.detach()),'regularization':float(reg.detach())})
    torch.cuda.synchronize();timings['baseline_training']=time.perf_counter()-tick
    data.save_audit(out/'training_access.json')
    torch.save(model.capture(),out/'baseline.pt')
    (out/'baseline_settings.json').write_text(json.dumps(vars(hidden),indent=2))
    rows=[];positions=[]
    tick=time.perf_counter()
    with torch.no_grad():
        for frame in range(10,31):
            x,cov,color,opacity=state(model,normalized_time(frame));positions.append(x.cpu().numpy())
            for split in ('view','time','joint'):
                for n,f in data.keys(split):
                    if f!=frame:continue
                    pred=render(x,cov,color,opacity,data.camera(n,'cuda'))['rgb']
                    rows.append({'split':split,'frame':frame,'camera':n,**image_metrics(pred,data.image(n,frame,split).cuda())})
        d=torch.cdist(x,endpoint.xyz).square()
        distribution=float(d.min(0).values.mean()+d.min(1).values.mean())
    visualize(positions,reference.ids.cpu().numpy(),out,{n:data.cameras[n] for n in ['cam00',data.train_cameras[0]]})
    torch.cuda.synchronize();timings['baseline_evaluation']=time.perf_counter()-tick;elapsed=time.perf_counter()-start
    summary={s:{k:float(np.mean([r[k] for r in rows if r['split']==s])) for k in ('psnr','ssim_uniform7','l1')} for s in ('view','time','joint')}
    report={'image':summary,'per_image':rows,'distribution':{'endpoint_symmetric_chamfer_squared':distribution},'trajectory_3d_error':None,
        'timing_seconds':timings,'total_seconds_including_preprocessing':elapsed,'allocated_gpu_hours':elapsed/3600,
        'estimated_usd':None if cfg.get('gpu_hourly_usd') is None else elapsed/3600*cfg['gpu_hourly_usd'],
        'smoke_only':cfg['smoke_only'],'material_correspondence_proven':False,'endpoint_geometry_verified':False,
        'baseline':'pinned upstream canonical 4DGS deformation, fixed count SH0, shared reference renderer',
        'published_recipe_reproduction':False,'paths_are_ode':False,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
    (out/'metrics.json').write_text(json.dumps(report,indent=2));(out/'losses.json').write_text(json.dumps(history,indent=2))
    data.save_audit(out/'all_access.json');(out/'status.json').write_text(json.dumps({'status':'completed','smoke_only':cfg['smoke_only']}))
