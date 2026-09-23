"""Stage A endpoints, optional B FM, C interval RGB, evaluation and artifacts."""
import argparse
import json
import time
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from data import N3DV,normalized_time
from geometry import Gaussians,triangulate,optimize_endpoint,initial_scales
from transport import Velocity,FlowMatching,flow,jacobian
from covariance_renderer import render
from evaluation import image_metrics,save_image,visualize,approximation_error
from tracks import build_tracks,track_loss,training_track_metrics


class Appearance(nn.Module):
    def __init__(self,n,rank,device):
        super().__init__();self.rank=rank
        if rank not in (0,1,2):raise ValueError('Only fixed or low rank time appearance allowed')
        self.coeff=nn.Parameter(torch.zeros(n,4,rank,device=device))
    def forward(self,g,t):
        if not self.rank:return g.color(),g.opacity()
        basis=g.xyz.new_tensor([t,t*(1-t)])[:self.rank]
        delta=.1*torch.tanh(self.coeff@basis)
        return (g.color_logits+delta[:,:3]).sigmoid(),(g.opacity_logits+delta[:,3]).sigmoid()


def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2))


def main(args):
    start=time.perf_counter();cfg=json.loads(Path(args.config).read_text())
    if args.data:cfg['data_root']=str(Path(args.data).expanduser().resolve())
    if args.variant:cfg['variant']=args.variant
    if args.steps is not None:cfg['trajectory_steps']=args.steps
    if args.device:cfg['device']=args.device
    if args.renderer:cfg['renderer']=args.renderer
    if args.appearance_rank is not None:cfg['appearance_time_rank']=args.appearance_rank
    if args.track_masks:cfg['track_weight']=args.track_weight
    out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Refuse overwriting {out}')
    out.mkdir(parents=True,exist_ok=True);dump(out/'config.json',cfg)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    try: execute(args,cfg,out,start)
    except Exception as exc:
        dump(out/'failure.json',{'status':'failed','type':type(exc).__name__,'error':str(exc),'seconds':time.perf_counter()-start})
        raise


def execute(args,cfg,out,start):
    data=N3DV(cfg);device=cfg['device']; timings={}
    dump(out/'inventory.json',data.inventory)
    if args.inspect:
        dump(out/'status.json',{'status':'inspected','training_cameras':data.train_cameras,'split_sizes':{s:len(data.keys(s)) for s in ('train','view','time','joint')}});return
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    def sync():
        if device.startswith('cuda'):torch.cuda.synchronize()
    endpoints=[];reports=[]
    # Higher initialization resolution is independent of reconstruction resolution.
    geo_cfg=dict(cfg,image_width=cfg.get('triangulation_width',640)); geo=N3DV(geo_cfg)
    for frame,key in [(10,'gaussians_start'),(30,'gaussians_end')]:
        tick=time.perf_counter();xyz,rgb,report=triangulate(geo,frame,cfg[key]);sync()
        timings[f'triangulate_{frame}']=time.perf_counter()-tick
        g=Gaussians(xyz.to(device),rgb.to(device),initial_scales(xyz,data.extent,cfg)).to(device)
        tick=time.perf_counter();report.update(optimize_endpoint(data,frame,g,cfg));sync()
        timings[f'endpoint_{frame}']=time.perf_counter()-tick
        endpoints.append(g);reports.append(report)
        with torch.no_grad():
            name=data.train_cameras[0];r=render(g.xyz,g.covariance(),g.color(),g.opacity(),data.camera(name,device))
            save_image(r['rgb'],out/f'endpoint_{frame}_rgb.png');save_image(data.image(name,frame),out/f'endpoint_{frame}_target.png')
            save_image(r['alpha'][...,None].expand(-1,-1,3),out/f'endpoint_{frame}_alpha.png')
            np.save(out/f'endpoint_{frame}_depth.npy',r['depth'].cpu().numpy())
        torch.save(g.state_dict(),out/f'endpoint_{frame}.pt')
    data.save_audit(out/'training_access_before_flow.json');geo.save_audit(out/'triangulation_access.json')
    dump(out/'endpoints.json',reports)
    inaccurate=not all(r['passes_rgb_gate'] for r in reports)
    if inaccurate and not cfg['allow_inaccurate_endpoints']:
        raise RuntimeError('Endpoint RGB gate failed; inspect saved RGB/depth/alpha before interval training')
    g,e=endpoints
    # Geometry is fixed after A, appearance remains trainable but time independent by default.
    g.xyz.requires_grad_(False);g.log_scale.requires_grad_(False);g.lower.requires_grad_(False)
    x0=g.xyz.detach();L0=g.factor().detach()
    if cfg['variant']=='canonical_4dgs':
        from baseline import run_baseline
        run_baseline(data,g,e,cfg,out,timings,start);return
    velocity=Velocity(cfg['velocity_width']).to(device)
    opt=torch.optim.Adam(velocity.parameters(),lr=cfg['velocity_lr'])
    tick=time.perf_counter();fm=None
    if cfg['variant'] in ('fm_rgb','fm_persistent'):
        fm=FlowMatching(x0,g.covariance(),e.xyz,e.covariance(),cfg)
        for _ in range(cfg['fm_steps']):
            loss=fm.loss(velocity);opt.zero_grad();loss.backward();opt.step()
        dump(out/'sinkhorn.json',fm.diagnostics)
    sync();timings['coupling_and_fm']=time.perf_counter()-tick
    tick=time.perf_counter()
    observations=build_tracks(data,g,args.track_masks) if cfg['track_weight']>0 else []
    if cfg['track_weight']>0 and not observations: raise RuntimeError('No reliable multiview tracks; do not claim track supervision')
    dump(out/'tracks.json',observations);timings['tracking']=time.perf_counter()-tick
    appearance=Appearance(len(x0),cfg['appearance_time_rank'],device)
    opt=torch.optim.Adam([{'params':velocity.parameters(),'lr':cfg['velocity_lr']},
        {'params':[g.color_logits,g.opacity_logits,*appearance.parameters()],'lr':cfg['appearance_lr']}])
    history=[];tick=time.perf_counter()
    for i in range(cfg['trajectory_steps']):
        # Deterministic coverage, all internal training times first; multi-camera RGB.
        frames=[f for f in cfg['train_frames'] if f not in (10,30)]+[10,30]
        frame=frames[i%len(frames)];t=normalized_time(frame)
        x,cov=flow(velocity,x0,L0,t,cfg['ode_method'],cfg['ode_step'])
        color,opacity=appearance(g,t)
        cameras=random.sample(data.train_cameras,min(cfg['views_per_step'],len(data.train_cameras)))
        rgb=torch.stack([(render(x,cov,color,opacity,data.camera(n,device))['rgb']-data.image(n,frame).to(device)).abs().mean() for n in cameras]).mean()
        # Material acceleration: dv/dt + (dv/dx)v, weak only.
        dt=1e-3;tm=max(0,t-dt);tp=min(1,t+dt)
        acc=(velocity(x,tp)-velocity(x,tm))/(tp-tm)+(jacobian(velocity,x,t)@velocity(x,t)[...,None]).squeeze(-1)
        reg=acc.square().mean(); tl=track_loss(x,frame,observations,data)
        loss=rgb+cfg['acceleration_weight']*reg+cfg['track_weight']*tl
        terminal=x.sum()*0
        if cfg['endpoint_weight']>0:
            end,_=flow(velocity,x0,L0,1,cfg['ode_method'],cfg['ode_step'])
            d=torch.cdist(end,e.xyz.detach()).square();terminal=d.min(1).values.mean()+d.min(0).values.mean()
            loss=loss+cfg['endpoint_weight']*terminal
        fml=x.sum()*0
        if cfg['variant']=='fm_persistent':fml=fm.loss(velocity);loss=loss+cfg['continued_fm_weight']*fml
        if appearance.rank:loss=loss+cfg['appearance_time_weight']*appearance.coeff.square().mean()
        opt.zero_grad();loss.backward()
        grad=sum(float(p.grad.norm()) for p in velocity.parameters() if p.grad is not None)
        if not np.isfinite(grad): raise FloatingPointError('Nonfinite velocity gradient')
        opt.step()
        history.append({'step':i,'frame':frame,'rgb_l1':float(rgb.detach()),'acceleration':float(reg.detach()),'track':float(tl.detach()),'fm':float(fml.detach()),'terminal_chamfer':float(terminal.detach()),'velocity_grad_norm':grad})
    sync();timings['interval_training']=time.perf_counter()-tick
    data.save_audit(out/'training_access.json')
    dump(out/'losses.json',history)
    torch.save({'reference':g.state_dict(),'endpoint':e.state_dict(),'velocity':velocity.state_dict(),'appearance':appearance.state_dict(),'config':cfg},out/'checkpoint.pt')
    tick=time.perf_counter(); rows=[];positions=[]
    with torch.no_grad():
        for frame in range(10,31):
            t=normalized_time(frame);x,cov=flow(velocity,x0,L0,t,cfg['ode_method'],cfg['ode_step']);positions.append(x.cpu().numpy())
            color,opacity=appearance(g,t)
            for split in ('view','time','joint'):
                for name,f in data.keys(split):
                    if f!=frame:continue
                    pred=render(x,cov,color,opacity,data.camera(name,device))['rgb']
                    target=data.image(name,frame,split).to(device)
                    rows.append({'split':split,'camera':name,'frame':frame,**image_metrics(pred,target)})
                    if name in ('cam00',data.train_cameras[0]) and frame in (10,20,30):save_image(pred,out/f'{split}_{name}_{frame}.png')
        end=torch.tensor(positions[-1],device=device);d=torch.cdist(end,e.xyz).square()
        distribution={'endpoint_symmetric_chamfer_squared':float(d.min(1).values.mean()+d.min(0).values.mean()),'note':'Distribution geometry only; not material correspondence.'}
    summary={s:{k:float(np.mean([r[k] for r in rows if r['split']==s])) for k in ('psnr','ssim_uniform7','l1')} for s in ('view','time','joint')}
    approximation=approximation_error(velocity,x0,L0,cfg)
    visualize(positions,g.ids.cpu().numpy(),out,{n:data.cameras[n] for n in ['cam00',data.train_cameras[0]]})
    sync();timings['evaluation_and_visualization']=time.perf_counter()-tick
    elapsed=time.perf_counter()-start;rate=cfg.get('gpu_hourly_usd')
    dump(out/'metrics.json',{'image':summary,'per_image':rows,'distribution':distribution,'trajectory_3d_error':None,
        'track_training_observations':len(observations),'track_consistency':training_track_metrics(positions,observations,data),'gaussian_approximation':approximation,
        'smoke_only':cfg['smoke_only'],'endpoint_rgb_gate_failed':inaccurate,'endpoint_geometry_verified':False,
        'timing_seconds':timings,'total_seconds_including_preprocessing':elapsed,'allocated_gpu_hours':elapsed/3600,
        'estimated_usd':None if rate is None else elapsed/3600*rate,
        'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
        'material_correspondence_proven':False})
    data.save_audit(out/'all_access.json');dump(out/'status.json',{'status':'completed','smoke_only':cfg['smoke_only']})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/smoke.json');p.add_argument('--data');p.add_argument('--out',required=True)
    p.add_argument('--variant',choices=['rgb','fm_rgb','fm_persistent','canonical_4dgs']);p.add_argument('--device');p.add_argument('--steps',type=int)
    p.add_argument('--renderer',choices=['reference','cuda']);p.add_argument('--inspect',action='store_true');p.add_argument('--track-masks');p.add_argument('--track-weight',type=float,default=.01);p.add_argument('--appearance-rank',type=int)
    main(p.parse_args())
