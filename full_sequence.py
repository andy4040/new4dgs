"""Continuous single-reference ODE training over a configurable complete sequence.

Uses every training time, fixed Gaussian identity, differentiable Jacobian covariance,
fixed-time appearance, deterministic training-only plateau monitor, resumable checkpoints.
"""
import argparse,json,time,random,copy,os
from pathlib import Path
import numpy as np
import torch
from data import N3DV,normalized_time
from geometry import Gaussians
from transport import SceneVelocity,chunked_flow,jacobian,flow
from covariance_renderer import render,project
from evaluation import image_metrics,save_image,approximation_error
from stopping import make_stopper,stopping_config


def dump(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def save(path,value):
    tmp=Path(str(path)+'.tmp');torch.save(value,tmp);tmp.replace(path)


def load_reference(path,device):
    saved=torch.load(path,map_location=device,weights_only=False);s=saved['reference']
    g=Gaussians(s['xyz'],s['color_logits'].sigmoid(),1.).to(device);g.load_state_dict(s)
    return g,saved


def main(a):
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and not a.resume:raise FileExistsError('Use --resume to continue an existing run')
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);start=time.perf_counter()
    device=cfg['device'];data=N3DV(cfg);g,reference=load_reference(cfg['reference_checkpoint'],device)
    if reference['config'].get('frame_start',10)!=cfg['frame_start']:raise ValueError('Reference keyframe does not match time origin')
    g.xyz.requires_grad_(False);g.log_scale.requires_grad_(False);g.lower.requires_grad_(False)
    x0=g.xyz.detach();L0=g.factor().detach();center=x0.median(0).values
    vc=cfg['velocity'];v=SceneVelocity(center,data.extent,**vc).to(device)
    opt=torch.optim.Adam([{'params':v.parameters(),'lr':cfg['velocity_lr']},
        {'params':[g.color_logits,g.opacity_logits],'lr':cfg['appearance_lr']}])
    sc=stopping_config(cfg);stopper=make_stopper(cfg);history=[];monitor=[];initial_step=0;elapsed_before=0.;reason='max_steps'
    monitor_frames=cfg['monitor_frames'];monitor_cameras=cfg['monitor_cameras']
    assert set(monitor_frames)<=set(cfg['train_frames']) and set(monitor_cameras)<=set(data.train_cameras)
    dump(out/'config.json',cfg)
    if a.resume:
        state=torch.load(a.resume,map_location=device,weights_only=False)
        if state['config']!=cfg:raise ValueError('Resume requires the same configuration')
        v.load_state_dict(state['velocity']);g.load_state_dict(state['reference']);opt.load_state_dict(state['optimizer'])
        initial_step=state['step'];elapsed_before=state['elapsed_seconds'];stopper.__dict__.update(state['stopper'])
        history=state['history'];monitor=state['monitor'];random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    torch.cuda.reset_peak_memory_stats()
    def transported(t,grad=True):return chunked_flow(v,x0,L0,t,cfg['ode_method'],cfg['ode_step'],cfg['flow_chunk_size'],checkpoint_grad=grad)
    @torch.no_grad()
    def evaluate_monitor(step):
        values=[]
        for f in monitor_frames:
            t=normalized_time(f,cfg);x,c=transported(t,False)
            for cam in monitor_cameras:
                pred=render(x,c,g.color(),g.opacity(),data.camera(cam,device))['rgb'];target=data.image(cam,f).to(device)
                values.append(image_metrics(pred,target))
                if cam==monitor_cameras[0] and f in cfg['preview_frames']:save_image(torch.cat([target,pred],1),out/f'monitor_{step:06d}_{cam}_{f:04d}.png')
        return {k:float(np.mean([row[k] for row in values])) for k in ['l1','psnr','ssim_uniform7']}
    def snapshot(step):
        return dict(config=cfg,reference=g.state_dict(),velocity=v.state_dict(),optimizer=opt.state_dict(),step=step,
                    stopper=stopper.__dict__.copy(),history=history,monitor=monitor,elapsed_seconds=elapsed_before+time.perf_counter()-start,
                    python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
    if not a.resume:
        # Gate is evaluated across ALL training cameras, before any motion optimization.
        gate=[]
        with torch.no_grad():
            for cam in data.train_cameras:
                pred=render(g.xyz,g.covariance(),g.color(),g.opacity(),data.camera(cam,device))['rgb']
                gate.append(image_metrics(pred,data.image(cam,cfg['frame_start']).to(device)))
        initial_psnr=float(np.mean([r['psnr'] for r in gate]));dump(out/'reference_gate.json',dict(mean_train_psnr=initial_psnr,required=cfg['reference_min_psnr'],per_camera=gate))
        if initial_psnr<cfg['reference_min_psnr']:raise RuntimeError('Reference below required PSNR; do not start trajectory training')
        m=evaluate_monitor(0);stopper.update(m['l1'],0);monitor.append(dict(step=0,**m));save(out/'best.pt',snapshot(0));save(out/'latest.pt',snapshot(0))
        print(json.dumps(dict(step=0,monitor=m,reference_psnr=initial_psnr)),flush=True)
    dump(out/'status.json',dict(status='training',step=initial_step,gaussians=len(x0)))
    frames=cfg['train_frames'];last_step=initial_step
    for i in range(initial_step,cfg['trajectory_steps']):
        step=i+1;last_step=step;f=frames[i%len(frames)];t=normalized_time(f,cfg)
        # Recompute at the current velocity parameters from the same frozen reference.
        tick=time.perf_counter();x,c=transported(t)
        cams=random.sample(data.train_cameras,cfg['views_per_step'])
        rgb=torch.stack([(render(x,c,g.color(),g.opacity(),data.camera(cam,device))['rgb']-data.image(cam,f).to(device)).abs().mean() for cam in cams]).mean()
        ids=torch.linspace(0,len(x0)-1,min(256,len(x0)),device=device).long();xx=x0[ids]
        dt=.001;tm=max(0,t-dt);tp=min(1,t+dt)
        # Weak Eulerian acceleration proxy at fixed spatial samples; explicitly reported.
        vel,J=v.value_and_jacobian(xx,t);acc=(v(xx,tp)-v(xx,tm))/(tp-tm)+(J@vel[...,None]).squeeze(-1)
        reg=acc.square().mean();loss=rgb+cfg['acceleration_weight']*reg
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {step}')
        opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(v.parameters(),cfg['gradient_clip'])
        if not torch.isfinite(grad):raise FloatingPointError(f'Nonfinite gradient at {step}')
        opt.step();entry=dict(step=step,frame=f,rgb_l1=float(rgb.detach()),acceleration=float(reg.detach()),velocity_grad_norm=float(grad),seconds=time.perf_counter()-tick)
        history.append(entry)
        if step%cfg['log_interval']==0 or step==1:
            print(json.dumps(entry),flush=True);dump(out/'status.json',dict(status='training',step=step,max_steps=cfg['trajectory_steps'],last_loss=entry,elapsed_seconds=elapsed_before+time.perf_counter()-start))
        if step%cfg['checkpoint_interval']==0:save(out/'latest.pt',snapshot(step));dump(out/'losses.json',history);data.save_audit(out/'training_access.json')
        if step%sc['eval_interval']==0 or step==cfg['trajectory_steps']:
            m=evaluate_monitor(step);improved,stop=stopper.update(m['l1'],step);monitor.append(dict(step=step,**m))
            if improved:save(out/'best.pt',snapshot(step))
            save(out/'latest.pt',snapshot(step));dump(out/'monitor.json',monitor);dump(out/'early_stopping.json',dict(reason='running',monitor_definition='fixed training-only frame/camera subset',**stopper.report()))
            print(json.dumps(dict(step=step,monitor=m,stopper=stopper.report())),flush=True)
            if sc['enabled'] and stop:reason='plateau';break
    data.save_audit(out/'training_access.json');dump(out/'losses.json',history)
    best=torch.load(out/'best.pt',map_location=device,weights_only=False);v.load_state_dict(best['velocity']);g.load_state_dict(best['reference'])
    dump(out/'early_stopping.json',dict(reason=reason,executed_steps=last_step,**stopper.report()))
    dump(out/'status.json',dict(status='evaluating',step=last_step,selected_step=best['step']))
    if cfg.get('image_cache'):
        import subprocess,sys
        subprocess.run([sys.executable,'scripts/cache_video_frames.py','--config',str(out/'config.json'),'--splits','view','time','joint'],check=True)
    evaluate_all(cfg,data,g,v,out)
    elapsed=elapsed_before+time.perf_counter()-start
    dump(out/'status.json',dict(status='completed',stop_reason=reason,executed_steps=last_step,selected_step=best['step'],elapsed_seconds=elapsed,allocated_gpu_hours=elapsed/3600,torch_peak_bytes=torch.cuda.max_memory_allocated(),material_correspondence_proven=False))


@torch.no_grad()
def evaluate_all(cfg,data,g,v,out):
    import cv2
    device=g.xyz.device;ids=torch.linspace(0,len(g.xyz)-1,min(128,len(g.xyz)),device=device).long();paths=[];rows=[]
    h=data.cameras['cam00']['height'];w=data.cameras['cam00']['width'];video=cv2.VideoWriter(str(out/'cam00_target_render.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),cfg.get('frame_rate',30), (w*2,h))
    selected={int(f) for f in cfg['preview_frames']};evaluation_start=time.perf_counter();static_rows=[]
    stationary=render(g.xyz,g.covariance(),g.color(),g.opacity(),data.camera('cam00',device))['rgb']
    for f in range(cfg['frame_start'],cfg['frame_end']+1):
        t=normalized_time(f,cfg);x,c=chunked_flow(v,g.xyz,g.factor(),t,cfg['ode_method'],cfg['ode_step'],cfg['flow_chunk_size'],False);paths.append(x[ids].cpu().numpy())
        # cam00: view split at training times; joint split at held-out times.
        split='view' if f in cfg['train_frames'] else 'joint';pred=render(x,c,g.color(),g.opacity(),data.camera('cam00',device))['rgb'];target=data.image('cam00',f,split).to(device)
        rows.append(dict(split=split,frame=f,camera='cam00',**image_metrics(pred,target)));static_rows.append(dict(split=split,frame=f,**image_metrics(stationary,target)));pair=torch.cat([target,pred],1)
        video.write(cv2.cvtColor((pair.clamp(0,1).cpu().numpy()*255).astype('uint8'),cv2.COLOR_RGB2BGR))
        if f in selected:save_image(pair,out/f'final_cam00_{f:04d}.png')
        if f in cfg['heldout_frames']:
            for cam in data.train_cameras:
                pred=render(x,c,g.color(),g.opacity(),data.camera(cam,device))['rgb'];target=data.image(cam,f,'time').to(device)
                rows.append(dict(split='time',frame=f,camera=cam,**image_metrics(pred,target)))
        if f%30==0:print(json.dumps({'evaluation_frame':f}),flush=True)
    video.release();np.savez_compressed(out/'trajectories_subset.npz',positions=np.array(paths),ids=g.ids[ids].cpu().numpy())
    from evaluation import visualize
    visualize(paths,g.ids[ids].cpu().numpy(),out,{'cam00':data.cameras['cam00']})
    # Step-size audit on a fixed ID subset, separate from trajectory ground truth.
    errors=[]
    for t in [.25,.5,.75,1.]:
        x,c=flow(v,g.xyz[ids],g.factor()[ids],t,cfg['ode_method'],cfg['ode_step']);y,d=flow(v,g.xyz[ids],g.factor()[ids],t,cfg['ode_method'],cfg['ode_step']/2)
        errors.append(dict(t=t,mean_position_difference=float((x-y).norm(dim=-1).mean()),relative_covariance_difference=float((c-d).flatten(1).norm(dim=-1).mean()/d.flatten(1).norm(dim=-1).mean().clamp_min(1e-12))))
    summary={s:{k:float(np.mean([r[k] for r in rows if r['split']==s])) for k in ['psnr','l1','ssim_uniform7']} for s in ['view','time','joint']}
    dump(out/'evaluation.json',dict(image=summary,stationary_reference_cam00=static_rows,per_image=rows,integrator_step_audit=errors,gaussian_approximation=approximation_error(v,g.xyz,g.factor(),cfg),trajectory_3d_error=None,material_correspondence_proven=False,evaluation_seconds=time.perf_counter()-evaluation_start))
    data.save_audit(out/'all_access.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--resume');a=p.parse_args()
    try:main(a)
    except Exception as exc:
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True);dump(out/'failure.json',dict(status='failed',type=type(exc).__name__,message=str(exc)));raise
