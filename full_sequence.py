"""Continuous single-reference ODE training over a configurable complete sequence.

Uses every training time, fixed Gaussian identity, differentiable Jacobian covariance,
view/time appearance, multi-frame reference refinement, deterministic training-only plateau monitor, resumable checkpoints.
"""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from data import N3DV,normalized_time
from sequence_model import load_sequence, preparation_settings
from transport import flow
from evaluation import image_metrics,save_image,approximation_error
from stopping import make_stopper,stopping_config


def dump(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def save(path,value):
    tmp=Path(str(path)+'.tmp');torch.save(value,tmp);tmp.replace(path)


def rng_state(device):
    return dict(python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if device.type == 'cuda' and state.get('cuda_rng') is not None:
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)


def main(a):
    cfg = json.loads(Path(a.config).read_text())
    prep = preparation_settings(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists() and not a.resume:
        raise FileExistsError('Use --resume to continue an existing run')
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    start = time.perf_counter()
    device = torch.device(cfg['device'])
    data = N3DV(cfg)
    resume = torch.load(a.resume, map_location=device, weights_only=False) if a.resume else None
    if resume is not None:
        if resume['config'] != cfg:
            raise ValueError('Resume requires the same configuration; new architecture needs a new run')
        source = resume
    else:
        source = torch.load(cfg['reference_checkpoint'], map_location=device, weights_only=False)
        if source['config'].get('frame_start', 10) != cfg['frame_start']:
            raise ValueError('Reference keyframe does not match time origin')
    model = load_sequence(cfg, source, data.extent)
    elapsed_before = resume['elapsed_seconds'] if resume is not None else 0.
    sc = stopping_config(cfg)
    monitor_frames, monitor_cameras = cfg['monitor_frames'], cfg['monitor_cameras']
    if (not monitor_frames or not monitor_cameras or
            not set(monitor_frames) <= set(cfg['train_frames']) or
            not set(monitor_cameras) <= set(data.train_cameras)):
        raise ValueError('Monitor must contain training frames/cameras only')
    dump(out/'config.json', cfg)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    @torch.no_grad()
    def evaluate_monitor(step, phase='motion'):
        values = []
        for frame in monitor_frames:
            t = normalized_time(frame, cfg)
            x, c = model.transport(t, False)
            for cam in monitor_cameras:
                pred = model.render(x, c, t, data.camera(cam, device))['rgb']
                target = data.image(cam, frame).to(device)
                values.append(image_metrics(pred, target))
                if cam == monitor_cameras[0] and frame in cfg['preview_frames']:
                    save_image(torch.cat([target, pred], 1), out/f'{phase}_{step:06d}_{cam}_{frame:04d}.png')
        return {k: float(np.mean([row[k] for row in values])) for k in ['l1', 'psnr', 'ssim_uniform7']}

    def snapshot(phase, step, opt, **extra):
        return dict(model.checkpoint(), config=cfg, phase=phase, step=step,
                    optimizer=opt.state_dict(), elapsed_seconds=elapsed_before+time.perf_counter()-start,
                    **rng_state(device), **extra)

    def rgb_loss(frame):
        t = normalized_time(frame, cfg)
        x, c = model.transport(t)
        cams = random.sample(data.train_cameras, cfg['views_per_step'])
        rgb = torch.stack([(model.render(x, c, t, data.camera(cam, device))['rgb']-
                            data.image(cam, frame).to(device)).abs().mean() for cam in cams]).mean()
        return rgb, t

    def backward(loss, opt):
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        opt.zero_grad()
        loss.backward()
        parameters = [p for group in opt.param_groups for p in group['params'] if p.grad is not None]
        if any(not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('Nonfinite training gradient')
        grad = torch.nn.utils.clip_grad_norm_(model.v.parameters(), cfg['gradient_clip'])
        if not torch.isfinite(grad):
            raise FloatingPointError('Nonfinite velocity gradient norm')
        return grad

    if resume is None:
        # Gate the supplied static reference before any multi-frame optimization.
        gate = []
        with torch.no_grad():
            for cam in data.train_cameras:
                pred = model.render(model.g.xyz, model.g.covariance(), 0., data.camera(cam, device))['rgb']
                gate.append(image_metrics(pred, data.image(cam, cfg['frame_start']).to(device)))
        initial_psnr = float(np.mean([r['psnr'] for r in gate]))
        dump(out/'reference_gate.json', dict(mean_train_psnr=initial_psnr,
                                             required=cfg['reference_min_psnr'], per_camera=gate))
        if initial_psnr < cfg['reference_min_psnr']:
            raise RuntimeError('Reference below required PSNR; do not start trajectory training')

    # A preparation checkpoint contains the changing geometry, optimizer, splitting
    # scores and ancestry. A motion checkpoint is created only after IDs are frozen.
    phase = resume.get('phase', 'motion') if resume is not None else 'preparation'
    if phase not in ('preparation', 'motion'):
        raise ValueError('Unknown checkpoint phase')
    if phase == 'preparation' and prep.get('steps', 0):
        model.freeze_geometry(False)
        opt = model.optimizer(preparation=True)
        scores = model.g.xyz.new_zeros(len(model.g.xyz))
        lineage, history, monitor, begin = [], [], [], 0
        if resume is not None:
            begin = resume['step']
            opt.load_state_dict(resume['optimizer'])
            scores.copy_(resume['scores'])
            lineage, history, monitor = resume['lineage'], resume['history'], resume['monitor']
            restore_rng(resume, device)
        else:
            monitor.append(dict(step=0, **evaluate_monitor(0, 'preparation')))

        def save_preparation(step):
            save(out/'preparation_latest.pt', snapshot('preparation', step, opt,
                 scores=scores, lineage=lineage, history=history, monitor=monitor))
            dump(out/'preparation.json', dict(step=step, gaussians=len(model.g.xyz),
                 frames=prep['frames'], lineage=lineage, monitor=monitor, geometry_frozen=False))
            data.save_audit(out/'preparation_access.json')

        if resume is None:
            save_preparation(0)
        for i in range(begin, prep['steps']):
            step = i+1
            frame = prep['frames'][i % len(prep['frames'])]
            rgb, _ = rgb_loss(frame)
            appearance_reg = model.appearance.regularization(rgb)
            backward(rgb+appearance_reg, opt)
            scores.add_(model.g.xyz.grad.detach().norm(dim=-1))
            opt.step()
            history.append(dict(step=step, frame=frame, rgb_l1=float(rgb.detach()),
                                appearance_regularization=float(appearance_reg.detach())))
            if step % prep['densify_interval'] == 0:
                available = prep['max_gaussians']-len(model.g.xyz)
                if step <= prep['densify_until'] and available > 0:
                    fraction = min(prep['split_fraction'], available/len(model.g.xyz))
                    event = model.split(scores, fraction)
                    lineage.append(dict(step=step, **event))
                    # Parameter shapes changed: reset Adam as in static keyframe splitting.
                    opt = model.optimizer(preparation=True)
                    scores = model.g.xyz.new_zeros(len(model.g.xyz))
                else:
                    scores.zero_()
            if step % cfg['log_interval'] == 0 or step == 1:
                print(json.dumps(dict(phase='preparation', **history[-1], gaussians=len(model.g.xyz))), flush=True)
                dump(out/'status.json', dict(status='preparing_geometry', step=step, max_steps=prep['steps']))
            if step % sc['eval_interval'] == 0 or step == prep['steps']:
                monitor.append(dict(step=step, **evaluate_monitor(step, 'preparation')))
            if step % cfg['checkpoint_interval'] == 0 or step == prep['steps']:
                save_preparation(step)
        model.freeze_geometry()
        dump(out/'preparation.json', dict(step=prep['steps'], gaussians=len(model.g.xyz),
             frames=prep['frames'], lineage=lineage, monitor=monitor, geometry_frozen=True,
             note='Joint multi-frame RGB refinement and splitting; no material correspondence guarantee.'))

    model.freeze_geometry()
    opt = model.optimizer()
    stopper = make_stopper(cfg)
    history, monitor, initial_step = [], [], 0
    reason = 'max_steps'
    if resume is not None and phase == 'motion':
        opt.load_state_dict(resume['optimizer'])
        initial_step = resume['step']
        stopper.__dict__.update(resume['stopper'])
        history, monitor = resume['history'], resume['monitor']
        restore_rng(resume, device)

    def motion_snapshot(step):
        return snapshot('motion', step, opt, stopper=stopper.__dict__.copy(), history=history, monitor=monitor)

    if resume is None or phase == 'preparation':
        m = evaluate_monitor(0)
        stopper.update(m['l1'], 0)
        monitor.append(dict(step=0, **m))
        save(out/'best.pt', motion_snapshot(0))
        save(out/'latest.pt', motion_snapshot(0))
        dump(out/'frozen_geometry.json', dict(gaussians=len(model.g.ids), ids=model.g.ids.cpu().tolist(),
                                             parent_ids=model.g.parent_ids.cpu().tolist()))
        print(json.dumps(dict(phase='motion', step=0, monitor=m)), flush=True)
    elif not (out/'best.pt').exists():
        if stopper.best_step != initial_step:
            raise FileNotFoundError('Resume needs best.pt from the original run as well as latest.pt')
        save(out/'best.pt', motion_snapshot(initial_step))

    dump(out/'status.json', dict(status='training', step=initial_step, gaussians=len(model.g.xyz)))
    frames, last_step = cfg['train_frames'], initial_step
    for i in range(initial_step, cfg['trajectory_steps']):
        step = i+1
        last_step = step
        frame = frames[i % len(frames)]
        tick = time.perf_counter()
        rgb, t = rgb_loss(frame)
        ids = torch.linspace(0, len(model.g.xyz)-1, min(256, len(model.g.xyz)), device=device).long()
        xx = model.g.xyz[ids]
        tm, tp = max(0, t-.001), min(1, t+.001)
        # Preserve the existing fixed-reference acceleration proxy in this change.
        vel, J = model.v.value_and_jacobian(xx, t)
        acc = (model.v(xx, tp)-model.v(xx, tm))/(tp-tm)+(J@vel[..., None]).squeeze(-1)
        reg = acc.square().mean()
        appearance_reg = model.appearance.regularization(rgb)
        grad = backward(rgb+cfg['acceleration_weight']*reg+appearance_reg, opt)
        opt.step()
        entry = dict(step=step, frame=frame, rgb_l1=float(rgb.detach()), acceleration=float(reg.detach()),
                     appearance_regularization=float(appearance_reg.detach()),
                     velocity_grad_norm=float(grad), seconds=time.perf_counter()-tick)
        history.append(entry)
        if step % cfg['log_interval'] == 0 or step == 1:
            print(json.dumps(entry), flush=True)
            dump(out/'status.json', dict(status='training', step=step, max_steps=cfg['trajectory_steps'],
                 last_loss=entry, elapsed_seconds=elapsed_before+time.perf_counter()-start))
        if step % sc['eval_interval'] == 0 or step == cfg['trajectory_steps']:
            m = evaluate_monitor(step)
            improved, stop = stopper.update(m['l1'], step)
            monitor.append(dict(step=step, **m))
            if improved:
                save(out/'best.pt', motion_snapshot(step))
            save(out/'latest.pt', motion_snapshot(step))
            dump(out/'monitor.json', monitor)
            dump(out/'early_stopping.json', dict(reason='running', monitor_definition='fixed training-only frame/camera subset', **stopper.report()))
            print(json.dumps(dict(step=step, monitor=m, stopper=stopper.report())), flush=True)
            if sc['enabled'] and stop:
                reason = 'plateau'
                break
        elif step % cfg['checkpoint_interval'] == 0:
            save(out/'latest.pt', motion_snapshot(step))
        if step % cfg['checkpoint_interval'] == 0:
            dump(out/'losses.json', history)
            data.save_audit(out/'training_access.json')
    data.save_audit(out/'training_access.json')
    dump(out/'losses.json', history)
    best = torch.load(out/'best.pt', map_location=device, weights_only=False)
    model = load_sequence(cfg, best, data.extent)
    dump(out/'early_stopping.json', dict(reason=reason, executed_steps=last_step, **stopper.report()))
    dump(out/'status.json', dict(status='evaluating', step=last_step, selected_step=best['step']))
    if cfg.get('image_cache'):
        import subprocess, sys
        subprocess.run([sys.executable, 'scripts/cache_video_frames.py', '--config', str(out/'config.json'),
                        '--splits', 'view', 'time', 'joint'], check=True)
    evaluate_all(cfg, data, model, out)
    elapsed = elapsed_before+time.perf_counter()-start
    dump(out/'status.json', dict(status='completed', stop_reason=reason, executed_steps=last_step,
         selected_step=best['step'], elapsed_seconds=elapsed, allocated_gpu_hours=elapsed/3600 if device.type == 'cuda' else 0.,
         torch_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
         material_correspondence_proven=False))


@torch.no_grad()
def evaluate_all(cfg,data,model,out):
    g,v=model.g,model.v
    import cv2
    device=g.xyz.device;ids=torch.linspace(0,len(g.xyz)-1,min(128,len(g.xyz)),device=device).long();paths=[];rows=[]
    h=data.cameras['cam00']['height'];w=data.cameras['cam00']['width'];video=cv2.VideoWriter(str(out/'cam00_target_render.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),cfg.get('frame_rate',30), (w*2,h))
    selected={int(f) for f in cfg['preview_frames']};evaluation_start=time.perf_counter();static_rows=[]
    stationary=model.render(g.xyz,g.covariance(),0.,data.camera('cam00',device))['rgb']
    for f in range(cfg['frame_start'],cfg['frame_end']+1):
        t=normalized_time(f,cfg);x,c=model.transport(t,False);paths.append(x[ids].cpu().numpy())
        # cam00: view split at training times; joint split at held-out times.
        split='view' if f in cfg['train_frames'] else 'joint';pred=model.render(x,c,t,data.camera('cam00',device))['rgb'];target=data.image('cam00',f,split).to(device)
        rows.append(dict(split=split,frame=f,camera='cam00',**image_metrics(pred,target)));static_rows.append(dict(split=split,frame=f,**image_metrics(stationary,target)));pair=torch.cat([target,pred],1)
        video.write(cv2.cvtColor((pair.clamp(0,1).cpu().numpy()*255).astype('uint8'),cv2.COLOR_RGB2BGR))
        if f in selected:save_image(pair,out/f'final_cam00_{f:04d}.png')
        if f in cfg['heldout_frames']:
            for cam in data.train_cameras:
                pred=model.render(x,c,t,data.camera(cam,device))['rgb'];target=data.image(cam,f,'time').to(device)
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
