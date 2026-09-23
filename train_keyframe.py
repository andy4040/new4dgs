"""Static keyframe reconstruction with target PSNR and deterministic plateau stopping."""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from data import N3DV
from geometry import Gaussians,triangulate,initial_scales,position_lr,split_gaussians
from covariance_renderer import render
from evaluation import image_metrics,save_image
from stopping import make_stopper,stopping_config


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--resume');p.add_argument('--evaluate-heldout',action='store_true')
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time.perf_counter()
    def dump(name,obj):(out/name).write_text(json.dumps(obj,indent=2))
    dump('config.json',cfg);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    device=cfg['device'];frame=10;data=N3DV(cfg)
    if args.resume:
        saved=torch.load(args.resume,map_location=device,weights_only=False);s=saved['reference']
        g=Gaussians(s['xyz'],s['color_logits'].sigmoid(),1.).to(device);g.load_state_dict(s)
        init={'source_checkpoint':str(Path(args.resume).resolve()),'selected':len(g.xyz),'optimizer_reset':True,'steps_are_additional':True}
    else:
        geo=N3DV(dict(cfg,image_width=cfg['triangulation_width']))
        xyz,rgb,init=triangulate(geo,frame,cfg['gaussians_start']);geo.save_audit(out/'triangulation_access.json')
        g=Gaussians(xyz.to(device),rgb.to(device),initial_scales(xyz,data.extent,cfg).to(device)).to(device)
    dump('initialization.json',init);print(json.dumps(init),flush=True)
    targets={n:data.image(n,frame).to(device) for n in data.train_cameras};cameras={n:data.camera(n,device) for n in data.train_cameras}
    def optimizer():
        return torch.optim.Adam([{'params':[g.xyz],'lr':position_lr(cfg,0,data.extent)},
            {'params':[g.log_scale,g.lower],'lr':cfg.get('endpoint_shape_lr',cfg['endpoint_other_lr'])},
            {'params':[g.color_logits,g.opacity_logits],'lr':cfg['endpoint_other_lr']}])
    opt=optimizer();history=[];losses=[];lineage=[];torch.cuda.reset_peak_memory_stats();stopper=make_stopper(cfg);sc=stopping_config(cfg)
    best_psnr=-float('inf');reason='max_steps';target=cfg.get('target_psnr',30.);scores=torch.zeros(len(g.xyz),device=device)
    @torch.no_grad()
    def evaluate(step,save=True):
        nonlocal best_psnr
        rows=[]
        for n in data.train_cameras:
            r=render(g.xyz,g.covariance(),g.color(),g.opacity(),cameras[n])
            rows.append(dict(camera=n,**image_metrics(r['rgb'],targets[n]),coverage=float((r['alpha']>.1).float().mean())))
            if save and n in data.train_cameras[:2]:save_image(torch.cat([targets[n],r['rgb']],dim=1),out/f'step_{step:05d}_{n}.png')
        summary=dict(step=step,gaussians=len(g.xyz),seconds=time.perf_counter()-start,**{k:float(np.mean([r[k] for r in rows])) for k in ['psnr','l1','ssim_uniform7','coverage']})
        # MSE is monitored because this experiment explicitly targets PSNR.
        summary['mse']=float(np.mean([10**(-r['psnr']/10) for r in rows]));history.append(summary);dump('progress.json',history)
        improved,plateau=stopper.update(summary['mse'],step)
        state={'reference':g.state_dict(),'optimizer':opt.state_dict(),'step':step,'config':cfg}
        torch.save(state,out/'latest.pt')
        if improved:torch.save(state,out/'best_loss.pt')
        if summary['psnr']>best_psnr:
            best_psnr=summary['psnr'];torch.save(state,out/'best.pt');dump('best_metrics.json',summary)
        dump('stopping.json',stopper.report());print(json.dumps(summary),flush=True)
        return summary,plateau
    summary,_=evaluate(0);step=0
    if summary['psnr']>=target:reason='target_psnr'
    else:
        for i in range(cfg['endpoint_steps']):
            step=i+1;n=data.train_cameras[i%len(data.train_cameras)];opt.param_groups[0]['lr']=position_lr(cfg,i,data.extent)
            r=render(g.xyz,g.covariance(),g.color(),g.opacity(),cameras[n]);diff=r['rgb']-targets[n]
            loss=diff.square().mean()+cfg.get('keyframe_l1_weight',.1)*diff.abs().mean()
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss step {i}')
            opt.zero_grad();loss.backward()
            scores.add_(g.xyz.grad.detach().norm(dim=-1));opt.step();losses.append(float(loss.detach()))
            densify=cfg.get('densify_until',0)
            densify_interval=cfg.get('densify_interval',1000)
            if densify_interval and step%densify_interval==0:
                if step<=densify and len(g.xyz)<cfg.get('max_gaussians',100000):
                    fraction=min(cfg.get('split_fraction',.25),(cfg['max_gaussians']-len(g.xyz))/len(g.xyz))
                    g,event=split_gaussians(g,scores,fraction);lineage.append(dict(step=step,**event));dump('lineage.json',lineage)
                    opt=optimizer();scores=torch.zeros(len(g.xyz),device=device);stopper.reset_patience()
                    print(json.dumps({'split_step':step,'gaussians':len(g.xyz)}),flush=True)
            if step%sc['eval_interval']==0 or step==cfg['endpoint_steps']:
                summary,plateau=evaluate(step)
                if summary['psnr']>=target:reason='target_psnr';break
                if sc['enabled'] and plateau and step>densify:reason='plateau';break
    executed=step;saved=torch.load(out/'best.pt',map_location=device,weights_only=False);s=saved['reference']
    g=Gaussians(s['xyz'],s['color_logits'].sigmoid(),1.).to(device);g.load_state_dict(s)
    summary=json.loads((out/'best_metrics.json').read_text());data.save_audit(out/'training_access.json');test=None
    with torch.no_grad():
        for n in data.train_cameras[:2]:
            r=render(g.xyz,g.covariance(),g.color(),g.opacity(),cameras[n]);save_image(torch.cat([targets[n],r['rgb']],dim=1),out/f'final_{n}.png')
            save_image(r['alpha'][...,None].expand(-1,-1,3),out/f'final_{n}_alpha.png');np.save(out/f'final_{n}_depth.npy',r['depth'].cpu().numpy())
        if args.evaluate_heldout:
            n=cfg['test_camera'];r=render(g.xyz,g.covariance(),g.color(),g.opacity(),data.camera(n,device));truth=data.image(n,frame,'view').to(device)
            test=image_metrics(r['rgb'],truth);save_image(torch.cat([truth,r['rgb']],dim=1),out/'final_cam00.png')
    dump('losses.json',losses);data.save_audit(out/'all_access.json')
    dump('result.json',dict(status='completed',stop_reason=reason,target_psnr=target,target_reached=summary['psnr']>=target,frame=frame,gaussians=len(g.xyz),executed_steps=executed,selected_step=saved['step'],train=summary,heldout_view=test,total_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),geometry_verified=False,material_correspondence_proven=False,densification=bool(lineage)))

if __name__=='__main__':main()
