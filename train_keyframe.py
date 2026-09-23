"""Standalone static keyframe training; held-out view is read only after training."""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from data import N3DV
from geometry import Gaussians,triangulate,initial_scales,position_lr
from covariance_renderer import render
from evaluation import image_metrics,save_image


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time.perf_counter()
    def dump(name,obj):(out/name).write_text(json.dumps(obj,indent=2))
    dump('config.json',cfg);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    device=cfg['device'];frame=10;data=N3DV(cfg);geo=N3DV(dict(cfg,image_width=cfg['triangulation_width']))
    xyz,rgb,init=triangulate(geo,frame,cfg['gaussians_start']);geo.save_audit(out/'triangulation_access.json')
    g=Gaussians(xyz.to(device),rgb.to(device),initial_scales(xyz,data.extent,cfg).to(device)).to(device)
    dump('initialization.json',init);print(json.dumps(init),flush=True)
    targets={n:data.image(n,frame).to(device) for n in data.train_cameras}
    cameras={n:data.camera(n,device) for n in data.train_cameras}
    opt=torch.optim.Adam([{'params':[g.xyz],'lr':position_lr(cfg,0,data.extent)},
        {'params':[g.log_scale,g.lower,g.color_logits,g.opacity_logits],'lr':cfg['endpoint_other_lr']}])
    history=[];losses=[];torch.cuda.reset_peak_memory_stats()
    @torch.no_grad()
    def evaluate(step):
        rows=[]
        for n in data.train_cameras:
            r=render(g.xyz,g.covariance(),g.color(),g.opacity(),cameras[n])
            rows.append(dict(camera=n,**image_metrics(r['rgb'],targets[n]),coverage=float((r['alpha']>.1).float().mean())))
            if n in data.train_cameras[:2]:
                save_image(torch.cat([targets[n],r['rgb']],dim=1),out/f'step_{step:05d}_{n}.png')
                if step==cfg['endpoint_steps']:
                    save_image(r['alpha'][...,None].expand(-1,-1,3),out/f'final_{n}_alpha.png');np.save(out/f'final_{n}_depth.npy',r['depth'].cpu().numpy())
        summary=dict(step=step,seconds=time.perf_counter()-start,**{k:float(np.mean([r[k] for r in rows])) for k in ['psnr','l1','ssim_uniform7','coverage']})
        history.append(summary);dump('progress.json',history);dump(f'metrics_{step:05d}.json',rows)
        torch.save({'reference':g.state_dict(),'optimizer':opt.state_dict(),'step':step,'config':cfg},out/'latest.pt')
        print(json.dumps(summary),flush=True)
    evaluate(0)
    for i in range(cfg['endpoint_steps']):
        n=data.train_cameras[i%len(data.train_cameras)];opt.param_groups[0]['lr']=position_lr(cfg,i,data.extent)
        r=render(g.xyz,g.covariance(),g.color(),g.opacity(),cameras[n]);loss=(r['rgb']-targets[n]).abs().mean()
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss step {i}')
        opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        if (i+1)%1000==0 or i+1==cfg['endpoint_steps']:evaluate(i+1)
    data.save_audit(out/'training_access.json')
    # Final-only view test. No cam00 data is read for initialization or optimization.
    with torch.no_grad():
        n=cfg['test_camera'];r=render(g.xyz,g.covariance(),g.color(),g.opacity(),data.camera(n,device));target=data.image(n,frame,'view').to(device)
        test=image_metrics(r['rgb'],target);save_image(torch.cat([target,r['rgb']],dim=1),out/'final_cam00.png')
    dump('losses.json',losses);data.save_audit(out/'all_access.json')
    dump('result.json',dict(status='completed',frame=frame,gaussians=len(xyz),steps=cfg['endpoint_steps'],train=history[-1],heldout_view=test,total_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),geometry_verified=False,material_correspondence_proven=False,densification=False))
    print(json.dumps(test),flush=True)

if __name__=='__main__':main()
