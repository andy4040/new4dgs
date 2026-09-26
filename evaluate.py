"""Reload an ODE checkpoint and evaluate exact view/time/joint holdouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from data import N3DV,normalized_time
from geometry import Gaussians
from transport import Velocity,flow
from run import Appearance
from covariance_renderer import render
from evaluation import image_metrics,visualize


def main(a):
    saved=torch.load(a.checkpoint,map_location=a.device,weights_only=True)
    cfg=dict(saved['config']);cfg['device']=a.device
    if a.data:cfg['data_root']=str(Path(a.data).expanduser().resolve())
    data=N3DV(cfg);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if saved.get('model_format')=='full_sequence_v2' or 'layers.0.weight' in saved.get('velocity',{}):
        from sequence_model import load_sequence
        from full_sequence import evaluate_all
        if saved.get('phase','motion')!='motion':
            raise ValueError('Evaluate a frozen motion checkpoint, not a preparation checkpoint')
        model=load_sequence(cfg,saved,data.extent)
        model.eval()
        evaluate_all(cfg,data,model,out)
        return
    ref=saved['reference'];g=Gaussians(ref['xyz'],torch.ones_like(ref['xyz'])*.5,.01).to(a.device);g.load_state_dict(ref)
    v=Velocity(cfg['velocity_width']).to(a.device);v.load_state_dict(saved['velocity']);v.eval()
    appearance=Appearance(len(g.xyz),cfg['appearance_time_rank'],a.device);appearance.load_state_dict(saved['appearance'])
    rows=[];positions=[]
    with torch.no_grad():
        for frame in range(10,31):
            t=normalized_time(frame);x,c=flow(v,g.xyz,g.factor(),t,cfg['ode_method'],cfg['ode_step']);positions.append(x.cpu().numpy())
            color,opacity=appearance(g,t)
            for split in ('view','time','joint'):
                for cam,f in data.keys(split):
                    if f!=frame:continue
                    pred=render(x,c,color,opacity,data.camera(cam,a.device))['rgb'];target=data.image(cam,f,split).to(a.device)
                    rows.append({'split':split,'frame':frame,'camera':cam,**image_metrics(pred,target)})
    summary={s:{k:float(np.mean([r[k] for r in rows if r['split']==s])) for k in ('psnr','ssim_uniform7','l1')} for s in ('view','time','joint')}
    (out/'evaluation.json').write_text(json.dumps({'image':summary,'per_image':rows,'trajectory_3d_error':None},indent=2))
    visualize(positions,g.ids.cpu().numpy(),out,{n:data.cameras[n] for n in ['cam00',data.train_cameras[0]]});data.save_audit(out/'evaluation_access.json')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data');p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');main(p.parse_args())
