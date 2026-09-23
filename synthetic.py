"""Known synthetic geometry experiment. RGB-only learned trajectories;
analytic motion is used to generate images and evaluate, never as a velocity loss.
This is NOT an N3DV result. Numerical oracle tests are in tests/test_numerics.py.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from transport import Velocity,flow,FlowMatching
from covariance_renderer import render
from evaluation import visualize,image_metrics,save_image,approximation_error


def truth(x,t): return x+x.new_tensor([.35*math.sin(math.pi*t),.3*t,.08*math.sin(2*math.pi*t)])


def main(args):
    torch.manual_seed(42);device=args.device;out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    x=torch.tensor([[-.4,-.25,2.],[0,-.25,2.1],[.4,-.25,2.],[-.4,.25,2.1],[0,.25,2.],[.4,.25,2.1]],device=device)
    L=torch.eye(3,device=device).expand(len(x),3,3)*.065;cov=L@L.transpose(-1,-2)
    color=torch.rand(len(x),3,device=device)*.7+.3;opacity=torch.ones(len(x),device=device)*.8
    cams={}
    for i,offset in enumerate([-.2,.2]):
        cams[str(i)]={'R':torch.eye(3,device=device),'t':torch.tensor([offset,0.,0.],device=device),'K':torch.tensor([[30.,0,12],[0,30.,12],[0,0,1.]],device=device),'width':24,'height':24}
    train_times=[0,.15,.3,.45,.6,.75,.9,1.];eval_times=[.1,.25,.5,.8]
    targets={(t,n):render(truth(x,t),cov,color,opacity,c)['rgb'].detach() for t in train_times for n,c in cams.items()}
    summaries={}
    cfg={'fm_batch':64,'sinkhorn_epsilon':.1,'sinkhorn_iterations':1000,'sinkhorn_tolerance':.002,'ode_method':'rk4','ode_step':.25}
    for variant in ['rgb','fm_rgb','fm_persistent']:
        tick=time.perf_counter();torch.manual_seed(1);v=Velocity(24).to(device)
        optimizer=torch.optim.Adam(v.parameters(),lr=.003)
        fm=FlowMatching(x,cov,truth(x,1),cov,cfg)
        if variant!='rgb':
            for _ in range(30):
                loss=fm.loss(v);optimizer.zero_grad();loss.backward();optimizer.step()
        for i in range(args.steps):
            t=train_times[i%len(train_times)];p,c=flow(v,x,L,t,step=.25)
            loss=torch.stack([(render(p,c,color,opacity,cam)['rgb']-targets[t,n]).square().mean() for n,cam in cams.items()]).mean()
            if variant=='fm_persistent':loss=loss+.01*fm.loss(v)
            optimizer.zero_grad();loss.backward();optimizer.step()
        preds=[];errors=[];image=[]
        with torch.no_grad():
            for t in np.linspace(0,1,21):preds.append(flow(v,x,L,float(t),step=.25)[0].cpu().numpy())
            for t in eval_times:
                p,c=flow(v,x,L,t,step=.25);errors.append(float((p-truth(x,t)).norm(dim=-1).mean()))
                for n,cam in cams.items():
                    pred=render(p,c,color,opacity,cam)['rgb'];target=render(truth(x,t),cov,color,opacity,cam)['rgb']
                    image.append(image_metrics(pred,target))
                    if t==.5 and n=='0':
                        dest=out/variant;dest.mkdir(exist_ok=True);save_image(pred,dest/'prediction.png');save_image(target,dest/'target.png')
        dest=out/variant;visualize(preds,np.arange(len(x)),dest,cams)
        torch.save(v.state_dict(),dest/'velocity.pt')
        summaries[variant]={'heldout_3d_mean_error':float(np.mean(errors)),
            'heldout_image':{k:float(np.mean([r[k] for r in image])) for k in image[0]},
            'approximation':approximation_error(v,x,L,cfg),'seconds_including_fm':time.perf_counter()-tick}
    report={'experiment':'synthetic_known_initial_geometry_rgb_training','steps':args.steps,'device':device,
        'oracle_velocity_supervision':False,'reference_geometry_oracle':True,'train_times':train_times,'evaluation_times':eval_times,'results':summaries}
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--out',default='runs/synthetic');p.add_argument('--steps',type=int,default=80)
    main(p.parse_args())
