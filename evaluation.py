import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from covariance_renderer import project
from transport import advect_points,flow


def image_metrics(pred,target):
    mse=(pred-target).square().mean()
    # Local SSIM, 7x7 uniform window, valid interior; records exact definition.
    x=pred.permute(2,0,1)[None]; y=target.permute(2,0,1)[None]
    pool=lambda z:F.avg_pool2d(z,7,stride=1)
    mx,my=pool(x),pool(y); vx=pool(x*x)-mx*mx; vy=pool(y*y)-my*my; xy=pool(x*y)-mx*my
    ssim=((2*mx*my+.01**2)*(2*xy+.03**2)/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean()
    return {'psnr':float(-10*torch.log10(mse.clamp_min(1e-12))),'ssim_uniform7':float(ssim),'l1':float((pred-target).abs().mean())}


def save_image(x,path):
    Image.fromarray((x.detach().cpu().clamp(0,1).numpy()*255).astype('uint8')).save(path)


def visualize(positions,ids,out,cameras=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=np.asarray(positions); ids=np.asarray(ids)
    colors=plt.get_cmap('hsv')((ids*0.61803398875)%1)
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(projection='3d')
    selection=np.linspace(0,len(ids)-1,min(64,len(ids))).astype(int)
    for i in selection:
        ax.plot(*p[:,i].T,color=colors[i],linewidth=1)
        ax.scatter(*p[0,i],color=colors[i],s=8)
    ax.set(xlabel='x',ylabel='y',zlabel='z',title='Persistent Gaussian IDs; paths from one reference state')
    fig.tight_layout();fig.savefig(out/'trajectories_3d.png');plt.close(fig)
    if cameras:
        for name,cam in cameras.items():
            uv=[]
            for pp in p:
                u,_=project(torch.tensor(pp,dtype=cam['R'].dtype,device=cam['R'].device),cam);uv.append(u.cpu().numpy())
            uv=np.array(uv);fig,ax=plt.subplots(figsize=(8,6))
            for i in selection:ax.plot(*uv[:,i].T,color=colors[i])
            ax.set(xlim=(0,cam['width']),ylim=(cam['height'],0),title=name+' ID projection trajectories',xlabel='pixel x',ylabel='pixel y')
            fig.tight_layout();fig.savefig(out/f'projection_{name}.png');plt.close(fig)
    np.savez_compressed(out/'trajectories.npz',positions=p,ids=ids,colors=colors)


@torch.no_grad()
def approximation_error(v,x,L,cfg,samples=1024):
    n=min(8,len(x));x=x[:n];L=L[:n]
    z=torch.randn(n,samples,3,device=x.device,dtype=x.dtype)
    samples0=x[:,None]+torch.einsum('nij,nsj->nsi',L,z)
    transported=advect_points(v,samples0.flatten(0,1),1,cfg['ode_method'],cfg['ode_step']).reshape(n,samples,3)
    m,c=flow(v,x,L,1,cfg['ode_method'],cfg['ode_step'])
    em=transported.mean(1);centered=transported-em[:,None]
    ec=centered.transpose(1,2)@centered/(samples-1)
    return {'samples_per_gaussian':samples,'gaussians':n,'mean_error':float((em-m).norm(dim=-1).mean()),
        'relative_covariance_frobenius':float(((ec-c).flatten(1).norm(dim=-1)/c.flatten(1).norm(dim=-1).clamp_min(1e-12)).mean()),
        'note':'Monte Carlo discrepancy includes sampling noise; not 3D trajectory ground truth.'}
