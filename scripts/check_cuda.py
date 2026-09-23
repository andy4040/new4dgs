"""Verify upstream scale/quaternion convention vs packed full covariance, then gradients."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'4DGaussians'))
import torch
from diff_gaussian_rasterization import GaussianRasterizer,GaussianRasterizationSettings
from utils.general_utils import build_scaling_rotation
from utils.graphics_utils import getProjectionMatrix
from covariance_renderer import render_cuda
from transport import Velocity,flow

torch.manual_seed(4)
P=getProjectionMatrix(.01,100.,.8,.8).cuda().T
settings=GaussianRasterizationSettings(image_height=32,image_width=32,tanfovx=__import__('math').tan(.4),tanfovy=__import__('math').tan(.4),bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=torch.eye(4,device='cuda'),projmatrix=P,sh_degree=0,campos=torch.zeros(3,device='cuda'),prefiltered=False,debug=False)
x=torch.tensor([[.1,.1,2.],[-.2,.2,2.4]],device='cuda');s=torch.tensor([[.08,.13,.2],[.1,.15,.07]],device='cuda')
q=torch.tensor([[.8,.2,-.1,.5],[1.,.1,.2,.3]],device='cuda');q=torch.nn.functional.normalize(q,dim=-1)
L=build_scaling_rotation(s,q);cov=L@L.transpose(-1,-2)
color=torch.tensor([[.8,.4,.2],[.3,.7,.9]],device='cuda');opacity=torch.ones(2,device='cuda')*.6
first=GaussianRasterizer(settings)(means3D=x,means2D=torch.zeros_like(x),shs=None,colors_precomp=color,opacities=opacity[:,None],scales=s,rotations=q,cov3D_precomp=None)[0]
second=render_cuda(x,cov,color,opacity,settings)[0]
error=float((first-second).abs().max())
assert error<1e-5,error
v=Velocity(8).cuda();px,pc=flow(v,x,L,.5,step=.25)
loss=render_cuda(px,pc,color,opacity,settings)[0].square().sum();loss.backward()
norm=sum(float(p.grad.norm()) for p in v.parameters() if p.grad is not None)
assert norm>0 and __import__('math').isfinite(norm)
report={'scale_rotation_vs_covariance_max_error':error,'rgb_to_velocity_gradient_norm':norm,'activated_scale':'stddev, not variance or log-scale','quaternion':'normalized wxyz','covariance_order':['xx','xy','xz','yy','yz','zz'],'status':'passed'}
Path('reports/cuda_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
