import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from geometry import Gaussians
from baseline import initialize,state
from covariance_renderer import render

cfg=json.loads(Path('configs/smoke.json').read_text())
torch.manual_seed(11)
x=torch.rand(16,3,device='cuda')*.5;x[:,2]+=2
model=Gaussians(x,torch.rand_like(x),.07).cuda()
b,hidden=initialize(model,cfg,1.)
p,c,color,o=state(b,.5)
cam={'R':torch.eye(3,device='cuda'),'t':torch.zeros(3,device='cuda'),'K':torch.tensor([[20.,0,12],[0,20.,12],[0,0,1.]],device='cuda'),'width':24,'height':24}
loss=render(p,c,color,o,cam)['rgb'].square().mean()+b.compute_regulation(hidden.time_smoothness_weight,hidden.l1_time_planes,hidden.plane_tv_weight)
b.optimizer.zero_grad();loss.backward();b.optimizer.step()
assert torch.isfinite(p).all() and torch.linalg.eigvalsh(c).min()>0
n=sum(float(p.grad.norm()) for p in b._deformation.parameters() if p.grad is not None)
assert n>0
report={'status':'passed','reference_count':len(x),'deformation_gradient_norm':n,'one_optimizer_step':True,'n3dv_execution':False,'baseline_type':'actual upstream HexPlane deformation / shared reference renderer'}
Path('reports/baseline_validation.json').write_text(json.dumps(report,indent=2));print(report)
