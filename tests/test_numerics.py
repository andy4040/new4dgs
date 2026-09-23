import json
from pathlib import Path
import math
import pytest
import torch
from torch import nn
from transport import flow,jacobian,Velocity,gaussian_w2,sinkhorn,ot_map,advect_points
from covariance_renderer import render,pack_covariance
from data import validate_split

torch.set_default_dtype(torch.float64)

class Translation(nn.Module):
    def forward(self,x,t):return torch.ones_like(x)*x.new_tensor([.3,-.2,.1])
class Rotation(nn.Module):
    def forward(self,x,t):return torch.stack([-x[...,1],x[...,0],x[...,2]*0],-1)
class Curve(nn.Module):
    def forward(self,x,t):
        tt=torch.as_tensor(t,dtype=x.dtype,device=x.device)
        return torch.ones_like(x)*torch.stack([tt*0+1,2*tt,3*tt*tt])

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('kind',[Translation,Rotation,Curve])
def test_analytic_motion(device,kind):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    x=torch.tensor([[.2,.3,1.],[-.4,.1,2.]],device=device)
    L=torch.diag_embed(torch.tensor([[.1,.2,.3],[.15,.1,.2]],device=device))
    p,c=flow(kind(),x,L,1,step=.05)
    if kind is Translation:expected=x+x.new_tensor([.3,-.2,.1]);A=torch.eye(3,device=device)
    elif kind is Curve:expected=x+1;A=torch.eye(3,device=device)
    else:
        A=x.new_tensor([[math.cos(1),-math.sin(1),0],[math.sin(1),math.cos(1),0],[0,0,1]])
        expected=x@A.T
    torch.testing.assert_close(p,expected,atol=1e-7,rtol=1e-7)
    torch.testing.assert_close(c,A@(L@L.transpose(-1,-2))@A.T+torch.eye(3,device=device)*1e-8,atol=1e-8,rtol=1e-7)
    assert torch.linalg.eigvalsh(c).min()>0


def test_step_convergence_and_same_reference():
    x=torch.tensor([[1.,0.,0.]]);L=torch.eye(3)[None]*.1
    gt=x.new_tensor([[math.cos(1),math.sin(1),0.]])
    coarse,_=flow(Rotation(),x,L,1,'euler',.1)
    fine,_=flow(Rotation(),x,L,1,'euler',.01)
    assert (fine-gt).norm()<(coarse-gt).norm()/5
    a,_=flow(Rotation(),x,L,.5,step=.02)
    b=advect_points(Rotation(),a,.5,step=.02)
    end,_=flow(Rotation(),x,L,1,step=.02)
    torch.testing.assert_close(b,end,atol=1e-9,rtol=1e-9)


def test_ot_rotation_invariance_unequal_marginals():
    torch.manual_seed(3)
    l=torch.randn(3,3,3);a=l@l.transpose(-1,-2)+torch.eye(3)*.2
    r=torch.linalg.qr(torch.randn(3,3))[0]
    torch.testing.assert_close(a,(l@r)@(l@r).transpose(-1,-2)+torch.eye(3)*.2)
    b=a.flip(0)*1.5;T=ot_map(a,b)
    torch.testing.assert_close(T@a@T.transpose(-1,-2),b,rtol=1e-6,atol=1e-7)
    x=torch.randn(3,3);y=torch.randn(5,3);b5=torch.eye(3).expand(5,3,3)*.3
    cost=gaussian_w2(x,a,y,b5);p,report=sinkhorn(cost,.1,1000,1e-6)
    torch.testing.assert_close(p.sum(1),torch.ones(3)/3,atol=1e-6,rtol=0)
    torch.testing.assert_close(p.sum(0),torch.ones(5)/5,atol=1e-6,rtol=0)
    torch.testing.assert_close(gaussian_w2(x,a,x,a).diagonal(),torch.zeros(3),atol=1e-7,rtol=0)


def camera(dtype=torch.float64):
    return {'R':torch.eye(3,dtype=dtype),'t':torch.zeros(3,dtype=dtype),'K':torch.tensor([[16.,0,8],[0,16.,8],[0,0,1]],dtype=dtype),'width':16,'height':16}


def test_rgb_gradient_through_jacobian_and_ode():
    class Affine(nn.Module):
        def __init__(self):super().__init__();self.a=nn.Parameter(torch.tensor(.2))
        def forward(self,x,t):return self.a*x
    v=Affine();x=torch.tensor([[.15,.1,2.]]);L=torch.eye(3)[None]*.12
    def objective():
        p,c=flow(v,x,L,.8,step=.2)
        # Deliberately detach center: remaining derivative must traverse covariance/J.
        return render(p.detach(),c,torch.tensor([[.8,.4,.2]]),torch.tensor([.7]),camera())['rgb'].square().sum()
    value=objective();value.backward();analytic=v.a.grad.item()
    # Finite difference covariance-only objective, keep center fixed for both perturbations.
    fixed=flow(v,x,L,.8,step=.2)[0].detach()
    def fd_obj():
        _,c=flow(v,x,L,.8,step=.2)
        return render(fixed,c,torch.tensor([[.8,.4,.2]]),torch.tensor([.7]),camera())['rgb'].square().sum()
    old=v.a.item();eps=1e-5
    with torch.no_grad():v.a.fill_(old+eps)
    plus=fd_obj().item()
    with torch.no_grad():v.a.fill_(old-eps)
    minus=fd_obj().item()
    assert abs(analytic)>1e-5
    assert analytic==pytest.approx((plus-minus)/(2*eps),rel=1e-4,abs=1e-6)


def test_mlp_rgb_backprop():
    torch.manual_seed(2);v=Velocity(8).double();x=torch.tensor([[.2,0.,2.]])
    p,c=flow(v,x,torch.eye(3)[None]*.1,.4,step=.2)
    loss=render(p,c,torch.ones(1,3)*.5,torch.ones(1)*.7,camera())['rgb'].square().sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in v.parameters())
    assert sum(p.grad.norm() for p in v.parameters())>0


def test_covariance_pack():
    c=torch.tensor([[[1.,2,3],[2,4,5],[3,5,6]]])
    torch.testing.assert_close(pack_covariance(c),torch.tensor([[1.,2,3,4,5,6]]))


def test_split():
    cfg=json.loads((Path(__file__).parents[1]/'configs/smoke.json').read_text());validate_split(cfg)
    cfg['train_frames'][1]=12
    with pytest.raises(AssertionError):validate_split(cfg)
