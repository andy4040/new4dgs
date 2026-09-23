import torch
from transport import Velocity,flow,chunked_flow


def test_checkpointed_chunked_flow_parameter_gradients():
    torch.manual_seed(44)
    v=Velocity(8).double();x=torch.randn(7,3,dtype=torch.double);L=torch.eye(3,dtype=torch.double).expand(7,3,3)*.1
    a,b=flow(v,x,L,.7,step=.25);loss=a.square().sum()+b.square().sum();loss.backward()
    expected=[p.grad.clone() for p in v.parameters()];v.zero_grad()
    c,d=chunked_flow(v,x,L,.7,step=.25,chunk_size=3)
    assert torch.allclose(a,c,atol=1e-12) and torch.allclose(b,d,atol=1e-12)
    (c.square().sum()+d.square().sum()).backward()
    for p,g in zip(v.parameters(),expected):assert torch.allclose(p.grad,g,atol=1e-9,rtol=1e-8)
    with torch.no_grad():
        c,d=chunked_flow(v,x,L,.7,step=.25,chunk_size=3)
        assert torch.allclose(a,c,atol=1e-12) and torch.allclose(b,d,atol=1e-12)


def test_scene_velocity_analytic_jacobian_and_parameter_gradients():
    from transport import SceneVelocity
    from torch.func import jacrev,vmap
    torch.manual_seed(7)
    v=SceneVelocity([1.,2.,3.],4.,width=8,depth=2).double();x=torch.randn(5,3,dtype=torch.double)
    value,j=v.value_and_jacobian(x,.3);truth=vmap(jacrev(lambda z:v(z,.3)))(x)
    assert torch.allclose(value,v(x,.3),atol=1e-12)
    assert torch.allclose(j,truth,atol=1e-12)
    a=torch.autograd.grad(j.square().sum(),tuple(v.parameters()),allow_unused=True)
    b=torch.autograd.grad(truth.square().sum(),tuple(v.parameters()),allow_unused=True)
    for u,w in zip(a,b):
        assert (u is None)==(w is None)
        if u is not None:assert torch.allclose(u,w,atol=1e-12)
