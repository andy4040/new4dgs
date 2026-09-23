"""Differentiable continuous flow, Gaussian W2 and log-domain Sinkhorn."""
import math
import torch
from torch import nn
from torch.func import jacrev, vmap


class Velocity(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4,width), nn.Tanh(), nn.Linear(width,width),
                                 nn.Tanh(),nn.Linear(width,3))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        t = torch.as_tensor(t,device=x.device,dtype=x.dtype)
        t = t.expand(x.shape[:-1]).unsqueeze(-1)
        return self.net(torch.cat([x,t],-1))


def jacobian(v, x, t):
    # jacrev/vmap preserves higher derivatives to network parameters and x.
    if hasattr(v,'value_and_jacobian'):return v.value_and_jacobian(x,t)[1]
    return vmap(jacrev(lambda z: v(z,t)))(x)


def flow(v, x0, factor0, t, method='rk4', step=0.05):
    """Integrate x and deformation gradient F from t=0 for EVERY request.
    Sigma(t)=F L0 L0^T F^T + jitter I; no eigendecomposition in RGB path.
    """
    if method not in ('euler','rk4') or step <= 0 or not 0 <= float(t) <= 1:
        raise ValueError('Invalid ODE settings/time')
    x=x0; F=torch.eye(3,device=x.device,dtype=x.dtype).expand(len(x),3,3)
    n=max(1,math.ceil(float(t)/step)); h=float(t)/n
    def rhs(xx,ff,tt):
        if hasattr(v,'value_and_jacobian'):
            velocity,J=v.value_and_jacobian(xx,tt)
            return velocity,J@ff
        return v(xx,tt),jacobian(v,xx,tt)@ff
    for i in range(n):
        tt=i*h
        a,b=rhs(x,F,tt)
        if method=='euler': x,F=x+h*a,F+h*b
        else:
            c,d=rhs(x+h*a/2,F+h*b/2,tt+h/2)
            e,f=rhs(x+h*c/2,F+h*d/2,tt+h/2)
            g,k=rhs(x+h*e,F+h*f,tt+h)
            x,F=x+h*(a+2*c+2*e+g)/6,F+h*(b+2*d+2*f+k)/6
    L=F@factor0
    cov=L@L.transpose(-1,-2)
    return x,cov + torch.eye(3,device=x.device,dtype=x.dtype)*1e-8


def advect_points(v, x, t, method='rk4', step=0.05):
    """Sample transport without Jacobian; used for Gaussian approximation audit."""
    n=max(1,math.ceil(float(t)/step)); h=float(t)/n
    for i in range(n):
        s=i*h; a=v(x,s)
        if method=='euler': x=x+h*a
        elif method=='rk4':
            b=v(x+h*a/2,s+h/2); c=v(x+h*b/2,s+h/2); d=v(x+h*c,s+h)
            x=x+h*(a+2*b+2*c+d)/6
        else: raise ValueError(method)
    return x


def sqrt_spd(c):
    vals,vecs=torch.linalg.eigh(c)
    return (vecs*vals.clamp_min(1e-10).sqrt().unsqueeze(-2))@vecs.transpose(-1,-2)


def gaussian_w2(x,a,y,b):
    """Squared W2, unequal set sizes. Eigh used only for detached FM endpoints."""
    ah=sqrt_spd(a)
    tr=lambda c:c.diagonal(dim1=-2,dim2=-1).sum(-1)
    blocks=[]
    for begin in range(0,len(x),128):
        root=ah[begin:begin+128]
        cross=root[:,None]@b[None]@root[:,None]
        trroot=torch.linalg.eigvalsh(cross).clamp_min(0).sqrt().sum(-1)
        dist=torch.cdist(x[begin:begin+128],y).square()
        blocks.append((dist+tr(a[begin:begin+128])[:,None]+tr(b)[None]-2*trroot).clamp_min(0))
    return torch.cat(blocks,0)


def sinkhorn(cost, epsilon=0.05, iterations=500, tolerance=2e-3):
    if epsilon <= 0 or not torch.isfinite(cost).all(): raise ValueError('Invalid cost/epsilon')
    n,m=cost.shape; loga=cost.new_full((n,),-math.log(n)); logb=cost.new_full((m,),-math.log(m))
    # epsilon is relative to positive median cost, recorded with diagnostics.
    scale=cost.detach().median().clamp_min(1e-6)
    K=-cost/(epsilon*scale); u=torch.zeros_like(loga); v=torch.zeros_like(logb)
    for _ in range(iterations):
        u=loga-torch.logsumexp(K+v[None],1)
        v=logb-torch.logsumexp(K+u[:,None],0)
    p=(K+u[:,None]+v[None]).exp()
    err=max((p.sum(1)-loga.exp()).abs().sum().item(),(p.sum(0)-logb.exp()).abs().sum().item())
    if err>tolerance: raise RuntimeError(f'Sinkhorn marginal L1 {err:.4g} > {tolerance}')
    return p,{'marginal_l1':err,'epsilon_absolute':float(epsilon*scale),'iterations':iterations}


def ot_map(a,b):
    ah=sqrt_spd(a); inv=torch.linalg.inv(ah)
    return inv@sqrt_spd(ah@b@ah)@inv


class FlowMatching:
    def __init__(self, x,a,y,b,cfg):
        self.x,self.a,self.y,self.b=[q.detach() for q in (x,a,y,b)]
        self.cost=gaussian_w2(self.x,self.a,self.y,self.b)
        self.p,self.diagnostics=sinkhorn(self.cost,cfg['sinkhorn_epsilon'],cfg['sinkhorn_iterations'],cfg['sinkhorn_tolerance'])
        self.batch=cfg['fm_batch']

    def loss(self,v):
        k=torch.multinomial(self.p.flatten(),self.batch,replacement=True)
        i=k//len(self.y); j=k%len(self.y)
        # Symmetric covariance root + Gaussian OT map: no arbitrary factor rotation.
        noise=torch.randn(self.batch,3,device=self.x.device,dtype=self.x.dtype)
        z0=self.x[i]+(sqrt_spd(self.a[i])@noise[...,None]).squeeze(-1)
        z1=self.y[j]+(ot_map(self.a[i],self.b[j])@(z0-self.x[i])[...,None]).squeeze(-1)
        s=torch.rand(self.batch,device=self.x.device,dtype=self.x.dtype)
        z=(1-s[:,None])*z0+s[:,None]*z1
        return (v(z,s)-(z1-z0)).square().mean()


def chunked_flow(v,x0,factor0,t,method='rk4',step=.05,chunk_size=2048,checkpoint_grad=True):
    """Same ODE and gradients, recomputed per independent Gaussian chunk.

    Reentrant checkpoint avoids the saved-tensor hooks incompatible with jacrev.
    A dummy differentiable input enables checkpoint with frozen reference geometry.
    Parameters remain in the shared velocity module; checkpoint backward accumulates
    their derivatives from each chunk. Never detach transported positions/covariance.
    """
    from torch.utils.checkpoint import checkpoint
    if chunk_size<=0:raise ValueError('chunk_size must be positive')
    positions=[];covariances=[]
    for xx,ll in zip(x0.split(chunk_size),factor0.split(chunk_size)):
        if torch.is_grad_enabled() and checkpoint_grad:
            dummy=x0.new_zeros((),requires_grad=True)
            # Bind t/method/step, not loop tensors: checkpoint supplies saved inputs.
            def integrate(a,b,d):
                x,c=flow(v,a,b,t,method,step)
                return x+d*0,c
            x,c=checkpoint(integrate,xx,ll,dummy,use_reentrant=True)
        else:x,c=flow(v,xx,ll,t,method,step)
        positions.append(x);covariances.append(c)
    return torch.cat(positions),torch.cat(covariances)


class SceneVelocity(nn.Module):
    """Shared smooth velocity MLP with normalized coordinates and Fourier features.

    Network predicts normalized-distance per normalized-time; output converts back
    to world-distance per normalized-time. Analytic spatial Jacobian retains the
    autograd graph for parameter derivatives through covariance transport.
    """
    def __init__(self,center,scale,width=64,depth=3,spatial_bands=3,time_bands=4):
        super().__init__();self.register_buffer('center',torch.as_tensor(center,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32).clamp_min(1e-6))
        self.register_buffer('spatial_freq',torch.pi*2**torch.arange(spatial_bands,dtype=torch.float32))
        self.register_buffer('time_freq',torch.pi*2**torch.arange(time_bands,dtype=torch.float32))
        dim=4+6*spatial_bands+2*time_bands
        self.layers=nn.ModuleList([nn.Linear(dim if i==0 else width,width) for i in range(depth)])
        self.output=nn.Linear(width,3);nn.init.normal_(self.output.weight,std=1e-4);nn.init.zeros_(self.output.bias)

    def value_and_jacobian(self,x,t):
        z=(x-self.center)/self.scale;tt=torch.as_tensor(t,device=x.device,dtype=x.dtype).expand(x.shape[:-1])[...,None]
        phase=z[...,None]*self.spatial_freq;tp=tt*self.time_freq
        features=torch.cat([z,tt,phase.sin().flatten(-2),phase.cos().flatten(-2),tp.sin(),tp.cos()],-1)
        eye=torch.eye(3,device=x.device,dtype=x.dtype).expand(*x.shape[:-1],3,3)
        zero=x.new_zeros(*x.shape[:-1],1+2*len(self.time_freq),3)
        ds=(phase.cos()*self.spatial_freq)[...,None]*eye[..., :,None,:]
        dc=(-phase.sin()*self.spatial_freq)[...,None]*eye[..., :,None,:]
        derivative=torch.cat([eye,zero[...,:1,:],ds.flatten(-3,-2),dc.flatten(-3,-2),zero[...,1:,:]],-2)
        # Derivative is with respect to normalized position; output scale cancels input scale.
        h=features
        for layer in self.layers:
            a=layer(h);sig=a.sigmoid();slope=sig*(1+a*(1-sig))
            derivative=(layer.weight@derivative)*slope[...,None];h=torch.nn.functional.silu(a)
        return self.output(h)*self.scale,self.output.weight@derivative

    def forward(self,x,t):
        # Forward alone avoids constructing unused Jacobian during sample transport.
        z=(x-self.center)/self.scale;tt=torch.as_tensor(t,device=x.device,dtype=x.dtype).expand(x.shape[:-1])[...,None]
        phase=z[...,None]*self.spatial_freq;tp=tt*self.time_freq
        h=torch.cat([z,tt,phase.sin().flatten(-2),phase.cos().flatten(-2),tp.sin(),tp.cos()],-1)
        for layer in self.layers:h=torch.nn.functional.silu(layer(h))
        return self.output(h)*self.scale
