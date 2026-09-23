"""Small differentiable reference splatter + optional upstream CUDA adapter.
World covariance is projected by the pinhole Jacobian; depth ordered alpha blend.
The reference backend is deliberately small-scale (O(N H W)).
"""
import torch


def project(x, camera):
    q=x@camera['R'].T+camera['t']
    z=q[:,2].clamp_min(1e-5); K=camera['K']
    uv=torch.stack([K[0,0]*q[:,0]/z+K[0,2],K[1,1]*q[:,1]/z+K[1,2]],-1)
    return uv,q


def render(x,cov,color,opacity,camera,pixel_chunk=4096):
    uv,q=project(x,camera); z=q[:,2].clamp_min(1e-5); K=camera['K']
    zero=torch.zeros_like(z)
    J=torch.stack([K[0,0]/z,zero,-K[0,0]*q[:,0]/z.square(),zero,K[1,1]/z,-K[1,1]*q[:,1]/z.square()],-1).reshape(-1,2,3)
    Rc=camera['R']; S=J@(Rc@cov@Rc.T)@J.transpose(-1,-2)
    S=S+torch.eye(2,device=x.device,dtype=x.dtype)*0.3
    inv=torch.linalg.inv(S)
    order=torch.argsort(q[:,2]); uv,inv,color,opacity=uv[order],inv[order],color[order],opacity[order]
    depth=q[order,2]; valid=(depth>1e-4).to(x.dtype)
    yy,xx=torch.meshgrid(torch.arange(camera['height'],device=x.device,dtype=x.dtype),torch.arange(camera['width'],device=x.device,dtype=x.dtype),indexing='ij')
    pixels=torch.stack([xx.flatten(),yy.flatten()],-1)
    rgb_parts=[]; dep_parts=[]; alpha_parts=[]
    for p in pixels.split(pixel_chunk):
        d=p[None]-uv[:,None]
        maha=torch.einsum('npi,nij,npj->np',d,inv,d)
        a=(opacity[:,None]*torch.exp(-0.5*maha)*valid[:,None]).clamp(0,0.99)
        trans=torch.cumprod(torch.cat([torch.ones_like(a[:1]),1-a+1e-10],0),0)[:-1]
        weight=trans*a
        rgb_parts.append(weight.T@color)
        dep_parts.append((weight*depth[:,None]).sum(0)/weight.sum(0).clamp_min(1e-8))
        alpha_parts.append(weight.sum(0))
    h,w=camera['height'],camera['width']
    return {'rgb':torch.cat(rgb_parts).reshape(h,w,3),'depth':torch.cat(dep_parts).reshape(h,w),
            'alpha':torch.cat(alpha_parts).reshape(h,w)}


def pack_covariance(cov):
    # Gaussian rasterizer symmetric order: xx,xy,xz,yy,yz,zz. Scale/rotation bypass.
    return cov[:,[0,0,0,1,1,2],[0,1,2,1,2,2]].contiguous()


def render_cuda(x,cov,color,opacity,settings):
    """settings is upstream GaussianRasterizationSettings; compiled extension required.
    cov3D_precomp avoids quaternion ordering and log/activated scale ambiguity.
    Settings scale_modifier MUST be 1. This adapter needs runtime parity testing.
    """
    if settings.scale_modifier != 1.0: raise ValueError('Use scale_modifier=1 for precomputed covariance')
    from diff_gaussian_rasterization import GaussianRasterizer
    return GaussianRasterizer(raster_settings=settings)(means3D=x,
        means2D=torch.zeros_like(x,requires_grad=True),shs=None,colors_precomp=color,
        opacities=opacity[:,None],scales=None,rotations=None,cov3D_precomp=pack_covariance(cov))
