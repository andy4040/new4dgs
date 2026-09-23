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
    if camera.get('renderer','reference') == 'cuda':
        return render_cuda_camera(x,cov,color,opacity,camera)
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


def render_cuda_camera(x,cov,color,opacity,camera):
    from diff_gaussian_rasterization import GaussianRasterizationSettings
    if x.device.type != 'cuda' or x.dtype != torch.float32:
        raise ValueError('Upstream rasterizer requires CUDA float32')
    h,w=camera['height'],camera['width'];K=camera['K']
    if abs(float(K[0,2])-w/2)>1e-4 or abs(float(K[1,2])-h/2)>1e-4:
        raise ValueError('CUDA adapter currently requires centered principal point')
    view=torch.eye(4,device=x.device,dtype=x.dtype)
    view[:3,:3]=camera['R'];view[:3,3]=camera['t'];view=view.T.contiguous()
    near,far=.01,1000.
    P=torch.zeros(4,4,device=x.device,dtype=x.dtype)
    P[0,0]=2*K[0,0]/w;P[1,1]=2*K[1,1]/h;P[2,2]=far/(far-near);P[2,3]=-far*near/(far-near);P[3,2]=1
    settings=GaussianRasterizationSettings(image_height=h,image_width=w,tanfovx=float(w/(2*K[0,0])),tanfovy=float(h/(2*K[1,1])),bg=x.new_zeros(3),scale_modifier=1.,viewmatrix=view,projmatrix=(view@P.T).contiguous(),sh_degree=0,campos=(-camera['R'].T@camera['t']).contiguous(),prefiltered=False,debug=False)
    rgb,_,depth=render_cuda(x,cov,color,opacity,settings)
    # Renderer returns weighted depth D and has no alpha output. White pass yields alpha.
    white=render_cuda(x,cov,torch.ones_like(color),opacity,settings)[0]
    alpha=white.mean(0)
    return {'rgb':rgb.permute(1,2,0),'alpha':alpha,'depth':depth.squeeze(0)/alpha.clamp_min(1e-8)}
