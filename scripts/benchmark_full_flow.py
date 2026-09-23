import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transport import SceneVelocity,chunked_flow
from geometry import Gaussians
from data import N3DV
from covariance_renderer import render
s=torch.load('runs/coffee_keyframe_30db_v2/best.pt',map_location='cuda',weights_only=False);r=s['reference'];cfg=s['config'];data=N3DV(cfg)
g=Gaussians(r['xyz'],r['color_logits'].sigmoid(),1.).cuda();g.load_state_dict(r);g.requires_grad_(False)
v=SceneVelocity(g.xyz.median(0).values,data.extent,width=64,depth=3).cuda();torch.set_num_threads(4)
result=[]
for step in [.25,.125]:
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    x,c=chunked_flow(v,g.xyz,g.factor(),1.,step=step,chunk_size=8192)
    im=render(x,c,g.color(),g.opacity(),data.camera(data.train_cameras[0],'cuda'))['rgb'];loss=im.mean();loss.backward();torch.cuda.synchronize()
    result.append(dict(gaussians=len(g.xyz),rk4_step=step,seconds=time.perf_counter()-start,torch_peak_bytes=torch.cuda.max_memory_allocated(),grad_norm=sum(float(p.grad.norm()) for p in v.parameters() if p.grad is not None)))
    print(json.dumps(result[-1]),flush=True);v.zero_grad();del x,c,im,loss
Path('reports/full_flow_benchmark.json').write_text(json.dumps({'measurements':result,'note':'Scalability only; frame-10 reference, untrained velocity; concurrent keyframe optimization on same GPU affects timings.'},indent=2))
