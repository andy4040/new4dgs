"""Evaluate saved old/new frame-10 models at the same resolution after training."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from data import N3DV
from geometry import Gaussians
from covariance_renderer import render
from evaluation import image_metrics,save_image
root=Path(__file__).resolve().parents[1];out=root/'runs/coffee_keyframe_10k'
cfg=json.loads((out/'config.json').read_text());data=N3DV(cfg);torch.set_num_threads(4)
models={}
for key,path in [('old',root/'runs/coffee_fm_rgb_v2/endpoint_10.pt'),('new',out/'latest.pt')]:
    s=torch.load(path,map_location='cuda',weights_only=False)
    if key=='new':s=s['reference']
    g=Gaussians(s['xyz'],s['color_logits'].sigmoid(),1.).cuda();g.load_state_dict(s);models[key]=g
rows=[]
with torch.no_grad():
    for n in data.names:
        split='view' if n=='cam00' else 'train';target=data.image(n,10,split).cuda();imgs=[target]
        for key,g in models.items():
            r=render(g.xyz,g.covariance(),g.color(),g.opacity(),data.camera(n,'cuda'))
            rows.append(dict(model=key,camera=n,split=split,**image_metrics(r['rgb'],target)));imgs.append(r['rgb'])
        if n in ['cam00','cam01','cam02']:save_image(torch.cat(imgs,dim=1),out/f'comparison_{n}.png')
summary={key:{split:{metric:float(np.mean([r[metric] for r in rows if r['model']==key and r['split']==split])) for metric in ['psnr','l1','ssim_uniform7']} for split in ['train','view']} for key in models}
result=dict(image_width=cfg['image_width'],frame=10,summary=summary,per_camera=rows,note='Target / old 512-point 68-step endpoint / new 12000-point 10000-step endpoint. Static keyframe only; no trajectory evaluation.')
(out/'comparison.json').write_text(json.dumps(result,indent=2));data.save_audit(out/'comparison_access.json');print(json.dumps(summary,indent=2))
