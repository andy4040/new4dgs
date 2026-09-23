"""Summarize measured official-data smoke results, including failed endpoints."""
import datetime
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
folders={'rgb':'coffee_rgb','fm_rgb':'coffee_fm_rgb_v2','fm_persistent':'coffee_fm_persistent','canonical_4dgs':'coffee_canonical_4dgs'}
rows=[]
for variant,folder in folders.items():
 run=root/'runs'/folder
 assert json.loads((run/'status.json').read_text())['status']=='completed'
 m=json.loads((run/'metrics.json').read_text());endpoints=json.loads((run/'endpoints.json').read_text())
 accesses=json.loads((run/'training_access.json').read_text())['accesses']+json.loads((run/'triangulation_access.json').read_text())['accesses']
 leaks=[a for a in accesses if a['camera']=='cam00' or a['frame'] in [12,16,20,24,28]]
 assert not leaks,(variant,leaks)
 rows.append({'variant':variant,'run':str(run.relative_to(root)),'image':m['image'],'distribution':m['distribution'],
   'trajectory_3d_error':None,'material_correspondence_proven':False,
   'endpoint_l1':[e['train_l1'] for e in endpoints],'endpoint_rgb_gate_failed':not all(e['passes_rgb_gate'] for e in endpoints),
   'endpoint_counts':[e['selected'] for e in endpoints],'training_access_leaks':0,
   'observed_training_frames':sorted({a['frame'] for a in accesses}),
   'total_seconds_including_preprocessing':m['total_seconds_including_preprocessing'],'stage_seconds':m['timing_seconds'],
   'peak_torch_allocated_gpu_bytes':m['peak_gpu_bytes'],
   'gaussian_approximation':m.get('gaussian_approximation'),
   'sinkhorn':json.loads((run/'sinkhorn.json').read_text()) if (run/'sinkhorn.json').exists() else None})
group=json.loads((root/'reports/coffee_followup_group.json').read_text())
last=max((root/'runs'/folders[v]/'status.json').stat().st_mtime for v in group['variants'])
end=datetime.datetime.fromtimestamp(last,datetime.timezone.utc)
start=datetime.datetime.fromisoformat(group['started_utc'])
group.update(finished_utc=end.isoformat(),wall_seconds=(end-start).total_seconds(),allocated_gpu_hours=(end-start).total_seconds()/3600,estimated_usd=None)
(root/'reports/coffee_followup_group.json').write_text(json.dumps(group,indent=2)+'\n')
report={'dataset':'official coffee_martini','config':'configs/coffee_smoke.json','scope':'functional smoke; inaccurate endpoints; no material-correspondence claim',
        'results':rows,'concurrent_group':group,'timing_comparison_valid':False,
        'shared_data_acquisition':json.loads((root/'reports/data_download.json').read_text()),
        'first_failed_run':json.loads((root/'runs/coffee_fm_rgb/failure.json').read_text())}
(root/'reports/coffee_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Actual coffee_martini smoke comparison','',
'All runs used the same 96×72 resolution, 512/640 endpoint Gaussian budgets, 68 endpoint steps, 16 RGB steps, fixed SH0/opacity over time and identical frame/camera split. FM variants used 20 initialization steps; persistent FM coefficient 0.01. The canonical architecture baseline uses the shared CUDA renderer and reduced smoke grid. These are not converged research runs.','',
'| Variant | View PSNR | Time PSNR | Joint PSNR | Endpoint squared Chamfer | Endpoint gate failed |',
'|---|---:|---:|---:|---:|---|']
for r in rows:
 im=r['image'];dist=r['distribution']['endpoint_symmetric_chamfer_squared']
 lines.append(f"| {r['variant']} | {im['view']['psnr']:.4f} | {im['time']['psnr']:.4f} | {im['joint']['psnr']:.4f} | {dist:.5f} | {r['endpoint_rgb_gate_failed']} |")
lines+=['','PSNR is in dB. Chamfer is world-coordinate units squared against an independently reconstructed endpoint distribution, not ground-truth trajectory error. The numeric 3D trajectory error is unavailable (null). No withheld training image accesses occurred.','',
'All endpoint RGB gates failed the L1≤0.08 threshold. Main 512/640 initialization endpoints had L1 approximately 0.2860/0.2840. Figures remain blurred and incomplete; these results only validate execution and expose configuration limitations.','',
'Initial 128/160-point run failed Sinkhorn at 500 iterations (marginal L1 0.03125). The revised configuration used 5,000 iterations and unchanged tolerance 0.002. Endpoint initialization changed from tiny extent-based scales to nearest-neighbor scales with an explicit scene-extent cap.','',
'Per-run stage timing includes triangulation, endpoint fitting, FM, RGB fitting and evaluation. RGB/no-FM, persistent-FM and baseline jobs ran concurrently on one GPU. Their wall times include contention; do not use them as method speed comparisons or sum their GPU hours. Group allocation time is recorded once in coffee_comparison.json. Initial download and extraction are reported separately. The GPU hourly price is unknown, so cost remains null.','',
'Next: reconstruct and visually validate denser endpoints at higher image resolution; address world-coordinate normalization and background/foreground coverage, then extend RGB training and check ODE step convergence before evaluating trajectory quality. Real N3DV tracking and temporal appearance ablations have not been run.']
(root/'reports/coffee_comparison.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[4:10]))
