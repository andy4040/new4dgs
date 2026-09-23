# Actual coffee_martini smoke comparison

All runs used the same 96×72 resolution, 512/640 endpoint Gaussian budgets, 68 endpoint steps, 16 RGB steps, fixed SH0/opacity over time and identical frame/camera split. FM variants used 20 initialization steps; persistent FM coefficient 0.01. The canonical architecture baseline uses the shared CUDA renderer and reduced smoke grid. These are not converged research runs.

| Variant | View PSNR | Time PSNR | Joint PSNR | Endpoint squared Chamfer | Endpoint gate failed |
|---|---:|---:|---:|---:|---|
| rgb | 9.7107 | 9.2254 | 9.7137 | 7.79875 | True |
| fm_rgb | 9.6941 | 9.1936 | 9.6971 | 7.67998 | True |
| fm_persistent | 9.6332 | 9.1321 | 9.6364 | 7.82006 | True |
| canonical_4dgs | 14.0719 | 13.3014 | 14.0669 | 7.81334 | True |

PSNR is in dB. Chamfer is world-coordinate units squared against an independently reconstructed endpoint distribution, not ground-truth trajectory error. The numeric 3D trajectory error is unavailable (null). No withheld training image accesses occurred.

All endpoint RGB gates failed the L1≤0.08 threshold. Main 512/640 initialization endpoints had L1 approximately 0.2860/0.2840. Figures remain blurred and incomplete; these results only validate execution and expose configuration limitations.

Initial 128/160-point run failed Sinkhorn at 500 iterations (marginal L1 0.03125). The revised configuration used 5,000 iterations and unchanged tolerance 0.002. Endpoint initialization changed from tiny extent-based scales to nearest-neighbor scales with an explicit scene-extent cap.

Per-run stage timing includes triangulation, endpoint fitting, FM, RGB fitting and evaluation. RGB/no-FM, persistent-FM and baseline jobs ran concurrently on one GPU. Their wall times include contention; do not use them as method speed comparisons or sum their GPU hours. Group allocation time is recorded once in coffee_comparison.json. Initial download and extraction are reported separately. The GPU hourly price is unknown, so cost remains null.

Next: reconstruct and visually validate denser endpoints at higher image resolution; address world-coordinate normalization and background/foreground coverage, then extend RGB training and check ODE step convergence before evaluating trajectory quality. Real N3DV tracking and temporal appearance ablations have not been run.
