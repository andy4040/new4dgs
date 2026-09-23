# Frame-10 30 dB target: achieved

Target definition: arithmetic mean of per-camera PSNR over the 17 training
cameras, frame 10, unchanged 384x288 resolution. Goal checked every 1000 updates;
cam00 and held-out times are excluded from training, stopping and selection.

| Metric | Previous 12000 / 10000 updates | New 97884 / 21000 cumulative updates |
|---|---:|---:|
| Training mean PSNR | 25.548 | 30.247 |
| cam00 PSNR (final test only) | 25.632 | 27.983 |
| Training L1 | .029984 | .016723 |
| cam00 L1 | .033054 | .024889 |
| Training SSIM (uniform7) | .826706 | .920562 |
| cam00 SSIM (uniform7) | .846931 | .915178 |

Stopped automatically at 11000 additional updates, reason `target_psnr`.
10/17 training cameras individually exceed 30 dB, range 26.270–33.384 dB.
The target is a mean, NOT 30 dB on every view and NOT 30 dB on cam00.
Mean-camera MSE is .00106261; PSNR computed from pooled MSE differs from mean PSNR.

Resumed previous optimized geometry; used MSE + .1 L1, lower shape LR .001,
color/opacity LR .003, extent-scaled XYZ LR .00004 -> .0000016 over 30000 updates.
Split 30% of points with greatest accumulated position-gradient magnitude at
1000-update intervals through update 8000. Explicit parent-child lineage is
saved. No arbitrary new surface points or time-varying appearance were introduced.
Point count is fixed after this geometry phase for subsequent trajectory work.

First attempt failed at split 1 due to float32 Cholesky on anisotropic covariance.
The fix computes factor products/decomposition in float64 plus relative jitter;
19 tests pass including this anisotropy case. Successful continuation took
171.60 seconds including image loading, optimization, densification, monitoring
and artifact saving. Previous reconstruction took 110.02 seconds. Failed-attempt,
post-run comparison and verification overhead are additional; full failed-attempt
duration was not captured, so total project GPU cost is not asserted. Rental
rate unknown. Reported peak PyTorch memory excludes native rasterizer/context.

Automatic stopping also implemented in endpoint RGB, optional FM, ODE RGB and
canonical baseline training. Endpoints monitor all training cameras; trajectories
monitor the full training camera/time grid, not randomly sampled image losses.
FM monitoring uses fixed random samples. Defaults: every 1000 steps, minimum
2000 steps, 5 checks without cumulative improvement >.0001 in monitored loss.
Best state is retained and restored. Maximum-step exhaustion is reported separately.
The target keyframe uses MSE min_delta=.000002, minimum 12000 updates, patience 5;
reaching the explicit target takes precedence over that minimum.

Synthetic control-flow checks intentionally set min_delta=1 and patience=2:
endpoint/FM/ODE/baseline all stop at update 3. ODE final state exactly equals its
saved best checkpoint. This validates stopping mechanics, not real-scene
convergence of those dynamic models. New real-data motion training was not run.

Artifacts: `runs/coffee_keyframe_30db_v2/best.pt` (model/optimizer/config),
`best_loss.pt`, `latest.pt`, `lineage.json`, training and final comparison audits,
loss/progress logs, depth/alpha, learning_curve.png and comparison_cam00.png
(left target / center prior 12000-point result / right new result).
cam00 comparison was evaluated only after the training process completed.

Visual detail improved substantially, but residual artifacts remain around glass,
hands and occlusion boundaries. No ground-truth depth or material correspondence
is established. Code/config/reports go to GitHub; checkpoint/render archive still
requires an off-instance download or a user-provided persistent destination.
