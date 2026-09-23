# Frame 10 extended reconstruction

Completed 12,000 fixed Gaussians, 10,000 Adam iterations at 384x288.
Initialization: 16,736 accepted pair triangulations at width 1280; selected
12,000 after spatial deduplication. Same N3DV coordinates and calibration.
17 training cameras, frame 10 only. Audits confirm no cam00 or held-out-time
images used in triangulation or training. cam00 evaluated after training.

Both saved endpoints evaluated at the SAME 384x288 resolution:

| Model | Train PSNR | cam00 PSNR | Train L1 | cam00 L1 |
|---|---:|---:|---:|---:|
| Old 512 points / 68 steps | 8.932 | 9.418 | .29458 | .27888 |
| New 12000 points / 10000 steps | 25.548 | 25.632 | .02998 | .03305 |

Train PSNR at 1000/5000/9000/10000: 21.772/24.885/25.516/25.548.
Final 1000 steps improved only .032 dB; further iterations alone have
uncertain benefit. Preprocessing + initialization + training + intermediate
outputs + final evaluation: 110.02 seconds on one RTX 3090 (~.03056 GPU hours).
Separate post-run old/new comparison and tests are not included in that time.
Hourly rental rate unknown; dollar cost unmeasured.

Clear overall image improvement, but hands, glass, thin structures and background
textures remain blurred. No ground-truth depth assessment or material trajectory
claim. This experiment increases points, iterations, resolutions and changes the
XYZ schedule duration together; it does not isolate individual causes.
No densification, pruning, time-varying appearance, frame-30 training or ODE
retraining performed. Next: consider density adaptation and better surface seeds
for residual detail before transporting this reference set.

Artifacts in `runs/coffee_keyframe_10k`: `latest.pt` (reference + Adam state),
config, initialization, loss/progress metrics, access audits, learning curve,
intermediate target/render pairs, final alpha/depth, comparison_cam00.png and
comparison_cam01.png (left: target; middle: old; right: new).
Code/config/report backed up by Git push; artifacts require external download.
Core tests: 16 passed. Full experiment completed without nonfinite loss.
