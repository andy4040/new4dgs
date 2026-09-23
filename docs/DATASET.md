# Actual coffee_martini acquisition (2026-09-23)

Downloaded from the original authors' release:
https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip

Official description and license: https://github.com/facebookresearch/Neural_3D_Video — CC-BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). This is the original synchronized video/calibration release, not a reprocessed single-frame/low-resolution mirror.

- External data root: `/workspace/datasets/coffee_martini`.
- Archive size: 1,186,324,684 bytes.
- Locally measured archive SHA256: `cbc31291ce143e6f31a00f23a049663a61cc8793af8dab259dbc6e338a676816`.
- ZIP member CRC integrity checks passed. No upstream cryptographic digest was supplied; the local hash records the acquired asset.
- Initial transfer took approximately 83 seconds. Integrity verification/extraction took 10.44 seconds. The compressed archive was removed after successful extraction to avoid retaining a second large copy.
- 18 videos: cam00, cam01, cam02, cam04, cam05, cam06, cam07, cam08, cam09, cam10, cam11, cam12, cam13, cam14, cam16, cam18, cam19, cam20.
- cam03/cam15/cam17 are absent. Official format explicitly maps calibration rows to **sorted existing video streams**, so missing numeric IDs are valid. The initial loader's overly strict contiguous-ID check was corrected, and a regression test covers that mapping.
- 300 frames per video, 30 FPS. Exact source dimensions/intrinsics and row mapping: `reports/coffee_inventory.json`.
- 17 training cameras after excluding cam00. No camera count was assumed before inspecting the release.
- 16 training frames × 17 cameras = 272 possible training observations. Spatial holdout: 16 images; temporal holdout: 85 images; joint holdout: 5 images.
- Frame indices 10–30 correspond to 1/3–1 second in the source video (zero-based indexing), a 2/3-second interval. Local ODE t=(frame−10)/20; world units/ODE-time multiplied by 1.5 become world units/video-second.

Read-only project access; raw videos are not copied into the Git repository or result backup. Dataset storage remains nonpersistent because this instance has no host volume. It can be reacquired with:

```bash
python scripts/download_n3dv.py --root /workspace/datasets
python run.py --data /workspace/datasets/coffee_martini --inspect --out runs/inventory_new
```
