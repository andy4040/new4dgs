# Origin and recovery record

No previous new4dgs or 4DGaussians checkout, endpoint checkpoint, or coffee_martini dataset was found in this instance during initial inspection (2026-09-23). This is a new implementation, **not a recovered implementation**. Initial `/workspace` contained only environment folders and guide links. Searches included `/workspace`, `/home`, `/root`, `/opt`, `/tmp`, and filesystem names down to six levels. A personal computer's Desktop is not mounted here.

The public GitHub API for `andy4040` returned `e2e_parking_mamba` and `To-do-list`; neither is the requested project. Private repositories were not accessible/inspected. No remote was selected or created. No push has occurred.

Upstream: https://github.com/hustvl/4DGaussians at **843d5ac636c37e4b611242287754f3d4ed150144**, cloned into a separate sibling `/workspace/4DGaussians`. Exact submodule URLs/commits are in `upstream-lock.json`. GLM nested commit: `5c46b9c07008ae65cb81ab79cd677ecc1934b903`. No tracked upstream source file was edited.

Reused/adapted conventions:

- `scene/neural_3D_dataset_NDC.py`: LLFF `poses_bounds.npy`, pose column permutation `[col1,-col0,col2,translation]`, subsequent camera y/z flip, transposed stored R and `T=-center @ R`. Our loader adapts this to strict frame/camera access, actual calibration H/W/focal, read-only videos/images and local normalized time. It does not call upstream's all-frame image extractor.
- `scene/gaussian_model.py`, `arguments/__init__.py`, `utils/general_utils.py`: endpoint position LR initial `1.6e-4`, final `1.6e-6`, exponential decay over **20,000** steps, both multiplied by training-camera extent. Upstream passes `lr_delay_mult=.01` but default `lr_delay_steps=0`, so that multiplier has no effect; we match that behavior.
- Upstream covariance convention: activated scale is standard deviation, rotation is normalized wxyz, `L=R diag(scale)`, covariance packed xx,xy,xz,yy,yz,zz. Our Cholesky-like factor and ODE deformation gradient use full covariance directly in rasterization.
- `baseline.py` imports upstream GaussianModel/HexPlane deformation directly; it shares our strict loader and render backend. Fixed Gaussian count, SH0, same endpoint initialization, no opacity/SH deformation, and no densification are deliberate controlled-protocol choices. It is a canonical 4DGS **architecture baseline**, not reproduction of the published full training recipe or paper numbers.

Licenses: upstream top-level LICENSE.md is Apache-2.0 (copy `UPSTREAM_APACHE_LICENSE.txt`). Several upstream files retain Inria/GRAPHDECO non-commercial research/evaluation notices. Rasterizer LICENSE.md is the separate research license copied into `RASTERIZER_LICENSE.txt`; simple-knn sources also contain Inria research-use notices and the pinned repository has no separate tracked LICENSE.md. Do not infer that the whole dependency stack is unrestricted Apache-2.0. Retain the original notices and consult the original maintainers for other uses. New code has not been assigned a blanket license covering upstream dependencies.

CUDA 12.8 compilation needed implicit standard headers made explicit through `NVCC_PREPEND_FLAGS='-include cfloat -include cstdint'`: first FLT_MAX and then uint32_t/uintptr_t were missing. This changes build flags, not source. Initial failed build logs and final success logs are retained. Python 3.12 / Torch 2.7.1+cu128 differs from upstream's old Torch 1.13.1 requirements; tested integration boundaries are listed in the README.


Follow-up acquisition: after the user authorized obtaining the data, the official coffee_martini release was downloaded directly. See DATASET.md. The initial failed discovery remains historical, not a current absence claim. Real data exposed an overly strict contiguous camera-ID check; it now follows the official sorted-existing-stream rule and has a regression test. Actual-scene smoke configuration also uses RMS 3-nearest-neighbor scale initialization (with explicit 0.1×camera-extent cap) after fixed small scales left most pixels empty, and 5,000 Sinkhorn iterations after the initial 500-iteration solve failed its unchanged tolerance.
