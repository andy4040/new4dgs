# new4dgs — continuous Gaussian trajectories

coffee_martini의 Gaussian 중심을 `dμ/dt=vθ(μ,t)`로 이동시키는 연구용 구현입니다. **이전 코드를 복구한 것이 아니라 새로 구현했습니다.** 기존 파일/공개 Git 백업 검색에서 해당 프로젝트를 찾지 못했습니다. 출처·기준 commit·라이선스는 [docs/PROVENANCE.md](docs/PROVENANCE.md)에 기록했습니다.

**2026-09-23 업데이트: 공식 coffee_martini 배포본을 직접 다운로드했습니다.** 외부 경로는 `/workspace/datasets/coffee_martini`이며, 18개 카메라(학습 17개), 각 300프레임, 2704×2028, 30fps를 실제 파일에서 확인했습니다. [다운로드 출처·무결성·분할](docs/DATASET.md). 아래 합성 검증 결과와 실제 N3DV 결과는 구분합니다. 물질 대응은 입증하지 않았습니다.

## 현재 확인된 결과와 상태

- RTX 3090 24GB, host driver 570.172.08, CUDA 12.8, Python 3.12.14, Torch 2.7.1+cu128. 처음에는 Torch가 없어서 설치했습니다. 드라이버는 수정하지 않았습니다.
- 핵심 수치/누출 차단 **16개 테스트 통과**. 평행이동, 회전, 곡선 운동을 CPU/CUDA에서 검사했습니다. Euler 수렴, RK4, Gaussian OT, 불균일 개수의 Sinkhorn, RGB→Jacobian 역전파의 유한차분 검사도 포함합니다.
- CUDA rasterizer scale/rotation 입력과 full covariance 입력의 최대 이미지 차이 **1.043e-7**. CUDA RGB→ODE 속도장 gradient norm **10.9368**. `reports/cuda_validation.json`.
- 원본 canonical HexPlane 모델의 gradient/optimizer 검사 통과. 합성 fixture에서 baseline 전체 학습/평가 실행 완료.
- 직접 생성한 3개 카메라 fixture로 삼각측량, 서로 다른 endpoint 개수(128/160), RGB 최적화, FM, ODE RGB, 분리 평가, 저장/재로드 및 궤적 시각화 완료. 훈련/삼각측량 접근 감사에서 cam00·보류 프레임 누출 **0건**. 저장 checkpoint 재평가의 이미지 지표가 원 실행과 정확히 일치했습니다.
- 실제 CUDA backend를 사용하는 fixture 학습, 신뢰 영역 마스크를 사용하는 LK track 버전, 제한된 시간 외관 변화 버전도 실행했습니다. 실제 N3DV에서의 검증은 아닙니다.
- fixture endpoint L1은 약 **0.4315 / 0.4216**으로 기준 0.08을 통과하지 못했습니다. smoke 설정이므로 계속 실행했으며 모든 관련 결과는 **부정확한 endpoint 기반 기능 검사**입니다. 장기 연구 성능으로 해석하지 마세요.
- 최초 실제 N3DV 시도는 데이터가 없어 실패했고 기록을 유지합니다(`reports/n3dv_failure.json`). 이후 공식 데이터를 다운로드했습니다. 첫 실제 데이터 실행에서는 Sinkhorn 500회가 주변분포 오차 0.03125로 실패하여 5,000회 설정으로 수정했습니다. 실제 추적/외관 ablation 및 장기 학습은 아직 실행하지 않았습니다.

실제 coffee_martini에서도 필수 네 설정의 smoke 학습·평가를 완료했습니다. 96×72, endpoint 512/640개, endpoint당 68 step, 구간 RGB 16 step입니다. [전체 비교와 제한](reports/coffee_comparison.md).

| 실제 N3DV 설정 | 공간 시점 PSNR | 시간 보간 PSNR |
|---|---:|---:|
| RGB only | 9.7107 | 9.2254 |
| FM 초기화 후 RGB | 9.6941 | 9.1936 |
| FM 계속 유지 | 9.6332 | 9.1321 |
| canonical 4DGS architecture baseline | 14.0719 | 13.3014 |

**모든 설정의 초기 endpoint가 품질 기준을 통과하지 못했습니다**(L1 약 0.2860/0.2840, 기준 0.08). 이 값은 동작 검증 결과이며 수렴한 연구 성능·물질 대응·공정한 최종 순위가 아닙니다. N3DV 정답 3D 오차는 null이며 모든 실행에서 학습/삼각측량 보류 영상 누출은 0건입니다. 초기 크기를 이웃 점 간격으로 설정하고, 공분산 크기 상한을 camera extent의 0.1배로 명시했습니다. 5,000회 Sinkhorn의 실제 FM 초기화 주변분포 L1은 약 1.35e-6입니다. 나머지 세 smoke 작업은 한 GPU에서 동시에 실행했으므로 개별 wall time을 속도 비교에 사용하거나 GPU 시간을 합산하면 안 됩니다. 그룹 wall time과 다운로드/추출 시간은 별도로 기록했습니다.

실제 결과: `runs/coffee_fm_rgb_v2/`, `runs/coffee_rgb/`, `runs/coffee_fm_persistent/`, `runs/coffee_canonical_4dgs/`. FM 결과에는 평가 RGB 위 고정 ID 투영 경로(`projection_overlay_cam00.png`)와 기준점 대비 변위 그림(`displacements_3d.png`)도 있습니다.

알려진 초기 기하를 사용한 별도의 작은 합성 곡선 영상 실험(각 80 RGB step, Gaussian 6개)의 보류 시간 결과:

| 설정 | PSNR (dB) | 평균 3D 궤적 오차 (합성 좌표 단위) |
|---|---:|---:|
| RGB only | 20.3213 | 0.25676 |
| FM 초기화 후 RGB | 20.2853 | 0.24905 |
| FM 계속 유지 | 20.2210 | 0.23826 |

초기 기하는 oracle이며, **정답 속도/궤적을 학습 손실에 넣지는 않았습니다.** 정답 운동으로 영상을 생성하고 보류 시간의 3D 오차를 평가했습니다. 수치 구현의 analytic oracle 테스트와 구분됩니다. 짧은 최적화의 궤적 오차는 크며 곡선 복원 성공이나 어느 방법의 우월성을 입증하지 않습니다. 검은 배경 비율이 높아 PSNR만으로 대응을 판단할 수 없습니다. 약 29초의 세 variant 실행 시간에는 FM이 포함됩니다. N3DV에서 이 3D 오차 항목은 **null**입니다.

## 실행 환경 및 명령

```bash
cd /workspace/new4dgs
source /venv/main/bin/activate
python -m pytest -q
python scripts/check_cuda.py
python scripts/check_baseline.py
python synthetic.py --steps 80 --out runs/synthetic_new
```

새 인스턴스에서는 `bash scripts/setup.sh`로 pinned upstream과 환경을 설치합니다. 기본 GPU arch는 이 RTX 3090의 8.6입니다. 다른 GPU에서는 해당 arch를 설정하고 CUDA wheel 호환성을 먼저 확인하세요. `requirements-core-lock.txt`는 핵심 직접 의존성 버전, `reports/environment-lock.txt`는 이 인스턴스의 전체 설치 스냅샷입니다. 후자의 확장 패키지 로컬 경로는 setup.sh로 재구축합니다. 기존 형제 4DGaussians가 다른 commit이면 setup은 수정하지 않고 중단합니다. `upstream_path`로 별도의 pinned checkout을 지정할 수 있습니다.

현재 데이터는 이미 `/workspace/datasets/coffee_martini`에 있습니다. 새 인스턴스에서는 `python scripts/download_n3dv.py --root /workspace/datasets`로 공식 배포본을 받을 수 있습니다. **프로젝트 내부에 데이터를 복사하지 않습니다.**

```bash
python run.py --data /workspace/datasets/coffee_martini --inspect --out runs/inventory
python run.py --data /workspace/datasets/coffee_martini --config configs/coffee_smoke.json \
  --variant fm_rgb --renderer cuda --out runs/coffee_smoke
python evaluate.py --checkpoint runs/coffee_smoke/checkpoint.pt \
  --data /workspace/datasets/coffee_martini --out runs/coffee_reloaded
python scripts/run_matrix.py --data /workspace/datasets/coffee_martini \
  --config configs/coffee_smoke.json --out runs/coffee_matrix
```

`run_matrix.py`는 `rgb`, `fm_rgb`, `fm_persistent`, `canonical_4dgs`를 같은 설정/분할/해상도로 순차 실행하고 비교 CSV를 생성합니다. `--out` 기존 결과 덮어쓰기를 막습니다. 각 실행은 seed를 재설정하고 endpoint부터 다시 실행하여 전처리와 endpoint/FM 시간을 총시간에 포함합니다. `metrics.json`의 `allocated_gpu_hours`는 전체 실행 wall time 기준이며 순수 kernel 시간은 아닙니다. 실제 시간당 요금을 `gpu_hourly_usd`에 넣으면 추정 비용을 계산합니다. 요금이 없으면 비용은 null입니다. 환경 설치 비용은 별도 환경 준비 시간이며 per-run 총시간에 포함되지 않습니다.

GPU 연구 설정 초안은 `configs/research_unvalidated.json`입니다. **10k/12k Gaussian도 충분하다고 검증하지 않았고, 해당 장기 설정은 실행하지 않았습니다.** endpoint 품질 기준 실패 시 중단합니다. 실제 데이터의 endpoint RGB/깊이/공간 배치 확인, 속도장 공간 좌표 스케일 점검, step-size 수렴 및 VRAM 측정 후 설정을 결정해야 합니다. 밀집 Sinkhorn은 여전히 O(NM) 메모리/계산이며, W2 3×3 중간 행렬만 블록 처리합니다. 장기 학습용 sparse/multiscale transport는 아직 없습니다.

## 데이터와 좌표

외부 폴더는 `poses_bounds.npy`와 `camNN.mp4` 또는 `camNN/images/0000.png` 구조를 지원합니다. 공식 규칙대로 calibration row를 존재하는 camNN 파일의 정렬 순서에 연결합니다. 빠진 카메라 번호는 허용하며, 카메라 수와 calibration row 수가 다르거나 중복 이름이 있으면 오류를 냅니다. 실제 카메라 수를 확인하며 **17개를 하드코딩하지 않습니다**. 이미지 폴더는 0번 프레임 존재를 확인하며, 영상은 필요한 frame만 읽습니다. 이미지가 이미 downsample된 경우에도 intrinsics는 calibration의 원래 H/W로부터 목표 해상도로 조정합니다.

- frame index는 0 기반.
- 학습: **10,11,13,14,15,17,18,19,21,22,23,25,26,27,29,30**.
- 보류: **12,16,20,24,28**, 모든 카메라의 학습에서 제외.
- `train`: cam00을 제외한 카메라 × 학습 16시간.
- `view`: cam00 × 학습 16시간. 공간 시점 평가.
- `time`: 학습 카메라 × 보류 5시간. 시간 보간 평가.
- `joint`: cam00 × 보류 5시간. 두 조건 동시 평가.

모든 학습 영상 접근은 allowlist로 검사됩니다. 삼각측량·endpoint·LK 추적도 이 로더를 사용합니다. eval 함수가 명시적으로 평가 split을 요청할 때만 보류 영상을 읽습니다. 카메라 calibration metadata는 이미 알려진 보정값으로 사용하며 RGB 보류 이미지와 구분합니다. `training_access.json`, `triangulation_access.json`, `all_access.json`으로 접근을 감사할 수 있습니다.

원본 4DGS pose 변환을 재사용했고 recenter/장면 축척 변경은 하지 않습니다. 원본 loader의 `frame/300` 대신 이 구간의 ODE 시간은 **t=(frame−10)/20 ∈ [0,1]**입니다. video FPS가 F라면 영상 시간은 frame/F초, ODE t에 대한 속도를 실제 초당 속도로 바꿀 때는 F/20을 곱합니다. 이미지 전용 폴더의 FPS는 임의로 30이라고 가정하지 않습니다. FM의 `s∼U[0,1]`는 두 endpoint 샘플 간 보조 보간 변수이며 초기화 시 t=s로 학습합니다. FM s는 보류 영상 관측이 아닙니다.

## 모델과 학습 해석

`geometry.py`는 학습 카메라의 SIFT mutual ratio match를 calibrated triangulation합니다. 양의 깊이, 최소 parallax, reprojection 오차로 필터링하고 중복 spatial voxel을 제거합니다. 조건을 만족하는 점이 너무 적으면 random geometry로 대체하지 않고 실패합니다. 두 endpoint는 독립 복원하므로 같은 개수/순서를 가정하지 않습니다.

기준 frame 10의 `ids=0..N-1`와 `parent_ids=-1`을 checkpoint에 저장합니다. 현 구현은 증식/제거를 하지 않습니다. ID 유지 자체는 물질점 대응의 증명이 아닙니다. 초기 endpoint 위치 LR는 카메라 extent를 곱한 원본 exponential schedule이며 smoke용으로 끝까지 급속 decay하지 않습니다.

`transport.py`는 위치·시간 MLP, Euler/RK4, Gaussian W2, log-domain Sinkhorn 및 조건부 FM을 구현합니다. FM은 uniform probability mass를 사용하며 **renderer opacity와 별도**입니다. 공분산 기반 OT map으로 샘플을 대응시켜 factor 임의 회전에 민감한 shared-noise 보간을 피합니다. 주변분포 L1 오차가 설정 tolerance를 초과하면 실패합니다. 이 coupling은 물질 대응 정답이 아닙니다.

매 시간은 같은 `(μ0,L0)`에서 적분합니다. frame별 독립 위치를 연결하지 않습니다. `F'=J_v F`, `Σ=F L0 L0ᵀ Fᵀ+1e-8 I`이며 covariance RGB 경로에서 eigendecomposition을 쓰지 않습니다. `torch.func.jacrev/vmap`은 파라미터에 대한 고차 미분 그래프를 유지합니다. 비선형 flow의 Gaussian 근사는 정확하지 않으므로 실제 Gaussian 샘플을 같은 ODE로 운반하여 평균/공분산 차이를 기록합니다. 1,024 sample 기반 오차에는 Monte Carlo 잡음이 포함됩니다. 비선형 analytic field의 별도 테스트는 실제 Gaussian 근사가 틀어지는 것도 확인합니다.

RGB 학습에서는 고정된 endpoint 기준 기하와 속도장을 사용합니다. 중심·공분산은 ODE로 이동하고 기준 SH0(상수 RGB 계수에 해당)와 opacity는 학습 가능하되 시간에 고정합니다. 임의 시간의 RGB reconstruction, 약한 material acceleration 제약을 사용합니다. 기본 계수: acceleration `1e-4`, terminal Chamfer `0`, continued FM `0.01`, tracking `0`. frame 30 분포 제약을 켜더라도 실제 대응 목표로 부르지 않습니다. `fm_rgb`는 초기화 이후 FM을 제거하며 `fm_persistent`만 계속 사용합니다. 실제 학습 횟수/계수는 각 `config.json`에 남습니다.

`covariance_renderer.py`는 작은 검증용 PyTorch splatter와 원본 CUDA rasterizer를 선택 지원합니다. 같은 공분산·pinhole 투영·depth alpha compositing을 사용하지만 픽셀 중심/가시성/cutoff 때문에 서로 완전히 같은 렌더러는 아닙니다. 비교 실험에서는 renderer를 통일하세요. CUDA의 activated scale/rotation convention과 full covariance packing은 실제 실행 검증했습니다. SH degree는 첫 비교에서 0입니다. higher-order SH 및 재질/조명 모델은 구현하지 않았습니다.

## 추적 및 외관 ablation

OpenCV LK를 사용하므로 별도의 pretrained 가중치를 가정하지 않습니다. 실제 액체/반사 영역은 자동으로 신뢰 판정하지 않습니다. 사용자가 제공한 **신뢰 영역만 흰색인 camNN.png 마스크**가 있어야 tracking을 켤 수 있습니다.

```bash
python run.py --data /workspace/datasets/coffee_martini --variant rgb \
  --track-masks /external/reliable_masks --track-weight 0.01 --out runs/coffee_tracks
python run.py --data /workspace/datasets/coffee_martini --variant fm_rgb \
  --appearance-rank 1 --out runs/coffee_time_appearance
```

frame 10 feature를 3px 이내의 Gaussian 투영과 연결하고 렌더 깊이·alpha로 초기 가시성을 검사합니다. 학습 시간만 LK로 추적하고 forward/backward 0.75px 검사, 영상 영역·신뢰 마스크, 최소 2개 학습 시점의 triangulation reprojection 2px 검사를 적용합니다. 겹침/가림과 calibration 오류를 완전히 해결하지는 못합니다. Gaussian-to-feature 초기 association도 근사입니다. 유효 track이 없으면 tracking 실험은 실패합니다. 추적 오차는 학습용 2D 일치도이며 독립 평가나 정답 3D 오차로 보고하지 않습니다.

시간 외관 변화는 rank 1/2 polynomial basis를 사용하며 logit 변화를 ±0.1로 제한합니다. 계수 L2 weight는 0.01입니다. fixed appearance와 별도 실험입니다. 실험의 움직임 오류를 외관 자유도로 숨길 수 있다는 한계는 남습니다.

## 산출물·한계·다음 순서

각 실행 폴더: 설정/카메라 inventory, endpoint RGB/target/alpha/depth, endpoint 품질 JSON, persistent ID checkpoint, Sinkhorn 진단, 학습 loss, 분리된 이미지/분포 지표, sample transport 오차, 접근 감사, 3D 궤적 PNG와 카메라 투영 PNG, ID/색/좌표 NPZ. 모든 궤적 그림은 ID별 색이 고정됩니다. baseline 궤적은 canonical deformation 출력이며 ODE라고 부르지 않습니다.

다음 단계는 실제 데이터의 endpoint RGB/깊이 검토 → 짧은 네 방법 비교 → 적분 step 수렴/geometry 수 증가 → 신뢰 마스크가 있을 때 tracking 비교 → 제한된 외관 ablation입니다. 이 구현의 수치 동작을 확인한 것과 실제 물질 궤적 복원을 입증한 것은 별개입니다. 실제 N3DV 정답 3D 궤적은 없고 추적 없는 RGB만으로 물질 대응을 단정할 수 없습니다.

## 백업

독립 Git 저장소는 `/workspace/new4dgs`입니다. 데이터/checkpoint/render/비밀정보는 Git에서 제외했습니다. **원격 저장소 `https://github.com/andy4040/new4dgs`를 확인하고 origin으로 연결했습니다.** 원격의 초기 README commit도 로컬 이력과 병합했습니다. 현재 이 Vast에는 HTTPS/SSH GitHub 인증이 없어 push가 실패했습니다. **원격 코드 백업은 아직 미완료**이며, 인증 후 `git push -u origin main`이 필요합니다.

**이 `/workspace`는 영구 볼륨이 아닙니다. 로컬 commit, git bundle, 다운로드 묶음 모두 이 인스턴스를 삭제하면 함께 사라집니다.** 별도 영구 저장 위치도 제공되지 않았습니다. `scripts/export_backup.py`가 `/workspace/new4dgs-backup.tar.gz`와 SHA256을 만듭니다. 사용자 PC로 반드시 내려받거나 영구 저장소로 복사해야 합니다.

```bash
python scripts/export_backup.py --out /workspace/new4dgs-backup.tar.gz
# 사용자 PC에서 실행 (이 인스턴스의 현재 SSH endpoint):
scp -P 51913 root@218.150.198.117:/workspace/new4dgs-backup.tar.gz .
```

묶음은 Git source archive, 전체 로컬 Git history bundle, 보존할 실행 결과/체크포인트/궤적을 포함합니다. 원본 데이터와 생성 fixture 영상은 포함하지 않습니다. 보존 목록: `docs/PRESERVE.md`. 원격 push 성공 여부와 별도 artifact 외부 보존 여부를 항상 따로 확인하세요.
