# 보존할 파일

인스턴스를 삭제하기 전에 아래를 외부로 복사하세요. `/workspace` 자체는 영구 저장소가 아닙니다.

1. `new4dgs` Git remote에 push한 모든 commit. 현재 URL 미제공이므로 **미완료**.
2. `/workspace/new4dgs-backup.tar.gz` 및 `.sha256`. source.tar(코드/설정/문서/환경명세), new4dgs.git.bundle(모든 로컬 commit), MANIFEST.json(결과별 SHA256), results/ 포함. 로컬 생성만으로 외부 백업 완료가 되지 않습니다.
3. 각 실제 연구 run의 `checkpoint.pt`, `endpoint_10.pt`, `endpoint_30.pt` 또는 `baseline.pt`; `config.json`, `inventory.json`, `metrics.json`, `endpoints.json`, `losses.json`, `sinkhorn.json`, `tracks.json`, `*_access.json`.
4. endpoint RGB/target/alpha/depth, trajectories.npz(좌표/ID/색), trajectories_3d.png, projection_*.png 및 평가 render.
5. 외부 원본 coffee_martini 데이터 및 calibration. 현재 인스턴스에 없으며 묶음에 포함하지 않습니다. 생성 fixture 영상은 `scripts/make_fixture.py`로 재생성하므로 묶음에서 제외합니다.
6. `docs/upstream-lock.json`과 upstream 라이선스. 원본 의존성 코드는 지정된 공개 원격과 commit으로 다시 받을 수 있습니다.

Git 복구 예:

```bash
mkdir recovered && cd recovered
tar -xzf ../new4dgs-backup.tar.gz
git clone new4dgs.git.bundle new4dgs
# 결과를 새 프로젝트로 옮기려면:
cp -a results new4dgs/runs
```

tar 내부 manifest의 commit과 `git rev-parse HEAD`를 비교하세요. 실제 커피 데이터가 준비된 뒤의 추가 run은 exporter의 selected 목록에 명시적으로 추가하거나 별도 보존해야 합니다. secrets/.env/SSH 키는 묶음에 넣지 않습니다.
