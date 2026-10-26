# 파지·인식 실험 도구 가이드

파지 좌표, 카메라 인식, 충돌 형상과 물리 동작을 단계별로 확인하는 도구다.
작업 지시와 카메라 영상으로 집기·놓기를 수행하려면 [Auto Picking 실행 가이드](../docs/known-object-pick-place.md)를 따른다.

## 실행 준비

프로젝트의 Pixi 환경을 구성한 뒤 프로젝트 루트에서 실행한다.
목록 조회·결과 조회·실행 미리보기는 GPU, 카메라, 추론 서버를 사용하지 않는다.

| 경로 | 용도 |
|---|---|
| `tiptop_outputs/experiments/grasping/20261020/` | 기본 입력. `results.json`, `plans/`, `runs/` 등 실험 자료 |
| `tiptop_outputs/experiments/work/` | 새 분석·계획·실행 결과의 기본 출력 위치 |
| `tiptop/config/known_objects.json`의 `cad_root`·`db_root` 및 로봇 자산 | 현재 도구가 참조하는 외부 CAD·DB·USD와 로봇 URDF 등 |

**실험 자료와 CAD·DB는 Git에 포함되지 않는다.** 다른 환경에서는 사용할 명령의 입력 자료를 별도로 준비해야 한다.
일부 계획·진단 도구는 `tiptop_outputs/m1013/tamp/diagnosis-wide/`의 `candidate-ik.json`, `initial.npz`도 참조한다.
입력 JSON에 기록된 CAD·DB·원본 파일 경로에 접근할 수 있어야 하며, 누락 자료를 자동으로 내려받지 않는다.
현재 물체 데이터 경로는 [실행 가이드](../docs/known-object-pick-place.md#물체-데이터)에 있다.

`--data-root`는 읽을 실험 자료, `--work-dir`는 새 결과 저장 위치를 지정한다.
두 옵션은 하위 명령 앞에 둔다. 기본값과 다른 폴더를 쓸 때의 예시는 다음과 같다.

```bash
pixi run python -m experiments \
  --data-root tiptop_outputs/experiments/grasping/20261020 \
  --work-dir tiptop_outputs/experiments/my-analysis \
  tool check-db --execute
```

분석·계획 도구는 같은 출력 폴더를 다시 사용하면 같은 이름의 결과를 덮어쓸 수 있다.
비교할 결과는 출력 폴더를 나눠 저장한다. 입력 폴더 내부를 작업 출력 위치로 지정할 수 없다.

## 도구 목록

```bash
pixi run python -m experiments list
```

| 명령 | 용도 |
|---|---|
| `results` | 저장된 실험 결과와 실패 원인 조회 |
| `run` | 저장된 입력·관절 경로로 물리 실험 재현 |
| `perception view` | 저장된 실제 영상과 DB 파지 후보를 HTML로 비교 |
| `perception observe` | 현재 가상 카메라에서 물체 자세·DB 후보 연결 확인 |
| `tool` | 좌표·충돌·계획 검사 및 결과 집계 |

## 실험 결과 조회

```bash
# 실패한 실험만 조회
pixi run python -m experiments results --failures-only

# 후추통 결과 조회
pixi run python -m experiments results --object pepper
```

입력은 `--data-root` 아래의 `results.json`이다. 파일을 수정하지 않고 터미널에 결과를 표시한다.
`PASS`는 해당 실험의 평가 기준 통과를 의미한다. 들기만 시험한 기록과 전체 집기·놓기 기록을 구분해서 집계한다.
원본의 `expected_place`는 놓기 포함 여부, `validated`는 기준 통과 여부다.
이 자료는 물리 실험 결과이며 작업 지시·인식·계획 전체 과정이나 실물의 성공률을 뜻하지 않는다.

## 물리 실험 재현

| `--case` | 대상 |
|---|---|
| `pepper` | 후추통 |
| `lotion` | 컨디셔너 병 |
| `dipper` | 미키 바가지 |

먼저 실행 명령과 입출력 경로를 확인한다. 이 단계에서는 Isaac을 시작하지 않는다.

```bash
pixi run python -m experiments run --case pepper
```

실제로 실행하려면 `--execute`와 새 출력 폴더를 지정한다.

```bash
pixi run python -m experiments run --case pepper \
  --output tiptop_outputs/experiments/manual-pepper-01 --execute
```

Isaac GUI에서 접근→닫기→들기→운반→내려놓기→열기→후퇴를 실행하고, 마지막 장면을 저장한 뒤 종료한다.
창 없이 실행하려면 `--headless`, Isaac 설치 위치를 바꾸려면 `--isaac-python /경로/python.sh`를 추가한다.
출력은 프로젝트 `tiptop_outputs/` 내부의 존재하지 않는 새 폴더여야 한다.

출력의 `check.json`에서 `success`, `physical_grasp_verified`, `physical_placement_verified`가 모두 `true`인지 확인한다.
다중 물체 사례는 `multi_object_verified`도 확인한다. 실패 단계와 원인은 `phase`, `error`, `trace`에 기록된다.
**이 명령은 기록된 경로의 물리 재현용이다.** 새 관측에서 경로를 계산하는 본체 실행과 구분한다.

## 인식 진단

### 저장 영상과 파지 DB 비교

입력 데이터셋에는 해당 장면의 RGB·깊이·마스크·카메라 설정·물체 정보·저장된 FoundationPose 결과와 CAD 겹침 영상,
그리고 merged/raw 파지 DB가 필요하다. 폴더 기준은 `review/real_v1/`과 `isaacsim_grasp_data_gen/`이다.
아래 경로와 장면·프레임은 보유 데이터에 맞춰 지정한다.

```bash
pixi run python -m experiments perception view -- \
  --dataset-root /media/uon/data1/Dataset_2026 \
  --scene Home/Uon_1F_Home/white_table --camera top_view_camera \
  --frame 0000 --object black_pepper_shaker \
  --output /tmp/tiptop-grasp-viewer.html
xdg-open /tmp/tiptop-grasp-viewer.html
```

브라우저에서 후보를 바꾸며 원점·접근축·손가락 접촉점을 비교한다. 새 추론이나 로봇 구동은 하지 않는다.
DB 접촉점은 해당 영상에서 실측한 접촉점이 아니며, 표시된 원점도 보정된 실물 TCP로 확정된 것은 아니다.

### 현재 카메라의 자세 추정·DB 연결 확인

먼저 [Auto Picking 실행 가이드](../docs/known-object-pick-place.md)의 추론 서버와 Isaac을 준비한다.
`--active-task`에는 본체 인식에서 생성한 `task/semantic-task.json`을 넣는다.
로봇을 움직이지 않고 이 파일을 준비하려면 본체 작업 명령에서 `--execute`를 생략한다.
아래는 본체 가이드의 출력 경로 예시이며, 실제 생성한 파일 경로로 바꿔 사용한다.

```bash
# 입력 파일과 로컬 자산만 확인. 카메라·추론 서버에는 접속하지 않는다.
pixi run python -m experiments perception observe -- \
  --active-task tiptop_outputs/m1013/isaac/manual-pick-place-01/task/semantic-task.json \
  --class-id black_pepper_shaker --check-inputs
```

입력 검사 결과가 `success=true`이면 실제 관측 진단을 실행한다.

```bash
pixi run python -m experiments perception observe -- \
  --active-task tiptop_outputs/m1013/isaac/manual-pick-place-01/task/semantic-task.json \
  --class-id black_pepper_shaker \
  --output tiptop_outputs/experiments/live-observe-01
```

기존 작업의 라벨·검출 영역을 사용하고, 현재 영상의 마스크·깊이·CAD 자세·DB 변환은 새로 계산한다.
Gemini를 호출하지 않으며, 물체가 검출 영역 밖으로 이동했다면 새 식별이 필요하다.
기본 카메라 URL은 `http://127.0.0.1:13345`, 자세 서버는 `http://127.0.0.1:13346`이다.
`--camera-url`, `--pose-url`, `--depth-url`로 변경할 수 있다. 깊이 서버의 기본값은 프로젝트 설정을 따른다.

Isaac에 표시되는 마스크·CAD 외곽과 출력의 `pose-overlay.png`를 확인한다.
`result.json`의 `success=true`, `candidate_count>0`은 인식·DB 연결 성공을 뜻한다.
수치 결과는 `current-observation-result.npz`에 저장한다. 후보 충돌 검사·경로 계획·파지는 수행하지 않는다.

## 좌표·충돌·계획 검사

`tool 이름`은 설명만 표시하고, `--execute`를 붙여야 실행한다. 추가 인자는 `--` 다음에 지정한다.
아래 표의 입력은 `--data-root`, 출력은 `--work-dir` 기준 상대 경로다.

| 도구 | 입력·자원 | 출력 및 확인 기준 |
|---|---|---|
| `check-frames` | 코드 내 좌표·물체 식별 예제, CPU | 터미널의 검사 결과 `OK` |
| `check-db` | `plans/multi-wide-*/`와 로컬 3종 CAD·DB, CPU | `audit/local-db-bridge.json`; 각 `max_frame_roundtrip_error < 1e-10` |
| `audit-close` | `plans/multi-*/pick.json`, 선택적 실행 기록, URDF, CPU | `audit/closure-sweep.json`; 각 `passed=true` |
| `audit-carry` | `plans/`의 대상 경로·후보·CAD, GPU | `plans/<사례>/full-geometry-audit.json`, `audit/multiscene-*-geometry.json`; 각 `full_object_box_audit_passed=true` |
| `plan-check` | `plans/scene-wide.json`, 3종 후보·계획·DB, GPU | `audit/cutamp-three-object-safe-world-probe.json`; `success=true`, `executor_contract_verified=true` |
| `summarize` | `runs/*/pick.json`과 `check.json`, CPU | `results.json`, `results.csv`; 각 실험의 `validated` 확인 |
| `diagnose` | 실행 기록·후보 파일·TCP 설정, CPU | `diagnostics.json`; 상대 이동·회전·배치 오차를 분석하는 자료이며 별도 합격 판정 없음 |

```bash
pixi run python -m experiments tool check-frames --execute
pixi run python -m experiments tool check-db --execute
pixi run python -m experiments tool audit-close --execute
pixi run python -m experiments tool audit-carry --execute
```

`audit-carry`에서 회전·이동·밀집 배치는 `-- --extensions`, 대체 후보는 `-- --alternatives`로 선택한다.
결과 항목이 0건이면 검증된 대상이 없는 것이다. 명령의 정상 종료만으로 통과를 판단하지 않는다.
`plan-check`는 계획 연결 검사이며 Isaac 물리 실행은 하지 않는다.

## 실험 계획 생성 및 일괄 실행

### 다중 물체 경로 준비

`prepare-scenes`는 로컬 3종 CAD·DB와 기준 IK 자료로 넓은/가까운 배치를 구성하고,
선별·IK를 통과한 첫 후보의 접근·들기 경로를 준비한다. 로봇은 움직이지 않는다.

```bash
pixi run python -m experiments --work-dir tiptop_outputs/experiments/new-study \
  tool prepare-scenes --execute
```

`plans/multi-summary.json`의 각 `status`가 `prepared`인지 확인한다.
거부된 항목은 `planning_rejected`, `error`와 해당 사례 폴더의 `error.txt`를 확인한다.
준비된 입력은 `plans/<사례>/pick.json`이다. 후속 검사에 사용하려면 입력 폴더를 명시적으로 바꾼다.

```bash
pixi run python -m experiments \
  --data-root tiptop_outputs/experiments/new-study \
  --work-dir tiptop_outputs/experiments/new-study-audit \
  tool audit-carry --execute
```

### 측정 자세 기반 운반·배치 경로 준비

`prepare-measured`는 성공한 다중 물체 들기 기록 `runs/multi-*/`의 `pick.json`, `check.json`을 읽어
운반·배치 경로를 준비한다. 여기서 측정 자세는 **이전 시뮬 실행에서 읽은 정답 자세**다.
카메라 기반 실물 제어용 입력으로 해석하지 않는다.

```bash
pixi run python -m experiments tool prepare-measured --execute -- alternatives
```

인자는 `initial`(기본), `extensions`, `alternatives` 중 선택한다.
출력은 `plans/measured-<모드>-placement-summary.json`이며 `status=prepared`인지 확인한다.
각 사례의 `pick.json`, `full-geometry-audit.json`을 함께 확인한다. 일치하는 입력이 없으면 준비되는 계획도 없다.

### 준비된 실험 일괄 실행

준비 목록(manifest) JSON의 `status=prepared` 항목을 대상으로, `pick_path`의 계획을
`repetitions` 횟수만큼 실행한다. 계획 파일 경로와 반복 횟수를 확인한 뒤 사용한다.

```bash
pixi run python -m experiments --work-dir tiptop_outputs/experiments/new-study-runs \
  tool batch --execute -- tiptop_outputs/experiments/new-study/plans/multi-summary.json
```

결과는 `runs/<사례>-rNN/`와 `logs/batch-<목록 이름>.json`에 저장된다.
이미 `check.json`이 있는 실행은 건너뛴다. 각 `check.json`의 성공 여부와 로그의 `exit_code`, `error`를 확인한다.
들기만 준비한 목록은 들기까지만 실행하며, 전체 집기·놓기는 놓기 경로까지 포함된 계획이 필요하다.

## 저장된 진단 자료

아래 자료는 문제 원인을 비교할 때 사용하는 입력·결과다. 모두 프로젝트 내부에 있지만 **Git에는 포함되지 않는다.**
다른 PC로 옮길 때는 코드와 별도로 보존한다. 각 폴더의 `migration.json`에는 원본 위치, 파일 크기와 SHA-256이 있다.

| 경로 (`tiptop_outputs/experiments/` 기준) | 용도와 주요 파일 |
|---|---|
| `diagnostics/camera-20261021/` | `asset-frame-check.json`: 3종 OBJ·USD의 축척·회전 정합. `depth-check.json`: 가상 양안 영상의 FoundationStereo 깊이 평가. `current-plan-selfmask/target-collision-comparison.json`: 목표 hull 포함 여부에 따른 IK 비교 |
| `grasping/20261020/supplement/` | `candidate-quality.json`, `candidate-coverage.json`: 후보·배치별 상세 집계. `original-experiment-sources.tar.gz`: 입력 준비·분석·일괄 실행 등에 사용한 원본 코드 51개 |
| `diagnostics/client-exit-20261022/` | 영상 저장 후 종료 충돌의 최소 재현 코드. 수정 결과는 `tiptop_outputs/m1013/isaac/prompt-pick-place-20261022/client-shutdown/`, 현재 회귀 검사는 `tests/test_perception_visualization.py` |
| `diagnostics/websocket-20261022/` | RGB+계산된 깊이 입력으로 복구한 websocket의 CPU 입력 검사 결과와 원본 검사 코드 |

카메라 자료의 `request.npz`·`foundation-depth.npz`는 깊이 평가용 촬영에 해당한다.
`live-runtime/planning-input/`은 별도의 후속 관측이며, 계획 자료와의 연결은 `observation_id`로 확인한다.
`evaluation-only.npz`의 Isaac 정답 깊이는 평가용이다. 실제 입력으로 사용하지 않는다.
장착안은 임시 형상 비교이며 실물 보정을 뜻하지 않고, 목표 hull을 끈 IK 비교도 실제 실행에서 충돌 검사를 끄라는 의미가 아니다.

`original-*.tar.gz`는 원본 조사 코드를 확인하기 위한 압축본이다. 제거된 함수나 옛 경로가 들어 있을 수 있으므로
그대로 실행하지 않는다. 평소에는 이 가이드의 유지보수되는 명령을 사용한다.

```bash
# 진단 결과 확인: 장치·서버·Gemini를 사용하지 않는다.
python3 -m json.tool tiptop_outputs/experiments/diagnostics/camera-20261021/asset-frame-check.json
tar -tzf tiptop_outputs/experiments/grasping/20261020/supplement/original-experiment-sources.tar.gz
```

## 검증 범위

- 물리 재현과 일괄 실행은 Isaac을 사용한다. 인식 진단은 GPU와 준비된 추론 서버를 사용한다.
- 물리 실험의 토크 상한 5 N·m는 해당 시뮬 조건이며 실물 정격값이 아니다.
- 물체가 sleep 상태이면 접촉 센서가 0으로 보일 수 있으므로 접촉력 하나로 파지 실패를 판단하지 않는다.
- 닫힘 검사는 고정된 팔 자세에서 그리퍼와 책상·주변 물체를 검사한다. 목표 물체 접촉과 전체 자기 충돌 검사는 포함하지 않는다.
- 운반 검사는 CAD와 구·박스 근사 및 이산 샘플을 사용한다. 연속 충돌 자유나 실물 동작을 보장하지 않는다.
