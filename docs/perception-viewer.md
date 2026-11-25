# 인식·파지 뷰어와 실행 패널

현재 작업의 RGB·깊이·마스크·계획된 최종 파지를 보여 주고, 같은 화면에서 서비스를 켜고 작업을 실행한다.
연속 영상이나 실측 도달 자세가 아니다.

## 사용

```bash
pixi run python -m tiptop.scripts.serve_perception_viewer   # http://127.0.0.1:8091
```

| 옵션 | 의미 |
|---|---|
| `--task-output tiptop_outputs/m1013/tasks/task-NNN` | 저장된 작업을 연다 (기본은 새 작업을 따라감) |
| `--root PATH` | 작업 루트 (기본 `tiptop_outputs/m1013/tasks`) |
| `--read-only` | 실행 패널·서비스 버튼 없이 화면만 표시 |
| `--port` | 기본 8091 |

터미널에서 작업을 실행할 때는 [작업 명령](known-object-pick-place.md#실행)에 `--viewer`를 추가한다.

## 표시

| 보기 | 내용 |
|---|---|
| RGB | 검출 마스크·박스·계획된 최종 그리퍼 |
| Depth | RGB에 정렬된 깊이. 검정은 누락 |
| Side view·파지점 확대 | 같은 파지의 CAD 손가락·DB 범위·높이 비교 |
| 3D Obstacles | 마스크 안 관측점과 충돌 hull. 주황이 파지 대상 |
| DB 후보 조회 | 최종 계획과 별개인 DB 후보·점수·원본 행 |

- 최종 파지는 `Pick.close` 직전 경로(`Pick.descend`, 합쳐졌으면 `Pick.approach`) 마지막 관절값의 FK다. DB 범위는 실측 접촉점이 아니다.
- 저장 작업은 기록된 계획과 DB 보정으로 재구성하며, 계획이 없으면 DB 후보로 대체하지 않는다.
- 3D Obstacles는 RGB와 같은 시점으로 시작해 드래그로 물체 중심 회전, `RGB 시점`으로 복귀한다. `관측점`·`Hull`은 각각 켜고 끌 수 있고, 관측점만 표시용으로 샘플링한다. 고정 구조물은 포함하지 않는다.
- 새 작업은 `viewer/obstacles.json`을 저장하며 저장 작업을 열 때도 쓴다.

## 실행 패널

물체·인식 모드·제약값을 고르면 실행 명령 미리보기가 나온다. 물체 목록을 클릭하면 지시문에 반영된다.

- **실행 버튼은 `--execute`를 포함한다.** 계획만 보거나 연결만 확인할 때는 터미널 명령을 쓴다.
- 기본값은 `tiptop/config/launcher_presets.json`이며 `이 값을 기본값으로 저장`으로 갱신한다. CLI 기본값과 다르다.
- `로봇 파라미터`: 가속도·jerk·종단 속도 허용치 ([로봇 한계](m1013-robot-limits.md#계획-한계)).
- `주변 물체 장애물 반영`: 끄면 파지 대상만 장애물로 등록한다. 인식 결과와 테이블·고정 구조물 검사는 유지하고, 3D Obstacles에는 등록된 물체만 표시한다.
- `데모 모드 (기록 최소화)`: 카메라 패킷·인식 이미지·후보 크롭·계획 입력 pickle을 저장하지 않아 저장 작업 검토와 실험 도구에는 쓸 수 없다.
- Gemini 키는 뷰어 서버 환경에서 읽는다. SAM3는 설정한 worker를 쓴다.
- 한 번에 한 작업만 실행한다. 중단 버튼은 스크립트에 Ctrl+C를 보내며 로봇 비상정지가 아니다.

### 작업 결과 로그

끝난 작업마다 한 줄이 `tiptop_outputs/m1013/task-log.jsonl`에 쌓이고 패널에 최근 100개가 표시된다.
줄에는 인식 모드·TASK 번호·물체·닫기 전류 상한·닫은 뒤 전류와 위치·소요 시간·결과가 들어간다.
`파지 성공/실패`는 닫은 위치가 빈 손 한계(315 count) 이하인지로 추정한 값이다.
`정상 인식/오인식`, `성공/파지 실패/이송 중 놓침/충돌`은 사람이 직접 표시하며 `task-marks.json`에 저장된다.

## 서비스 시작·종료

패널 위쪽 칩은 서비스를 두 줄로 보여 준다. 줄은 `launcher_presets.json`의 `row`로 정한다. SAM3 서버와 작업 worker는 Gemini 모드에서도 보이며(켜 둔 채 잊지 않도록), 이때 두 칩이 `SAM3 모드에서만 사용` 상자로 묶인다.

| 줄 | 서비스 |
|---|---|
| 1 | ros2-dev 컨테이너, FoundationPose, SAM3, 작업 worker |
| 2 | bringup, 컨트롤러, 그리퍼, gateway, 관측 서버 |

| 표시 | 의미 |
|---|---|
| 회색 ○ | 꺼짐 |
| 노랑 ◐ | 뷰어가 띄운 프로세스가 준비 중(포트 닫힘) 또는 종료 중 |
| 파랑 ● | 준비 완료(포트·소켓·ROS 노드 확인) |
| 빨강 ✕ | 종료를 요청하지 않았는데 프로세스가 끝남. 다시 시작하면 해제 |

- ▶/■ 버튼은 `launcher_presets.json`의 `command`를 `bash -ic`로 실행한다. 새 터미널과 같이 `~/.bashrc`가 적용되므로 아래가 필요하다.
  - alias `fp='conda activate foundationpose'`, `sam='conda activate sam3'`
  - 환경 변수 `FOUNDATIONPOSE_PYTHON`, `SAM3_PYTHON` (conda가 miniforge가 아닐 때)
- ■는 Ctrl+C와 같다. 프로세스 그룹(컨테이너 안 서비스는 컨테이너 안의 그룹)에 SIGINT를 보내고, 10초 안에 안 끝나면 SIGTERM, 20초에 SIGKILL을 보낸다.
- 컨테이너 안 서비스는 시작할 때 `/tmp/tiptop-<id>.pid`에 pid를 남기고 새 세션으로 실행하며, 종료는 이 pid의 그룹에 보낸다.
- `ros2-dev`와 `팔 컨트롤러`는 한 번 실행하고 끝나는 명령이다. 파랑은 상태 확인(컨트롤러는 `dsr_moveit_controller`가 active)이 정하며, 이미 active면 버튼이 없다.
- 서비스는 뷰어를 재시작해도 계속 실행되고 `Log/services/<id>/current.json`으로 이어받는다.
- 시작 순서는 강제하지 않는다. 관측 서버는 시작 때 gateway `/state`를 조회하므로 gateway를 먼저 띄운다.
- 출력은 `Log/services/<id>/<시각>.log`에 ANSI 색 그대로 저장하고 서비스마다 최근 10개만 남긴다. `less -R`로 본다.
- 터미널에서 직접 띄운 서버도 파랑으로 보이지만 버튼은 없다.

### 시작·종료 순서

시작: ros2-dev → FoundationPose → SAM3 → 작업 worker → bringup → 컨트롤러 → 그리퍼 → gateway → 관측 서버.
종료는 거꾸로 한다. 지켜야 할 규칙은 세 가지다.

- 관측 서버는 gateway보다 먼저 끈다. 캡처마다 gateway에서 로봇 상태를 읽으므로, gateway가 먼저 꺼지면 캡처가 실패한다.
- gateway는 bringup·그리퍼보다 먼저 끈다.
- ros2-dev 컨테이너는 마지막에 끈다. 먼저 끄면 안의 ROS 서비스가 한꺼번에 내려간다.

worker·SAM3·FoundationPose는 로봇 쪽과 의존이 없어 아무 때나 꺼도 된다. ros2-dev와 컨트롤러는 ■가 없으므로 컨테이너는 `docker stop ros2-dev`로, 컨트롤러는 bringup을 끄면 함께 사라진다.
