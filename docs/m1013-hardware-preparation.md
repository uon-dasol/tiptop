# M1013 실물 준비

팔·그리퍼 기동은 [실물 기동](m1013-real-connection.md), 인식·계획은 [작업 실행](known-object-pick-place.md)을 따른다.
운영 설정은 사용자 선언값을 포함하며 `calibrated:true`가 정밀 실측 완료를 뜻하지 않는다.

## 설정 파일

| 파일 | 용도 |
|---|---|
| `tiptop/config/m1013_fixed_camera.declared.json` | 카메라 설치 자세·일련번호·영상 180° 회전 |
| `tiptop/config/m1013_fixed_hardware.declared.json` | 그리퍼 장착·TCP·DB 보정·카메라 충돌 형상·작업 영역 |
| `ros2_ws/src/execution/tiptop_execution/config/uon_native.measured.json` | 그리퍼 위치·전류 count와 완료 조건 |

- 길이는 m, 관절각은 rad, 작업 좌표계는 `base_link`다.
- 카메라의 `parent_from_camera`는 회전 전 컬러 광학 자세이며 `image_rotation_deg`를 따로 적용한다. 충돌 형상의 `parent_from_rig`와 혼용하지 않는다.
- 고정 카메라 본체·브래킷 상자는 자동 생성하므로 `workspace.fixed_obstacles`에 넣지 않는다.

## 상태 전용 gateway

보정·계획 검토만 할 때 ROS 컨테이너에서 실행한다. 동작 명령은 거부한다.

```bash
ros2 run tiptop_execution gateway --namespace dsr01 --robot-system real --read-only
```

실제 실행에는 그리퍼 설정을 포함한 gateway가 필요하다([실물 기동](m1013-real-connection.md#그리퍼와-gateway)).

## 설정 검사와 관측

두 검사는 장치를 열지 않는다.

```bash
pixi run python -m tiptop.scripts.prepare_m1013_hardware \
  tiptop/config/m1013_fixed_hardware.declared.json
pixi run python -m tiptop.scripts.serve_orbbec_observation \
  --calibration tiptop/config/m1013_fixed_camera.declared.json --check
```

실제 카메라 관측 서버는 별도 호스트 터미널에서 실행한다. 같은 카메라를 쓰는 preview 프로그램은 먼저 종료한다.

```bash
pixi run python -m tiptop.scripts.serve_orbbec_observation \
  --calibration tiptop/config/m1013_fixed_camera.declared.json \
  --depth-source sensor --port 13348 \
  --output tiptop_outputs/m1013/real-observe-review
```

- 관측 서버는 읽기 전용이다. 기본 `--depth-source`는 `foundation_stereo`이며, 이 경우에만 별도 깊이 서버가 필요하다.
- 시작 때 gateway `/state`를 조회하므로 gateway를 먼저 띄운다.
- [작업 실행](known-object-pick-place.md#실행)에서 `--execute` 없이 인식·계획을 먼저 확인한다.

## 확인할 사항

- 카메라 외부 자세·TCP·DB 정렬·작업 영역을 현장 기준점과 대조한다.
- 그리퍼 position·effort는 native 위치·전류 count다. m·N으로 해석하지 않는다.
- 전류·정지 상태만으로 물체 접촉과 기구 걸림을 구분할 수 없다. 파지 성공은 따로 확인한다.
- 고정 카메라의 로봇 깊이 제거는 관측 관절값과 모델 충돌 구를 쓰므로 장착·보정 오차의 영향을 받는다.
- 모델·보정을 바꾸면 기존 계획을 재사용하지 않고 다시 관측·계획한다.

추가 측정은 `pixi run python -m tiptop.scripts.calibrate_m1013 --help`로 확인한다. pivot은 TCP 위치만 구하고, 고정 카메라의 외부 자세는 별도 기준점으로 측정한다.

카메라는 고정(`base_link`)만 사용한다. 손목(`link_6`) 장착 경로(`mount: wrist`, 손눈 보정, `m1013_hardware.example.json`)는 구현만 돼 있고 사용·검증하지 않았다.
