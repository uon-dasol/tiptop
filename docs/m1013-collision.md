# M1013 충돌 설정과 검증

## 모델

- M1013 팔과 UON 그리퍼의 형상을 링크별 충돌 구로 표현하고, 파지한 물체용 부착 슬롯(`attached_object`)을 둔다.
  cuTAMP의 부착 슬롯 32개를 확보하는 설정에서도 실험 자기충돌 커널을 쓴다.
- 구는 원본 충돌 메쉬의 모든 삼각형을 1 mm 여유로 덮도록 배치했다. 이 여유는 구가 메쉬 밖으로 튀어나오는 양의 상한이 아니다.
- 일반 링크는 적은 수의 구로 근사하고, 바닥·두 번째 링크만 좁은 간격을 보존하도록 국소 보정한다. 해당 링크 쌍은 계속 검사하며 `self_collision_ignore`는 유지한다.
- 계획 모델의 그리퍼는 열린 형상으로 고정하고 실행 때만 개폐한다. showcase는 같은 링크별 구를 열린/닫힌 각도로 변환한다.
- 작업대·파지한 물체·장착/TCP 보정은 별도 검증이 필요하다.

모델 재생성은 [fit_m1013_collision_spheres.py](../scripts/fit_m1013_collision_spheres.py)다.
기본 실행은 계산해 보고만 하고(`total_spheres_including_attachment` 등), `--write`를 주면 모델 YAML과 cuTAMP 그리퍼 구를 갱신한다. `--output`은 후보 YAML만 따로 저장한다.

```bash
pixi run python -m scripts.fit_m1013_collision_spheres
```

일반적인 자동 구 생성 → 문제 부위 보정 방식은 [cuRobo 설정 안내](https://curobo.org/tutorials/1_robot_configuration.html)와
[NVIDIA 편집기 안내](https://docs.isaacsim.omniverse.nvidia.com/latest/manipulators/manipulators_robot_description_editor.html)를 참고한다.

## 원본 모델과 메쉬

계획용 [URDF](../tiptop/doosan/assets/m1013_uon/m1013_uon.urdf)는 [ROS2 통합 모델](../ros2_ws/src/description/m1013_uon_3f_description/urdf/m1013_uon_3f.urdf.xacro)을 펼친 것이다. 메쉬는 복사하지 않고 아래 원본을 상대 경로로 참조한다.

| 원본 위치 (`ros2_ws/src/` 기준) | 참조 파일 |
|---|---|
| `drivers/manipulator/doosan-robot2/dsr_description2/meshes/` | 두산 DAE |
| `description/uon_3f_description/meshes/` | UON STL |

- 원본 준비는 [ROS2 설치 안내](../ros2_ws/README_CUSTOM.md)의 외부 레포 가져오기(`vcs import`)까지 필요하다. 메쉬를 읽는 데 ROS2 실행·colcon 빌드는 필요 없다.
- 원본 메쉬를 바꿔도 계획용 URDF·충돌 구는 자동 갱신되지 않으므로 정합성을 다시 확인한다.
- 두산 모델의 원본 저작권·이용 조건은 [LICENSE.doosan](../tiptop/doosan/assets/m1013_uon/LICENSE.doosan)에 보관한다. UON 메쉬는 사용자 제공 자산(Proprietary)이다.

## 외부 패키지 패치

`curobo/`와 `cutamp/`는 루트 Git에서 제외된 별도 저장소이며, 변경은 패치 파일로 재현한다. 설치 스크립트가 패치를 적용하고, 이미 적용돼 있으면 건너뛰며, 상위 코드와 맞지 않으면 중단한다.

| 대상 | 패치 | 내용 |
|---|---|---|
| cuRobo (`b5fad1d`로 고정) | [curobo-self-collision.patch](../patches/curobo-self-collision.patch) | `motion_gen.py`의 `check_start_state`가 장애물 충돌을 반환하기 전에 일시적으로 끈 자기충돌 검사를 복구 |
| cuTAMP (v0.0.6) | [cutamp-ordered-grasps.patch](../patches/cutamp-ordered-grasps.patch) | `ordered_grasps` 설정: 입자마다 주어진 파지를 순서대로 하나씩 배정 (랜덤 샘플링 대신) |

설치는 [install-curobo.sh](../install/install-curobo.sh), [install-cutamp.sh](../install/install-cutamp.sh)(`pixi run setup-planners`)가 한다.
M1013 로더는 [model.py](../tiptop/doosan/model.py), URDF·충돌 구 설정은 [자산 폴더](../tiptop/doosan/assets/m1013_uon/)에 있다.

## 회귀 검사

로봇에 명령을 보내지 않는다. GPU 환경의 프로젝트 루트에서 실행한다.

```bash
pixi run python -m pytest tests/test_m1013_collision.py -v
```

실험 커널 선택, 정상 자세 통과, 장애물 검사 뒤 자기충돌 검사 유지, 구가 원본 메쉬를 덮는지, 실제 메쉬가 교차하는 자세의 거부, 열린/닫힌 그리퍼 링크 위치와 URDF FK의 일치, cuTAMP 그리퍼 구 정합성을 확인한다.
시험용 손가락 박스는 검사 안에서만 추가·제거한다. 박스 유무 경로 비교는 [경로 계획](m1013-planning.md), 단일 물체 계획은 [cuTAMP 연결 검사](m1013-tamp.md)를 따른다.
