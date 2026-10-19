# M1013 충돌 설정과 검증

## 현재 모델

- M1013 팔과 UON 그리퍼의 실제 형상을 구 96개로 표현한다. 기본 부착 슬롯 4개까지 포함하면 100개다.
  cuTAMP의 부착 슬롯 32개를 확보해도 실험 자기충돌 커널을 사용한다(총 128개, 검사 쌍 5,608개).
  기존 showcase는 같은 링크별 구를 열린/닫힌 그리퍼 각도로 변환한다.
  현재 cuTAMP 최소 시험의 계획 모델은 열린 그리퍼 형상으로 고정하며 실행 시에만 개폐한다.
- 바닥·두 번째 링크의 메쉬 근사로 발생한 0° 자세의 자기충돌 오탐을 수정했다.
  해당 링크 쌍은 계속 검사하며, 기존 `self_collision_ignore`는 유지한다.
- 모든 링크의 원본 충돌 메쉬 삼각형을 1mm 여유로 덮도록 구를 배치했다.
  이 여유는 구가 메쉬 밖으로 튀어나오는 양의 상한을 뜻하지 않는다.
- 일반 링크는 적은 수의 구로 근사하고, 바닥·두 번째 링크만 좁은 간격을 보존하도록 국소 보정한다.
  실환경 작업대·파지한 물체와 장착·TCP 보정은 별도 검증이 필요하다.

모델 재생성 도구는 [fit_m1013_collision_spheres.py](../scripts/fit_m1013_collision_spheres.py)다.
기본 실행은 96개 구 모델을 계산해 보고만 하며,
`--write`를 지정하면 모델 YAML과 이에 대응하는 cuTAMP 그리퍼 구(17개)를 갱신한다.

```bash
pixi run python -m scripts.fit_m1013_collision_spheres
```

일반적인 자동 구 생성→문제 부위 보정 방식은
[cuRobo 설정 안내](https://curobo.org/tutorials/1_robot_configuration.html)와
[NVIDIA 편집기 안내](https://docs.isaacsim.omniverse.nvidia.com/latest/manipulators/manipulators_robot_description_editor.html)를 참고했다.

## 원본 모델과 메쉬

계획용 [URDF](../tiptop/doosan/assets/m1013_uon/m1013_uon.urdf)는
[ROS2 통합 모델](../ros2_ws/src/description/m1013_uon_3f_description/urdf/m1013_uon_3f.urdf.xacro)을
펼친 것이며, 메쉬는 복사하지 않고 아래 원본을 상대 경로로 참조한다.

| 원본 위치 (`ros2_ws/src/` 기준) | 참조 파일 |
|---|---|
| `drivers/manipulator/doosan-robot2/dsr_description2/meshes/` | 두산 DAE 20개 |
| `description/uon_3f_description/meshes/` | UON STL 5개 |

검증 기준은 `ros2_ws@81b83d0fc4a6eb135eef34c2c476b75b1b137c40`,
`doosan-robot2@3f4649bc645477c7a76aa7b07e272bc06c8c504e`다.
원본 준비는 [ROS2 설치 안내](../ros2_ws/README_CUSTOM.md)의 외부 레포 가져오기까지 필요하며,
메쉬를 읽는 데 ROS2 실행·colcon 빌드는 필요하지 않다. 두산 설치는 `jazzy` 브랜치를 받으므로
정확한 재현에는 위 리비전을 사용한다. 별도 PC·단독 wheel 설치에서도 원본 파일 준비가 필요하다.
원본 메쉬를 변경해도 계획용 URDF·충돌 구는 자동 갱신되지 않으므로 정합성을 다시 확인한다.

두산 모델의 원본 저작권·이용 조건은 [원본 LICENSE 사본](../tiptop/doosan/assets/m1013_uon/LICENSE.doosan)에
보관한다. UON 메쉬는 사용자 제공 자산이며 원본 ROS2 패키지의 라이선스 표기는 Proprietary다.

## cuRobo 패치

적용 기준은 프로젝트에서 사용하는 cuRobo fork의 `b5fad1d`다.
설치 스크립트는 재현을 위해 이 리비전을 고정한다.

| 수정 위치 | 수정 이유 |
|---|---|
| `motion_gen.py`의 `check_start_state` | 장애물 충돌을 반환하기 전에 일시적으로 끈 자기충돌 검사를 복구 |

수정은 [curobo-self-collision.patch](../patches/curobo-self-collision.patch)에 보관하며,
[설치 스크립트](../install/install-curobo.sh)가 빌드 전에 적용한다.
이미 적용된 패치는 건너뛰고, 상위 코드와 맞지 않으면 검토하도록 중단한다.
현재 M1013 모델은 실험 자기충돌 커널을 사용한다.

`curobo/`와 `cutamp/`는 TiPToP 루트 Git에서 제외된 별도 저장소다.
cuRobo는 패치 파일과 설치 적용 코드로 변경을 재현한다.
cuTAMP는 원본 v0.0.6을 사용하며 내부 수정은 필요하지 않다.
M1013 로더는 [model.py](../tiptop/doosan/model.py), URDF·충돌 구 설정은
[자산 폴더](../tiptop/doosan/assets/m1013_uon/)에 포함한다.
메쉬는 복사하지 않고 `ros2_ws`의 두산·UON 원본을 참조하므로 해당 소스 준비가 필요하다.
단일 물체의 cuTAMP 작업 계획과 가상 실행은 [작업 계획 안내](m1013-tamp.md)를 따른다.

## 회귀 검사 — 로봇 명령 전송 없음

호스트의 최상위 TiPToP 폴더에서 실행한다. GPU 환경에서 **8 passed**를 확인한다.

```bash
pixi run python -m pytest tests/test_m1013_collision.py -v
```

실험 커널 선택, 정상 자세 통과, 장애물 검사 후 자기충돌 검사 유지,
원본 메쉬 포함 여부와 실제 메쉬가 교차하는 자세의 거부, 그리퍼 구 정합성을 확인한다.
시험용 손가락 박스는 검사 안에서만 추가·제거하며, 실제 `plan` 명령에 포함되지 않는다.
열린/닫힌 그리퍼의 각 링크 위치가 URDF의 FK와 일치하는지도 확인한다.
박스가 없는 경로와 우회 경로의 실제 비교는 아래 계획 실행 안내의 "박스 유무 비교"를 따른다.
계획 생성·가상 실행 절차는 [계획 실행 안내](m1013-planning.md)를 따른다.
