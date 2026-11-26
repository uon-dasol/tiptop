# M1013 실물 기동

설치·빌드는 [ROS 워크스페이스 안내](../ros2_ws/README_CUSTOM.md)를 따른다.
ROS 명령은 `ros2-dev` 컨테이너에서 실행하고, 같은 로봇의 노드는 같은 ROS domain을 쓴다.
Bringup은 제어권·서보·RT 통신을 설정하므로 단순 상태 조회가 아니다. 기존 드라이버와 중복 실행하지 않는다.

[뷰어](perception-viewer.md#서비스-시작종료)의 서비스 칩으로도 아래 순서대로 시작·종료할 수 있다.

## 팔과 컨트롤러

```bash
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
  name:=dsr01 mode:=real host:=192.168.1.30 rt_host:=192.168.1.30 \
  port:=12345 model:=m1013 gui:=false
```

별도 ROS 터미널에서 컨트롤러를 확인하고, 비활성이면 시작한다.

```bash
ros2 control list_controllers -c /dsr01/controller_manager
# dsr_moveit_controller가 active가 아닐 때만 실행
ros2 run controller_manager spawner dsr_moveit_controller -c /dsr01/controller_manager
```

상태 확인 (`robot_system=0`이 실물):

```bash
ros2 service call /dsr01/dsr_controller2/system/get_robot_system dsr_msgs2/srv/GetRobotSystem '{}'
ros2 service call /dsr01/dsr_controller2/aux_control/get_current_posj dsr_msgs2/srv/GetCurrentPosj '{}'
```

`GetCurrentPosj`는 degree, ROS JointState·TiPToP 관절값은 rad다.

## 그리퍼와 gateway

각각 별도 ROS 터미널에서 실행한다. 그리퍼 드라이버는 워크스페이스의 패치를 적용한 빌드를 쓴다.

```bash
ros2 launch uon_3f_gripper uon_3f_gripper_node.launch.py
```

```bash
ros2 run tiptop_execution gateway --namespace dsr01 --robot-system real \
  --gripper-config /workspaces/ros2_ws/src/execution/tiptop_execution/config/uon_native.measured.json
```

- gateway는 `127.0.0.1:8090`에서 요청을 받는다. 상태 전용은 `--read-only`를 쓴다.
- 관측 서버·작업 명령은 [실물 준비](m1013-hardware-preparation.md), 위치·전류 단위는 [그리퍼 설정](m1013-gripper-check.md)을 따른다.
- RViz로 모델·그리퍼를 확인하려면 [모델 안내](../ros2_ws/docs/m1013-uon-3f-model.md)를 따른다. 실물 그리퍼(`--gripper-config`)를 제어하는 데는 필요하지 않다.

## 상태 확인과 정지

실행기는 팔 경로를 HTTP로 gateway에 한 번에 보내고, gateway가 ROS2 `FollowJointTrajectory` 액션으로 바꿔 `dsr_moveit_controller`에 전달한다.
호스트 프로젝트 루트에서 gateway에 직접 명령할 수도 있다.

```bash
pixi run python -m tiptop.doosan status   # 관절·모션 상태, 그리퍼 피드백
pixi run python -m tiptop.doosan stop     # 팔 액션 취소, 두산 stop 요청, 그리퍼 정지
```

- `status`가 `Connection refused`이면 gateway가 없다. `timed out`이면 실행 여부를 알 수 없으므로 기존 gateway 터미널의 오류를 먼저 확인하고 중복 실행하지 않는다.
- `stop`으로 중단된 동작은 성공으로 처리하지 않는다. 정지 뒤 `status`에서 `motion_status: 0`을 확인한다.
- `stop`은 그리퍼 유지, 궤적 취소, 팔 정지를 각각 시도한다. 하나라도 실패하면 응답의 `success`가 `false`이고 `gripper_hold_error`·`trajectory_cancel_error`·`arm_stop_error` 중 해당 항목에 사유가 들어간다. 팔 정지 호출이 오류 없이 끝났는지는 `arm_stop_acknowledged`로 확인한다.
- 같은 CLI의 `open`·`close`는 그리퍼를 실제로 개폐하고, `run PLAN.json`은 gateway에 연결된 팔을 실제로 움직인다. 계획의 `robot_system`이 gateway와 다르면 거부하며, 실물 실행은 실측 하드웨어 프로필로 만든 계획만 받는다.

## 변경 적용과 로그

- ROS 코드를 바꾸면 해당 패키지를 빌드하고 기존 노드를 재시작한다.
- 두산 드라이버의 최소 명령 변화량은 코드 상수 `1e-9 rad`이며 환경 변수는 없다.
- 실패·정지 응답은 ROS 터미널 로그와 작업 폴더의 `physical-execution.json`에서 확인한다.
- gateway는 요청을 보낸 것만으로 팔 단계를 완료로 보지 않는다. 액션이 성공하고 `check_motion=0`이며 모든 관절의 목표-피드백 오차가 `0.5°` 이내인 상태가 연속 세 번 확인돼야 한다.
- 단계별 제어 완료는 실제 물체의 파지·방출 성공과 구분한다.
