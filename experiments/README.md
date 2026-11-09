# 실험·진단 도구

저장된 결과 조회와 계획 재현용이다. 새 관측의 집기·놓기는 [작업 실행](../docs/known-object-pick-place.md)을 사용한다.
명령은 프로젝트 루트에서 실행한다.

```bash
pixi run python -m experiments list
```

## 결과 조회·재현

```bash
pixi run python -m experiments results --failures-only
pixi run python -m experiments run --case pepper
```

`run`은 기본적으로 명령만 미리 보여 준다. Isaac을 실행하려면 `--execute`와 새 출력 폴더를 지정한다.

```bash
pixi run python -m experiments run --case pepper \
  --output tiptop_outputs/experiments/pepper-01 --execute
```

대상은 `pepper`·`lotion`·`dipper`다. 결과는 `check.json`에서 확인한다.
입력은 기본 `tiptop_outputs/experiments/grasping/20261020`, 새 출력은 `tiptop_outputs/experiments/work`를 사용한다.
다른 경로는 상위 옵션 `--data-root`·`--work-dir`로 지정한다. 입력·결과 파일은 Git에 포함하지 않는다.

## 인식 진단

| 명령 | 용도 |
|---|---|
| `perception view` | 저장 영상과 DB 후보를 HTML로 비교 |
| `perception observe` | 현재 가상 스테레오 영상의 인식·자세·DB 연결 확인 |

하위 옵션은 `pixi run python -m experiments perception view -- --help`처럼 확인한다.
`observe`에는 작업의 `semantic-task.json`을 `--active-task`로 지정한다.
이 도구는 스테레오 입력용이며 현재 `--lab`의 RGB-D에는 본체 작업 명령과 뷰어를 사용한다.

## 책상 지지면 방식 비교

저장된 작업에서 `ransac-cad`·`ransac-mask`·`declared`의 지지 높이와 CAD 바닥 차이를 비교한다.
카메라·추론 서버·로봇에 연결하지 않는다.

```bash
pixi run python -m experiments perception table-support -- \
  --task-range 230 262 --declared-support-z-m 0.0 0.006
```

입력은 작업의 관측·깊이·마스크·CAD 자세다. 필요한 파일이 없는 작업은 제외한다.
`--declared-support-z-m`을 생략하면 하드웨어 설정의 책상 윗면을 사용한다.
`results.json`에 방식별 통과 여부와 높이 차이를 저장한다. 저장 깊이의 양자화와 RANSAC 난수 때문에 실제 실행과 차이가 날 수 있다.

## 좌표·충돌·계획 검사

```bash
pixi run python -m experiments tool check-frames
pixi run python -m experiments tool check-frames --execute
```

`tool`은 `--execute` 전에는 설명과 입출력 경로만 표시한다.
`list`에서 도구와 CPU/GPU/Isaac 요구 사항을 확인하고, 하위 인자는 `--` 뒤에 지정한다.
모델 기반 접촉·충돌 판정은 실물 성공률이나 연속 충돌 자유를 보장하지 않는다.
