# SAM3 reference 인식

`sam3_reference`는 등록 물체의 reference 이미지와 현재 RGB로 마스크를 구한다. Gemini 없이 FoundationPose·파지 DB·계획·실행으로 이어진다.
임계값을 넘는 모든 인스턴스의 마스크를 장면에 유지하고, 그중 하나를 파지 대상으로 고른다.
설치는 [README_CUSTOM](../README_CUSTOM.md#sam3)을 따른다.

## 서버

```bash
bash scripts/run_sam3_server.sh --check   # 추론 없이 파일·모듈 검사. SAM3_READY가 나오면 정상
bash scripts/run_sam3_server.sh
```

환경을 찾지 못하면 `SAM3_PYTHON`에 conda Python 경로를 지정한다. 가중치는 로컬 캐시를 쓰며 `--checkpoint PATH`로 지정할 수 있다.

| 설정 | 기본값 | 변경 |
|---|---|---|
| 포트 | `13347` | `--port` |
| reference root | `/nas/dasolbyun/sanjabu/test_v2` | `--reference-root` |
| metadata | `/nas/dasolbyun/sanjabu/2026_real_world_test_objects_attr2.csv` | `--metadata` |
| scene | `Home/Uon_1F_Home/floor` | 작업 CLI의 `--reference-scene` |
| 신뢰도 임계값 | `0.5` | 작업 CLI의 `--sam3-confidence-threshold` |

실험실 작업은 `--reference-scene Lab/Uon_1F_Lab/table`을 쓴다.
임계값을 요청마다 바꾸려면 서버가 그 값을 받는 버전이어야 하며, 서버가 적용한 값이 요청과 다르면 `[SAM3]` 로그에 경고가 나온다.
서버 코드나 reference를 바꾸면 서버를 재시작한다.

## 작업 명령

```bash
pixi run python -m tiptop.scripts.run_known_pick_place \
  --perception-mode sam3_reference --reference-scene Lab/Uon_1F_Lab/table \
  --task 'black_pepper_shaker 집고 박스에 놓아' --check
```

`--check`를 빼면 계획하고, `--execute`를 더하면 동작한다. 나머지 옵션은 [작업 실행](known-object-pick-place.md#실행)을 따른다.

## 후보 선택

SAM3의 점수는 참조 박스와 비슷한 정도일 뿐 물체 식별 근거가 아니어서, 최고 점수 후보가 다른 물체일 수 있다. 그래서 후보를 하나씩 검증한다.

1. 점수가 임계값을 넘는 모든 마스크를 장면에 유지하고, 점수 순 상위 최대 8개를 검증한다. 동점은 서버가 준 순서를 따른다.
2. 후보마다 FoundationPose로 포즈를 추정하고 CAD 실루엣과 마스크의 IoU(`--cad-mask-min-iou`)를 검사한다.
3. 처음 통과한 후보를 대상으로 쓴다. 모두 거부되면 임계값을 낮추지 않고 중단한다.

- 인스턴스 이름은 `<class>__instance_<서버 인덱스>`이고 canonical class는 `class_id`로 유지한다.
- `--appearance-ranking`은 reference 박스와의 색 유사도(`--appearance-weight` 배)를 점수에 더해 검증 순서만 바꾼다.
- 임계값을 낮추면 후보가 늘 뿐 순위는 바뀌지 않는다. 진짜 물체 점수가 임계값 이하면 후보에 오르지 못한다.

### 주변 물체 (`object` reference)

`--surrounding-obstacles`(기본 켜짐)일 때만 SAM3를 한 번 더 호출한다.
`<reference scene>/reference/object/`에 여러 종류의 물체 예시를 박스로 표시해 두면, 대상이 아닌 물체도 `object__instance_<번호>`로 검출해 장애물 hull로 등록한다.

- 이 호출의 임계값은 0.3 고정이다. 장애물은 거짓 양성보다 미검출이 위험하다.
- 대상 종류의 검출과 겹치면(한쪽 마스크의 절반 이상이 다른 쪽 안) 같은 물체로 보고 제외한다.
- `object`는 카탈로그에 없는 범용 이름이라 서버가 카탈로그·ID 검증을 건너뛴다.
- 대상 검출 직후 시작해 FoundationPose 검증과 동시에 돌고, 대상 후보가 모두 거부되면 취소한다.

### 진단

- 터미널: `[SAM3]`(후보 목록·검증 대상), `[Perception]`(후보별 IoU·CAD 대비 크기·채택/거부).
- `semantic-task.json`의 `reference_diagnostics.candidate_trials`: 시도한 후보와 IoU.
- 작업 폴더 `candidates/`: 모든 후보의 크롭(`idx<번호>_<accepted|rejected|skipped>_score<점수>.png`)과 전체를 표시한 `overview.png`.

## Reference 규칙

- scene의 `reference/reference_index.json`에서 이미지와 박스를 읽는다.
- 물체는 canonical class 이름으로 연결한다. CAD CSV의 `Class_name`과 행 번호 `No`를 구분한다.
- 한 이미지에 박스가 여럿이면 `bboxes`에 넣고 `bbox`는 첫 번째 박스와 같게 둔다.
- `GET /health`는 준비 상태, `GET /reference?class_id=…&scene=…`는 reference 유효성을 확인한다.
