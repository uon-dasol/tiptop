# SAM3 reference 인식 모드

`run_known_pick_place`는 `--perception-mode gemini_er|sam3_reference`로 인식 방식을 선택한다.
기본값은 `tiptop/config/tiptop.yml`의 `perception.mode: gemini_er`다. CLI가 설정 파일보다 우선한다.

| 모드 | 인식 | 지시 | Gemini 사용 |
| --- | --- | --- | --- |
| `gemini_er` | Gemini ER 검출 → SAM2 마스크 | 기존 자연어 지시 | 새 지시마다 최대 1회, 명시적 허용 필요 |
| `sam3_reference` | reference stitching → SAM3 검출·마스크 | 등록 class ID + 집기·빈자리 놓기 | 0회 |

두 경로는 현재 RGB 해상도의 마스크부터 깊이·FoundationPose·파지 DB·빈자리 선택·계획·실행을 공유한다.
SAM3 모드는 SAM2 모델과 Gemini 클라이언트를 초기화하지 않는다. 클래스당 인스턴스 하나 제약을 유지한다.

## 독립 소스와 환경

SAM3는 `git@github.com:uonrobotics/sam3.git`의 독립 체크아웃 `SAM3/`를 사용한다.
서버는 `SAM3/tiptop_server/server.py`, TiPToP 실행기는 `tiptop/scripts/sam3_server.py`다.
data_gen 프로젝트를 실행 의존성으로 사용하지 않는다.
모델·전처리·후처리는 SAM3 프로세스에서 실행하며 TiPToP Pixi 환경은 HTTP/NumPy 결과만 받는다.
서버용 모듈은 기존 data_gen batch 예제와 분리돼 있으며, 선택 설치 항목 `tiptop-server`를 사용한다.
프로젝트 루트에서 서브모듈을 준비하고 별도 환경에 설치한다.

```bash
git submodule update --init SAM3
conda create -n sam3 python=3.12 -y
conda activate sam3
python -m pip install 'setuptools<81' numpy==1.26.4 ninja
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e './SAM3[tiptop-server]'
python -m pip install 'git+https://github.com/ronghanghu/cc_torch.git@446e9360ed10d3c8d03925ccb5f220ccb62009de'
```

CUDA 확장 빌드 조건은 [SAM3 README](../SAM3/README.md)의 optional dependencies를 따른다.
[SAM3 가중치](https://huggingface.co/facebook/sam3)의 접근 권한을 확보하고 `sam3.pt`를 미리 준비한다.
`SAM3_PYTHON`으로 다른 인터프리터를 지정할 수 있다. 실행기는 항상 프로젝트의 `SAM3/` 소스를 우선한다.
체크포인트는 기존 Hugging Face 로컬 캐시의 `sam3.pt`를 사용하며 `--checkpoint PATH`로 지정 가능하다.
실행 중 다운로드하지 않는다. 캐시에 여러 버전이 있으면 사용할 체크포인트를 명시한다.

## 서버 시작

```bash
bash scripts/run_sam3_server.sh --check
bash scripts/run_sam3_server.sh
curl -s http://127.0.0.1:13347/health
```

`--check`는 GPU 추론 없이 파일·모듈 존재를 확인한다. `SAM3_READY`는 모델 로드와 포트 바인딩 뒤 출력된다.
서버는 localhost에만 바인딩하고 GPU 요청을 하나씩 처리한다. 모델은 한 번 로드하고 reference는 캐시한다.
지시마다 현재 영상을 새로 보내며 이전 ROI·마스크·추론 상태를 재사용하지 않는다.
reference 원본을 수정한 경우 서버를 재시작해 캐시를 갱신한다.
`GET /reference?class_id=black_pepper_shaker&scene=Home/Uon_1F_Home/floor`는
reference 파일·BBox·identity를 검사하며 모델 추론을 하지 않는다. 공통 CLI의 `--check`가 사용한다.

기본 서버 설정:

- 포트: `13347` (`--port`)
- reference root: `/media/uon/data1/Dataset_2026/review/real_v1` (`--reference-root`)
- metadata: `/media/uon/data1/Dataset_2026/review/2026_real_world_test_objects_metadata.csv` (`--metadata`)
- confidence: `0.5` (`--confidence-threshold`)

reference scene 기본값은 `Home/Uon_1F_Home/floor`이며 아래 CLI의 `--reference-scene`으로 선택한다.
각 scene의 `reference/reference_index.json`에서 이미지·BBox를 찾고 reference metadata identity/hash를 검증한다.
CAD CSV와 reference CSV의 `obj_NNN` 번호가 다르므로 연결에는 canonical class ID를 쓴다.
reference 원본은 읽기만 하고 결과는 TiPToP 작업 폴더에 저장한다.

## 서버 API

| 요청 | 기능 |
| --- | --- |
| `GET /health` | 모델 준비 상태·소스 경로·추론 횟수 조회 |
| `GET /reference?class_id=…&scene=…` | reference 이미지·BBox·identity 확인, 추론 없음 |
| `POST /segment-reference` | 현재 RGB와 reference로 검출·마스크 추론 |

추론 입력은 NumPy NPZ의 `rgb`(uint8 H×W×3), `class_id`(canonical 이름),
`reference_scene`(reference root 상대 경로)다. 출력 NPZ는 `boxes_xyxy`(N×4 원본 픽셀),
`masks`(bool N×1×H×W), `scores`(N), `metadata`(JSON 문자열)다.
미검출은 N=0이며 클라이언트가 작업 실패로 처리한다.
1008×1008 canvas의 위쪽에는 reference, 아래쪽에는 현재 영상을 letterbox하고,
아래쪽 검출만 원래 영상 좌표로 복원한다. 현재 영상의 모델 입력 상태는 요청마다 새로 만든다.

## 모드 선택

Isaac GUI·관측 서버와 FoundationStereo·FoundationPose는 [가상 실행 가이드](known-object-pick-place.md#실행-절차)를 따른다.
모드 선택은 시뮬레이션/실물 어댑터 선택과 독립적이다. 아래 명령은 계획까지만 수행한다.

```bash
# Reference 모드: Gemini 허용 옵션·API 키 불필요
pixi run python -m tiptop.scripts.run_known_pick_place \
  --perception-mode sam3_reference \
  --task 'black_pepper_shaker를 집고 빈곳에 놓아'

# 기존 Gemini ER 모드
pixi run python -m tiptop.scripts.run_known_pick_place \
  --perception-mode gemini_er --allow-gemini-once \
  --task '후추통 집고 빈 곳에 놓아'
```

가상 물리 실행은 GUI를 띄운 뒤 `--execute`를 추가한다. 실물은 기존 측정 프로파일과 명시적 gateway 요건을 유지한다.
`--check`를 추가하면 새 영상과 서버/reference 연결만 확인하고 종료한다. `--execute`와 함께 줘도
추론·동작은 수행하지 않는다. 자세한 항목은 [연결 사전 검사](known-object-pick-place.md#연결-사전-검사)를 참고한다.
접근각 상한은 기본3°이며 자동 확대하지 않는다. 후보가 없으면 실패하며 필요 시 사용자가 조정한다.
`--interactive`를 추가하면 새 지시마다 관측·추론·계획을 수행한다.
`--sam3-url`은 서버 주소를, `--reference-scene`은 reference 환경을 바꾼다.
반복 입력은 선택한 모드를 유지한다. 모드를 바꾸려면 작업 CLI를 해당 옵션으로 다시 실행하면 되고 Isaac 장면을 리셋할 필요는 없다.

reference 모드의 지시는 `<canonical_class_id>를 집고 빈곳에 놓아` 형식이다.
`--output`은 생략 가능하다. 기본값은 `tiptop_outputs/m1013/tasks/task-001`이며,
기존 기록이 있으면 번호를 자동 증가시킨다.
실제 저장 경로는 `TASK_OUTPUT`(대화형 세션 루트는 `SESSION_OUTPUT`)으로 출력한다.
한국어 물체명 해석·임의 목적지 지시는 이 모드에서 지원하지 않는다.
등록되지 않은 클래스, reference 누락, 미검출, 잘못된 마스크는 해당 작업 실패로 기록한다.
자동 Gemini 전환·다른 환경 reference 재시도·confidence 완화·장면 변경은 하지 않는다.
SAM3 모드에서는 `--allow-gemini-once`와 진단용 `--active-task`를 함께 사용할 수 없다.

`semantic-task.json`의 `semantic_source`와 `reference_diagnostics`에 reference 경로·identity·점수·선택 후보가 남는다.
미검출의 reference 진단 정보는 `task-result.json`에 남고 다음 명시적 지시를 받을 수 있다.
검출 후보가 여러 개이면 최고 점수 하나를 선택한다.
