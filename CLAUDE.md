# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 입력 데이터 특성 (중요 — OMR 알고리즘 설계의 전제 조건)

- **출처**: 출판사로부터 입수한 **디지털 PDF** (스캔본이 아님). 노이즈, 기울어짐(skew),
  그림자, 종이 질감 등이 사실상 없다고 가정 가능. Deskewing/Dewarping/디노이징처럼
  "지저분한 스캔"을 전제로 한 전처리 기법은 이 데이터에는 불필요하거나 효과가 미미함.
- **악보 난이도**: 중·고등학생 음악 교과서 수준. 복잡한 다성 오케스트라 악보가 아님.
- **성부 수**: **이성부(2성부) 이하**. 단선율이거나 소프라노/알토 2성부 정도까지만 존재.
  화성학적으로 복잡한 화음 진행이나 다중 보표(그랜드 스태프) 케이스는 거의 없음.
- **이 조건이 의미하는 것**: 범용 OMR(임의의 손글씨·사진·복잡한 오케스트라 악보까지
  다루는)에 필요한 견고성(robustness)은 여기서 우선순위가 낮다. 대신 "깨끗한 디지털
  인쇄물 + 단순한 성부 구조"라는 좁고 명확한 도메인에 특화된 정밀도를 높이는 쪽이 ROI가
  훨씬 높다. OpenCV 알고리즘 튜닝 시 이 가정을 적극 활용할 것 (예: 임계값을 노이즈
  대응용으로 보수적으로 잡을 필요 없음, 오선/음표 형태가 폰트 단위로 일관적이라는
  가정 활용 가능).

## OMR 엔진 방향 (homr는 보류)

- **homr(`homr_runner.py`)는 현재 사용하지 않음 (보류 상태).** 애초 도입 동기가
  "Audiveris보다 정교한 인식"이었으나, homr 0.6.2가 슬러/타이 인식 결과를 MusicXML
  출력에서 의도적으로 비활성화한 상태임을 확인 (TODO 섹션 참고). 이 문제가 해결되지
  않는 한 GPU/모델 다운로드 비용을 들여 쓸 이유가 약함. **GPU 환경이 갖춰지고 homr가
  업데이트되어 타이 출력이 복원되면 재검토.** 코드는 그대로 보존하되 기본 워크플로우
  (`main.py full`, `python main.py run`)에서는 Audiveris만 사용.
- **현재 핵심 방향: `note_recognition/` 패키지 — Audiveris를 쓰지 않는 자체 OpenCV
  음표 인식기.** 위 "입력 데이터 특성"(디지털 PDF, 2성부 이하, 중고등학생 수준)을
  적극 활용해, 범용 OMR이 아닌 좁은 도메인 특화 인식기를 새로 구축 중. 목표 음가:
  온음표/2분음표/4분음표/8분음표/16분음표(+ 부속 점음표·쉼표는 추후).

  **진행 상황** (단계별로 `tests/test_*.py`에서 검증):
  1. ✅ **합성 테스트 이미지 생성기** (`tests/fixtures/synthetic_score.py`):
     실제 PDF 없이도 ground truth가 100% 확실한 검증용 악보 이미지를 직접
     렌더링. 표준 음표 모양(채워진/빈 머리, 기둥, 깃발, 덧줄)을 OpenCV로
     그려 5종 음가(whole/half/quarter/eighth/sixteenth) 전부 생성 가능.
  2. ✅ **오선 제거** (`note_recognition/staff_removal.py`):
     `detect_staff_line_thickness()` - 오선 한 줄의 실제 두께를 run-length
     최빈값으로 측정 (버그 이력: run 시작점이 아닌 모든 y에서 재측정해
     최빈값이 항상 실제보다 작게 나오던 버그 발견·수정함, 두께 3을 1로
     오판했었음).
     `remove_staff_lines()` - 2단계 검증(가로 연속성 → 세로 run-length)으로
     오선만 정밀 제거. 1차 구현(세로 run-length만 사용)은 빈 음표머리(2분/
     온음표)의 타원 테두리가 오선과 두께가 비슷해 통째로 지워지는 회귀가
     실험으로 발견되어, 가로 연속성 기준을 추가해 해결. `tests/test_staff_removal.py`
     6개로 검증 (빈 영역 100% 제거, 채워진 머리 보존율 ≥70%, 빈 머리가
     오선으로 오인되지 않음, 5종 음가 전부 생존 확인).
  3. ✅ **음표 객체 분리 및 음가 분류** (`note_recognition/note_detector.py`):
     연결성분 → 기둥 유무/머리 밀도/깃발 개수로 5종 음가 분류.
     `note_recognition/beam_splitter.py` 신규: 빔으로 묶인 컴포넌트를
     세로 투영 피크 기반으로 개별 음표로 분할. 3개 묶음까지 검증.
     `tests/test_note_detector.py` 17개로 검증 (stem_up/down 전 음가,
     빔 그룹 2/3개, 혼합 마디). `pytest 43/43 통과`.
  4. ✅ **음높이 판정** (`note_recognition/note_pitcher.py`):
     `head_y_to_staff_step()` - head_y 픽셀 → staff_step 역산
     (`round((line4_y - head_y) * 2 / staff_gap)`).
     `staff_step_to_pitch()` - 높은/낮은음자리표별 온음계 매핑.
     `Pitch` 데이터클래스 - step/octave/accidental + music21 호환
     `name_with_octave`(예: "C4") + MIDI 번호 변환.
     `tests/test_note_pitcher.py` 13개로 검증 (landmarks E4~F5,
     Middle C=C4, 7step=1옥타브, MIDI 오름차순, 픽셀 오차 스냅,
     합성이미지 왕복 테스트, end-to-end E4/B4/D5/C4 판정 확인).
     `pytest 56/56 통과`.
  5. ✅ **MusicXML 생성** (`note_recognition/xml_builder.py`):
     ✅ **바라인 기반 마디 분리**: `barlines: list[int]` 파라미터로
     StaffZone.barlines를 직접 받아 head_x 비교로 마디 배정.
     barlines 없으면 박자 누산 fallback. 음표/쉼표 x좌표 기준 통합 정렬.
     `_assign_events_to_measures_by_barlines()` / `_by_beat()` 두 전략.
     ✅ **조표(key signature)**: `key_sig` 파라미터로 음이름에 accidental 반영.
     `key_signature.py` 신규: `get_accidental_map()` (Circle of Fifths 샵/플랫),
     `apply_key_signature()` (임시표 우선). `tests/test_key_signature.py` 14개.
     ✅ **점음표(dotted)**: is_dotted=True이면 quarterLength×1.5 + dots=1.
     ✅ **전/2분쉼표(rest)**: DetectedRest → music21 note.Rest() 변환.
     `pytest 89/89 통과`.
  6. ✅ **opencv 엔진 main.py 연동** (`opencv_runner.py`):
     barlines, x_start 모두 연결. `compare-engines` 3개 엔진 비교.
  7. ✅ **패키지 공개 API** (`note_recognition/__init__.py`):
     `from note_recognition import detect_notes, save_musicxml` 한 줄로 사용 가능.
     헤더 영역 마스킹: `detect_notes(..., x_start=N)`으로 음자리표/박자표 영역 제외.

## Commands

```bash
# 단일 PDF 변환 + XML 비교 + HTML 리포트 (가장 많이 쓰는 명령)
python main.py full --pdf "path/to/score.pdf" --orig "path/to/original.mxl"

# config.ini 기준 폴더 일괄 처리
python main.py run

# XML 두 파일 직접 비교 (PDF OCR 포함 시 --pdf-source 추가)
python main.py compare --pdf converted.mxl --orig original.mxl --pdf-source score.pdf

# 단(System) 단위 시각 비교 HTML 뷰어 (verovio 렌더링)
python main.py visual --pdf score.pdf --xml original.mxl

# Finale PDF 직접 사용 (MuseScore 불필요)
python main.py visual --pdf score.pdf --finale-pdf finale_export.pdf

# 슬라이스 이미지를 파일로 덤프 (디버깅)
python main.py visual --pdf score.pdf --xml original.mxl --dump-slices

# 현재 설정 확인
python main.py config

# Audiveris vs homr 엔진 비교 (음높이/리듬 정확도 대조용, 타이는 homr가 미지원)
python main.py compare-engines --pdf score.pdf --orig original.mxl

# 단위 테스트 (xml_comparator 타이 비교, 음표 인식 등 전체)
python -m pytest tests/ -v

# 합성 악보 이미지 직접 생성해서 눈으로 확인 (디버깅용)
python tests/fixtures/synthetic_score.py
```

## 데이터 경로 (config.ini)

```
Finale_Ref/pdfs/           교과서 스캔본 PDF (~293개)
Finale_Ref/xmls/           Finale 원본 MusicXML (.mxl)
Finale_Ref/xmls_converted/ Audiveris OMR 변환 결과
Finale/reports/            HTML 리포트 출력
```

파일명이 100% 일치해야 매칭된다 (`I Have a Dream D (중등 음악1 천재).pdf` ↔ `.mxl`). 출판사(천재, 비상, 음악과 생활 등)가 다르면 완전히 다른 악보다.

## 아키텍처

두 개의 독립적인 파이프라인이 있다.

### 파이프라인 1: 기호 비교 (Symbolic)

`full` / `run` / `compare` 커맨드가 실행하는 3-트랙 비교:

```
PDF → Audiveris (pdf_to_xml.py) → OMR XML
PDF → pdf_parser.py (OCR) → 코드 기호, 가사
                              ↓
xml_comparator.compare()   ← 원본 Finale XML
  ├─ 트랙1: 음표/쉼표/화음  (music21 파싱, offset 단위 비교)
  ├─ 트랙2: 코드 기호       (Tesseract OCR vs ChordSymbol)
  └─ 트랙3: 가사            (EasyOCR vs Lyric, recall 기반 유사도)
                              ↓
report_generator.py → HTML 리포트 (무시/수정완료/재확인 버튼)
```

### 파이프라인 2: 시각 비교 (Visual)

`visual` 커맨드가 실행하는 이미지 단위 비교:

```
교과서 PDF → system_slicer.slice_pdf_to_systems()  (dpi=600 필수)
               └─ pdf_parser._detect_staves() 재사용
               └─ _detect_barlines()로 단별 마디 수 계산 → measures_per_system

Finale 소스 (택1):
  [XML]  → xml_to_systems.xml_to_systems(measures_per_system=...)
              └─ _force_system_layout(): <print new-page="yes"/> 삽입
              └─ verovio breaks="encoded" → 페이지당 1단 SVG
  [PDF]  → system_slicer.slice_pdf_to_systems() 직접 슬라이싱

system_slicer.pair_systems() → 단 번호로 1:1 매칭
visual_report.save_visual_html() → HTML 뷰어 (Space=통과, Enter=수정필요)
```

## 핵심 알고리즘 주의사항

**바코드(마디선) 감지** (`pdf_parser._detect_barlines`):
- 95% raw col_sum 임계값 사용. MORPH_OPEN이나 수평선 제거 후 방식은 모두 실패 이력 있음
- 시스템 오른쪽 경계선 동적 제거: `parse_page()` 내에서 spread < 3% AND avg > 78% 조건으로 제거
- `len(_detect_barlines(...))` = 실제 마디 수 (NOT +1)

**오선 감지 DPI**: `_detect_staves()`는 **반드시 600 DPI** 필요. 300 DPI에서는 오선 1개만 감지되는 실패 이력 있음 (`slice_pdf_to_systems(dpi=600)` 기본값 유지).

**verovio 레이아웃 강제** (`xml_to_systems._force_system_layout`):
- `breaks="encoded"` 모드에서는 `<print new-system="yes"/>`가 **무시**된다. 반드시 `<print new-page="yes"/>`를 사용해야 verovio가 페이지 브레이크로 인식함
- ET로 직렬화하면 XML 선언/DOCTYPE이 사라져 verovio 파싱이 실패할 수 있음 → **문자열 조작(regex)으로만** 삽입. ET는 measure number 읽기에만 사용
- `systemMaxPerPage`는 `breaks="encoded"` 모드에서 무시된다 (auto/smart 전용)

**ChordSymbol 서브클래스 버그** (`xml_comparator._note_dict`):
- `ChordSymbol`은 `chord.Chord`의 서브클래스 → `getElementsByClass`에 포함됨
- `isinstance(el, ChordSymbol): continue`로 명시적 제외 필수

**가사 비교** (`xml_comparator`):
- `_lyric_list_by_verse()`: 절(verse) 번호별 분리, 최고 유사도 절 선택
- `_lyric_similarity()`: Jaccard가 아닌 recall 방식 (`len(ok & pk) / len(ok)`)
- 원본 한글 2자 미만 마디는 비교 생략 (`len(orig_korean) < 2: continue`)

**Pickup measure**: `.number == 0`은 Python에서 falsy → `_m_num()` 헬퍼로 `is not None` 체크

**verovio 한글 경로**: 사용자명에 한글 포함 시 C++ 레이어에서 data 경로 오류 발생. `C:/verovio_data` 폴더에 패키지 data 복사해야 함 (`xml_to_systems._init_verovio()` 참조). `verovio.setDefaultResourcePath()`는 `loadData()` 이전에 호출해야 함.

## 외부 도구 설정

| 도구 | 용도 | config.ini 키 |
|------|------|--------------|
| Audiveris | PDF → MusicXML OMR | `[audiveris] path` |
| Tesseract | 코드 기호 OCR | 경로 하드코딩 (`C:\Program Files\Tesseract-OCR\`) |
| EasyOCR | 가사 OCR (한글+영문) | Python 패키지, GPU 없음 |
| verovio | MusicXML → SVG (파이프라인 2 전용) | Python 패키지 |
| MuseScore 4 | (미설치) `musescore_renderer.py`에 코드 있지만 현재 미사용 | `[musescore] path` |

OCR 설정: Tesseract PSM 6, whitelist `ABCDEFGabcdefgmM#b1234567/`, 신뢰도 > 40. EasyOCR 신뢰도 > 0.2.

## TODO (미구현)

- HTML 리포트에서 오류 클릭 시 PDF 원본 위치 하이라이트 (설계 문서: `~/.claude/projects/.../todo_pdf_highlight.md`)
  - ✅ 좌표 매핑 기반 작업 완료: `pdf_parser.py`에 `StaffZone.measure_to_x_range()`/
    `measure_bbox()`(마디 번호 → 픽셀 bbox, `x_to_measure()`의 역함수),
    `build_measure_location_map()`(여러 페이지를 가로지르는 절대 마디 번호 →
    페이지+bbox 전역 매핑, `main._extract_pdf_data()`의 절대 마디 번호
    누산 규칙과 동일하게 맞춤) 추가. `tests/test_measure_location.py`
    9개로 검증 (역함수 일관성, 페이지/오선 경계 넘는 번호 이어짐, y좌표
    음수 클램프 등).
  - 아직 안 한 것: (1) 이 매핑을 실제 HTML 리포트(`report_generator.py`)에
    연결 - 클릭 시 PDF를 이미지로 보여주고 bbox 위치에 박스를 그리는
    프론트엔드 작업. (2) PDF를 페이지 이미지로 변환해 HTML에 내장하거나
    별도 서빙하는 방식 결정 필요 (base64 inline vs 별도 정적 파일).
  - ✅ `main._extract_pdf_data()`의 절대 마디 번호 누산 로직을
    `pdf_parser.iter_zones_with_start_measure()`(신규 공유 헬퍼)로
    리팩터링해 중복 제거 완료. `build_measure_location_map()`이 쓰는
    `iter_absolute_measures()`와 같은 누산 규칙을 공유하는지
    교차 검증 테스트(`test_iter_zones_with_start_measure_matches_iter_absolute_measures`,
    `test_extract_pdf_data_style_chord_numbering_matches_location_map`)로 확인.
  - 여전히 음표 단위 정밀 좌표는 없음 (Audiveris 결과에 좌표 정보가 없어
    마디 단위 bbox가 한계. 음표 단위로 가려면 Audiveris .omr 중간 파일이나
    homr `--write-staff-positions` 옵션 활용 검토 필요 - 별도 트랙).

- **이음줄(tie/slur) 인식 개선** — 1단계 완료, 다음 단계 정리:
  1. ✅ `xml_comparator.py`에 PDF측 `el.tie` 직접 비교 로직 추가 완료
     (`tie_missing`/`tie_extra` kind 신설, `tests/test_tie_comparison.py`로 검증)
  2. ✅ 인접 동일음높이 음표 합산 비교(`_detect_split_tie`) 추가 완료
  3. ⏸️ **보류**: `homr` 엔진 연동(`homr_runner.py`)을 Audiveris 대안으로
     추가했으나, **homr 0.6.2(PyPI)는 slur/tie를 MusicXML에 출력하지 않도록
     의도적으로 비활성화되어 있음** (`build_note_chord()` 내 "Disabled
     slurs and ties until the detection is more robust" 주석, 호출부
     주석 처리됨). GPU 비용 대비 이득이 없어 보류 결정 (상단 "OMR 엔진
     방향" 참고). GPU 환경 구축 + homr 쪽 타이 출력 복원 시 재검토.
  - **현재 권장 방향**: 1·2번(순수 XML 후처리)을 실제 Audiveris 결과로
    검증 — 지금까지는 music21로 합성한 가짜 시나리오로만 테스트함
    (`tests/test_tie_comparison.py`). 실제 교과서 PDF 1개로
    `python main.py full --pdf ... --orig ...` 돌려서 `tie_missing`/
    `tie_suspect` 건수가 실제 채보 오류와 맞는지 사람이 확인 필요.
    병행해서 OpenCV 기반 음표/오선 인식 정밀도를 높이는 작업(아래 신규
    섹션) 진행.

- `homr_runner.py`: ✅ 다중 페이지 PDF의 페이지별 결과(.musicxml) 병합 비교 구현 완료
  (`merge_page_musicxmls()` - 마디 번호 1부터 재부여하여 단일 합본 XML 생성,
  `main.py`의 `full --engine homr`/`compare-engines`에 자동 연결됨).
  `tests/test_homr_merge.py`로 검증 (실제 모델 없이 가짜 페이지 XML로 테스트).
  알려진 한계: 페이지 경계에서 조표/박자표가 실제로 바뀌는 악보는 부정확할 수 있음
  (단순 이어붙이기라 attributes 재선언을 하지 않음) - 실측 필요.
  ⏸️ 코드는 완성 상태로 보존하되 **사용은 보류** (상단 "OMR 엔진 방향" 참고).

- **정리 후보 (삭제하지 않고 보류, 본인 확인 필요)**:
  - `check_pdf.py`: 최초 커밋(`f065f7b`) 이후 한 번도 수정 안 된 초기
    프로토타입. docstring엔 "PaddleOCR" 사용한다고 적혀있지만 실제 코드는
    EasyOCR을 import함 - 코드와 설명이 어긋난 죽은 스크립트로 보임.
    `pdf_parser.py`가 사실상 이 로직을 흡수해 정식화한 상태. 개인 경로
    하드코딩(`C:\Users\강우현\...`)도 포함. 더 이상 안 쓰면 삭제 또는
    `scratch/`로 이동 권장.
  - `musescore_renderer.py`: CLAUDE.md 외부 도구 표에도 "미설치, 현재
    미사용"이라 명시됨. `xml_to_systems.py`(verovio 경로)로 대체된 듯.
    실제 사용 여부 확인 후 정리 검토.

- `homr_runner.py`: ⏸️ 보류 상태라 우선순위 낮음. 실제 모델 다운로드/추론을
  거친 통합 테스트는 아직 없음 (이 컨테이너는
  `release-assets.githubusercontent.com`이 네트워크 화이트리스트에 없어
  `homr --init` 모델 다운로드 불가). GPU 환경 구축 후 재검토 시
  `homr --init` 후 `python main.py compare-engines --pdf ... --orig ...`
  실행 결과로 검증.
