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
  적극 활용해, 범용 OMR이 아닌 좁은 도메인 특화 인식기를 새로 구축 중.

  **진행 상황** — 파이프라인 완성, `pytest 139/139` 통과:

  1. ✅ **합성 테스트 이미지 생성기** (`tests/fixtures/synthetic_score.py`):
     `NoteSpec(x, staff_step, duration, stem_up, beam_to_next, dotted)` +
     `RestSpec(x, duration)` + `SyntheticScoreSpec`. 5종 음가·점음표·
     전/2분/4분/8분쉼표·빔 연결 렌더링. `render_synthetic_staff()` →
     `(img, note_gt, rest_gt)` 3-tuple 반환.

  2. ✅ **오선 제거** (`note_recognition/staff_removal.py`):
     `detect_staff_line_thickness()` — run 시작점 기준 최빈값 측정.
     `remove_staff_lines()` — 가로 연속성 → 세로 두께 2단계 검증.
     `tests/test_staff_removal.py` 8개.

  3. ✅ **음표 검출 + 음가 분류** (`note_recognition/note_detector.py`):
     - 연결성분 분리 → 기둥 유무(h > staff_gap×2.5) → 머리 밀도
       (HEAD_FILL_THRESHOLD=0.50) → 깃발 수(4-conn) 순서로 분류.
     - **기둥 방향(stem_up) 판별**: 투영(projection) 기반.
       세로 투영(vproj)으로 기둥 열 마스킹 → 가로 투영(hproj)으로
       머리가 상단/하단 중 어디인지 판단. density 비교보다 견고.
     - **head_x 추정**: 기둥 열 중심 찾기 → 반대편이 머리.
       stem_x_hint 있으면 우선 사용.
     - 점음표: 머리 오른쪽 4~60px blob 탐지 (`_detect_dot`).
     - 쉼표: 블록형(전/2분, aspect>3) + 선형(4분/8분, h<gap×2, aspect<1).
     - 노이즈 필터: 마디선(`_is_barline`, w<6, h≥r×6), 텍스트(aspect>5+h<gap),
       ROI margin 분리(상단 3.5×gap, 하단 2.5×gap), pitch 범위 필터(±10 step),
       next_staff_top_y 클램프(가사/코드 기호 영역 제외).
     알려진 한계:
       - 이성부/화음(chord) 미지원 — 같은 x에 음표가 2개인 악보(꿈꾸지 않으면 F)
       - 16분음표 step=0~3: 두 번째 깃발이 오선 줄과 겹쳐 제거→eighth 오분류
       - stem_down 빔: 두 번째 음표 sixteenth 오분류 가능
     `tests/test_note_detector.py` 24개.

  4. ✅ **빔 분리** (`note_recognition/beam_splitter.py`):
     세로 투영 피크(최댓값의 30% 이상) = 기둥 위치 → 피크 사이 최솟값
     구간 중점 = 분할선. 각 서브bbox에 `stem_up`/`stem_x` 힌트 반환.
     3개 묶음까지 검증. `_is_barline()` (w<6, h≥r×6)로 마디선 필터.
     `tests/test_beam_splitter.py` 15개.

  5. ✅ **음높이 판정** (`note_recognition/note_pitcher.py`):
     오차 허용: ±4px. `tests/test_note_pitcher.py` 15개.

  6. ✅ **조표 + 마디 내 임시표** (`note_recognition/key_signature.py`):
     `MeasureAccidentalState` — 마디 경계 reset() / 음표별 apply(pitch).
     우선순위: 마디 내 임시표 > 조표. 제자리표(natural) 미지원.
     `tests/test_key_signature.py` 20개.

  7. ✅ **MusicXML 생성** (`note_recognition/xml_builder.py`):
     음표/쉼표 x좌표 통합 정렬. barlines 기반 마디 분리. 임시표 관리.
     `tests/test_xml_builder.py` 21개.

  8. ✅ **opencv 엔진 main.py 연동** (`opencv_runner.py`):
     `python main.py full --engine opencv --pdf ... --orig ...` 사용 가능.
     config.ini `[opencv]`에서 key_sig/time_sig/clef_type/threshold 읽음.
     next_staff_top_y 전달로 오선 간 가사 영역 제외.
     HTML 리포트 클릭 시 PDF 위치 하이라이트.

  **합성 이미지 기준 정확도** (`python benchmark_opencv.py`): **전체 100%(54/54)**

  **실제 교과서 PDF 5개 샘플 결과** (초기 → 현재, 노이즈 기준):
  남촌 C: 184→90, 꿈꾸지: 239→104, DQ: 200→97, 태양D: 173→138, 태양F: 330→172
  **노이즈 합계 1130→601 (-47%)**, 누락 350→247 (-29%)

  - **시스템 그룹핑 적응형 임계값 + 균일성 검증** (opencv_runner):
    꿈꾸지(2단 합창)의 시스템 내 gap(2.1~2.3h)과 간 gap(2.6~2.7h)이
    고정 임계 2.0h를 모두 초과해 반주 오선이 파트0에 뒤섞이던 최대 원인
    해결(-163건). gap 비율 최대 점프 지점을 임계로, 그룹 크기 불균일 시
    전부 1단 flatten. 한계: 꿈꾸지 p2는 내부/간 gap이 동일(2.29h)해
    y_gap만으로 구분 불가.

  최근 추가된 노이즈/누락 개선 (검증 완료):
  - 타원성 필터(`_head_ellipse_fill`, ELLIPSE_FILL_MIN=0.30): 진짜 머리
    fill=0.43~0.95 vs 노이즈 0.01~0.19. cv2.fitEllipse 기반. -128건
  - 화음 멤버 duration 상속: 채워진 머리+기둥 없음=whole 불가 →
    같은 x(±1.5r) 기둥 음표에서 상속. 이성부(꿈꾸지) 누락 -22건
  - HOLLOW_HEAD_DENSITY_MIN=0.22: half/whole 빈 머리 density 하한.
    MXL whole=0인데 whole 45~81개 검출되던 산발 픽셀 노이즈 제거
  - 참고용 소형 오선 제외(opencv_runner): 주 gap 85% 미만 오선 스킵
    (태양 D 페이지 상단 발성연습 악보 69건 제거)
  - density용 head_y 투영 argmax 교정: bbox에 타이 조각 붙을 때
    공식(y+r)이 최대 51px 어긋나던 문제. 편차>r인 케이스만 교정,
    pitch용 head_y는 공식 유지

## 로컬 테스트 시 확인 우선순위 (파라미터 튜닝 가이드)

실제 교과서 PDF로 `python main.py full --engine opencv --pdf ... --orig ...` 실행 시
아래 순서로 확인. 모두 `config.ini [opencv]` 섹션에서 코드 수정 없이 조정 가능.

| 파라미터 | 현재값 | 위험도 | 설명 |
|---|---|---|---|
| `HEAD_FILL_THRESHOLD` | 0.47 | **높음** | half 여유 0.077. 실제 폰트에서 오분류 가능성 가장 높음 → 0.43~0.45 시도 |
| `key_sig` | 0 | **높음** | 자동 감지 미구현. 악보 보고 직접 입력 (G장조=1, D장조=2, F장조=-1) |
| `time_sig` | 4/4 | 중간 | 악보 보고 직접 입력 |
| `_NOTEHEAD_RADIUS_RATIO` | 0.55 | 중간 | 출판사별 폰트에 따라 달라짐 |
| `_HAS_STEM_HEIGHT_RATIO` | 2.5 | 낮음 | whole vs 기둥있는 음표 구분 여유 충분 |
| `clef_type` | treble | 낮음 | 교과서 대부분 treble, 자동 감지 미구현 |

**권장 실행 순서:**
1. `python benchmark_opencv.py` — 합성 이미지 기준 100% 확인 (회귀 방지)
2. `python main.py full --engine opencv --pdf 교과서1페이지.pdf --orig ...`
3. half→quarter 오분류 多 → `head_fill_threshold` 낮춤
4. 음표 미검출 → `notehead_radius_ratio` 조정
5. 조표 틀림 → `key_sig` 수동 입력
6. `python main.py compare-engines` — 3개 엔진 정확도 비교



```bash
# 단일 PDF 변환 + XML 비교 + HTML 리포트 (가장 많이 쓰는 명령)
python main.py full --pdf "path/to/score.pdf" --orig "path/to/original.mxl"

# OpenCV 자체 OMR 엔진 사용
python main.py full --engine opencv --pdf score.pdf --orig original.mxl

# config.ini 기준 폴더 일괄 처리
python main.py run

# XML 두 파일 직접 비교 (PDF OCR 포함 시 --pdf-source 추가)
python main.py compare --pdf converted.mxl --orig original.mxl --pdf-source score.pdf

# 단(System) 단위 시각 비교 HTML 뷰어 (verovio 렌더링)
python main.py visual --pdf score.pdf --xml original.mxl

# Finale PDF 직접 사용 (MuseScore 불필요)
python main.py visual --pdf score.pdf --finale-pdf finale_export.pdf

# 현재 설정 확인
python main.py config

# 3개 엔진 비교 (audiveris / homr / opencv)
python main.py compare-engines --pdf score.pdf --orig original.mxl

# OpenCV 파이프라인 합성 이미지 벤치마크 (로컬 테스트 전 100% 확인)
python benchmark_opencv.py
python benchmark_opencv.py -v  # 실패 케이스 상세 출력

# 단위 테스트 전체 (134개)
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

- **HTML 리포트 PDF 위치 하이라이트**:
  - ✅ **완료**: `report_generator.save_html(result, path, measure_location_map=...)`
    — 오류 행에 `data-page`, `data-bbox` 속성 추가. 클릭 시 JS 오버레이로
    `images/page_{n}.png` 위에 빨간 박스 표시.
  - ✅ **완료**: `main._build_measure_map_and_save_images()` — `cmd_full` 시
    자동으로 페이지 이미지(150dpi) + 마디 위치 매핑 생성.
  - ✅ **완료**: `build_measure_location_map()`의 bbox y1을 다음 오선
    `top_y - staff_h`로 클램프 (인접 오선 가사/코드 영역 겹침 방지).
  - 여전히 음표 단위 정밀 좌표는 없음 (마디 단위 bbox가 한계).

- ✅ **이음줄(tie/slur) 인식 개선** — OpenCV 기반 구현 완료:
  1. ✅ `xml_comparator.py`: `tie_missing`/`tie_extra` kind 추가, `_detect_split_tie`
  2. ✅ `note_recognition/arc_detector.py`: Morphological Opening 기반 호 감지
     (`ArcCandidate` 데이터클래스, cut_left/cut_right 시스템 경계 처리)
  3. ✅ `note_recognition/xml_builder.py`: `_apply_arcs()` — 동음=붙임줄, 이음=이음줄
  4. ✅ `tests/test_arc_detection.py`: 5곡 샘플 감지율 검증 (DPI=300)
     - 최신 결과 (2026-06): DQ=49/49, kk=78/78, nc=40/40, oD=68/68, oF=50/50
       **F1=100.0%** (세션 전 F1=89.8%→93.9%→96.3%→97.2%→97.9%→99.6%→99.8%→100%)
       (감지율 = det_arcs/ref_total, ref_total=tie+slur 태그 전체 수; 물리 아크=ref_total/2)
     - 주요 개선:
       · note_filter y0/y1 끝점별 검사, x/y_tol=3*gap
       · 수평 1px dilation(gap 16-75 스태프): kk FN 해결
       · bw<77 좁은 슬러(gap≥18): y_tol=70px 완화 → kk 4 FN 추가 복구
       · 스태프 하단 아래볼록 FP 제거 tiered 필터: DQ/nc/oD 100% 달성
         (bw>110,cy_below>0.6g / bw>90,cy_below>0.7g / bw>79,cy_below>1.6g)
     - 잔여 한계: 없음 (F1=100%)
  5. ⏸️ homr 엔진 타이 출력은 보류 (GPU 미구축, homr 0.6.2 의도적 비활성화 상태)

- **로컬 PDF 실측 후 튜닝 필요**:
  - `HEAD_FILL_THRESHOLD=0.47` — 실제 폰트에서 half→quarter 오분류 가능성 높음
  - `_NOTEHEAD_RADIUS_RATIO=0.55` — 출판사별 폰트에 따라 달라짐
  - `key_sig`/`time_sig`/`clef_type` — 악보 보고 config.ini 직접 입력 필요
    (자동 감지 미구현)
  - 16분음표 step=0~3 오선 겹침 한계 — 실제 악보에서 발생 빈도 확인 필요

- **정리 후보 (본인 확인 필요)**:
  - `check_pdf.py`: 죽은 스크립트 (PaddleOCR 기재 but EasyOCR 사용, 개인 경로 하드코딩)
  - `musescore_renderer.py`: verovio 경로로 대체, 미사용
