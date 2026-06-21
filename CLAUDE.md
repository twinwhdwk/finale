# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

# 단위 테스트 (xml_comparator 타이 비교 로직)
python -m pytest tests/ -v
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
  3. `homr` 엔진 연동(`homr_runner.py`)을 Audiveris 대안으로 추가했으나,
     **homr 0.6.2(PyPI)는 slur/tie를 MusicXML에 출력하지 않도록 의도적으로
     비활성화되어 있음** (`build_note_chord()` 내 "Disabled slurs and ties
     until the detection is more robust" 주석, 호출부 주석 처리됨).
     → 현재 버전으로는 이음줄 개선 목적 달성 불가. `compare-engines`
       커맨드는 음높이/리듬 정확도 비교용으로만 유효.
     → GitHub `liebharc/homr` main 브랜치가 이후 업데이트되어 이 기능이
       풀렸는지 재확인 필요 (이 저장소 작업 시점엔 PyPI 0.6.2만 검증함,
       컨테이너 네트워크 제약으로 모델 다운로드/실제 추론 테스트는 못 함).
  - **권장 다음 단계**: 실제 Audiveris 변환 결과(`Finale_Ref/xmls_converted/`)로
    1·2번 로직을 검증 — 지금까지는 music21로 합성한 가짜 시나리오로만
    테스트함 (`tests/test_tie_comparison.py`). 실제 교과서 PDF 1개로
    `python main.py full --pdf ... --orig ...` 돌려서 `tie_missing`/
    `tie_suspect` 건수가 실제 채보 오류와 맞는지 사람이 확인 필요.
    homr는 모델 다운로드 가능한 로컬 PC에서 실측 후 재평가.

- `homr_runner.py`: ✅ 다중 페이지 PDF의 페이지별 결과(.musicxml) 병합 비교 구현 완료
  (`merge_page_musicxmls()` - 마디 번호 1부터 재부여하여 단일 합본 XML 생성,
  `main.py`의 `full --engine homr`/`compare-engines`에 자동 연결됨).
  `tests/test_homr_merge.py`로 검증 (실제 모델 없이 가짜 페이지 XML로 테스트).
  알려진 한계: 페이지 경계에서 조표/박자표가 실제로 바뀌는 악보는 부정확할 수 있음
  (단순 이어붙이기라 attributes 재선언을 하지 않음) - 실측 필요.

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

- `homr_runner.py`: 실제 모델 다운로드/추론을 거친 통합 테스트 아직 없음
  (이 컨테이너는 `release-assets.githubusercontent.com`이 네트워크
  화이트리스트에 없어 `homr --init` 모델 다운로드 불가). 로컬 PC에서
  `homr --init` 후 `python main.py compare-engines --pdf ... --orig ...`
  실행 결과로 검증 필요.
