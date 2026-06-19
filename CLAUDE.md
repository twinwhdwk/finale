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

- HTML 리포트에서 오류 클릭 시 PDF 원본 위치 하이라이트 (bbox 좌표 필요, 설계 문서: `~/.claude/projects/.../todo_pdf_highlight.md`)
