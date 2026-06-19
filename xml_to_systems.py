"""
MusicXML → 단(System) 단위 SVG 변환 모듈 (verovio 사용).

MuseScore 없이 MXL 파일을 단 단위로 렌더링합니다.
verovio 데이터 폴더가 한글 경로면 C:/verovio_data 로 복사 후 사용합니다.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import verovio

from system_slicer import SlicedScore, SystemSlice

# verovio 폰트 경로: 사용자명에 한글이 포함되면 C:/verovio_data 로 fallback
_VEROVIO_DATA_FALLBACK = "C:/verovio_data"
_PKG_DATA = str(Path(verovio.__file__).parent / "data")


def _init_verovio() -> None:
    """verovio 리소스 경로를 설정합니다. 한글 경로 문제를 자동으로 처리합니다."""
    if Path(_VEROVIO_DATA_FALLBACK).exists():
        verovio.setDefaultResourcePath(_VEROVIO_DATA_FALLBACK)
    else:
        verovio.setDefaultResourcePath(_PKG_DATA)


def _read_mxl(mxl_path: str) -> str:
    """MXL(압축 MusicXML) 또는 일반 XML 파일을 UTF-8 문자열로 읽습니다."""
    p = Path(mxl_path)
    if p.suffix.lower() in (".mxl",):
        with zipfile.ZipFile(mxl_path) as z:
            names = z.namelist()
            if "p1.musicxml" in names:
                entry = "p1.musicxml"
            else:
                entry = next(
                    (n for n in names if n.endswith((".musicxml", ".xml")) and "META" not in n),
                    None,
                )
                if entry is None:
                    raise RuntimeError(f"MXL 내부에서 악보 파일을 찾을 수 없습니다: {mxl_path}")
            return z.read(entry).decode("utf-8")
    else:
        return p.read_text(encoding="utf-8")


def _force_system_layout(xml_str: str, measures_per_system: list[int]) -> str:
    """
    MusicXML에 <print new-system="yes"/>를 삽입해 단 레이아웃을 강제합니다.
    measures_per_system = [6, 6, 5, 4, 5, 6] 형태로 각 단의 마디 수를 받습니다.

    ET 직렬화 대신 문자열 조작을 사용해 XML 선언/DOCTYPE을 원본 그대로 보존합니다.
    """
    # ET로는 오직 measure 번호 목록만 읽음 (직렬화하지 않음)
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return xml_str

    part = root.find('part')
    if part is None:
        return xml_str

    measures = part.findall('measure')
    total = len(measures)
    if total == 0:
        return xml_str

    # 시스템 브레이크가 필요한 0-based 인덱스 계산
    break_indices: set[int] = set()
    cumsum = 0
    for count in measures_per_system[:-1]:
        cumsum += count
        if cumsum < total:
            break_indices.add(cumsum)

    if not break_indices:
        return xml_str

    # 브레이크가 필요한 마디의 number 속성값 수집
    break_numbers: set[str] = set()
    for idx in break_indices:
        num = measures[idx].get('number')
        if num:
            break_numbers.add(num)

    # ── 문자열 조작으로 원본 XML 수정 ──────────────────────────────
    result = xml_str

    # 1. 기존 new-system 단독 print 요소 제거
    result = re.sub(r'[ \t]*<print\s+new-system="yes"\s*/>\s*', '', result)
    result = re.sub(r'[ \t]*<print\s+new-system="yes"\s*></print>\s*', '', result)
    # 2. 다른 속성과 함께 있는 new-system 속성만 제거
    result = re.sub(r'\s+new-system="yes"', '', result)

    # 3. 해당 마디 여는 태그 직후에 <print new-system="yes"/> 삽입
    for num in break_numbers:
        # <measure number="7"> 또는 <measure number="7" width="123.45">
        pattern = rf'(<measure\b[^>]*\bnumber="{re.escape(num)}"[^>]*>)'
        replacement = r'\1<print new-system="yes"/>'
        result = re.sub(pattern, replacement, result, count=1)

    return result


def xml_to_systems(
    xml_path: str,
    measures_per_system: list[int] | None = None,
    page_width: int = 2800,
    scale: int = 35,
) -> SlicedScore:
    """
    MusicXML / MXL 파일을 단(System) 단위 SVG로 렌더링하여 SlicedScore로 반환합니다.

    measures_per_system 지정 시 breaks="encoded"로 레이아웃을 강제합니다.
    각 SystemSlice.png_bytes 에 SVG 바이트(UTF-8)가 저장되고 is_svg=True 입니다.
    """
    _init_verovio()

    xml_data = _read_mxl(xml_path)

    if measures_per_system:
        xml_data = _force_system_layout(xml_data, measures_per_system)
        breaks_mode = "encoded"
    else:
        breaks_mode = "auto"

    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth":        page_width,
        "pageHeight":       600,
        "scale":            scale,
        "adjustPageHeight": True,
        "systemMaxPerPage": 1,
        "breaks":           breaks_mode,
        "footer":           "none",
        "header":           "none",
        "spacingStaff":     6,
        "spacingSystem":    3,
    })

    ok = tk.loadData(xml_data)
    if not ok:
        raise RuntimeError(
            "verovio: MusicXML 로드 실패.\n"
            "1. C:/verovio_data 폴더가 있는지 확인하세요.\n"
            "2. 없다면: Python 패키지 내 data 폴더를 C:/verovio_data 로 복사하세요."
        )

    total = tk.getPageCount()
    expected = len(measures_per_system) if measures_per_system else total
    if measures_per_system and total != expected:
        print(
            f"  [verovio] 경고: PDF 기준 {expected}단이나 verovio가 {total}단 렌더링."
            " (breaks=encoded 미적용 가능성)"
        )

    result = SlicedScore(source_path=str(xml_path))
    for page_num in range(total):
        svg_str = tk.renderToSVG(page_num + 1)
        result.systems.append(SystemSlice(
            abs_system  = page_num + 1,
            source_page = page_num,
            staff_idx   = 1,
            png_bytes   = svg_str.encode("utf-8"),
            is_svg      = True,
        ))

    result.total_systems = total
    print(f"  [verovio] {Path(xml_path).name}: {total}단 렌더링 완료")
    return result
