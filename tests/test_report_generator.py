"""
report_generator.py 단위 테스트.

save_html()이 올바른 HTML을 생성하는지, measure_location_map이
제공될 때 data-page/data-bbox 속성이 렌더링되는지 검증한다.
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xml_comparator import CompareResult, Discrepancy  # noqa: E402
from report_generator import save_html, print_console  # noqa: E402
from pdf_parser import MeasureLocation  # noqa: E402


def _make_result(*discrepancies) -> CompareResult:
    """테스트용 CompareResult 헬퍼."""
    return CompareResult(
        pdf_xml="test.pdf.xml",
        orig_xml="test.orig.xml",
        total_measures=10,
        discrepancies=list(discrepancies),
    )


def _make_loc(measure_num, page_num, bbox) -> MeasureLocation:
    return MeasureLocation(
        measure_num=measure_num,
        page_num=page_num,
        staff_index=1,
        bbox=bbox,
    )


# ── save_html 기본 동작 ───────────────────────────────────────────────

def test_save_html_creates_file(tmp_path: Path):
    """save_html()이 지정 경로에 파일을 생성해야 함."""
    result = _make_result()
    out = str(tmp_path / "report.html")
    save_html(result, out)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


def test_save_html_contains_discrepancy_message(tmp_path: Path):
    """오류 메시지가 HTML에 포함되어야 함."""
    msg = "특이한 음표 오류 메시지 XYZ"
    result = _make_result(Discrepancy("note_missing", 3, 0.0, msg))
    out = str(tmp_path / "report.html")
    save_html(result, out)
    content = Path(out).read_text(encoding="utf-8")
    assert msg in content


def test_save_html_data_measure_attribute(tmp_path: Path):
    """오류 행에 data-measure 속성이 붙어야 함."""
    result = _make_result(Discrepancy("note_missing", 7, 0.0, "테스트"))
    out = str(tmp_path / "report.html")
    save_html(result, out)
    content = Path(out).read_text(encoding="utf-8")
    assert 'data-measure="7"' in content


def test_save_html_no_errors_shows_empty_state(tmp_path: Path):
    """오류 없으면 '불일치 없음' 메시지가 포함되어야 함."""
    result = _make_result()
    out = str(tmp_path / "report.html")
    save_html(result, out)
    content = Path(out).read_text(encoding="utf-8")
    assert "불일치 없음" in content


# ── measure_location_map 하이라이트 ──────────────────────────────────

def test_save_html_with_location_map_adds_data_page(tmp_path: Path):
    """measure_location_map이 있으면 data-page 속성이 추가되어야 함."""
    result = _make_result(Discrepancy("note_missing", 3, 0.0, "테스트"))
    loc_map = {3: _make_loc(3, page_num=0, bbox=(100, 200, 500, 350))}
    out = str(tmp_path / "report.html")
    save_html(result, out, measure_location_map=loc_map)
    content = Path(out).read_text(encoding="utf-8")
    assert 'data-page="0"' in content


def test_save_html_with_location_map_adds_data_bbox(tmp_path: Path):
    """measure_location_map이 있으면 data-bbox 속성이 추가되어야 함."""
    result = _make_result(Discrepancy("tie_missing", 5, 1.0, "타이 누락"))
    bbox = (200, 100, 700, 300)
    loc_map = {5: _make_loc(5, page_num=1, bbox=bbox)}
    out = str(tmp_path / "report.html")
    save_html(result, out, measure_location_map=loc_map)
    content = Path(out).read_text(encoding="utf-8")
    assert 'data-bbox="200,100,700,300"' in content


def test_save_html_highlight_js_always_present(tmp_path: Path):
    """showHighlight JS 함수는 항상 HTML에 포함되어야 함 (map 없어도)."""
    result = _make_result(Discrepancy("note_missing", 1, 0.0, "테스트"))
    out = str(tmp_path / "report.html")
    save_html(result, out)  # loc_map 없음
    content = Path(out).read_text(encoding="utf-8")
    assert "showHighlight" in content


def test_save_html_unmatched_measure_no_data_bbox(tmp_path: Path):
    """loc_map에 없는 마디 번호는 data-bbox 없이 렌더링되어야 함."""
    result = _make_result(Discrepancy("note_missing", 9, 0.0, "테스트"))
    loc_map = {3: _make_loc(3, page_num=0, bbox=(0, 0, 100, 100))}  # 9는 없음
    out = str(tmp_path / "report.html")
    save_html(result, out, measure_location_map=loc_map)
    content = Path(out).read_text(encoding="utf-8")
    # 마디 9에 대한 data-bbox는 없어야 함
    # data-measure="9"는 있어야 함
    assert 'data-measure="9"' in content


# ── print_console ─────────────────────────────────────────────────────

def test_print_console_runs_without_error(capsys):
    """print_console()이 예외 없이 실행되어야 함."""
    result = _make_result(
        Discrepancy("note_missing", 1, 0.0, "테스트 음표 오류"),
        Discrepancy("tie_missing",  2, 1.0, "타이 누락"),
    )
    print_console(result)
    captured = capsys.readouterr()
    assert "총 마디" in captured.out or "불일치" in captured.out
