"""
붙임줄(tie) / 이음줄(slur) 감지 진단 테스트.

tests/fixtures/scores/ 의 샘플 5곡에 대해:
  1. PDF 각 페이지 → note_recognition 파이프라인으로 arc 감지
  2. 참조 MXL에서 실제 tie/slur 개수 추출
  3. 감지된 arc 수 vs 참조값 비교 리포트 출력

실행:
    python tests/test_arc_detection.py
    python -m pytest tests/test_arc_detection.py -v -s
"""

from __future__ import annotations

import sys
import zipfile
import re
from pathlib import Path

import pytest

SCORES_DIR = Path(__file__).parent / "fixtures" / "scores"
SAMPLES = [f.stem for f in SCORES_DIR.glob("*.pdf")]


# ── 참조 MXL 파싱 ─────────────────────────────────────────────────────

def _read_mxl_content(mxl_path: Path) -> str:
    """mxl(zip) 에서 가장 큰 musicxml 엔트리를 UTF-8 문자열로 반환."""
    with zipfile.ZipFile(mxl_path) as zf:
        entries = [e for e in zf.infolist()
                   if e.filename.endswith(('.musicxml', '.xml'))
                   and e.file_size > 1000]
        if not entries:
            return ""
        entry = max(entries, key=lambda e: e.file_size)
        return zf.read(entry.filename).decode("utf-8", errors="replace")


def _count_ref(mxl_path: Path) -> tuple[int, int]:
    """(tie 개수, slur 개수) 반환."""
    content = _read_mxl_content(mxl_path)
    ties  = len(re.findall(r'<tied ',  content))
    slurs = len(re.findall(r'<slur ',  content))
    return ties, slurs


# ── PDF arc 감지 ──────────────────────────────────────────────────────

def _detect_arcs_from_pdf(pdf_path: Path, dpi: int = 200) -> dict:
    """
    PDF 전 페이지에서 arc를 감지하고 결과 요약을 반환.

    OCR(Tesseract)이 필요한 parse_page() 대신
    _detect_staves() / _detect_barlines() 만으로 오선 위치를 파악한다.

    Returns:
        {
            'arc_count': int,
            'note_count': int,
            'pages': int,
            'zones_per_page': list[int],
        }
    """
    import fitz
    from pdf_parser import _pdf_page_to_np, _detect_staves, _detect_barlines
    from note_recognition.staff_removal import detect_staff_line_thickness
    from note_recognition.note_detector import detect_notes

    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)
    doc.close()

    total_arcs  = 0
    total_notes = 0
    zones_per_page: list[int] = []

    for page_num in range(n_pages):
        img_rgb = _pdf_page_to_np(str(pdf_path), page_num=page_num, dpi=dpi)
        import cv2 as _cv2
        img = (_cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2GRAY)
               if img_rgb.ndim == 3 else img_rgb)
        staves = _detect_staves(img)        # list[(top_y, bot_y)]
        zones_per_page.append(len(staves))
        if not staves:
            continue

        first_top, first_bot = staves[0]
        line_thickness = detect_staff_line_thickness(
            img, [(first_top, first_bot)]
        )

        for top_y, bot_y in staves:
            staff_gap = max(1, (bot_y - top_y) // 4)
            barlines  = _detect_barlines(img, top_y, bot_y)
            x_start   = (max(0, barlines[0] - staff_gap * 3)
                         if barlines else img.shape[1] // 10)
            result = detect_notes(
                img,
                staff_top_y=top_y,
                staff_bot_y=bot_y,
                staff_gap=staff_gap,
                line_thickness=line_thickness,
                x_start=x_start,
            )
            total_arcs  += len(result.arcs)
            total_notes += len(result.notes)

    return {
        "arc_count":      total_arcs,
        "note_count":     total_notes,
        "pages":          n_pages,
        "zones_per_page": zones_per_page,
    }


# ── pytest 파라미터화 테스트 ──────────────────────────────────────────

@pytest.mark.parametrize("stem", SAMPLES)
def test_arc_detection(stem: str):
    pdf_path = SCORES_DIR / f"{stem}.pdf"
    mxl_path = SCORES_DIR / f"{stem}.mxl"

    assert pdf_path.exists(), f"PDF 없음: {pdf_path}"
    assert mxl_path.exists(), f"MXL 없음: {mxl_path}"

    ref_ties, ref_slurs = _count_ref(mxl_path)
    ref_total = ref_ties + ref_slurs

    det = _detect_arcs_from_pdf(pdf_path)
    det_arcs  = det["arc_count"]
    det_notes = det["note_count"]

    recall = det_arcs / ref_total if ref_total > 0 else None
    recall_str = f"{recall:.1%}" if recall is not None else "N/A (참조 0개)"

    print(f"\n{'─'*60}")
    print(f"곡명 : {stem}")
    print(f"페이지: {det['pages']}  오선계: {det['zones_per_page']}")
    print(f"감지된 음표 : {det_notes}")
    print(f"감지된 arc  : {det_arcs}")
    print(f"참조 tie    : {ref_ties}")
    print(f"참조 slur   : {ref_slurs}")
    print(f"참조 합계   : {ref_total}")
    print(f"arc 감지율  : {recall_str}")

    # 최소 기준: 참조가 있을 때 arc가 1개 이상 감지돼야 함
    if ref_total > 0:
        assert det_arcs > 0, (
            f"arc를 하나도 감지하지 못함 (참조 {ref_total}개)"
        )


# ── 단독 실행 모드 ────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"샘플 악보 경로: {SCORES_DIR}")
    print(f"발견된 샘플  : {len(SAMPLES)}곡\n")

    for stem in sorted(SAMPLES):
        pdf_path = SCORES_DIR / f"{stem}.pdf"
        mxl_path = SCORES_DIR / f"{stem}.mxl"
        if not pdf_path.exists() or not mxl_path.exists():
            print(f"[SKIP] {stem} — 파일 누락")
            continue

        ref_ties, ref_slurs = _count_ref(mxl_path)
        ref_total = ref_ties + ref_slurs

        print(f"▶ {stem}")
        print(f"  참조: tie={ref_ties}, slur={ref_slurs}, 합={ref_total}")
        print(f"  감지 중...", flush=True)

        try:
            det = _detect_arcs_from_pdf(pdf_path)
            det_arcs  = det["arc_count"]
            det_notes = det["note_count"]
            recall = det_arcs / ref_total if ref_total > 0 else float("nan")
            status = "[OK]" if det_arcs > 0 else "[미감지]"
            print(f"  감지: arc={det_arcs} / 음표={det_notes}  ->  감지율 {recall:.1%}  {status}")
        except Exception as e:
            print(f"  [ERROR] {e}")
        print()
