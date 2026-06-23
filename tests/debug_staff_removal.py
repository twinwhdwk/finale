"""
staff_removal이 붙임줄/이음줄 호를 지우는지 디버그.

각 샘플 PDF 첫 페이지의 첫 오선에 대해:
  1. 오선 제거 전 binary → arc 감지
  2. 오선 제거 후 binary → arc 감지
  3. 두 결과 비교 + 컬러 오버레이 이미지 저장

실행:
    python tests/debug_staff_removal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_parser import _pdf_page_to_np, _detect_staves, _detect_barlines
from note_recognition.staff_removal import (
    detect_staff_line_thickness,
    remove_staff_lines,
)
from note_recognition.arc_detector import detect_arcs

SCORES_DIR = Path(__file__).parent / "fixtures" / "scores"
OUT_DIR     = Path(__file__).parent / "debug_arc"
DPI = 300


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def _binary_inv(gray: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return b


def _draw_arcs(bgr: np.ndarray, arcs, color: tuple, thickness: int = 2) -> None:
    for a in arcs:
        cv2.rectangle(bgr,
                      (a.x0, a.cy - 4), (a.x1, a.cy + 4),
                      color, thickness)
        label = "R" if a.cut_right else ("L" if a.cut_left else "")
        if label:
            cv2.putText(bgr, label, (a.x0, a.cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def debug_one(pdf_path: Path) -> None:
    stem = pdf_path.stem
    print(f"\n[{stem}]")

    img_rgb = _pdf_page_to_np(str(pdf_path), page_num=0, dpi=DPI)
    img     = _to_gray(img_rgb)
    staves  = _detect_staves(img)

    if not staves:
        print("  오선 감지 실패")
        return

    top_y, bot_y   = staves[0]
    staff_gap      = max(1, (bot_y - top_y) // 4)
    line_thickness = detect_staff_line_thickness(img, [(top_y, bot_y)])
    barlines       = _detect_barlines(img, top_y, bot_y)

    print(f"  오선: top={top_y}, bot={bot_y}, gap={staff_gap}, thickness={line_thickness}")
    print(f"  마디선: {len(barlines)}개")

    # ── 1단계: 오선 제거 전 binary에서 arc 감지 ──────────────────────
    margin = int(staff_gap * 3.5)
    roi_top = max(0, top_y - margin)
    roi_bot = min(img.shape[0], bot_y + margin)

    binary_before = _binary_inv(img)
    binary_before_roi = binary_before.copy()
    binary_before_roi[:roi_top, :] = 0
    binary_before_roi[roi_bot:, :]  = 0

    arcs_before = detect_arcs(binary_before_roi, staff_gap, top_y, bot_y,
                               img_width=img.shape[1])
    print(f"  제거 전  arc: {len(arcs_before)}개")

    # ── 2단계: 오선 제거 후 binary에서 arc 감지 ─────────────────────
    removed      = remove_staff_lines(img, top_y, bot_y,
                                      line_thickness=line_thickness)
    binary_after = _binary_inv(removed)
    binary_after_roi = binary_after.copy()
    binary_after_roi[:roi_top, :] = 0
    binary_after_roi[roi_bot:, :]  = 0

    arcs_after = detect_arcs(binary_after_roi, staff_gap, top_y, bot_y,
                              img_width=img.shape[1])
    print(f"  제거 후  arc: {len(arcs_after)}개")
    print(f"  손실된  arc: {len(arcs_before) - len(arcs_after)}개")

    # ── 3단계: 오선 제거로 사라진 픽셀 영역 계산 ────────────────────
    lost_pixels = ((binary_before_roi > 0) & (binary_after_roi == 0)).sum()
    print(f"  오선 제거로 사라진 픽셀: {lost_pixels}개")

    # ── 4단계: 컬러 오버레이 이미지 저장 ────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)

    # 오선 크롭 범위만 저장 (전체 페이지는 너무 큼)
    y0_vis = max(0, top_y - margin)
    y1_vis = min(img.shape[0], bot_y + margin)

    # 이미지 1: 제거 전 + arc (파란색)
    bgr_before = cv2.cvtColor(binary_before_roi[y0_vis:y1_vis], cv2.COLOR_GRAY2BGR)
    arcs_b_shifted = [
        type(a)(x0=a.x0, x1=a.x1, cy=a.cy - y0_vis,
                width=a.width, convex=a.convex,
                cut_left=a.cut_left, cut_right=a.cut_right)
        for a in arcs_before
    ]
    _draw_arcs(bgr_before, arcs_b_shifted, (255, 100, 0))
    cv2.imwrite(str(OUT_DIR / f"{stem}_before.png"), bgr_before)

    # 이미지 2: 제거 후 + arc (녹색)
    bgr_after = cv2.cvtColor(binary_after_roi[y0_vis:y1_vis], cv2.COLOR_GRAY2BGR)
    arcs_a_shifted = [
        type(a)(x0=a.x0, x1=a.x1, cy=a.cy - y0_vis,
                width=a.width, convex=a.convex,
                cut_left=a.cut_left, cut_right=a.cut_right)
        for a in arcs_after
    ]
    _draw_arcs(bgr_after, arcs_a_shifted, (0, 200, 0))
    cv2.imwrite(str(OUT_DIR / f"{stem}_after.png"), bgr_after)

    # 이미지 3: 제거로 사라진 픽셀 (빨간색) 하이라이트
    lost_mask = ((binary_before_roi > 0) & (binary_after_roi == 0)).astype(np.uint8) * 255
    bgr_diff  = cv2.cvtColor(binary_after_roi[y0_vis:y1_vis], cv2.COLOR_GRAY2BGR)
    bgr_diff[lost_mask[y0_vis:y1_vis] > 0] = (0, 0, 255)  # 빨간색 = 지워진 픽셀
    _draw_arcs(bgr_diff, arcs_b_shifted, (255, 100, 0))    # 파란색 = 제거 전 arc
    _draw_arcs(bgr_diff, arcs_a_shifted, (0, 200, 0))      # 녹색 = 제거 후 arc
    cv2.imwrite(str(OUT_DIR / f"{stem}_diff.png"), bgr_diff)

    print(f"  저장: {OUT_DIR}/{stem}_before/after/diff.png")


if __name__ == "__main__":
    samples = sorted(SCORES_DIR.glob("*.pdf"))
    if not samples:
        print(f"샘플 없음: {SCORES_DIR}")
        sys.exit(1)

    for pdf_path in samples:
        debug_one(pdf_path)

    print(f"\n완료. 이미지 저장 위치: {OUT_DIR}")
