"""
악보 PDF / PNG를 단(System) 단위로 슬라이스하는 모듈.

pdf_parser의 기존 함수(_pdf_page_to_np, _detect_staves)를 재사용하여
OCR 없이 빠르게 가로 띠(오선 단) 이미지를 추출합니다.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import fitz
import numpy as np

from pdf_parser import _detect_barlines, _detect_staves, _pdf_page_to_np

# 오선 위·아래 여백 비율 (staff_h 기준)
_PAD_TOP = 1.2   # 코드 기호 영역 포함
_PAD_BOT = 3.5   # 2절 가사 + 작사/작곡 영역까지 포함; next_top-4 제약으로 다음 오선은 침범 안 함


# ── 데이터 클래스 ────────────────────────────────────────────────────

@dataclass
class SystemSlice:
    abs_system:   int    # 전체 기준 절대 단 번호 (1-based)
    source_page:  int    # 페이지 또는 PNG 인덱스 (0-based)
    staff_idx:    int    # 페이지 내 오선 인덱스 (1-based)
    png_bytes:    bytes
    measure_count: int  = 0     # 이 단의 마디 수 (PDF 바코드 감지 기준)
    is_svg:        bool = False # True이면 png_bytes에 SVG(UTF-8) 저장

    @property
    def base64_src(self) -> str:
        b64 = base64.b64encode(self.png_bytes).decode("ascii")
        if self.is_svg:
            return f"data:image/svg+xml;base64,{b64}"
        return f"data:image/png;base64,{b64}"


@dataclass
class SlicedScore:
    source_path:   str
    total_systems: int = 0
    systems:       list[SystemSlice] = field(default_factory=list)
    warnings:      list[str]         = field(default_factory=list)

    @property
    def measures_per_system(self) -> list[int]:
        return [s.measure_count for s in self.systems]


@dataclass
class PairedSystem:
    abs_system: int
    textbook:   SystemSlice | None
    finale:     SystemSlice | None


# ── 공통 슬라이서 ────────────────────────────────────────────────────

def _group_staves(
    staves: list[tuple[int, int]],
    img_gray: "np.ndarray | None" = None,
) -> list[list[tuple[int, int]]]:
    """오선을 시스템 단위로 묶는다 (그랜드 스태프 / 다성부 대응).

    y-gap 기반: 직전 오선과의 수직 간격이 staff_h의 2배 이내면 같은 시스템.
    """
    if not staves:
        return []
    groups: list[list[tuple[int, int]]] = [[staves[0]]]
    for staff in staves[1:]:
        prev_top, prev_bot = groups[-1][-1]
        staff_h = max(1, prev_bot - prev_top)
        y_gap   = staff[0] - prev_bot
        if y_gap < staff_h * 2.0:
            groups[-1].append(staff)
        else:
            groups.append([staff])
    return groups


def _slice_img(
    img_rgb: np.ndarray,
    page_num: int,
    result: SlicedScore,
    abs_system: int,
) -> int:
    """단일 페이지 numpy 이미지에서 모든 단을 추출해 result에 추가."""
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    page_h, page_w = img_gray.shape

    staves = _detect_staves(img_gray)
    if not staves:
        result.warnings.append(f"page/img {page_num + 1}: 오선 미감지 — 건너뜀")
        return abs_system

    systems = _group_staves(staves)

    for sys_idx, sys_staves in enumerate(systems):
        top_y = sys_staves[0][0]   # 첫 오선 상단
        bot_y = sys_staves[-1][1]  # 마지막 오선 하단
        staff_h = sys_staves[0][1] - sys_staves[0][0]  # 첫 오선 높이 기준

        # 마디선 감지는 첫 번째(트레블) 오선 기준
        barlines      = _detect_barlines(img_gray, top_y, sys_staves[0][1])
        measure_count = len(barlines)

        next_sys_top = systems[sys_idx + 1][0][0] if sys_idx + 1 < len(systems) else page_h
        y0 = max(0, top_y - int(staff_h * _PAD_TOP))
        y1 = min(next_sys_top - 4, bot_y + int(staff_h * _PAD_BOT))
        staff_idx = sys_idx + 1

        if y1 <= y0:
            result.warnings.append(
                f"page/img {page_num + 1} staff {staff_idx}: 수직 범위 무효"
            )
            continue

        crop     = img_rgb[y0:y1, 0:page_w]
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        ok, buf  = cv2.imencode(".png", crop_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        if not ok:
            result.warnings.append(
                f"system {abs_system}: PNG 인코딩 실패"
            )
            continue

        result.systems.append(SystemSlice(
            abs_system    = abs_system,
            source_page   = page_num,
            measure_count = measure_count,
            staff_idx   = staff_idx,
            png_bytes   = buf.tobytes(),
        ))
        abs_system += 1

    return abs_system


# ── 교과서 PDF 처리 ──────────────────────────────────────────────────

def slice_pdf_to_systems(pdf_path: str, dpi: int = 600) -> SlicedScore:
    """
    교과서 스캔본 PDF → 단(System)별 슬라이스.

    _pdf_page_to_np / _detect_staves 를 재사용하여
    타이틀·일러스트·페이지번호를 자동으로 무시하고 오선 띠만 추출합니다.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    result = SlicedScore(source_path=str(pdf_path))
    abs_system = 1

    for page_num in range(total_pages):
        try:
            img_rgb = _pdf_page_to_np(pdf_path, page_num, dpi)
        except Exception as e:
            result.warnings.append(f"page {page_num + 1}: 렌더링 실패 ({e})")
            continue
        abs_system = _slice_img(img_rgb, page_num, result, abs_system)

    result.total_systems = abs_system - 1
    return result


# ── Finale PNG 처리 ──────────────────────────────────────────────────

def slice_pngs_to_systems(png_paths: list[str]) -> SlicedScore:
    """
    MuseScore 출력 PNG 목록 → 단(System)별 슬라이스.

    Finale XML을 MuseScore로 변환한 PNG 파일들을 입력으로 받습니다.
    """
    result = SlicedScore(source_path=str(png_paths[0]) if png_paths else "")
    abs_system = 1

    for page_num, png_path in enumerate(png_paths):
        img_bgr = cv2.imread(png_path)
        if img_bgr is None:
            result.warnings.append(f"PNG 로드 실패: {png_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        abs_system = _slice_img(img_rgb, page_num, result, abs_system)

    result.total_systems = abs_system - 1
    return result


# ── 매칭 ─────────────────────────────────────────────────────────────

def pair_systems(
    textbook: SlicedScore,
    finale:   SlicedScore,
) -> list[PairedSystem]:
    """단 번호가 같으면 자동 매칭. 한쪽에 없으면 None."""
    tb_by_s = {s.abs_system: s for s in textbook.systems}
    fn_by_s = {s.abs_system: s for s in finale.systems}
    total   = max(textbook.total_systems, finale.total_systems, 1)

    return [
        PairedSystem(
            abs_system = n,
            textbook   = tb_by_s.get(n),
            finale     = fn_by_s.get(n),
        )
        for n in range(1, total + 1)
    ]


# ── 디버깅용 파일 저장 ────────────────────────────────────────────────

def save_slices_to_disk(sliced: SlicedScore, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in sliced.systems:
        name = f"sys{s.abs_system:03d}_p{s.source_page + 1}_s{s.staff_idx}.png"
        (out / name).write_bytes(s.png_bytes)
    print(f"  {len(sliced.systems)}개 단 이미지 저장: {out}")
