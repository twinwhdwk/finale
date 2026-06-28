"""
악보 헤더(음자리표·조표·박자표) 자동 감지.

입력 조건:
- 클린 디지털 PDF (스캔본 아님)
- 600dpi 기준으로 튜닝됨
- 교과서 악보 (단순 기호, 이성부 이하)

반환: HeaderInfo (clef / key_sig / time_num / time_den)
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    _TESS_OK = True
except ImportError:
    _TESS_OK = False


_COMMON_TIME_SIGS: set[tuple[int, int]] = {
    (4, 4), (3, 4), (2, 4), (6, 8), (9, 8), (2, 2), (3, 8), (12, 8),
}


@dataclass
class HeaderInfo:
    """악보 헤더에서 감지된 음악 기호 정보."""
    clef:     str = 'G'  # 'G'=높은음자리표  'F'=낮은음자리표
    key_sig:  int = 0    # 샵 개수(양수) / 플랫 개수(음수), C장조=0
    time_num: int = 4    # 박자표 분자
    time_den: int = 4    # 박자표 분모

    @property
    def time_sig(self) -> str:
        return f"{self.time_num}/{self.time_den}"

    def __str__(self) -> str:
        clef_name = '높은음' if self.clef == 'G' else '낮은음'
        if self.key_sig > 0:
            ks = f"샵{self.key_sig}"
        elif self.key_sig < 0:
            ks = f"플랫{-self.key_sig}"
        else:
            ks = "제자리(C)"
        return f"{clef_name}자리표 / {ks} / {self.time_sig}"


# ── 공개 API ─────────────────────────────────────────────────────────

def detect_header(
    img_gray: np.ndarray,
    staff_ys: list[int],
) -> HeaderInfo:
    """
    단(System) 그레이스케일 이미지에서 음자리표·조표·박자표 자동 감지.

    Args:
        img_gray : 전체 단 이미지 (그레이스케일)
        staff_ys : 5개 오선 y 좌표 리스트 (위→아래, len ≥ 5)
    Returns:
        HeaderInfo
    """
    if len(staff_ys) < 5:
        return HeaderInfo()

    _, binary = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)

    gap   = (staff_ys[-1] - staff_ys[0]) / 4.0
    top_y = int(staff_ys[0])
    bot_y = int(staff_ys[-1])
    w_img = img_gray.shape[1]

    # 헤더 존 경계 (gap 기반 — DPI가 달라도 비율 유지)
    # 클레프: 0 ~ 5.5gap  /  조표: ~10gap  /  박자표: ~14gap
    clef_end = min(int(gap * 5.5),  w_img // 4)
    key_end  = min(int(gap * 10.0), w_img // 4)
    ts_end   = min(int(gap * 14.0), w_img // 4)

    clef     = _detect_clef(binary, top_y, bot_y, gap, clef_end)
    key_sig  = _detect_key_sig(binary, top_y, bot_y, gap, clef_end, key_end)
    num, den = _detect_time_sig(img_gray, binary, top_y, bot_y, gap, key_end, ts_end)

    return HeaderInfo(clef=clef, key_sig=key_sig, time_num=num, time_den=den)


# ── 음자리표 감지 ─────────────────────────────────────────────────────

def _detect_clef(
    binary: np.ndarray,
    top_y: int, bot_y: int,
    gap: float,
    clef_end: int,
) -> str:
    """
    높은음자리표(G): 오선 하단(bot_y) 아래로 내려오는 세로 획이 존재.
    낮은음자리표(F): 오선 상부에 콤팩트하게 배치, 하단 초과 없음.
    """
    y0 = max(0, top_y - int(gap * 1.5))
    y1 = min(binary.shape[0] - 1, bot_y + int(gap * 2.5))

    region = binary[y0:y1, :clef_end]
    if region.size == 0:
        return 'G'

    n, _, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    if n < 2:
        return 'G'

    # 면적 최대 블롭 = 클레프 본체
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    blob_bot = stats[best, cv2.CC_STAT_TOP] + stats[best, cv2.CC_STAT_HEIGHT]
    staff_bot_in_region = bot_y - y0

    # 높은음자리표: 블롭 하단이 오선 하단보다 0.3gap 이상 아래로 내려옴
    return 'G' if blob_bot > staff_bot_in_region + gap * 0.3 else 'F'


# ── 조표 감지 ─────────────────────────────────────────────────────────

def _detect_key_sig(
    binary: np.ndarray,
    top_y: int, bot_y: int,
    gap: float,
    key_start: int, key_end: int,
) -> int:
    """
    조표 블롭 카운팅.

    샵(#): 종횡비(bw/bh) ≥ 0.55, 격자형 획
    플랫(b): 종횡비 < 0.55, 세로로 긴 형태
    """
    if key_start >= key_end:
        return 0

    y0 = max(0, top_y - int(gap))
    y1 = min(binary.shape[0] - 1, bot_y + int(gap))
    region = binary[y0:y1, key_start:key_end]
    if region.size == 0:
        return 0

    n, _, stats, centroids = cv2.connectedComponentsWithStats(region, connectivity=8)
    if n < 2:
        return 0

    valid: list[dict] = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bh   = stats[i, cv2.CC_STAT_HEIGHT]
        bw   = stats[i, cv2.CC_STAT_WIDTH]

        if area < gap * gap * 0.4:  continue
        if area > gap * gap * 5.0:  continue
        if bh   < gap * 0.8:        continue
        if bh   > gap * 3.5:        continue
        if bw   > gap * 2.0:        continue

        valid.append({'bw': bw, 'bh': bh})

    if not valid:
        return 0

    count = len(valid)
    avg_aspect = sum(v['bw'] / max(v['bh'], 1) for v in valid) / count

    # 샵: aspect ≥ 0.55  /  플랫: aspect < 0.55
    return count if avg_aspect >= 0.55 else -count


# ── 박자표 감지 ───────────────────────────────────────────────────────

def _detect_time_sig(
    img_gray: np.ndarray,
    binary: np.ndarray,
    top_y: int, bot_y: int,
    gap: float,
    ts_start: int, ts_end: int,
) -> tuple[int, int]:
    """
    박자표 분자/분모 인식.

    1차: Tesseract OCR (이미지 3× 확대, 숫자 whitelist)
    2차: 블롭 형태 기반 폴백
    """
    if ts_start >= ts_end:
        return 4, 4

    y0 = max(0, top_y)
    y1 = min(img_gray.shape[0] - 1, bot_y)
    if y0 >= y1:
        return 4, 4

    gray_ts = img_gray[y0:y1, ts_start:ts_end]
    bin_ts  = binary[y0:y1, ts_start:ts_end]

    if _TESS_OK:
        result = _ocr_time_sig(gray_ts)
        if result:
            return result

    return _blob_time_sig(bin_ts, gap)


def _ocr_time_sig(gray_region: np.ndarray) -> tuple[int, int] | None:
    """Tesseract로 박자표 숫자 인식."""
    inv = cv2.bitwise_not(gray_region)
    h, w = inv.shape
    if h == 0 or w == 0:
        return None

    # 3× 확대 후 OTSU 이진화 (오선 포함 상태에서도 인식)
    inv_big = cv2.resize(inv, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    _, thr = cv2.threshold(inv_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        cfg = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thr, config=cfg)
        digits = [int(c) for c in text if c.isdigit()]
    except Exception:
        return None

    if len(digits) < 2:
        return None

    # 인접 쌍 중 유효한 박자표 찾기
    for i in range(len(digits) - 1):
        pair = (digits[i], digits[i + 1])
        if pair in _COMMON_TIME_SIGS:
            return pair

    # 유효 범위 내 첫 두 숫자 반환
    n, d = digits[0], digits[1]
    if n in {2, 3, 4, 6, 9, 12} and d in {2, 4, 8, 16}:
        return n, d

    return None


def _blob_time_sig(region_bin: np.ndarray, gap: float) -> tuple[int, int]:
    """블롭 형태로 박자표 추정 (OCR 폴백)."""
    h = region_bin.shape[0]
    if h == 0:
        return 4, 4

    mid = h // 2
    num = _classify_digit_blob(region_bin[:mid], gap)
    den = _classify_digit_blob(region_bin[mid:], gap)
    return num, den


def _classify_digit_blob(region: np.ndarray, gap: float) -> int:
    """블롭 하나 또는 여럿에서 숫자 추정."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    blobs = [
        stats[i] for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA]   > gap * gap * 0.15
        and stats[i, cv2.CC_STAT_HEIGHT] > gap * 0.4
    ]
    if not blobs:
        return 4

    # 두 블롭 → 두 자리 수 (예: 12)
    if len(blobs) >= 2:
        blobs.sort(key=lambda s: s[cv2.CC_STAT_LEFT])
        d1 = _shape_digit(blobs[0], region)
        d2 = _shape_digit(blobs[1], region)
        return d1 * 10 + d2

    return _shape_digit(blobs[0], region)


def _shape_digit(stat: np.ndarray, region: np.ndarray) -> int:
    """블롭 형태 특징으로 단일 숫자(0-9) 추정."""
    bx = stat[cv2.CC_STAT_LEFT]
    by = stat[cv2.CC_STAT_TOP]
    bw = stat[cv2.CC_STAT_WIDTH]
    bh = stat[cv2.CC_STAT_HEIGHT]
    if bw == 0 or bh == 0:
        return 4

    aspect = bw / bh
    crop = region[by:by + bh, bx:bx + bw]

    _, thr = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = sum(
        1 for h in (hierarchy[0] if hierarchy is not None else [])
        if h[3] != -1
    )

    top_px = float(np.sum(crop[:bh // 2] > 0))
    bot_px = float(np.sum(crop[bh // 2:] > 0))

    if holes >= 2:    return 8
    if holes == 1:
        if bot_px > top_px * 1.2:  return 6
        if top_px > bot_px * 1.2:  return 9
        return 0

    if aspect < 0.35:               return 1
    if aspect > 0.90:               return 0
    if top_px < bot_px * 0.6:      return 2
    if bot_px < top_px * 0.6:      return 3
    return 4
