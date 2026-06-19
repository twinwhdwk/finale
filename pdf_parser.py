"""
PDF 악보 파서 모듈

오선지 영역을 자동 감지(ROI Zoning)한 뒤 3개 트랙으로 분리 추출합니다.
  - 코드 기호: 오선 위 영역 → Tesseract (영문)
  - 가사:     오선 아래 영역 → EasyOCR (한글+영문)
  - 음표:     오선 영역 → Audiveris (별도 모듈)

마디선(barline)을 감지하여 OCR 결과를 정확한 마디 번호로 매핑합니다.
"""

import re
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract
import easyocr
import fitz
from dataclasses import dataclass, field
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_NOISE_WORDS = re.compile(
    r'학습|목표|작곡|작사|감상|느끼|보통|빠르게|Moderato|Andante|Allegro'
    r'|교과서|단원|마디|QR|코드번호|출처|저작권'
)
# slash chord (G/D, A/C#) 포함, 두 자리 숫자 (7, 11, 13) 포함
_CHORD_PATTERN = re.compile(
    r'^[A-G][#b]?'
    r'(m|M|maj|min|dim|aug|sus[24]?|add)?'
    r'[0-9]{0,2}'
    r'(/[A-G][#b]?)?$'
)
# whitelist에 '/' 추가 (slash chord용)
_TESSERACT_CHORD = r"--psm 6 -c tessedit_char_whitelist=ABCDEFGabcdefgmM#b1234567/"

_ocr_reader: easyocr.Reader | None = None


def _get_ocr() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False)
    return _ocr_reader


@dataclass
class StaffZone:
    index:   int        # 오선 번호 (페이지 내 1부터)
    top_y:   int        # 오선 최상단 픽셀
    bot_y:   int        # 오선 최하단 픽셀
    staff_h: int        # 오선 높이 (bot_y - top_y)
    barlines: list[int] = field(default_factory=list)
    # barlines: 오선 내 마디선 x좌표 목록 (오선 경계 제외)
    chords: list[tuple[int, int, str, float]] = field(default_factory=list)
    # chords: [(measure_in_staff, x_pixel, chord_text, confidence), ...]
    lyrics: list[tuple[int, str]] = field(default_factory=list)
    # lyrics: [(measure_in_staff, text), ...]

    @property
    def measure_count(self) -> int:
        """이 오선에 포함된 마디 수 (마디선+1, 최소 1)"""
        return max(1, len(self.barlines) + 1)

    def x_to_measure(self, x: int) -> int:
        """x 좌표 → 오선 내 마디 번호 (1-based)"""
        for i, bx in enumerate(self.barlines):
            if x < bx:
                return i + 1
        return len(self.barlines) + 1


@dataclass
class PageParseResult:
    pdf_path:    str
    page_num:    int
    staff_count: int
    zones: list[StaffZone] = field(default_factory=list)

    def all_chords(self) -> list[tuple[int, int, int, str, float]]:
        """(staff_idx, measure_in_staff, x, chord, conf) 전체 리스트"""
        out = []
        for z in self.zones:
            for m, x, ch, cf in z.chords:
                out.append((z.index, m, x, ch, cf))
        return out

    def all_lyrics(self) -> list[tuple[int, int, str]]:
        """(staff_idx, measure_in_staff, lyric_text) 전체 리스트"""
        return [(z.index, m, t) for z in self.zones for m, t in z.lyrics]


# ── 이미지 변환 ───────────────────────────────────────────────────────

def _pdf_page_to_np(pdf_path: str, page_num: int = 0, dpi: int = 600) -> np.ndarray:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


# ── 오선 감지 ────────────────────────────────────────────────────────

def _detect_staves(img_gray: np.ndarray) -> list[tuple[int, int]]:
    """오선 감지 → [(top_y, bot_y), ...] 반환"""
    h, w = img_gray.shape
    _, binary = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 6, 1))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    rows = np.where(mask.sum(axis=1) > w * 0.3)[0]

    if not len(rows):
        return []
    merged, cluster = [], [rows[0]]
    for r in rows[1:]:
        if r - cluster[-1] <= 20:
            cluster.append(r)
        else:
            merged.append(int(np.mean(cluster)))
            cluster = [r]
    merged.append(int(np.mean(cluster)))

    staves = []
    i = 0
    while i + 4 < len(merged):
        five = merged[i:i+5]
        gaps = [five[j+1] - five[j] for j in range(4)]
        avg = np.mean(gaps)
        if max(gaps) < avg * 1.8:
            staves.append((five[0], five[-1]))
            i += 5
        else:
            i += 1
    return staves


# ── 마디선 감지 ──────────────────────────────────────────────────────

def _detect_barlines(img_gray: np.ndarray, top_y: int, bot_y: int) -> list[int]:
    """
    오선 내 세로줄(마디선) x좌표 목록 반환.

    알고리즘:
    1. Raw binary 열별 합계 사용 (수평선 제거 없이)
    2. 임계값 = h * 255 * 0.95: 오선 높이의 95% 이상이 검은색인 열만 선택
       → 음표 기둥(통상 staff_h의 85~90%)과 마디선(100%) 분리
    3. NMS: 50px 이내 인접 열 → 중심값으로 병합
    4. 공간 필터:
       - 좌측 15%: 클레프 / 조표 / 박자표 영역 제외
       - 우측 89%: 오선 끝 경계선 제외
    5. 비정상적으로 좌측에 있는 첫 마디선(< 25%) 제거
       → 비첫 오선(클레프만, 조표/박자표 반복 없음)의 왼쪽 경계선
    6. 300px 미만 간격 제거 (겹침표 이중선)
    """
    staff_crop = img_gray[top_y:bot_y, :]
    h, w = staff_crop.shape
    if h < 5 or w < 40:
        return []

    _, binary = cv2.threshold(staff_crop, 128, 255, cv2.THRESH_BINARY_INV)

    # 1. 열별 합계 (수평선 제거 없음)
    col_sums = binary.sum(axis=0).astype(np.float32)

    # 2. 95% 임계값: 음표 기둥 제거 (barline은 100%, stem은 85~90%)
    threshold = h * 255 * 0.95
    barline_cols = np.where(col_sums >= threshold)[0]

    if not len(barline_cols):
        return []

    # 3. NMS: 50px 이내 인접 열 병합
    merged_x: list[int] = []
    cluster: list[int] = [int(barline_cols[0])]
    for c in barline_cols[1:]:
        if c - cluster[-1] <= 50:
            cluster.append(int(c))
        else:
            merged_x.append(int(np.mean(cluster)))
            cluster = [int(c)]
    merged_x.append(int(np.mean(cluster)))

    # 4. 공간 필터 (클레프/조표 좌측, 오선 끝 우측 제외)
    filtered = [x for x in merged_x if x > w * 0.15 and x < w * 0.89]

    # 5. 비첫오선 왼쪽 경계선 제거 (< 25% 위치의 첫 마디선)
    if filtered and filtered[0] < w * 0.25:
        filtered.pop(0)

    # 6. 겹침표 이중선 제거 (300px 미만 간격)
    result: list[int] = []
    for x in filtered:
        if not result or x - result[-1] >= 300:
            result.append(x)

    return result


# ── 크롭 헬퍼 ────────────────────────────────────────────────────────

def _crop_chord_zone(img_gray: np.ndarray, top_y: int, staff_h: int) -> np.ndarray:
    """오선 위 코드 기호 영역 크롭"""
    c_top = max(0, top_y - staff_h)
    c_bot = max(0, top_y - 2)
    return img_gray[c_top:c_bot, :]


def _crop_lyric_zone(img_gray: np.ndarray, bot_y: int, staff_h: int, next_top: int) -> np.ndarray:
    """오선 아래 가사 영역 크롭"""
    l_top = bot_y + max(5, staff_h // 4)
    l_bot = min(next_top - 5, bot_y + int(staff_h * 2.5))
    if l_bot <= l_top:
        return np.array([])
    return img_gray[l_top:l_bot, :]


# ── 전처리 ───────────────────────────────────────────────────────────

def _preprocess_chord(crop: np.ndarray) -> Image.Image:
    pil = Image.fromarray(crop)
    pil = ImageEnhance.Contrast(pil).enhance(2.5)
    return pil.filter(ImageFilter.SHARPEN)


def _preprocess_lyric(crop: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(crop, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=25, C=8
    )
    w = crop.shape[1]
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1))
    lines = cv2.morphologyEx(cv2.bitwise_not(bw), cv2.MORPH_OPEN, horiz)
    bw = cv2.add(bw, lines)
    return cv2.resize(bw, (bw.shape[1]*2, bw.shape[0]*2), interpolation=cv2.INTER_CUBIC)


# ── OCR 추출 ─────────────────────────────────────────────────────────

def _extract_chords(
    crop: np.ndarray,
    barlines: list[int],
) -> list[tuple[int, int, str, float]]:
    """코드 기호 추출 → [(measure_in_staff, x, text, conf), ...]"""
    if crop.shape[0] < 5:
        return []
    img = _preprocess_chord(crop)
    data = pytesseract.image_to_data(
        img, lang="eng", config=_TESSERACT_CHORD,
        output_type=pytesseract.Output.DICT
    )
    results = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = int(data["conf"][i])
        if _CHORD_PATTERN.match(text) and conf > 40:
            x = data["left"][i]
            m_in_staff = sum(1 for bx in barlines if x >= bx) + 1
            results.append((m_in_staff, x, text, conf / 100))
    return results


def _extract_lyrics(
    crop_gray: np.ndarray,
    barlines: list[int],
) -> list[tuple[int, str]]:
    """가사 추출 → [(measure_in_staff, text), ...]"""
    if crop_gray.size == 0 or crop_gray.shape[0] < 5:
        return []
    proc = _preprocess_lyric(crop_gray)
    raw = _get_ocr().readtext(proc)
    texts: list[tuple[int, str]] = []
    for (bbox, text, conf) in raw:
        if conf < 0.2:
            continue
        cleaned = re.sub(r'[^가-힣a-zA-Z\s\-]', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if not cleaned or _NOISE_WORDS.search(cleaned) or len(cleaned) <= 1:
            continue
        # bbox center x — proc는 2배 업스케일이므로 /2 로 원본 좌표 복원
        center_x = int((bbox[0][0] + bbox[2][0]) / 2 / 2)
        m_in_staff = sum(1 for bx in barlines if center_x >= bx) + 1
        texts.append((m_in_staff, cleaned))
    return texts


# ── 공개 API ──────────────────────────────────────────────────────────

def parse_page(pdf_path: str, page_num: int = 0, dpi: int = 600) -> PageParseResult:
    """
    PDF 한 페이지를 파싱하여 코드 기호와 가사를 추출합니다.

    마디선을 감지하여 OCR 결과를 정확한 마디 번호(오선 내)로 매핑합니다.

    Args:
        pdf_path:  PDF 파일 경로
        page_num:  페이지 번호 (0부터)
        dpi:       렌더링 해상도

    Returns:
        PageParseResult
    """
    img_np   = _pdf_page_to_np(pdf_path, page_num, dpi)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h        = img_gray.shape[0]

    staves_raw = _detect_staves(img_gray)
    result = PageParseResult(
        pdf_path=str(pdf_path),
        page_num=page_num,
        staff_count=len(staves_raw),
    )

    w = img_gray.shape[1]

    # 1패스: 마디선 감지
    raw_barlines: list[list[int]] = []
    for top_y, bot_y in staves_raw:
        raw_barlines.append(_detect_barlines(img_gray, top_y, bot_y))

    # 시스템 오른쪽 경계선 일괄 제거
    # 각 오선의 마지막 바코드 x 위치가 일관되게 같으면(spread < 3%) → 경계선
    rightmost = [bl[-1] / w for bl in raw_barlines if bl]
    if len(rightmost) >= 2:
        avg_r  = sum(rightmost) / len(rightmost)
        spread = max(abs(x - avg_r) for x in rightmost)
        if spread < 0.03 and avg_r > 0.78:          # 일관되고 우측 78% 이상
            raw_barlines = [
                (bl[:-1] if bl and abs(bl[-1] / w - avg_r) < 0.03 else bl)
                for bl in raw_barlines
            ]

    for idx, (top_y, bot_y) in enumerate(staves_raw):
        staff_h  = bot_y - top_y
        next_top = staves_raw[idx+1][0] if idx+1 < len(staves_raw) else h
        barlines = raw_barlines[idx]

        zone = StaffZone(
            index=idx+1, top_y=top_y, bot_y=bot_y,
            staff_h=staff_h, barlines=barlines,
        )

        chord_crop = _crop_chord_zone(img_gray, top_y, staff_h)
        zone.chords = _extract_chords(chord_crop, barlines)

        lyric_crop = _crop_lyric_zone(img_gray, bot_y, staff_h, next_top)
        zone.lyrics = _extract_lyrics(lyric_crop, barlines)

        result.zones.append(zone)

    return result


def parse_all_pages(pdf_path: str, dpi: int = 600) -> list[PageParseResult]:
    """PDF 전체 페이지를 파싱합니다."""
    doc   = fitz.open(pdf_path)
    total = len(doc)
    doc.close()
    results = []
    for p in range(total):
        print(f"  페이지 {p+1}/{total} 파싱 중...")
        results.append(parse_page(pdf_path, p, dpi))
    return results
