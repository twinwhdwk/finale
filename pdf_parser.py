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

    def measure_to_x_range(self, measure_in_staff: int, staff_width: int) -> tuple[int, int]:
        """
        오선 내 마디 번호(1-based) → (x_start, x_end) 픽셀 범위.

        x_to_measure()의 역방향. PDF 하이라이트 기능에서 "이 마디가
        화면의 어느 가로 구간에 있는지"를 구할 때 사용한다.

        barlines는 마디 사이의 경계만 담고 있으므로 (오선 좌우 끝은
        제외, _detect_barlines 참고), 첫 마디의 시작은 0, 마지막 마디의
        끝은 staff_width로 보정한다. staff_width는 오선 크롭 이미지의
        너비(보통 페이지 전체 너비, parse_page에서 img_gray.shape[1])를
        넘겨주면 된다.
        """
        if measure_in_staff < 1:
            raise ValueError(f"마디 번호는 1 이상이어야 함: {measure_in_staff}")

        x_start = 0 if measure_in_staff == 1 else self.barlines[measure_in_staff - 2]
        if measure_in_staff - 1 < len(self.barlines):
            x_end = self.barlines[measure_in_staff - 1]
        else:
            x_end = staff_width
        return (x_start, x_end)

    def measure_bbox(self, measure_in_staff: int, staff_width: int) -> tuple[int, int, int, int]:
        """
        오선 내 마디 번호(1-based) → (x0, y0, x1, y1) 픽셀 bbox.

        y0/y1은 코드 기호 영역까지 포함하도록 오선 위 staff_h만큼,
        가사 영역까지 포함하도록 오선 아래 staff_h*2.5만큼 여유를 둔다
        (pdf_parser._crop_chord_zone/_crop_lyric_zone과 동일한 범위 규칙).
        실제 다음 오선과 겹치지 않도록 호출부에서 next_top으로 클램프하는
        것을 권장 (parse_page에서 각 zone을 만들 때 이미 알고 있는 값).
        """
        x0, x1 = self.measure_to_x_range(measure_in_staff, staff_width)
        y0 = max(0, self.top_y - self.staff_h)
        y1 = self.bot_y + int(self.staff_h * 2.5)
        return (x0, y0, x1, y1)


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
    """오선 감지 → [(top_y, bot_y), ...] 반환

    병합 임계값 5px: 같은 오선 줄 내 두꺼운 픽셀(보통 2~3px)은 합치되
    인접 오선 줄(간격 보통 15~80px)은 각각 독립 점으로 유지한다.
    임계값이 20px이면 오선 5줄 전체가 하나의 점으로 묶여 시스템 간격을
    staff_gap으로 오인하는 버그가 발생한다.
    """
    h, w = img_gray.shape
    _, binary = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 6, 1))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    rows = np.where(mask.sum(axis=1) > w * 0.3)[0]

    if not len(rows):
        return []
    # 같은 오선 줄 픽셀(2~3px 두께)만 합침 — 5px 이하 간격만 병합
    merged, cluster = [], [rows[0]]
    for r in rows[1:]:
        if r - cluster[-1] <= 5:
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
    # bot_y는 오선 하단 선 y좌표이므로 포함해야 함 (top_y:bot_y는 bot_y 행 제외).
    # off-by-one이면 95% 임계값이 13px 차로 통과 실패하는 경우 발생.
    staff_crop = img_gray[top_y:bot_y + 1, :]
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

    # 5. 비첫오선 왼쪽 경계선 제거 (< 20% 위치의 첫 마디선)
    #    25%에서 20%로 낮춤: 첫 시스템의 좁은 첫 마디선(~23%)이 제거되던 문제 해결.
    #    비첫 오선의 클레프 경계선은 대체로 6~15% 범위이므로 영향 없음.
    if filtered and filtered[0] < w * 0.20:
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


def iter_absolute_measures(pages: list[PageParseResult]):
    """
    여러 페이지의 모든 (오선, 오선 내 마디 번호)를 절대 마디 번호와 함께
    순서대로 순회하는 이터레이터.

    "페이지를 순서대로 돌면서 zone.measure_count만큼 절대 마디 번호를
    누산한다"는 규칙이 main._extract_pdf_data()(코드/가사를 절대 마디
    번호로 변환)와 build_measure_location_map()(마디 자체의 위치를
    절대 마디 번호로 매핑) 두 곳에서 각각 따로 구현되어 있었다.
    이 함수와 iter_zones_with_start_measure()가 그 누산 규칙의 단일
    진실 공급원(single source of truth) 역할을 하도록 양쪽에서 재사용한다.

    Yields:
        (abs_measure_num, page, zone, measure_in_staff) 튜플.
        abs_measure_num: 전체 악보 기준 절대 마디 번호 (1부터)
        page:            해당 PageParseResult
        zone:            해당 StaffZone
        measure_in_staff: 오선 내 마디 번호 (1부터, zone.x_to_measure()와 동일 기준)
    """
    abs_measure = 1
    for page in pages:
        for zone in page.zones:
            for m_in_staff in range(1, zone.measure_count + 1):
                yield (abs_measure + m_in_staff - 1, page, zone, m_in_staff)
            abs_measure += zone.measure_count


def iter_zones_with_start_measure(pages: list[PageParseResult]):
    """
    여러 페이지의 모든 오선(zone)을, 그 오선이 시작하는 절대 마디 번호와
    함께 순회하는 이터레이터.

    zone.chords/zone.lyrics처럼 "오선 내 마디 번호"로 이미 인덱싱된
    데이터를 절대 마디 번호로 변환할 때 사용한다 (이런 데이터는 마디
    하나하나가 아니라 zone 단위로 들고 있어 iter_absolute_measures와는
    순회 단위가 다름). iter_absolute_measures()와 동일한 누산 규칙을
    공유한다.

    Yields:
        (zone_start_abs_measure, page, zone) 튜플.
        zone_start_abs_measure: 이 오선의 1번째 마디가 해당하는 절대 마디 번호
    """
    abs_measure = 1
    for page in pages:
        for zone in page.zones:
            yield (abs_measure, page, zone)
            abs_measure += zone.measure_count


@dataclass
class MeasureLocation:
    """절대 마디 번호 1개의 PDF 상 위치 (페이지 + 픽셀 bbox)."""
    measure_num: int        # 전체 악보 기준 절대 마디 번호 (1부터)
    page_num:    int        # 0-based 페이지 번호 (fitz 인덱스와 동일)
    staff_index: int        # 페이지 내 오선 번호 (1부터)
    bbox:        tuple[int, int, int, int]   # (x0, y0, x1, y1) 픽셀 좌표


def build_measure_location_map(
    pages: list[PageParseResult],
    page_width: int,
) -> dict[int, MeasureLocation]:
    """
    parse_all_pages() 결과로부터 "절대 마디 번호 → PDF 상 위치" 매핑을 만든다.

    main._extract_pdf_data()의 절대 마디 번호 누산 로직(abs_measure)과
    동일한 규칙을 사용해, 코드/가사 추출과 위치 매핑이 항상 같은 마디
    번호 기준을 공유하도록 한다. PDF 하이라이트 기능(report_generator의
    HTML에서 오류 클릭 시 원본 위치 표시)의 핵심 데이터 구조로 쓰기 위함.

    주의:
        - page_width는 페이지마다 다를 수 있으나(스캔 PDF는 보통 동일),
          이 함수는 단순화를 위해 모든 페이지에 같은 너비를 가정한다.
          페이지별 너비가 다르면 호출부에서 페이지별로 따로 호출할 것.
        - bbox y1은 다음 오선의 상단(next_top)으로 클램프되어 인접 오선
          가사/코드 영역과 겹치지 않도록 보정된다.

    Args:
        pages:      parse_all_pages()의 반환값
        page_width: 픽셀 단위 페이지 너비 (parse_page 내부의
                    img_gray.shape[1]과 동일한 값을 넘겨야 정확함.
                    호출부에서 알 수 없다면 pdf_parser._pdf_page_to_np로
                    동일 dpi로 한 번 더 렌더링해 shape를 구해야 함)

    Returns:
        {절대_마디_번호: MeasureLocation}
    """
    location_map: dict[int, MeasureLocation] = {}
    abs_measure = 1

    for page in pages:
        zones = page.zones
        # 각 zone의 y1 상한: 다음 zone 상단에서 staff_h만큼 위(코드 영역 경계)
        # 마지막 zone은 클램프 없음 (None)
        next_top_limits = []
        for i, zone in enumerate(zones):
            if i + 1 < len(zones):
                # 다음 오선의 top_y - 현재 오선 한 줄 높이 ≈ 코드 기호 영역 경계
                limit = zones[i + 1].top_y - zone.staff_h
            else:
                limit = None
            next_top_limits.append(limit)

        for zone_idx, zone in enumerate(zones):
            y1_limit = next_top_limits[zone_idx]
            for m_in_staff in range(1, zone.measure_count + 1):
                abs_num = abs_measure + m_in_staff - 1
                x0, y0, x1, y1 = zone.measure_bbox(m_in_staff, page_width)
                if y1_limit is not None:
                    y1 = min(y1, y1_limit)  # 다음 오선과 겹치지 않도록 클램프
                location_map[abs_num] = MeasureLocation(
                    measure_num=abs_num,
                    page_num=page.page_num,
                    staff_index=zone.index,
                    bbox=(x0, y0, x1, y1),
                )
            abs_measure += zone.measure_count

    return location_map
