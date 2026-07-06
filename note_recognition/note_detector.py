"""
음표 객체 분리 및 음가(duration) 분류 모듈.

파이프라인 2단계: 오선이 제거된 이미지(staff_removal.py 출력)에서
개별 음표를 연결성분으로 분리하고, 각 음표의 음가를 판별한다.

## 분류 알고리즘 (합성 이미지 실험으로 검증된 규칙)

### 1. 기둥 유무 → whole 구분
   - bbox.h < notehead_radius * 4 이면 → whole (기둥 없음)
   - 아니면 기둥 있음 (half / quarter / eighth / sixteenth)

### 2. 머리 채움 여부 → half vs quarter/eighth/sixteenth
   - 음표머리 추정 영역(중심 ± notehead_radius px)에서 픽셀 밀도 계산
   - density > HEAD_FILL_THRESHOLD(0.50) → 채워진 머리(quarter 이하)
   - density ≤ 0.47 → 빈 머리(half)

### 3. 깃발 개수 → quarter / eighth / sixteenth
   - stem_x를 추정 후(기둥 방향에 따라 머리 왼쪽 또는 오른쪽)
   - 기둥 끝~머리 사이 strip을 cv2.connectedComponentsWithStats(4-conn)
   - 성분 수(배경 제외) = 깃발 수
     0 → quarter, 1 → eighth, 2 → sixteenth, 3+ → 32분 이하(추후 확장)

### 4. 기둥 방향
   - head_y - bbox.top vs bbox.bottom - head_y 중 긴 쪽이 기둥 방향
   - 판정된 기둥 방향으로 stem_x 추정 위치가 달라짐 (stem_up → 오른쪽, stem_down → 왼쪽)

## 알려진 한계 (현재 버전)
- 빔(beam, 여러 음표를 잇는 가로 선)으로 묶인 케이스: beam_splitter.py로 처리 (2~3개 묶음 검증 완료)
- 점음표(.dotted): _detect_dot()으로 지원 (is_dotted 필드)
- 쉼표: 전/2분쉼표(블록형) + 4분/8분쉼표(선형) 지원.
- 음높이(pitch) 판정: note_pitcher.py에서 처리 (이 모듈의 담당 아님)
- 코드(chord), 이성부 분리 미구현
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from note_recognition.arc_detector import ArcCandidate


# ── 상수 (config.ini [opencv] 섹션으로 오버라이드 가능) ───────────────

HEAD_FILL_THRESHOLD = 0.50   # 채워진 머리 판정 임계값 (quarter 이하)
# 합성 이미지 실측: half 최대=0.486(칸 위치), quarter 최소=0.576 → 여유 0.09
# 0.47에서는 half가 칸(홀수 step) 위치에서 quarter로 오분류됨
MIN_NOTE_AREA = 50            # 노이즈/아티팩트 제거용 최소 픽셀 수
ELLIPSE_FILL_MIN = 0.30       # 음표머리 타원성 최소값 (미만이면 노이즈로 제거)
HOLLOW_HEAD_DENSITY_MIN = 0.22  # half/whole(빈 머리) density 하한
# 진짜 빈 타원 테두리: 0.3~0.5 (합성 0.39~0.49). 미만은 산발 픽셀 노이즈.
# MXL whole=0인 악보에서 whole 45~81개 검출되던 문제의 주 원인.
# 실측(오 나의 태양 F, 300dpi): 진짜 채워진 머리 fill=0.83~0.95,
# 진짜 half(빈 머리) fill=0.43~0.72, 노이즈(텍스트/기호) fill=0.01~0.19


def _head_ellipse_fill(binary: np.ndarray, head_x: int, head_y: int,
                       notehead_radius: int) -> float:
    """
    head 위치 패치에서 최대 contour를 타원 피팅해 fill 비율을 반환.

    진짜 음표머리는 타원형이라 contour 면적/타원 면적이 높고(0.4+),
    텍스트·기호 조각은 불규칙해서 낮다(0.2 미만).
    판단 불가(빈 패치, 극소 contour)일 때는 1.0을 반환해 필터를 통과시킨다
    (보수적: 확실한 노이즈만 제거).
    """
    r = notehead_radius
    y0, y1 = max(0, head_y - r - 2), head_y + r + 2
    x0, x1 = max(0, head_x - r - 2), head_x + r + 2
    patch = binary[y0:y1, x0:x1]
    if patch.size == 0 or patch.sum() == 0:
        return 1.0
    contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 20 or len(c) < 5:
        return 1.0
    (_, _), (MA, ma), _ = cv2.fitEllipse(c)
    ellipse_area = np.pi * MA * ma / 4.0
    if ellipse_area <= 0:
        return 1.0
    return float(area / ellipse_area)
_NOTEHEAD_RADIUS_RATIO = 0.55  # notehead 반지름 / staff_gap 비율
_HAS_STEM_HEIGHT_RATIO = 2.5   # 기둥 판정 높이 배율

# config.ini가 있으면 [opencv] 섹션 값으로 덮어쓰기
try:
    import config_loader as _cl
    _ocv = _cl.get_opencv_params()
    HEAD_FILL_THRESHOLD    = _ocv["head_fill_threshold"]
    _NOTEHEAD_RADIUS_RATIO = _ocv["notehead_radius_ratio"]
    _HAS_STEM_HEIGHT_RATIO = _ocv["has_stem_height_ratio"]
except Exception:
    pass  # config.ini 없거나 형식 오류 → 기본값 유지


@dataclass
class DetectedNote:
    """오선 제거된 이미지에서 검출된 음표 1개의 정보."""
    bbox: tuple[int, int, int, int]
    head_x: int
    head_y: int
    duration: str
    n_flags: int
    stem_up: bool | None
    head_fill_density: float
    component_area: int
    is_dotted: bool = False


@dataclass
class DetectedRest:
    """오선 제거된 이미지에서 검출된 쉼표 1개의 정보."""
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    center_x: int
    center_y: int
    duration: str    # "whole" | "half" | "quarter" | "eighth"
    aspect: float    # w/h 비율 (디버그용)


@dataclass
class NoteDetectionResult:
    """한 오선 시스템에 대한 음표 검출 결과."""
    notes: list[DetectedNote] = field(default_factory=list)
    rests: list[DetectedRest] = field(default_factory=list)
    arcs: list["ArcCandidate"] = field(default_factory=list)  # 붙임줄/이음줄 호 후보
    staff_top_y: int = 0
    staff_bot_y: int = 0
    line_thickness: int = 2
    staff_gap: int = 20           # 검출 시 사용된 오선 간격 추정값 (notehead_radius 추정 기반)


def _classify_rest(
    bbox: tuple[int, int, int, int],
    staff_top_y: int,
    staff_gap: int,
) -> DetectedRest | None:
    """
    연결성분 bbox가 쉼표인지 판별하고 종류를 반환한다.

    ## 판별 규칙

    전/2분쉼표는 납작한 가로 직사각형 블록 (aspect w/h > 3):
      - 전쉼표: bbox 하단이 맨 아래 오선(line4_y) 근처 + 두꺼운 블록
      - 2분쉼표: bbox가 line3 근처에 위치 + 더 얇은 블록

    4분/8분쉼표는 복잡한 선형 모양으로 aspect < 3이지만 높이가 낮음.
    현재 단순화: 4분/8분은 aspect < 3 + area가 음표보다 작고
    세로 높이가 오선 간격의 1.5배 이하인 경우.

    음표(전체 bbox h가 큰 경우)와 쉼표(h가 작은 경우)는 h로 1차 구분.

    Args:
        bbox:         (x, y, w, h) 연결성분 bbox
        staff_top_y:  오선 맨 위줄 y좌표
        staff_gap:    오선 간격

    Returns:
        DetectedRest (쉼표로 판별된 경우), None (음표이거나 판별 불가)
    """
    x, y, w, h = bbox
    if h == 0 or w == 0:
        return None
    aspect = w / h
    area = w * h   # bbox 면적 (실제 픽셀 수 근사)
    cx, cy = x + w // 2, y + h // 2
    line4_y = staff_top_y + 4 * staff_gap
    line3_y = staff_top_y + 3 * staff_gap

    # 전/2분쉼표: 가로로 넓은 블록 (aspect > 3, h < staff_gap * 0.8)
    if aspect > 3 and h < staff_gap * 0.8 and w > staff_gap:
        bottom_y = y + h
        dist_to_line4 = abs(bottom_y - line4_y)
        dist_to_line3 = abs(bottom_y - line3_y)

        if dist_to_line4 < dist_to_line3:
            return DetectedRest(bbox=bbox, center_x=cx, center_y=cy,
                                duration="whole", aspect=aspect)
        else:
            return DetectedRest(bbox=bbox, center_x=cx, center_y=cy,
                                duration="half", aspect=aspect)

    # 4분/8분쉼표: 세로로 길고 좁은 선형 기호
    # 핵심 조건: h < staff_gap*2 (기둥 있는 음표는 h > staff_gap*3) AND aspect < 1.0
    # 온음표는 aspect > 1.0이라 자동 제외. 기둥 음표는 h > 60 (staff_gap=20 기준)이라 제외.
    # 오선 중간 영역에 위치해야 함 (악보 헤더/장식 제외).
    if (h < staff_gap * 2.0 and aspect < 1.0 and
            area > 50 and area < staff_gap * staff_gap):
        mid_staff_y = staff_top_y + 2 * staff_gap
        if abs(cy - mid_staff_y) < staff_gap * 2.0:
            # 8분쉼표는 4분쉼표보다 더 작고 단순 (area 기준)
            if area < staff_gap * staff_gap * 0.5:
                return DetectedRest(bbox=bbox, center_x=cx, center_y=cy,
                                    duration="eighth", aspect=aspect)
            else:
                return DetectedRest(bbox=bbox, center_x=cx, center_y=cy,
                                    duration="quarter", aspect=aspect)

    return None


def _estimate_notehead_radius(staff_gap: int) -> int:
    """오선 간격(staff_gap)으로 음표머리 반지름을 추정한다.

    표준 음악 표기 비율: notehead 단축 ≈ staff_gap * 0.55~0.65.
    합성 이미지 실측값: staff_gap=20 → notehead_radius=11 (ratio=0.55).
    """
    return max(6, int(staff_gap * _NOTEHEAD_RADIUS_RATIO))


def _classify_duration(
    binary: np.ndarray,
    bbox: tuple[int, int, int, int],
    notehead_radius: int,
    stem_up_hint: bool | None = None,
    stem_x_hint: int | None = None,
    staff_top_y: int | None = None,
    staff_bot_y: int | None = None,
) -> DetectedNote:
    """
    단일 연결성분(bbox)으로부터 음표 정보를 분류해 DetectedNote를 반환.

    Args:
        binary:          오선 제거된 이진 이미지
        bbox:            (x, y, w, h) 연결성분 바운딩 박스
        notehead_radius: 음표머리 반지름 추정값 (픽셀)
        stem_up_hint:    빔 분리에서 얻은 기둥 방향 힌트 (None이면 자체 추정)
        stem_x_hint:     빔 분리에서 얻은 기둥 x 좌표 힌트.
                         이 값이 있으면 head_x 추정에 사용 (bbox 중심보다 정확함).
                         기둥은 머리의 가장자리에 붙으므로:
                           stem_up → 기둥은 오른쪽 → head_x ≈ stem_x - notehead_radius
                           stem_down → 기둥은 왼쪽 → head_x ≈ stem_x + notehead_radius
    """
    x, y, w, h, = bbox
    bot = y + h

    # ── 1단계: 기둥 유무로 whole 구분 ──
    # staff_gap을 직접 알 수 없으므로, bbox h와 notehead_radius의 비율로 추정.
    # whole: h ≈ notehead_radius*2 (타원만)
    # 기둥 있음: h ≈ notehead_radius*2 + stem_length (stem_length ≈ staff_gap*3.5)
    # 임계값: notehead_radius * (2 + 2.5) = notehead_radius * 4.5
    # (staff_gap ≈ notehead_radius / 0.55 이므로, staff_gap * 2.5 = notehead_radius * 4.5)
    has_stem = (h > int(notehead_radius * (_HAS_STEM_HEIGHT_RATIO / _NOTEHEAD_RADIUS_RATIO)))

    if not has_stem:
        # whole note: bbox가 머리만 감쌈
        head_x = x + w // 2
        head_y = y + h // 2
        head_region = binary[
            max(0, head_y - notehead_radius): head_y + notehead_radius,
            max(0, head_x - notehead_radius): head_x + notehead_radius
        ]
        density = _pixel_density(head_region)
        is_dotted = _detect_dot(binary, bbox, head_x, head_y, notehead_radius)
        return DetectedNote(
            bbox=(x, y, w, h), head_x=head_x, head_y=head_y,
            duration="whole", n_flags=0, stem_up=None,
            head_fill_density=density, is_dotted=is_dotted,
            component_area=binary[y:bot, x:x+w].sum() // 255,
        )

    # ── 2단계: 기둥 방향 판정 ──
    head_y_if_up   = bot - notehead_radius   # stem_up이면 머리는 bbox 하단
    head_y_if_down = y   + notehead_radius   # stem_down이면 머리는 bbox 상단

    if stem_up_hint is not None:
        stem_up_guess = stem_up_hint
    else:
        # 투영(projection) 기반 판별:
        #   세로 투영(vproj)으로 기둥 열 찾기 → 기둥 제외
        #   가로 투영(hproj)으로 머리가 상단/하단 중 어디인지 판단
        # density 비교보다 기둥 픽셀 혼재 영향을 덜 받아 안정적
        region = binary[y:bot, x:x + w]
        vproj = region.sum(axis=0) / 255           # 열별 픽셀 수
        stem_mask = vproj > h * 0.5                # 높이 절반 이상 → 기둥
        region_no_stem = region.copy()
        stem_cols = np.where(stem_mask)[0]
        if len(stem_cols) > 0:
            region_no_stem[:, stem_cols] = 0       # 기둥 열 제거
        hproj = region_no_stem.sum(axis=1) / 255  # 행별 픽셀 수 (기둥 제외)
        top_mass = hproj[:h // 2].sum()
        bot_mass = hproj[h // 2:].sum()

        # 머리가 하단에 더 많으면 stem_up, 상단이면 stem_down
        if abs(top_mass - bot_mass) < max(2.0, h * 0.1):
            # 불분명할 때: 오선 기준 보조 판별
            if staff_top_y is not None and staff_bot_y is not None:
                up_in  = staff_top_y <= head_y_if_up   <= staff_bot_y
                dn_in  = staff_top_y <= head_y_if_down <= staff_bot_y
                if up_in and not dn_in:
                    stem_up_guess = True
                elif dn_in and not up_in:
                    stem_up_guess = False
                else:
                    stem_up_guess = True  # 기본값
            else:
                stem_up_guess = True
        else:
            stem_up_guess = (bot_mass > top_mass)

    head_y = head_y_if_up if stem_up_guess else head_y_if_down

    # head_x: stem_x_hint 우선, 없으면 기둥 위치에서 역산
    if stem_x_hint is not None:
        if stem_up_guess:
            head_x = stem_x_hint - (notehead_radius - 2)
        else:
            head_x = stem_x_hint + (notehead_radius - 2)
        head_x = int(np.clip(head_x, x, x + w))
    else:
        # 기둥 열 찾기 → 반대편에 머리
        region_hx = binary[y:bot, x:x + w]
        vproj_hx = region_hx.sum(axis=0) / 255
        stem_mask_hx = vproj_hx > h * 0.5
        stem_cols_hx = np.where(stem_mask_hx)[0]
        if len(stem_cols_hx) > 0:
            stem_cx = float(np.mean(stem_cols_hx))   # 기둥 열 중심
            if stem_up_guess:
                # stem_up: 기둥이 오른쪽 → 머리는 기둥 왼쪽
                head_x = x + max(0, int(stem_cx) - notehead_radius)
            else:
                # stem_down: 기둥이 왼쪽 → 머리는 기둥 오른쪽
                head_x = x + min(w - 1, int(stem_cx) + notehead_radius)
        else:
            head_x = x + w // 2
        head_x = int(np.clip(head_x, x, x + w))

    # ── 3단계: 머리 밀도 → half vs (quarter/eighth/sixteenth) 구분 ──
    head_region = binary[
        max(0, head_y - notehead_radius): head_y + notehead_radius,
        max(0, head_x - notehead_radius): head_x + notehead_radius
    ]
    head_density = _pixel_density(head_region)
    is_filled = (head_density >= HEAD_FILL_THRESHOLD)
    is_dotted = _detect_dot(binary, bbox, head_x, head_y, notehead_radius)

    if not is_filled:
        return DetectedNote(
            bbox=(x, y, w, h), head_x=head_x, head_y=head_y,
            duration="half", n_flags=0, stem_up=stem_up_guess,
            head_fill_density=head_density, is_dotted=is_dotted,
            component_area=binary[y:bot, x:x+w].sum() // 255,
        )

    # ── 4단계: 깃발 개수 → quarter / eighth / sixteenth ──
    n_flags = _count_flags(binary, bbox, head_x, head_y, notehead_radius, stem_up_guess)
    duration = {0: "quarter", 1: "eighth", 2: "sixteenth"}.get(n_flags, "sixteenth")

    return DetectedNote(
        bbox=(x, y, w, h), head_x=head_x, head_y=head_y,
        duration=duration, n_flags=n_flags, stem_up=stem_up_guess,
        head_fill_density=head_density, is_dotted=is_dotted,
        component_area=binary[y:bot, x:x+w].sum() // 255,
    )


def _pixel_density(region: np.ndarray) -> float:
    """영역 내 픽셀 밀도 (검정 픽셀 수 / 전체 픽셀 수). 빈 영역이면 0."""
    total = region.size
    if total == 0:
        return 0.0
    return float(region.sum()) / (255 * total)


def _detect_dot(
    binary: np.ndarray,
    bbox: tuple[int, int, int, int],
    head_x: int,
    head_y: int,
    notehead_radius: int,
) -> bool:
    """
    음표 오른쪽에서 점음표의 점(augmentation dot)을 탐지한다.

    점은 음표머리 오른쪽 가장자리에서 1~2배 반지름 거리에 위치하는
    작은 원(2~4px). 해당 영역에서 크기가 적절한 연결성분이 있으면 점으로 본다.

    Args:
        binary:          BINARY_INV 기준 이진 이미지
        bbox:            음표 연결성분 bbox (x,y,w,h)
        head_x, head_y:  음표머리 중심 추정값
        notehead_radius: 음표머리 반지름

    Returns:
        점음표이면 True
    """
    h_img, w_img = binary.shape

    # 점이 위치할 x 범위: 머리 오른쪽 끝(head_x + r)부터 ~2*r 범위
    dot_x0 = max(0, head_x + notehead_radius)
    dot_x1 = min(w_img, head_x + notehead_radius * 3)

    # y 범위: 머리 중심 ± half_step (칸 중간쯤에 위치)
    half_step = notehead_radius // 2
    dot_y0 = max(0, head_y - notehead_radius)
    dot_y1 = min(h_img, head_y + notehead_radius)

    if dot_x0 >= dot_x1 or dot_y0 >= dot_y1:
        return False

    dot_region = binary[dot_y0:dot_y1, dot_x0:dot_x1].copy()
    if not dot_region.any():
        return False

    n, _, stats, _ = cv2.connectedComponentsWithStats(dot_region, connectivity=8)
    # 점의 크기: area 4~60px (너무 크면 음표머리나 다른 음표, 너무 작으면 노이즈)
    DOT_MIN_AREA = 4
    DOT_MAX_AREA = 60
    for i in range(1, n):
        area = int(stats[i][4])
        dw, dh = int(stats[i][2]), int(stats[i][3])
        # 점: 정사각형에 가까운 작은 blob
        aspect = max(dw, dh) / max(1, min(dw, dh))
        if DOT_MIN_AREA <= area <= DOT_MAX_AREA and aspect < 2.5:
            return True
    return False


def _count_flags(
    binary: np.ndarray,
    bbox: tuple[int, int, int, int],
    head_x: int,
    head_y: int,
    notehead_radius: int,
    stem_up: bool,
) -> int:
    """
    기둥 옆 깃발 영역에서 4-connectivity 연결성분 수 = 깃발 수.

    stem_x 탐색 전략: bbox 추정값(오차 ~10px) 대신, 기둥 중간 높이 행의
    픽셀에서 기둥의 실제 rightmost x를 직접 읽는다. 이렇게 하면 bbox 추정
    오차로 인해 깃발 strip이 기둥 픽셀을 포함하거나 깃발을 놓치는 버그를
    방지할 수 있다.
    """
    x, y, w, h = bbox
    bot = y + h

    # ── 기둥 실제 x를 기둥 중간 행 픽셀에서 탐색 ──
    if stem_up:
        mid_y = (y + max(y, head_y - notehead_radius)) // 2
    else:
        mid_y = (min(bot, head_y + notehead_radius) + bot) // 2
    mid_y = int(np.clip(mid_y, 0, binary.shape[0] - 1))

    row = binary[mid_y, x: x + w]
    stem_pixels = [x + i for i, v in enumerate(row) if v > 0]
    if stem_pixels:
        stem_rightmost_x = max(stem_pixels)
        stem_leftmost_x  = min(stem_pixels)
    else:
        # 기둥을 못 찾으면 bbox 가장자리로 폴백
        stem_rightmost_x = x + w - 3
        stem_leftmost_x  = x + 2

    # 깃발 y 범위
    if stem_up:
        strip_y0 = max(0, y)
        strip_y1 = max(0, head_y - notehead_radius)
    else:
        strip_y0 = min(binary.shape[0], head_y + notehead_radius)
        strip_y1 = min(binary.shape[0], bot)

    if strip_y0 >= strip_y1:
        return 0

    # 깃발 x 범위:
    #   stem_up   → 깃발이 기둥 오른쪽 (표준 음악 표기법)
    #   stem_down → 깃발이 기둥 왼쪽  (표준 음악 표기법)
    if stem_up:
        strip_x0 = max(0, stem_rightmost_x + 1)
        strip_x1 = min(binary.shape[1], stem_rightmost_x + 18)
    else:
        strip_x0 = max(0, stem_leftmost_x - 18)
        strip_x1 = max(0, stem_leftmost_x)  # 기둥 픽셀 바로 왼쪽까지

    if strip_x0 >= strip_x1:
        return 0

    strip = binary[strip_y0:strip_y1, strip_x0:strip_x1].copy()
    if not strip.any():
        return 0

    n_components, _, _, _ = cv2.connectedComponentsWithStats(strip, connectivity=4)
    return n_components - 1  # 배경(0) 제외


def detect_notes(
    img_gray: np.ndarray,
    staff_top_y: int,
    staff_bot_y: int,
    staff_gap: int,
    line_thickness: int,
    min_horizontal_run: int | None = None,
    x_start: int = 0,
    barlines: list[int] | tuple[int, ...] = (),
    next_staff_top_y: int | None = None,
) -> NoteDetectionResult:
    """
    오선 제거 → 연결성분 분리 → 음가 분류를 한 번에 수행.

    Args:
        img_gray:           그레이스케일 원본 (전체 페이지 또는 오선 크롭)
        staff_top_y:        오선 5줄 최상단 y (_detect_staves 결과)
        staff_bot_y:        오선 5줄 최하단 y
        staff_gap:          인접 오선 줄 사이 간격(픽셀)
        line_thickness:     오선 두께 (detect_staff_line_thickness 결과)
        min_horizontal_run: staff_removal.remove_staff_lines 파라미터.
                           None이면 자동(이미지 폭의 5%)
        x_start:            음표 검출 시작 x 좌표 (기본 0). 오선 왼쪽에
                           위치한 음자리표/박자표/조표 기호를 음표로 오분류하지
                           않도록 실제 악보 시작 x를 지정한다.
                           예) StaffZone.barlines가 있으면 첫 마디 시작을
                           barlines[0] - staff_gap*2 정도로 추정하거나,
                           pdf_parser가 감지한 첫 음표 x를 넘겨주면 됨.

    Returns:
        NoteDetectionResult (x순 정렬된 DetectedNote 리스트 포함)
    """
    from note_recognition.staff_removal import remove_staff_lines
    from note_recognition.beam_splitter import split_all_beam_components

    if staff_gap <= 0:
        staff_gap = max(10, (staff_bot_y - staff_top_y) // 4)

    notehead_radius = _estimate_notehead_radius(staff_gap)

    removed = remove_staff_lines(
        img_gray, staff_top_y, staff_bot_y,
        line_thickness=line_thickness,
        min_horizontal_run=min_horizontal_run or max(20, notehead_radius * 2),
    )

    _, binary = cv2.threshold(removed, 128, 255, cv2.THRESH_BINARY_INV)

    # y 범위: 음표 기둥 + 덧줄 여유
    # 상단: stem_up 덧줄 기둥이 오선 위로 올라가므로 3.5×gap 여유 필요
    # 하단: stem_down 기둥 끝까지 + 여유, 단 MAX_STAFF_STEP 필터로 극단 케이스 차단됨
    #        2.5×gap으로 충분 (가사/코드 텍스트 제외 효과)
    margin_top = int(staff_gap * 3.5)
    margin_bot = int(staff_gap * 2.5)
    roi_top = max(0, staff_top_y - margin_top)
    if next_staff_top_y is not None:
        max_bot = next_staff_top_y - int(staff_gap * 0.5)
        roi_bot = min(img_gray.shape[0], min(staff_bot_y + margin_bot, max_bot))
    else:
        roi_bot = min(img_gray.shape[0], staff_bot_y + margin_bot)
    binary_roi = binary.copy()
    binary_roi[:roi_top, :] = 0
    binary_roi[roi_bot:, :] = 0

    # x 범위: x_start 이전(음자리표/박자표/조표 기호 영역) 마스킹
    if x_start > 0:
        binary_roi[:, :x_start] = 0

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_roi, connectivity=8)

    # 빔으로 묶인 컴포넌트는 분할, 나머지는 그대로 → dict 목록
    all_items = split_all_beam_components(binary_roi, stats, notehead_radius, min_area=MIN_NOTE_AREA)

    notes: list[DetectedNote] = []
    rests: list[DetectedRest] = []

    # 오선 범위 밖 너무 먼 음표 제거 기준
    # 실제 악보에서 덧줄은 보통 3개 이하 → step ±10 이내
    # step > 10이면 코드 기호/텍스트 등 오인식 가능성 높음
    MAX_STAFF_STEP = 10
    line4_y_ref = staff_top_y + 4 * staff_gap  # step=0 기준 y

    for item in all_items:
        bbox = item["bbox"]
        bx, by, bw, bh = bbox

        # 텍스트/가사/코드 기호 필터
        # 음표머리보다 훨씬 넓거나 낮은 컴포넌트 = 텍스트
        if bw > 0 and bh > 0:
            aspect = bw / bh
            if aspect > 5 and bh < staff_gap:  # 가로로 극단적으로 넓고 얇음
                continue

        # pitch 범위 필터: 음표머리 y가 오선에서 너무 멀면 노이즈로 제거
        # stem_up(기본)이면 머리가 bbox 하단, whole이면 bbox 중심
        # 보수적으로 bbox 하단을 머리 위치로 추정 (stem_up 가정)
        approx_head_y = by + bh - notehead_radius
        approx_step = abs(round((line4_y_ref - approx_head_y) * 2 / staff_gap))
        if approx_step > MAX_STAFF_STEP:
            continue  # 오선 범위 밖 → 노이즈로 제거

        # 쉼표 판별 먼저 시도 (음표보다 h가 훨씬 작은 납작한 컴포넌트)
        rest = _classify_rest(bbox, staff_top_y, staff_gap)
        if rest is not None:
            rests.append(rest)
            continue

        note = _classify_duration(
            binary_roi, bbox, notehead_radius,
            stem_up_hint=item["stem_up"],
            stem_x_hint=item["stem_x"],
            staff_top_y=staff_top_y,
            staff_bot_y=staff_bot_y,
        )
        # 타원성 검증: 음표머리가 타원형이 아니면 텍스트/기호 노이즈로 제거
        fill = _head_ellipse_fill(binary_roi, note.head_x, note.head_y,
                                  notehead_radius)
        if fill < ELLIPSE_FILL_MIN:
            continue

        # 빈 머리(half/whole) density 하한: 진짜 빈 타원 테두리는 0.3~0.5.
        # 그 미만이면 타원조차 아닌 산발 픽셀(가사 잔재/기호) → 제거.
        if note.duration in ("half", "whole") and \
                note.head_fill_density < HOLLOW_HEAD_DENSITY_MIN:
            continue

        notes.append(note)

    # x순 정렬 (악보 읽기 순서)
    notes.sort(key=lambda n: n.head_x)
    rests.sort(key=lambda r: r.center_x)

    # ── 화음(chord) 멤버 duration 상속 ──
    # whole은 빈 머리여야 하는데, 채워진 머리(density≥threshold)인데 기둥이
    # 없어 whole로 분류된 성분 = 화음에서 기둥이 다른 머리에 연결된 멤버.
    # 같은 x(±1.5r)의 기둥 있는 음표에서 duration을 상속한다.
    # 이성부 악보(꿈꾸지 않으면 F 등)의 누락 개선.
    for i, n in enumerate(notes):
        if n.duration != "whole" or n.head_fill_density < HEAD_FILL_THRESHOLD:
            continue
        # 인접 탐색 (x 근접 + 기둥 있는 음표)
        mate = None
        for m in notes:
            if m is n or m.duration == "whole":
                continue
            if abs(m.head_x - n.head_x) <= notehead_radius * 1.5:
                mate = m
                break
        if mate is not None:
            notes[i] = DetectedNote(
                bbox=n.bbox, head_x=n.head_x, head_y=n.head_y,
                duration=mate.duration, n_flags=mate.n_flags,
                stem_up=mate.stem_up,
                head_fill_density=n.head_fill_density,
                is_dotted=n.is_dotted,
                component_area=n.component_area,
            )
        else:
            # 화음 짝이 없어도 채워진 머리는 whole일 수 없음 → quarter로 추정
            notes[i] = DetectedNote(
                bbox=n.bbox, head_x=n.head_x, head_y=n.head_y,
                duration="quarter", n_flags=0, stem_up=None,
                head_fill_density=n.head_fill_density,
                is_dotted=n.is_dotted,
                component_area=n.component_area,
            )

    # 붙임줄/이음줄 호 감지.
    # 음표 감지용 x_start(바라인 기반)는 첫 마디 전체를 마스킹할 수 있으므로
    # 아크 감지는 clef/key/time sig 영역만 제외하는 작은 값을 사용.
    # 대신, 첫 마디 영역(x < x_start)에서 오선 아래 가사 위치에 있는 구조는
    # 가사 곡선으로 간주하여 제거한다.
    from note_recognition.arc_detector import detect_arcs
    binary_for_arcs = binary.copy()
    binary_for_arcs[:roi_top, :] = 0
    binary_for_arcs[roi_bot:, :] = 0
    img_w = img_gray.shape[1]
    # img_w의 1/8 ≈ 325px: treble clef + 조표 + 박자표 전체 영역을 커버하면서
    # 첫 마디 아크 감지는 허용. min(x_start, cap)으로 x_start가 너무 큰 경우 보정.
    x_arc_start = min(x_start, max(80, img_w // 8))
    if x_arc_start > 0:
        binary_for_arcs[:, :x_arc_start] = 0
    arcs = detect_arcs(
        binary_for_arcs, staff_gap, staff_top_y, staff_bot_y,
        img_width=binary_for_arcs.shape[1],
    )

    # 음표 위치 기반 아크 필터: 양 끝점 근방에 음표가 없으면 가사 extender 등 오감지.
    # y0/y1(끝점별 y)로 검사하여 피치 변화 슬러에서도 정확하게 동작.
    # 예외: cut 아크, x_start 미만 영역(음표 마스킹으로 미검출) 시작 아크.
    if notes:
        x_tol = staff_gap * 3
        y_tol_base = staff_gap * 3   # 짧은 슬러 기본 y 허용 범위
        filtered: list = []
        for arc in arcs:
            if arc.cut_left or arc.cut_right:
                filtered.append(arc)
                continue
            # bw<77px 좁은 슬러(grace note 등): gap≥18 스태프에서만 y_tol=70으로 완화.
            # gap<18(oD=14, oF=17 소형 스태프)에선 기준값 유지 → 소형 스태프 FP 방지.
            y_tol = 70 if (arc.width < 77 and staff_gap >= 18) else y_tol_base
            # x_start 미만 영역은 음표 마스킹 → 해당 끝점 면제.
            # 마디선 직전(1gap 이내): 음표머리가 슬러에 흡수되어 미검출될 수 있음 → 오른쪽 면제.
            # 이미지 우측 경계(89% 초과, _detect_barlines 제외 구간): 최종 마디선 끝점도 면제.
            bar_tol = staff_gap
            img_w = img_gray.shape[1]
            right_near_bar = bool(barlines) and any(
                abs(arc.x1 - bx) <= bar_tol for bx in barlines
            )
            right_near_edge = arc.x1 > int(img_w * 0.89) - 2 * staff_gap
            left_exempt  = (arc.x0 < x_start)
            right_exempt = (arc.x1 < x_start) or right_near_bar or right_near_edge
            near_left   = left_exempt or any(
                abs(n.head_x - arc.x0) <= x_tol and abs(n.head_y - arc.y0) <= y_tol
                for n in notes
            )
            near_right  = right_exempt or any(
                abs(n.head_x - arc.x1) <= x_tol and abs(n.head_y - arc.y1) <= y_tol
                for n in notes
            )
            if near_left and near_right:
                filtered.append(arc)
        arcs = filtered

    return NoteDetectionResult(
        notes=notes,
        rests=rests,
        arcs=arcs,
        staff_top_y=staff_top_y,
        staff_bot_y=staff_bot_y,
        line_thickness=line_thickness,
        staff_gap=staff_gap,
    )
