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
   - density > HEAD_FILL_THRESHOLD(0.47) → 채워진 머리(quarter 이하)
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
- 빔(beam, 여러 음표를 잇는 가로 선)이 있는 경우 연결성분이 합쳐질 수 있음.
  현재는 단일 음표(깃발 개별) 케이스만 다루며, 빔 분리는 향후 단계에서 처리.
- 점음표(.dotted), 두잇음표(duplet), 쉼표는 미구현 (향후 단계).
- 음높이(pitch) 판정은 이 모듈의 담당이 아님 → note_pitcher.py (미구현).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


# ── 상수 ──────────────────────────────────────────────────────────────

HEAD_FILL_THRESHOLD = 0.47   # 이 값 이상이면 채워진 머리 (quarter 이하)
MIN_NOTE_AREA = 50            # 노이즈/아티팩트 제거용 최소 픽셀 수
# 표준 음악 표기 비율: notehead 높이(단축) ≈ staff_gap * 0.55~0.65.
# (합성 이미지 실측: staff_gap=20, notehead_radius=11 → 11/20=0.55)
_NOTEHEAD_RADIUS_RATIO = 0.55
# 기둥 판정 임계값 배율: whole(기둥 없음)의 bbox h ≈ notehead_radius*2,
# quarter(기둥 있음)의 bbox h ≈ staff_gap*3.5 + notehead_radius*2.
# staff_gap*2.5가 두 케이스 사이 중간에 위치 (합성 이미지 실측 확인).
_HAS_STEM_HEIGHT_RATIO = 2.5


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

    # 4분/8분쉼표는 온음표와 특성이 겹쳐 false positive가 많음.
    # 현재 버전에서는 블록형(전/2분)만 탐지하고 선형 쉼표는 미지원.
    # TODO: 4분쉼표(지그재그)와 8분쉼표(사선+점)의 정밀 분류기 추가.

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
        region_up   = binary[head_y_if_up - notehead_radius:   head_y_if_up   + notehead_radius, x:x+w]
        region_down = binary[head_y_if_down - notehead_radius: head_y_if_down + notehead_radius, x:x+w]
        density_up   = _pixel_density(region_up)
        density_down = _pixel_density(region_down)
        if abs(density_up - density_down) < 0.05:
            stem_up_guess = True
        else:
            stem_up_guess = (density_up >= density_down)

    head_y = head_y_if_up if stem_up_guess else head_y_if_down

    if stem_x_hint is not None:
        if stem_up_guess:
            head_x = stem_x_hint - (notehead_radius - 2)
        else:
            head_x = stem_x_hint + (notehead_radius - 2)
        head_x = int(np.clip(head_x, x, x + w))
    else:
        head_x = x + w // 2

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
) -> NoteDetectionResult:
    """
    오선 제거 → 연결성분 분리 → 음가 분류를 한 번에 수행.

    Args:
        img_gray:           그레이스케일 원본 (전체 페이지 또는 오선 크롭)
        staff_top_y:        오선 5줄 최상단 y (_detect_staves 결과)
        staff_bot_y:        오선 5줄 최하단 y
        staff_gap:          인접 오선 줄 사이 간격(픽셀). pitch 판정에도 필요하므로
                           정확하게 넘겨주는 것을 권장. 0이면 bbox에서 추정 시도.
        line_thickness:     오선 두께 (detect_staff_line_thickness 결과)
        min_horizontal_run: staff_removal.remove_staff_lines로 전달할 파라미터.
                           None이면 자동(이미지 폭의 5%)

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

    # 오선 범위에만 집중 (코드 기호, 가사 영역 제외)
    # y 범위를 오선 영역 ± 오선간격*1.5으로 제한
    margin = int(staff_gap * 3.5)  # 기둥 길이만큼 여유 (staff_gap*3.5가 기둥 길이)
    roi_top = max(0, staff_top_y - margin)
    roi_bot = min(img_gray.shape[0], staff_bot_y + margin)
    binary_roi = binary.copy()
    binary_roi[:roi_top, :] = 0
    binary_roi[roi_bot:, :] = 0

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_roi, connectivity=8)

    # 빔으로 묶인 컴포넌트는 분할, 나머지는 그대로 → dict 목록
    all_items = split_all_beam_components(binary_roi, stats, notehead_radius, min_area=MIN_NOTE_AREA)

    notes: list[DetectedNote] = []
    rests: list[DetectedRest] = []

    for item in all_items:
        bbox = item["bbox"]
        bx, by, bw, bh = bbox

        # 쉼표 판별 먼저 시도 (음표보다 h가 훨씬 작은 납작한 컴포넌트)
        rest = _classify_rest(bbox, staff_top_y, staff_gap)
        if rest is not None:
            rests.append(rest)
            continue

        note = _classify_duration(
            binary_roi, bbox, notehead_radius,
            stem_up_hint=item["stem_up"],
            stem_x_hint=item["stem_x"],
        )
        notes.append(note)

    # x순 정렬 (악보 읽기 순서)
    notes.sort(key=lambda n: n.head_x)
    rests.sort(key=lambda r: r.center_x)

    return NoteDetectionResult(
        notes=notes,
        rests=rests,
        staff_top_y=staff_top_y,
        staff_bot_y=staff_bot_y,
        line_thickness=line_thickness,
        staff_gap=staff_gap,
    )
