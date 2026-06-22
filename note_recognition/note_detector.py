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
STEM_DIRECTION_RATIO = 1.5   # stem 길이가 반대 방향의 이 배 이상이면 해당 방향으로 확정
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
    bbox: tuple[int, int, int, int]  # (x, y, w, h) 연결성분 bbox
    head_x: int                       # 추정 음표머리 중심 x
    head_y: int                       # 추정 음표머리 중심 y
    duration: str                     # "whole"|"half"|"quarter"|"eighth"|"sixteenth"
    n_flags: int                      # 깃발 수 (0~2, whole/half는 항상 0)
    stem_up: bool | None              # True=기둥 위, False=기둥 아래, None=기둥 없음(whole)
    head_fill_density: float          # 머리 영역 픽셀 밀도 (디버그용)
    component_area: int               # 연결성분 전체 픽셀 수 (디버그용)


@dataclass
class NoteDetectionResult:
    """한 오선 시스템에 대한 음표 검출 결과."""
    notes: list[DetectedNote] = field(default_factory=list)
    staff_top_y: int = 0
    staff_bot_y: int = 0
    line_thickness: int = 2
    staff_gap: int = 20           # 검출 시 사용된 오선 간격 추정값 (notehead_radius 추정 기반)


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
) -> DetectedNote:
    """
    단일 연결성분(bbox)으로부터 음표 정보를 분류해 DetectedNote를 반환.

    Args:
        binary:          오선 제거된 이진 이미지 (255=음표, 0=배경, BINARY_INV 기준)
        bbox:            (x, y, w, h) 연결성분 바운딩 박스
        notehead_radius: 음표머리 반지름 추정값 (픽셀)
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
        return DetectedNote(
            bbox=(x, y, w, h), head_x=head_x, head_y=head_y,
            duration="whole", n_flags=0, stem_up=None,
            head_fill_density=density,
            component_area=binary[y:bot, x:x+w].sum() // 255,
        )

    # ── 2단계: 기둥 방향 판정 ──
    # 음표머리는 기둥이 끝나는 반대쪽에 가까이 있음.
    # stem_up=True  → 기둥이 위로 → bbox 하단 근처에 머리
    # stem_up=False → 기둥이 아래로 → bbox 상단 근처에 머리
    # 머리 y = bbox 상단 또는 하단에서 notehead_radius만큼 안쪽 추정
    head_y_if_up   = bot - notehead_radius          # 기둥이 위라면 머리는 아래쪽
    head_y_if_down = y   + notehead_radius           # 기둥이 아래라면 머리는 위쪽

    region_up   = binary[head_y_if_up - notehead_radius:   head_y_if_up   + notehead_radius, x:x+w]
    region_down = binary[head_y_if_down - notehead_radius: head_y_if_down + notehead_radius, x:x+w]

    density_up   = _pixel_density(region_up)
    density_down = _pixel_density(region_down)

    # 더 밀한 쪽이 머리 위치. 같으면 기둥 길이로 다시 판정.
    if abs(density_up - density_down) < 0.05:
        # 애매하면 bbox의 위아래 비대칭으로 판단 (기둥이 있는 쪽이 더 길다)
        stem_up_guess = True  # 기본값
    else:
        stem_up_guess = (density_up >= density_down)

    head_y = head_y_if_up if stem_up_guess else head_y_if_down
    head_x = x + w // 2   # 가로 중심 (기둥은 머리 좌우 가장자리에 붙으므로 큰 오차 없음)

    # ── 3단계: 머리 밀도 → half vs (quarter/eighth/sixteenth) 구분 ──
    head_region = binary[
        max(0, head_y - notehead_radius): head_y + notehead_radius,
        max(0, head_x - notehead_radius): head_x + notehead_radius
    ]
    head_density = _pixel_density(head_region)
    is_filled = (head_density >= HEAD_FILL_THRESHOLD)

    if not is_filled:
        return DetectedNote(
            bbox=(x, y, w, h), head_x=head_x, head_y=head_y,
            duration="half", n_flags=0, stem_up=stem_up_guess,
            head_fill_density=head_density,
            component_area=binary[y:bot, x:x+w].sum() // 255,
        )

    # ── 4단계: 깃발 개수 → quarter / eighth / sixteenth ──
    n_flags = _count_flags(binary, bbox, head_x, head_y, notehead_radius, stem_up_guess)
    duration = {0: "quarter", 1: "eighth", 2: "sixteenth"}.get(n_flags, "sixteenth")

    return DetectedNote(
        bbox=(x, y, w, h), head_x=head_x, head_y=head_y,
        duration=duration, n_flags=n_flags, stem_up=stem_up_guess,
        head_fill_density=head_density,
        component_area=binary[y:bot, x:x+w].sum() // 255,
    )


def _pixel_density(region: np.ndarray) -> float:
    """영역 내 픽셀 밀도 (검정 픽셀 수 / 전체 픽셀 수). 빈 영역이면 0."""
    total = region.size
    if total == 0:
        return 0.0
    return float(region.sum()) / (255 * total)


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
    else:
        # 기둥을 못 찾으면 bbox 오른쪽 가장자리로 폴백
        stem_rightmost_x = x + w - 3

    # 깃발 y 범위
    if stem_up:
        strip_y0 = max(0, y)
        strip_y1 = max(0, head_y - notehead_radius)
    else:
        strip_y0 = min(binary.shape[0], head_y + notehead_radius)
        strip_y1 = min(binary.shape[0], bot)

    if strip_y0 >= strip_y1:
        return 0

    # 깃발 x 범위: 기둥 오른쪽 끝 바로 다음부터 ~15px
    strip_x0 = max(0, stem_rightmost_x + 1)
    strip_x1 = min(binary.shape[1], stem_rightmost_x + 18)

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

    notes: list[DetectedNote] = []
    for i in range(1, n):  # 0은 배경
        area = stats[i][4]
        if area < MIN_NOTE_AREA:
            continue  # 노이즈/아티팩트 제거

        cx, cy, cw, ch = stats[i][0], stats[i][1], stats[i][2], stats[i][3]
        note = _classify_duration(binary_roi, (cx, cy, cw, ch), notehead_radius)
        notes.append(note)

    # x순 정렬 (악보 읽기 순서)
    notes.sort(key=lambda n: n.head_x)

    return NoteDetectionResult(
        notes=notes,
        staff_top_y=staff_top_y,
        staff_bot_y=staff_bot_y,
        line_thickness=line_thickness,
        staff_gap=staff_gap,
    )
