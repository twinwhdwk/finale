"""
빔(beam) 분리 모듈.

목표: 빔(beam, 여러 음표를 잇는 굵은 가로선)으로 연결되어 하나의
연결성분으로 합쳐진 음표 그룹을 개별 음표로 분리한다.

## 문제

8분/16분음표가 연속으로 올 때, 각 음표의 기둥 끝이 굵은 가로선(빔)으로
연결된다. 오선 제거 후 cv2.connectedComponentsWithStats를 적용하면
이 빔 때문에 여러 음표가 하나의 연결성분으로 합쳐져서 `_classify_duration`이
개별 음표를 처리할 수 없다.

## 전략 (세로 투영 기반 분할)

빔으로 묶인 컴포넌트는 bbox.w가 넓고, 세로 방향 투영(vertical projection:
각 x좌표에서의 픽셀 수)에서 기둥(stem) 위치에 강한 피크가 생긴다.
  - 기둥은 세로로 긴 직선 → 해당 x에서 픽셀 수가 급격히 증가
  - 빔은 가로로 긴 선이지만 기둥보다 얇음 → 기준값(피크의 30% 이상) 이하
  - 음표 사이 빈 공간 → 매우 적은 픽셀(빔만 지나가는 픽셀)

피크 위치(=기둥 x)를 찾고, 피크 사이 골짜기의 최저점에서 컴포넌트를
세로로 잘라 개별 음표 서브이미지를 만든다.

## 빔 컴포넌트 판별 기준

bbox.w > single_note_max_width 이면 빔 컴포넌트 후보로 본다.
single_note_max_width ≈ notehead_radius * 4 (머리 지름 + 기둥 폭 + 여유)

## 알려진 한계
- 3개 이상 묶인 빔도 동일 알고리즘으로 처리 가능하지만 테스트가 없음.
- 빔 그룹의 음표들이 모두 같은 음가여야 한다(표준 표기법 준수 가정).
  실제론 8분+16분이 섞일 수 있는데, 이 경우 개별 음표 분류 후 깃발 수로
  판별하므로 큰 문제 없음.
- stem_down 빔은 별도로 검증 필요.
"""

import cv2
import numpy as np


def is_beam_component(
    w: int,
    h: int,
    notehead_radius: int,
) -> bool:
    """
    연결성분이 빔으로 묶인 그룹인지 판단한다.

    단일 음표의 최대 가로폭(머리 지름 + 기둥 + 깃발) ≈ notehead_radius * 4.
    그보다 넓으면 빔 컴포넌트 후보.
    """
    single_note_max_w = notehead_radius * 4
    return w > single_note_max_w and h > notehead_radius * 2


def _vertical_projection(region: np.ndarray) -> np.ndarray:
    """각 x좌표에서의 픽셀 수 (세로 투영)."""
    return (region > 0).sum(axis=0).astype(np.int32)


def _find_stem_peaks(vproj: np.ndarray, peak_threshold_ratio: float = 0.3) -> list[int]:
    """
    세로 투영에서 기둥(stem) 위치 피크를 찾는다.

    Args:
        vproj: 세로 투영 배열
        peak_threshold_ratio: 최댓값의 이 비율 이상이면 피크 구간

    Returns:
        각 피크의 중심 x 인덱스 목록 (컴포넌트 내 로컬 좌표)
    """
    if vproj.max() == 0:
        return []
    threshold = int(vproj.max() * peak_threshold_ratio)
    peaks = []
    in_peak = False
    peak_start = 0
    for i, v in enumerate(vproj):
        if v >= threshold and not in_peak:
            in_peak = True
            peak_start = i
        elif v < threshold and in_peak:
            in_peak = False
            center = (peak_start + i - 1) // 2
            peaks.append(center)
    if in_peak:
        center = (peak_start + len(vproj) - 1) // 2
        peaks.append(center)
    return peaks


def _find_split_positions(vproj: np.ndarray, peaks: list[int]) -> list[int]:
    """
    피크 사이에서 픽셀 수가 최소인 x(분할점)를 찾는다.

    각 인접 피크 쌍 사이에서 vproj가 최소인 지점을 분할선으로 사용.
    최솟값이 여러 개 연속으로 같다면 그 구간의 중간을 선택.

    Returns:
        컴포넌트 내 로컬 x 좌표 목록 (분할선)
    """
    splits = []
    for i in range(len(peaks) - 1):
        lo, hi = peaks[i], peaks[i + 1]
        if lo >= hi:
            splits.append((lo + hi) // 2)
            continue
        segment = vproj[lo:hi]
        min_val = segment.min()
        # 최솟값 연속 구간의 중간을 분할점으로
        min_indices = np.where(segment == min_val)[0]
        mid_local = int(min_indices[len(min_indices) // 2])
        splits.append(lo + mid_local)
    return splits


def split_beam_component(
    binary: np.ndarray,
    bbox: tuple[int, int, int, int],
    notehead_radius: int,
    min_sub_width: int | None = None,
) -> list[tuple[tuple[int, int, int, int], bool]] | None:
    """
    빔으로 묶인 연결성분(bbox)을 개별 음표 bbox + stem_up 힌트로 분할한다.

    Returns:
        [(서브bbox, stem_up_hint), ...] 또는 분할 불가 시 None.
        stem_up_hint: 빔의 y위치로 판정한 기둥 방향
            - 빔이 bbox 상단에 있으면(y < bbox 중심) stem_up=True (기둥이 위)
            - 빔이 bbox 하단에 있으면(y > bbox 중심) stem_up=False (기둥이 아래)
    """
    x, y, w, h = bbox
    region = binary[y: y + h, x: x + w]

    vproj = _vertical_projection(region)
    peaks = _find_stem_peaks(vproj)

    if len(peaks) < 2:
        return None

    splits = _find_split_positions(vproj, peaks)
    if not splits:
        return None

    if min_sub_width is None:
        min_sub_width = notehead_radius

    # 빔 방향 힌트: 가로 투영(각 y행 픽셀 수)에서 빔이 어디 있는지 판단
    # 빔은 bbox의 상단(stem_up) 또는 하단(stem_down)에 집중됨
    hproj = (region > 0).sum(axis=1)  # 각 y행의 픽셀 수
    bbox_mid_local = h // 2
    upper_sum = int(hproj[:bbox_mid_local].sum())
    lower_sum = int(hproj[bbox_mid_local:].sum())
    # 위쪽에 픽셀이 더 많으면 빔이 위(stem_up=True), 아래면 stem_down
    beam_stem_up = (upper_sum >= lower_sum)

    boundaries = [0] + splits + [w]
    results = []
    for i, peak in enumerate(peaks):
        if i >= len(boundaries) - 1:
            break
        sub_x0 = boundaries[i]
        sub_x1 = boundaries[i + 1]
        sub_w = sub_x1 - sub_x0
        if sub_w < min_sub_width:
            continue

        sub_region = region[:, sub_x0:sub_x1]
        rows_with_pixel = np.where(sub_region.any(axis=1))[0]
        if len(rows_with_pixel) == 0:
            continue

        sub_y0 = int(rows_with_pixel[0])
        sub_y1 = int(rows_with_pixel[-1]) + 1
        sub_h = sub_y1 - sub_y0

        # 기둥 x (전체 이미지 좌표계): 피크 위치에서 음표 머리 x를 추정하는 데 사용
        stem_x_global = x + peak

        results.append({
            "bbox": (x + sub_x0, y + sub_y0, sub_w, sub_h),
            "stem_up": beam_stem_up,
            "stem_x": stem_x_global,   # head_x 추정용
        })

    return results if len(results) >= 2 else None


def _is_barline(w: int, h: int, notehead_radius: int) -> bool:
    """
    연결성분이 마디선(세로 직선)인지 판별한다.

    마디선은 폭이 1~3px으로 매우 좁고, 오선 전체 높이보다 훨씬 길다.
    notehead_radius × 8 이상의 h + w < 6이면 마디선으로 본다.
    """
    return w < 6 and h > notehead_radius * 8


def split_all_beam_components(
    binary: np.ndarray,
    stats: np.ndarray,
    notehead_radius: int,
    min_area: int = 50,
) -> list[dict]:
    """
    모든 연결성분을 검사해 빔 컴포넌트는 분할하고, 나머지는 그대로 반환.

    Returns:
        [{"bbox": (x,y,w,h), "stem_up": bool|None, "stem_x": int|None}, ...]
        stem_up: 빔에서 추정한 기둥 방향 (빔 아닌 컴포넌트는 None)
        stem_x:  기둥 x좌표 힌트 (head_x 추정용, 빔 아닌 컴포넌트는 None)
    """
    result = []
    n = len(stats)

    for i in range(1, n):
        area = int(stats[i][4])
        if area < min_area:
            continue

        bx, by, bw, bh = int(stats[i][0]), int(stats[i][1]), int(stats[i][2]), int(stats[i][3])
        bbox = (bx, by, bw, bh)

        # 마디선(세로 직선) 제거
        if _is_barline(bw, bh, notehead_radius):
            continue

        if is_beam_component(bw, bh, notehead_radius):
            sub_results = split_beam_component(binary, bbox, notehead_radius)
            if sub_results:
                result.extend(sub_results)
                continue

        result.append({"bbox": bbox, "stem_up": None, "stem_x": None})

    return result
