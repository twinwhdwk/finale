"""
붙임줄(tie) / 이음줄(slur) 호 감지 모듈.

오선 제거 후 binary 이미지에서 얇고 넓은 곡선 구조를 찾아 ArcCandidate 목록을 반환한다.
음높이 비교(tie vs slur 판별)는 xml_builder.py에서 처리한다.

## 동작 원리

1. 수평 Morphological Opening (kernel: staff_gap*1.2 × 1px)
   - 수평 연속 픽셀이 min_arc_w 이상인 구조만 남긴다.
   - 빔(beam)·줄기(stem)는 너무 두껍거나 수직이라 필터됨.
2. 외곽선 컨투어에서 bbox 및 곡률(curvature) 계산.
   - 곡률: 양 끝점 y 평균 vs 최상단(또는 최하단) y의 차이 ≥ 2px 필수.
   - 직선(빔 아티팩트)는 곡률이 ≈0이라 자동 제외.
3. 시스템 경계 판정:
   - x1 ≥ img_width - edge_thr → cut_right (다음 시스템으로 이어짐)
   - x0 ≤ edge_thr → cut_left  (이전 시스템에서 이어짐)
   - 잘린 호는 너비·종횡비 기준을 절반으로 완화.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ArcCandidate:
    """감지된 붙임줄/이음줄 호 후보."""
    x0: int          # 왼쪽 끝 x
    x1: int          # 오른쪽 끝 x
    cy: int          # 양 끝점 y 평균 (음높이 매칭용)
    width: int       # 호 전체 너비 (x1 - x0)
    convex: str      # 'up' (위로 볼록) | 'down' (아래로 볼록)
    cut_left: bool = False    # 왼쪽 시스템 경계에서 잘린 호
    cut_right: bool = False   # 오른쪽 시스템 경계에서 잘린 호


def detect_arcs(
    binary: np.ndarray,
    staff_gap: int,
    staff_top_y: int,
    staff_bot_y: int,
    img_width: int = 0,
) -> list[ArcCandidate]:
    """
    오선 제거 후 binary 이미지에서 붙임줄/이음줄 호 후보를 검출한다.

    Args:
        binary:       BINARY_INV 기준 이진 이미지 (오선 제거 후).
        staff_gap:    오선 간격(픽셀).
        staff_top_y:  오선 맨 위줄 y.
        staff_bot_y:  오선 맨 아래줄 y.
        img_width:    이미지 너비 (0이면 binary.shape[1] 사용).
                      시스템 경계 잘린 호 판정에 사용.

    Returns:
        ArcCandidate 목록.
    """
    h, w = binary.shape
    if img_width == 0:
        img_width = w

    edge_thr = max(int(staff_gap * 1.5), 50)

    # 오선 위·아래 각 2.5칸. 3.5칸으로 넓힐 경우 오선 아래 가사 곡선이
    # 아크로 오감지되어 FP 증가. 2.5칸으로 FP/TP 균형이 가장 좋음.
    arc_y_min = max(0, staff_top_y - int(staff_gap * 2.5))
    arc_y_max = min(h, staff_bot_y + int(staff_gap * 2.5))

    # 수평 Opening 커널 폭: staff_gap 비례, 최소 30px.
    min_arc_w = max(int(staff_gap * 1.2), 30)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (min_arc_w, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)

    # y 범위 마스킹
    mask = np.zeros_like(horiz)
    mask[arc_y_min:arc_y_max, :] = 255
    horiz = cv2.bitwise_and(horiz, mask)

    cnts, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    arcs: list[ArcCandidate] = []
    for cnt in cnts:
        x, y, bw, bh = cv2.boundingRect(cnt)

        cut_right = (x + bw >= img_width - edge_thr)
        cut_left  = (x <= edge_thr)
        is_cut = cut_right or cut_left

        # 너비 필터: OPEN 커널과 동일 기준 (잘린 호는 절반 완화)
        if bw < (min_arc_w // 2 if is_cut else min_arc_w):
            continue

        # 높이: 오선·빔보다 얇아야 함
        if bh > staff_gap * 0.9:
            continue

        # 종횡비: 수평으로 충분히 길어야 함
        aspect_min = 1.5 if is_cut else 2.5
        if bw / max(bh, 1) < aspect_min:
            continue

        # 곡률 계산
        pts = cnt.reshape(-1, 2)
        apex_y  = float(pts[:, 1].min())
        base_y  = float(pts[:, 1].max())
        left_pt  = pts[pts[:, 0].argmin()]
        right_pt = pts[pts[:, 0].argmax()]
        end_avg  = (float(left_pt[1]) + float(right_pt[1])) / 2.0
        up_ness   = end_avg - apex_y
        down_ness = base_y  - end_avg

        # 직선(빔 아티팩트)은 곡률 ≈ 0 → 제외
        if max(up_ness, down_ness) < 2.0:
            continue

        convex = 'up' if up_ness > down_ness else 'down'
        cy = int(end_avg)

        arcs.append(ArcCandidate(
            x0=x, x1=x + bw, cy=cy, width=bw, convex=convex,
            cut_left=cut_left, cut_right=cut_right,
        ))

    return arcs
