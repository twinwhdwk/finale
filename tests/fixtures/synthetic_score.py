"""
합성 악보 이미지 생성기 (음표 인식 알고리즘 검증용).

실제 PDF/스캔본 없이도 OpenCV 기반 오선 제거·음표 검출·음가 판정
로직을 검증하기 위해, 표준 5선보 위에 음표를 직접 그려서 ground
truth(정확한 위치/음가/음높이)가 100% 확실한 테스트 이미지를 만든다.

디지털 PDF(노이즈 없음) 조건을 반영해 안티에일리어싱 없는 순수 흑백
선화로 그린다 - 실제 입력 데이터 특성과 동일 (CLAUDE.md "입력 데이터
특성" 참고).

좌표계: OpenCV/numpy 표준 (y는 아래로 증가). 오선은 위에서부터
line0(가장 위) ~ line4(가장 아래) 5줄. 표준 음악 표기 규칙:
  - line4(맨 아래 줄) = 미(E4, 높은음자리표 기준)
  - 줄과 줄 사이 간격(staff_gap)마다 한 음씩 상행
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class NoteSpec:
    """그릴 음표 하나의 명세 (ground truth)."""
    x: int                  # 음표머리 중심 x좌표
    staff_step: int         # 오선 기준 위치. 0=맨 아래줄(line4), 1=그 위 칸,
                             # 2=line3, ... 위로 갈수록 +1 (반음계 아님, 온음계 디아토닉 스텝)
    duration: str            # "whole" | "half" | "quarter" | "eighth" | "sixteenth"
    stem_up: bool = True     # 기둥 방향 (staff_step이 중간(4) 이상이면 보통 아래로 그리는게 정석이나
                             # 단순화를 위해 명시적으로 지정)


@dataclass
class SyntheticScoreSpec:
    """합성 악보 1개 시스템(오선 1줄) 명세."""
    notes: list[NoteSpec] = field(default_factory=list)
    width: int = 1600
    height: int = 400
    staff_top: int = 150        # line0(맨 위 줄)의 y좌표
    staff_gap: int = 20         # 인접한 두 줄 사이 간격 (px)
    line_thickness: int = 2
    notehead_radius: int = 11   # 음표머리 타원 반지름 (가로)


def _staff_step_to_y(staff_step: int, staff_top: int, staff_gap: int) -> int:
    """
    staff_step(0=맨 아래줄) -> y 픽셀 좌표.

    오선은 5줄(line0~line4), line4가 staff_top + 4*staff_gap (맨 아래).
    staff_step 0 = line4, 1 = line4와 line3 사이 칸, 2 = line3, ...
    한 staff_step당 staff_gap/2 만큼 y가 감소(위로 이동).
    """
    line4_y = staff_top + 4 * staff_gap
    return int(round(line4_y - staff_step * (staff_gap / 2)))


def render_synthetic_staff(spec: SyntheticScoreSpec) -> tuple[np.ndarray, list[dict]]:
    """
    명세에 따라 오선 + 음표를 그린 흑백 이미지를 생성.

    Returns:
        (img_gray, ground_truth)
        img_gray:     uint8 그레이스케일 이미지 (255=흰 배경, 0=검정 선)
        ground_truth: 각 음표의 실제 위치/음가 정보 dict 리스트
                      (검증 시 검출 결과와 비교할 정답 데이터)
    """
    img = np.full((spec.height, spec.width), 255, dtype=np.uint8)

    # ── 오선 5줄 그리기 ──
    margin = 60
    for i in range(5):
        y = spec.staff_top + i * spec.staff_gap
        cv2.line(img, (margin, y), (spec.width - margin, y), 0, spec.line_thickness)

    ground_truth = []

    for note in spec.notes:
        head_y = _staff_step_to_y(note.staff_step, spec.staff_top, spec.staff_gap)
        gt = _draw_note(img, note, head_y, spec)
        ground_truth.append(gt)

    return img, ground_truth


def _draw_note(img: np.ndarray, note: NoteSpec, head_y: int, spec: SyntheticScoreSpec) -> dict:
    """음표 하나를 그리고 ground truth 딕셔너리를 반환."""
    r = spec.notehead_radius
    x = note.x

    is_filled = note.duration not in ("whole", "half")
    has_stem = note.duration != "whole"
    n_flags = {"eighth": 1, "sixteenth": 2}.get(note.duration, 0)

    # ── 음표머리: 채워진 음표는 -20도 기울어진 타원 (실제 인쇄 음표 모양에 가깝게) ──
    axes = (r, int(r * 0.72))
    angle = -20
    if is_filled:
        cv2.ellipse(img, (x, head_y), axes, angle, 0, 360, 0, -1)  # 채움
    else:
        cv2.ellipse(img, (x, head_y), axes, angle, 0, 360, 0, 2)   # 빈 머리(테두리만)

    stem_x = x + r - 2 if note.stem_up else x - r + 2
    stem_top = head_y
    stem_bot = head_y

    if has_stem:
        stem_len = spec.staff_gap * 3.5
        if note.stem_up:
            stem_top = int(head_y - stem_len)
            cv2.line(img, (stem_x, head_y), (stem_x, stem_top), 0, 2)
        else:
            stem_bot = int(head_y + stem_len)
            cv2.line(img, (stem_x, head_y), (stem_x, stem_bot), 0, 2)

    # ── 깃발(flag): 기둥 끝에서 짧은 사선 ──
    # 표준 음악 표기법:
    #   stem_up  → 깃발이 기둥 오른쪽 아래 방향으로 뻗음 (fx1 = stem_x + 12)
    #   stem_down → 깃발이 기둥 왼쪽 위 방향으로 뻗음  (fx1 = stem_x - 12)
    flag_end_y = stem_top if note.stem_up else stem_bot
    for f in range(n_flags):
        offset = f * 8
        if note.stem_up:
            fy0 = flag_end_y + offset
            fy1 = fy0 + 14
            fx1 = stem_x + 12
        else:
            fy0 = flag_end_y - offset
            fy1 = fy0 - 14
            fx1 = stem_x - 12
        cv2.line(img, (stem_x, fy0), (fx1, fy1), 0, 2)

    # ── 오선 밖 음표를 위한 덧줄(ledger line) ──
    if note.staff_step < 0 or note.staff_step > 8:
        # 짝수 staff_step(=오선 줄과 같은 위치)에만 덧줄 필요
        step = note.staff_step
        ledger_steps = []
        if step < 0:
            s = -2
            while s >= step:
                ledger_steps.append(s)
                s -= 2
        else:
            s = 10
            while s <= step:
                ledger_steps.append(s)
                s += 2
        for s in ledger_steps:
            ly = _staff_step_to_y(s, spec.staff_top, spec.staff_gap)
            cv2.line(img, (x - r - 4, ly), (x + r + 4, ly), 0, 2)

    return {
        "x": x,
        "head_y": head_y,
        "staff_step": note.staff_step,
        "duration": note.duration,
        "is_filled": is_filled,
        "has_stem": has_stem,
        "n_flags": n_flags,
        "stem_up": note.stem_up,
    }


if __name__ == "__main__":
    # 간단한 동작 확인: 4분음표 몇 개를 그려서 저장
    spec = SyntheticScoreSpec(notes=[
        NoteSpec(x=200, staff_step=4, duration="quarter"),
        NoteSpec(x=300, staff_step=6, duration="eighth"),
        NoteSpec(x=400, staff_step=2, duration="half"),
        NoteSpec(x=500, staff_step=4, duration="whole"),
        NoteSpec(x=600, staff_step=8, duration="sixteenth"),
    ])
    img, gt = render_synthetic_staff(spec)
    cv2.imwrite("/tmp/synthetic_staff_demo.png", img)
    print("저장 완료: /tmp/synthetic_staff_demo.png")
    for g in gt:
        print(g)
