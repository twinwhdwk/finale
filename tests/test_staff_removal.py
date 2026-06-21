"""
note_recognition.staff_removal 단위 테스트.

합성 악보 이미지(tests/fixtures/synthetic_score.py)로 오선 제거 알고리즘을
검증한다. ground truth가 픽셀 단위로 확실하므로 "오선이 완전히 사라졌는가"
"음표가 충분히 보존됐는가"를 정량적으로 측정할 수 있다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from synthetic_score import SyntheticScoreSpec, NoteSpec, render_synthetic_staff  # noqa: E402
from note_recognition.staff_removal import (  # noqa: E402
    detect_staff_line_thickness,
    remove_staff_lines,
)

# 합성 이미지의 실제 렌더링 두께 (cv2.line(thickness=2)가 실제로는 3px로
# 그려짐 - Bresenham 알고리즘 특성). synthetic_score.py의 line_thickness
# 기본값과 별개로, 여기서는 직접 측정한 실측치를 씀.
ACTUAL_RENDERED_THICKNESS = 3


def _staff_y_range(spec: SyntheticScoreSpec) -> tuple[int, int]:
    top_y = spec.staff_top
    bot_y = spec.staff_top + 4 * spec.staff_gap
    return top_y, bot_y


def test_thickness_detection_matches_actual_rendering():
    """detect_staff_line_thickness가 실제 렌더링된 두께(3px)를 정확히 측정해야 함.

    회귀 방지: 과거 버전은 run의 시작점이 아닌 모든 y에서 재측정해
    최빈값이 항상 실제보다 작게(1px) 나오는 버그가 있었음.
    """
    spec = SyntheticScoreSpec(notes=[])
    img, _ = render_synthetic_staff(spec)
    top_y, bot_y = _staff_y_range(spec)

    detected = detect_staff_line_thickness(img, [(top_y, bot_y)])
    assert detected == ACTUAL_RENDERED_THICKNESS, (
        f"두께 측정값 {detected}이 실제 렌더링 두께 {ACTUAL_RENDERED_THICKNESS}와 다름"
    )


def test_thickness_detection_robust_to_notes_present():
    """음표가 섞여 있어도 오선 두께 측정이 음표 두께에 휘둘리지 않아야 함
    (오선이 다수이므로 최빈값 채택이 음표보다 우세해야 함)."""
    spec = SyntheticScoreSpec(notes=[
        NoteSpec(x=200, staff_step=4, duration="quarter"),
        NoteSpec(x=400, staff_step=2, duration="half"),
        NoteSpec(x=500, staff_step=4, duration="whole"),
    ])
    img, _ = render_synthetic_staff(spec)
    top_y, bot_y = _staff_y_range(spec)

    detected = detect_staff_line_thickness(img, [(top_y, bot_y)])
    assert detected == ACTUAL_RENDERED_THICKNESS


def test_staff_lines_fully_removed_in_empty_region():
    """음표가 없는 영역에서는 오선이 100% 제거되어야 함 (잔여 픽셀 0)."""
    spec = SyntheticScoreSpec(notes=[
        NoteSpec(x=200, staff_step=4, duration="quarter"),
    ])
    img, _ = render_synthetic_staff(spec)
    top_y, bot_y = _staff_y_range(spec)

    removed = remove_staff_lines(img, top_y, bot_y,
                                  line_thickness=ACTUAL_RENDERED_THICKNESS,
                                  min_horizontal_run=40)

    _, after = cv2.threshold(removed, 128, 255, cv2.THRESH_BINARY_INV)
    # 음표(x=200 근방)와 충분히 떨어진 오른쪽 빈 영역에서 측정
    empty_region = after[top_y - 5:bot_y + 5, 900:1000]
    assert empty_region.sum() == 0, (
        f"빈 오선 영역에 잔여 픽셀 {empty_region.sum() // 255}개 남음 (0이어야 함)"
    )


def test_filled_notehead_mostly_preserved():
    """채워진 음표머리(4분음표)는 오선 제거 후에도 대부분 보존되어야 함
    (최소 70% 픽셀 유지 - 가장자리 일부 깎이는 것은 허용)."""
    spec = SyntheticScoreSpec(notes=[
        NoteSpec(x=200, staff_step=4, duration="quarter"),
    ])
    img, gt = render_synthetic_staff(spec)
    top_y, bot_y = _staff_y_range(spec)

    removed = remove_staff_lines(img, top_y, bot_y,
                                  line_thickness=ACTUAL_RENDERED_THICKNESS,
                                  min_horizontal_run=40)

    _, before = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    _, after = cv2.threshold(removed, 128, 255, cv2.THRESH_BINARY_INV)

    head_x, head_y = gt[0]["x"], gt[0]["head_y"]
    r = spec.notehead_radius + 3
    region_before = before[head_y - r:head_y + r, head_x - r:head_x + r]
    region_after = after[head_y - r:head_y + r, head_x - r:head_x + r]

    pixels_before = region_before.sum() // 255
    pixels_after = region_after.sum() // 255
    preservation_ratio = pixels_after / pixels_before

    assert preservation_ratio >= 0.70, (
        f"채워진 음표머리 보존율 {preservation_ratio:.2%}이 70% 미만 "
        f"(제거전 {pixels_before}px -> 제거후 {pixels_after}px)"
    )


def test_hollow_notehead_not_erased_as_staff_line():
    """
    빈 음표머리(2분/온음표)가 오선으로 오인되어 사라지지 않아야 함.

    회귀 방지: 세로 run-length만으로 판정하던 1차 구현은 빈 머리의 타원
    테두리(두께 2~3px)가 오선과 두께가 비슷해 통째로 지워지는 버그가 있었음
    (가로 연속성 검증을 추가해 해결).
    """
    spec = SyntheticScoreSpec(notes=[
        NoteSpec(x=500, staff_step=4, duration="whole"),
    ])
    img, gt = render_synthetic_staff(spec)
    top_y, bot_y = _staff_y_range(spec)

    removed = remove_staff_lines(img, top_y, bot_y,
                                  line_thickness=ACTUAL_RENDERED_THICKNESS,
                                  min_horizontal_run=40)

    _, before = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    _, after = cv2.threshold(removed, 128, 255, cv2.THRESH_BINARY_INV)

    head_x, head_y = gt[0]["x"], gt[0]["head_y"]
    r = spec.notehead_radius + 3
    region_before = before[head_y - r:head_y + r, head_x - r:head_x + r]
    region_after = after[head_y - r:head_y + r, head_x - r:head_x + r]

    pixels_before = region_before.sum() // 255
    pixels_after = region_after.sum() // 255
    preservation_ratio = pixels_after / pixels_before if pixels_before else 0

    assert preservation_ratio >= 0.50, (
        f"빈 음표머리(온음표) 보존율 {preservation_ratio:.2%}이 50% 미만 - "
        f"오선으로 오인되어 지워졌을 가능성 (제거전 {pixels_before}px -> 제거후 {pixels_after}px)"
    )


def test_all_duration_types_survive_staff_removal():
    """4분/8분/16분/2분/온음표 5종 모두 오선 제거 후 검출 가능한 수준으로 남아야 함."""
    spec = SyntheticScoreSpec(notes=[
        NoteSpec(x=200, staff_step=4, duration="quarter"),
        NoteSpec(x=300, staff_step=6, duration="eighth"),
        NoteSpec(x=400, staff_step=2, duration="half"),
        NoteSpec(x=500, staff_step=4, duration="whole"),
        NoteSpec(x=600, staff_step=8, duration="sixteenth"),
    ])
    img, gt = render_synthetic_staff(spec)
    top_y, bot_y = _staff_y_range(spec)

    removed = remove_staff_lines(img, top_y, bot_y,
                                  line_thickness=ACTUAL_RENDERED_THICKNESS,
                                  min_horizontal_run=40)
    _, after = cv2.threshold(removed, 128, 255, cv2.THRESH_BINARY_INV)

    for note_gt in gt:
        head_x, head_y = note_gt["x"], note_gt["head_y"]
        r = spec.notehead_radius + 3
        region = after[head_y - r:head_y + r, head_x - r:head_x + r]
        pixel_count = region.sum() // 255
        assert pixel_count > 0, (
            f"{note_gt['duration']} 음표(x={head_x})가 완전히 사라짐"
        )


if __name__ == "__main__":
    tests = [
        test_thickness_detection_matches_actual_rendering,
        test_thickness_detection_robust_to_notes_present,
        test_staff_lines_fully_removed_in_empty_region,
        test_filled_notehead_mostly_preserved,
        test_hollow_notehead_not_erased_as_staff_line,
        test_all_duration_types_survive_staff_removal,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}개 통과, {failed}개 실패")
    sys.exit(1 if failed else 0)
