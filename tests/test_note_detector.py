"""
note_recognition.note_detector 단위 테스트.

합성 악보 이미지(tests/fixtures/synthetic_score.py)로 음표 검출 및
음가(duration) 분류 알고리즘을 검증한다.

검증 범주:
- 5종 음가 기본 분류 (whole/half/quarter/eighth/sixteenth)
- 여러 음표가 섞인 마디에서 개수 및 순서(x 정렬) 검증
- 오선 높이에 따른 음표 위치 변화에 대한 견고성
- stem_down(기둥 아래 방향) 음표 분류
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

from synthetic_score import SyntheticScoreSpec, NoteSpec, render_synthetic_staff  # noqa: E402
from note_recognition.staff_removal import detect_staff_line_thickness  # noqa: E402
from note_recognition.note_detector import detect_notes  # noqa: E402


def _run(notes_spec: list[NoteSpec], **kwargs) -> tuple:
    """합성 이미지 생성 + 오선 두께 측정 + 음표 검출을 한 번에 수행."""
    spec = SyntheticScoreSpec(notes=notes_spec, **kwargs)
    img, gt, _ = render_synthetic_staff(spec)
    top_y = spec.staff_top
    bot_y = spec.staff_top + 4 * spec.staff_gap
    t = detect_staff_line_thickness(img, [(top_y, bot_y)])
    result = detect_notes(img, top_y, bot_y, staff_gap=spec.staff_gap, line_thickness=t)
    return result, gt, spec


# ── 5종 음가 개별 분류 테스트 ─────────────────────────────────────────

def test_whole_note_classified_correctly():
    result, gt, _ = _run([NoteSpec(x=200, staff_step=4, duration="whole")])
    assert len(result.notes) == 1
    assert result.notes[0].duration == "whole"
    assert result.notes[0].stem_up is None  # 기둥 없음


def test_half_note_classified_correctly():
    result, gt, _ = _run([NoteSpec(x=200, staff_step=4, duration="half")])
    assert len(result.notes) == 1
    n = result.notes[0]
    assert n.duration == "half"
    assert n.n_flags == 0
    assert n.stem_up is not None


def test_quarter_note_classified_correctly():
    result, gt, _ = _run([NoteSpec(x=200, staff_step=4, duration="quarter")])
    assert len(result.notes) == 1
    n = result.notes[0]
    assert n.duration == "quarter"
    assert n.n_flags == 0


def test_eighth_note_classified_correctly():
    result, gt, _ = _run([NoteSpec(x=200, staff_step=4, duration="eighth")])
    assert len(result.notes) == 1
    n = result.notes[0]
    assert n.duration == "eighth"
    assert n.n_flags == 1


def test_sixteenth_note_classified_correctly():
    result, gt, _ = _run([NoteSpec(x=200, staff_step=4, duration="sixteenth")])
    assert len(result.notes) == 1
    n = result.notes[0]
    assert n.duration == "sixteenth"
    assert n.n_flags == 2


# ── 혼합 마디 테스트 ─────────────────────────────────────────────────

def test_all_five_durations_in_one_measure():
    """5종 음가가 섞인 마디에서 5개 모두 정확히 분류되어야 함 (5/5)."""
    notes_spec = [
        NoteSpec(x=200, staff_step=4, duration="quarter"),
        NoteSpec(x=350, staff_step=6, duration="eighth"),
        NoteSpec(x=500, staff_step=2, duration="half"),
        NoteSpec(x=650, staff_step=4, duration="whole"),
        NoteSpec(x=800, staff_step=8, duration="sixteenth"),
    ]
    result, gt, _ = _run(notes_spec)

    assert len(result.notes) == 5, f"5개 음표 기대, 검출: {len(result.notes)}개"
    correct = sum(n.duration == t["duration"] for n, t in zip(result.notes, gt))
    assert correct == 5, (
        f"5/5 기대, {correct}/5 정확 - "
        + ", ".join(f"{n.duration}(정답:{t['duration']})" for n, t in zip(result.notes, gt)
                    if n.duration != t["duration"])
    )


def test_notes_sorted_by_x_position():
    """검출된 음표는 x좌표(악보 읽기 순서) 오름차순으로 정렬되어야 함."""
    notes_spec = [
        NoteSpec(x=500, staff_step=4, duration="quarter"),
        NoteSpec(x=200, staff_step=4, duration="whole"),
        NoteSpec(x=800, staff_step=4, duration="half"),
    ]
    result, _, _ = _run(notes_spec)
    assert len(result.notes) == 3
    xs = [n.head_x for n in result.notes]
    assert xs == sorted(xs), f"x 정렬 실패: {xs}"


def test_note_count_matches_spec():
    """마디 내 음표 수가 명세한 수와 정확히 일치해야 함."""
    for count in [1, 2, 3, 4]:
        notes_spec = [
            NoteSpec(x=200 + i * 200, staff_step=4, duration="quarter")
            for i in range(count)
        ]
        result, _, _ = _run(notes_spec)
        assert len(result.notes) == count, (
            f"음표 {count}개 기대, 검출: {len(result.notes)}개"
        )


# ── 위치 변화 견고성 테스트 ──────────────────────────────────────────

def test_note_at_various_staff_positions():
    """오선의 다양한 높이(step 0~8)에 있는 음표도 동일한 음가로 분류되어야 함."""
    for step in [0, 2, 4, 6, 8]:
        result, gt, _ = _run([NoteSpec(x=200, staff_step=step, duration="quarter")])
        assert len(result.notes) >= 1, f"staff_step={step}에서 음표 미검출"
        assert result.notes[0].duration == "quarter", (
            f"staff_step={step}에서 오분류: {result.notes[0].duration}"
        )


def test_stem_down_quarter_classified_correctly():
    """기둥이 아래로 내려간 4분음표도 'quarter'로 분류되어야 함."""
    result, gt, _ = _run([NoteSpec(x=200, staff_step=4, duration="quarter", stem_up=False)])
    assert len(result.notes) == 1
    assert result.notes[0].duration == "quarter"
    assert result.notes[0].n_flags == 0


def test_stem_down_eighth_classified_correctly():
    """
    기둥이 아래로 내려간 8분음표 분류.

    회귀 방지: 합성 이미지가 stem_down 깃발을 기둥 오른쪽에 잘못 그리고 있었고
    (표준은 왼쪽), _count_flags도 항상 오른쪽만 탐색했어서 stem_down 케이스가
    우연히 통과하고 있었음. 두 버그를 동시에 수정해 이 테스트가 진짜로 왼쪽
    깃발 탐색을 검증하도록 함.
    """
    result, gt, _ = _run([NoteSpec(x=300, staff_step=4, duration="eighth", stem_up=False)])
    assert len(result.notes) == 1
    n = result.notes[0]
    assert n.duration == "eighth", f"stem_down eighth 오분류: {n.duration}"
    assert n.n_flags == 1


def test_stem_down_sixteenth_classified_correctly():
    """기둥이 아래로 내려간 16분음표 - 깃발 2개 탐지 검증."""
    result, gt, _ = _run([NoteSpec(x=300, staff_step=4, duration="sixteenth", stem_up=False)])
    assert len(result.notes) == 1
    n = result.notes[0]
    assert n.duration == "sixteenth", f"stem_down sixteenth 오분류: {n.duration}"
    assert n.n_flags == 2


def test_stem_down_half_classified_correctly():
    """기둥이 아래로 내려간 2분음표."""
    result, gt, _ = _run([NoteSpec(x=300, staff_step=4, duration="half", stem_up=False)])
    assert len(result.notes) == 1
    assert result.notes[0].duration == "half"
    assert result.notes[0].n_flags == 0


# ── 빔(beam) 연결 테스트 ─────────────────────────────────────────────

def test_beamed_eighth_notes_split_correctly():
    """
    빔으로 연결된 8분음표 2개가 개별 음표로 분리되어야 함.

    회귀 방지: 빔이 없으면 두 음표는 별개 연결성분으로 잡히지만, 빔이 있으면
    기둥 끝이 연결되어 하나의 큰 컴포넌트가 됨. beam_splitter.py의 세로
    투영 피크 기반 분할 알고리즘이 이를 처리해야 함.
    """
    result, gt, _ = _run([
        NoteSpec(x=200, staff_step=4, duration="eighth", beam_to_next=True),
        NoteSpec(x=320, staff_step=6, duration="eighth"),
    ])
    assert len(result.notes) == 2, f"빔 분리 실패: {len(result.notes)}개 검출"
    assert result.notes[0].duration == "eighth"
    assert result.notes[1].duration == "eighth"


def test_beamed_sixteenth_notes_split_correctly():
    """빔으로 연결된 16분음표 2개 - 분리 및 sixteenth 분류 검증.

    주의: 빔 그룹에서 분리된 서브bbox에 빔이 일부 포함돼 깃발이 2개 이상
    잡힐 수 있지만, _classify_duration의 {2: sixteenth}.get(n, 'sixteenth')
    로직으로 2+ 모두 sixteenth로 처리되므로 분류 정확도는 유지됨.
    """
    result, gt, _ = _run([
        NoteSpec(x=200, staff_step=4, duration="sixteenth", beam_to_next=True),
        NoteSpec(x=320, staff_step=6, duration="sixteenth"),
    ])
    assert len(result.notes) == 2
    assert all(n.duration == "sixteenth" for n in result.notes)
    assert all(n.n_flags >= 2 for n in result.notes)  # 빔 포함으로 2 이상일 수 있음


def test_mixed_beamed_and_independent_notes():
    """빔 그룹과 독립 음표가 섞인 마디 - 모두 정확히 분류되어야 함."""
    notes_spec = [
        NoteSpec(x=150, staff_step=4, duration="quarter"),
        NoteSpec(x=280, staff_step=4, duration="eighth", beam_to_next=True),
        NoteSpec(x=400, staff_step=6, duration="eighth"),
        NoteSpec(x=530, staff_step=2, duration="half"),
    ]
    result, gt, _ = _run(notes_spec)
    assert len(result.notes) == 4
    assert [n.duration for n in result.notes] == ["quarter", "eighth", "eighth", "half"]


def test_three_beamed_eighth_notes():
    """3개 연속 빔 묶음도 개별 분리되어야 함."""
    result, gt, _ = _run([
        NoteSpec(x=150, staff_step=4, duration="eighth", beam_to_next=True),
        NoteSpec(x=270, staff_step=6, duration="eighth", beam_to_next=True),
        NoteSpec(x=390, staff_step=2, duration="eighth"),
    ])
    assert len(result.notes) == 3, f"3개 빔 분리 실패: {len(result.notes)}개"
    assert all(n.duration == "eighth" for n in result.notes)


if __name__ == "__main__":
    tests = [
        test_whole_note_classified_correctly,
        test_half_note_classified_correctly,
        test_quarter_note_classified_correctly,
        test_eighth_note_classified_correctly,
        test_sixteenth_note_classified_correctly,
        test_all_five_durations_in_one_measure,
        test_notes_sorted_by_x_position,
        test_note_count_matches_spec,
        test_note_at_various_staff_positions,
        test_stem_down_quarter_classified_correctly,
        test_stem_down_eighth_classified_correctly,
        test_stem_down_sixteenth_classified_correctly,
        test_stem_down_half_classified_correctly,
        test_beamed_eighth_notes_split_correctly,
        test_beamed_sixteenth_notes_split_correctly,
        test_mixed_beamed_and_independent_notes,
        test_three_beamed_eighth_notes,
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
    import sys
    sys.exit(1 if failed else 0)


# ── 쉼표(rest) 탐지 테스트 ──────────────────────────────────────────

def test_whole_rest_detected_separately_from_notes():
    """전쉼표(whole rest)가 음표와 별도로 rests 목록에 분리 탐지되어야 함."""
    from synthetic_score import RestSpec
    spec = SyntheticScoreSpec(
        notes=[NoteSpec(x=200, staff_step=4, duration="quarter")],
        rests=[RestSpec(x=500, duration="whole")],
    )
    img, gt, rest_gt = render_synthetic_staff(spec)
    top_y = spec.staff_top
    bot_y = spec.staff_top + 4 * spec.staff_gap
    t = detect_staff_line_thickness(img, [(top_y, bot_y)])
    result = detect_notes(img, top_y, bot_y, staff_gap=spec.staff_gap, line_thickness=t)

    assert len(result.notes) == 1, f"음표 {len(result.notes)}개 (기대 1개)"
    assert len(result.rests) >= 1, "전쉼표가 탐지되지 않음"
    assert any(r.duration == "whole" for r in result.rests), "전쉼표 duration 오분류"


def test_half_rest_detected():
    """2분쉼표(half rest)가 rests 목록에 탐지되어야 함."""
    from synthetic_score import RestSpec
    spec = SyntheticScoreSpec(rests=[RestSpec(x=400, duration="half")])
    img, gt, rest_gt = render_synthetic_staff(spec)
    top_y = spec.staff_top
    bot_y = spec.staff_top + 4 * spec.staff_gap
    t = detect_staff_line_thickness(img, [(top_y, bot_y)])
    result = detect_notes(img, top_y, bot_y, staff_gap=spec.staff_gap, line_thickness=t)

    assert len(result.rests) >= 1, "2분쉼표가 탐지되지 않음"
    assert any(r.duration == "half" for r in result.rests), "2분쉼표 duration 오분류"


def test_whole_note_not_misclassified_as_rest():
    """
    온음표(whole note)가 쉼표로 오분류되지 않아야 함.

    회귀 방지: _classify_rest의 초기 구현에서 4분/8분쉼표 판별 조건이
    온음표와 겹쳐 whole note가 DetectedRest로 빠지는 버그가 있었음.
    """
    result, gt, _ = _run([NoteSpec(x=200, staff_step=4, duration="whole")])
    assert len(result.notes) == 1, (
        f"whole note가 음표로 검출되지 않음 (notes={len(result.notes)}, "
        f"rests={len(result.rests)})"
    )
    assert result.notes[0].duration == "whole"
    assert len(result.rests) == 0, f"온음표가 쉼표로 오분류됨: {result.rests}"


def test_x_start_masks_header_region():
    """
    x_start 파라미터로 지정한 x 이전 영역의 음표는 검출에서 제외되어야 함.

    실제 PDF에서 오선 왼쪽에 위치한 음자리표/박자표/조표 기호가
    음표로 오분류되는 문제를 방지하기 위한 기능.
    """
    spec = SyntheticScoreSpec(notes=[
        NoteSpec(x=100, staff_step=4, duration="quarter"),  # 헤더 영역 (제외 대상)
        NoteSpec(x=400, staff_step=4, duration="quarter"),  # 실제 악보 영역
    ])
    img, gt, _ = render_synthetic_staff(spec)
    top_y = spec.staff_top
    bot_y = spec.staff_top + 4 * spec.staff_gap
    t = detect_staff_line_thickness(img, [(top_y, bot_y)])

    # x_start=0: 2개 모두 검출
    r_all = detect_notes(img, top_y, bot_y, staff_gap=spec.staff_gap,
                         line_thickness=t, x_start=0)
    assert len(r_all.notes) == 2

    # x_start=250: x=100 음표 제외
    r_masked = detect_notes(img, top_y, bot_y, staff_gap=spec.staff_gap,
                            line_thickness=t, x_start=250)
    assert len(r_masked.notes) == 1, f"x_start 마스킹 실패: {len(r_masked.notes)}개 검출"
    assert r_masked.notes[0].head_x > 250
