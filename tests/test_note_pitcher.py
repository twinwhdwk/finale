"""
note_recognition.note_pitcher 단위 테스트.

높은음자리표(treble clef) 기준 pitch 매핑과 head_y → staff_step 역산을
검증한다. 이 테스트들은 외부 의존성 없이 순수 수학 계산만 검증한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sys as _sys_pitcher
_sys_pitcher.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from synthetic_score import _staff_step_to_y  # noqa: E402
from note_recognition.note_pitcher import (  # noqa: E402
    head_y_to_staff_step,
    staff_step_to_pitch,
    head_y_to_pitch,
    Pitch,
)

# 테스트용 오선 파라미터 (synthetic_score.py 기본값과 동일)
STAFF_TOP = 150
STAFF_GAP = 20
LINE4_Y   = STAFF_TOP + 4 * STAFF_GAP   # = 230


# ── staff_step → pitch 매핑 (treble clef) ────────────────────────────

def test_treble_clef_landmark_pitches():
    """높은음자리표 기준 주요 음들이 올바른 음이름을 반환해야 함."""
    landmarks = {
        0:  ("E", 4),   # 맨 아래줄
        1:  ("F", 4),   # 아래줄 위 칸
        2:  ("G", 4),   # 아래서 2번째 줄
        3:  ("A", 4),
        4:  ("B", 4),   # 가운데 줄
        5:  ("C", 5),   # 가운데와 위 사이 칸
        6:  ("D", 5),
        7:  ("E", 5),
        8:  ("F", 5),   # 맨 위줄
        9:  ("G", 5),
        -1: ("D", 4),   # 맨 아래 줄 아래 칸
        -2: ("C", 4),   # 가운데 C (middle C)
        -3: ("B", 3),
        -4: ("A", 3),
    }
    for step, (expected_name, expected_oct) in landmarks.items():
        p = staff_step_to_pitch(step, clef="treble")
        assert p.step == expected_name and p.octave == expected_oct, (
            f"step={step}: 기대={expected_name}{expected_oct}, "
            f"실제={p.step}{p.octave}"
        )


def test_middle_c_is_c4():
    """Middle C(C4)는 staff_step=-2에서 나와야 함 (오선 맨 아래줄 아래 2칸)."""
    p = staff_step_to_pitch(-2, clef="treble")
    assert p.step == "C"
    assert p.octave == 4
    assert p.name_with_octave == "C4"


def test_octave_increases_every_7_steps():
    """7 staff_step마다 정확히 1옥타브 올라가야 함."""
    for base_step in [0, 7, 14]:
        base = staff_step_to_pitch(base_step)
        higher = staff_step_to_pitch(base_step + 7)
        assert higher.step == base.step, (
            f"step {base_step}→{base_step+7}: 음 이름이 달라짐 ({base.step}→{higher.step})"
        )
        assert higher.octave == base.octave + 1, (
            f"step {base_step}→{base_step+7}: 옥타브 증가 없음"
        )


def test_name_with_octave_format():
    """name_with_octave 포맷이 music21 호환 형식이어야 함."""
    p = staff_step_to_pitch(0)
    assert p.name_with_octave == "E4"
    p2 = staff_step_to_pitch(-2)
    assert p2.name_with_octave == "C4"


def test_midi_note_c4_is_60():
    """C4의 MIDI 노트 번호는 60이어야 함 (표준)."""
    p = staff_step_to_pitch(-2)  # C4
    assert p.midi_note == 60, f"C4 MIDI = {p.midi_note} (기대: 60)"


def test_midi_note_ascending():
    """높은 pitch는 더 높은 MIDI 번호를 가져야 함."""
    steps = list(range(0, 9))  # E4~F5
    midi_notes = [staff_step_to_pitch(s).midi_note for s in steps]
    for i in range(len(midi_notes) - 1):
        assert midi_notes[i] < midi_notes[i + 1], (
            f"step {steps[i]}→{steps[i+1]}: MIDI가 내려감 ({midi_notes[i]}→{midi_notes[i+1]})"
        )


# ── bass clef ────────────────────────────────────────────────────────

def test_bass_clef_step0_is_g2():
    """낮은음자리표 step=0은 G2여야 함."""
    p = staff_step_to_pitch(0, clef="bass")
    assert p.step == "G"
    assert p.octave == 2


# ── head_y → staff_step 역산 ─────────────────────────────────────────

def test_head_y_on_line4_gives_step0():
    """head_y가 정확히 line4_y이면 staff_step=0 (E4)."""
    step = head_y_to_staff_step(LINE4_Y, STAFF_TOP, STAFF_GAP)
    assert step == 0


def test_head_y_to_staff_step_all_lines():
    """오선 5줄의 y좌표가 정확히 짝수 step을 반환해야 함 (줄 위치는 짝수)."""
    for line_i in range(5):
        y = STAFF_TOP + line_i * STAFF_GAP
        expected_step = (4 - line_i) * 2  # line0=step8, line4=step0
        step = head_y_to_staff_step(y, STAFF_TOP, STAFF_GAP)
        assert step == expected_step, (
            f"line{line_i}(y={y}): step={step}, 기대={expected_step}"
        )


def test_head_y_snaps_to_nearest_step():
    """head_y에 픽셀 오차가 있어도 가장 가까운 step으로 스냅되어야 함."""
    # line4_y=230, step0=E4
    # ±2px 이내는 같은 step으로
    for offset in [-2, -1, 0, 1, 2]:
        step = head_y_to_staff_step(LINE4_Y + offset, STAFF_TOP, STAFF_GAP)
        assert step == 0, (
            f"y={LINE4_Y+offset} (오차={offset}px): step={step}, 기대=0"
        )


def test_head_y_to_pitch_e4():
    """line4_y에서 head_y_to_pitch는 E4를 반환해야 함."""
    p = head_y_to_pitch(LINE4_Y, STAFF_TOP, STAFF_GAP)
    assert p.name_with_octave == "E4"


# ── 합성 이미지 연동 검증 ─────────────────────────────────────────────

def test_synthetic_score_staff_step_round_trip():
    """
    synthetic_score._staff_step_to_y → head_y_to_staff_step 왕복이
    동일한 step을 반환해야 함 (합성 이미지로 생성한 음표의 head_y를
    역산하면 원래 staff_step이 복원되는지 확인).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
    from synthetic_score import _staff_step_to_y

    for step in range(-4, 13):
        y = _staff_step_to_y(step, STAFF_TOP, STAFF_GAP)
        recovered = head_y_to_staff_step(y, STAFF_TOP, STAFF_GAP)
        assert recovered == step, (
            f"staff_step={step}: y={y}, 역산 결과={recovered}"
        )


def test_full_pipeline_note_pitch_from_synthetic():
    """
    합성 이미지에서 음표를 검출하고 pitch까지 판정하는 end-to-end 검증.
    detect_notes + head_y_to_pitch의 결합이 예상 음이름을 반환해야 함.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
    from synthetic_score import SyntheticScoreSpec, NoteSpec, render_synthetic_staff
    from note_recognition.staff_removal import detect_staff_line_thickness
    from note_recognition.note_detector import detect_notes

    # staff_step=0=E4, step=4=B4, step=6=D5, step=-2=C4
    note_specs = [
        NoteSpec(x=200, staff_step=0,  duration="quarter"),   # E4
        NoteSpec(x=350, staff_step=4,  duration="quarter"),   # B4
        NoteSpec(x=500, staff_step=6,  duration="quarter"),   # D5
        NoteSpec(x=650, staff_step=-2, duration="quarter"),   # C4 (아래 덧줄)
    ]
    spec = SyntheticScoreSpec(notes=note_specs)
    img, gt, _ = render_synthetic_staff(spec)
    top_y = spec.staff_top
    bot_y = spec.staff_top + 4 * spec.staff_gap
    t = detect_staff_line_thickness(img, [(top_y, bot_y)])
    result = detect_notes(img, top_y, bot_y, staff_gap=spec.staff_gap, line_thickness=t)

    expected_pitches = ["E4", "B4", "D5", "C4"]
    assert len(result.notes) == len(expected_pitches), (
        f"음표 수 불일치: {len(result.notes)} != {len(expected_pitches)}"
    )
    for note, gt_spec, expected in zip(result.notes, note_specs, expected_pitches):
        p = head_y_to_pitch(note.head_y, top_y, spec.staff_gap)
        assert p.name_with_octave == expected, (
            f"staff_step={gt_spec.staff_step}: 기대={expected}, "
            f"실제={p.name_with_octave} (head_y={note.head_y})"
        )


if __name__ == "__main__":
    tests = [
        test_treble_clef_landmark_pitches,
        test_middle_c_is_c4,
        test_octave_increases_every_7_steps,
        test_name_with_octave_format,
        test_midi_note_c4_is_60,
        test_midi_note_ascending,
        test_bass_clef_step0_is_g2,
        test_head_y_on_line4_gives_step0,
        test_head_y_to_staff_step_all_lines,
        test_head_y_snaps_to_nearest_step,
        test_head_y_to_pitch_e4,
        test_synthetic_score_staff_step_round_trip,
        test_full_pipeline_note_pitch_from_synthetic,
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
    import sys as _sys
    _sys.exit(1 if failed else 0)


def test_head_y_error_tolerance_within_safe_range():
    """
    head_y 오차가 ±4px 이내에서는 음이름이 정확해야 함.

    실측: staff_gap=20일 때 짝수 step은 ±5px, 홀수 step은 ±4px까지 허용.
    보수적으로 ±4px를 공통 안전 범위로 사용.
    합성 이미지 실측 오차: 최대 2px (여유 충분).
    """
    staff_top, staff_gap = 150, 20
    safe_margin = 4  # 짝수/홀수 step 공통 최소 안전 범위

    for step in range(0, 9):
        y_exact = _staff_step_to_y(step, staff_top, staff_gap)
        for offset in range(-safe_margin, safe_margin + 1):
            detected = head_y_to_staff_step(y_exact + offset, staff_top, staff_gap)
            assert detected == step, (
                f"staff_step={step} offset={offset}px: "
                f"검출={detected} (±{safe_margin}px 안전 범위 내 오류)"
            )


def test_head_y_error_causes_wrong_pitch_at_boundary():
    """
    staff_gap/2 픽셀 오차에서 음이름이 틀려야 함 (경계값 테스트).
    """
    staff_top, staff_gap = 150, 20
    half_gap = staff_gap // 2  # = 10px

    # 정확히 half_gap 오차 → 인접 step으로 이동
    y_step1 = _staff_step_to_y(1, staff_top, staff_gap)  # F4
    detected = head_y_to_staff_step(y_step1 + half_gap, staff_top, staff_gap)
    assert detected != 1, (
        f"half_gap={half_gap}px 오차에서도 step=1이 유지됨 - "
        "경계값 동작 이상"
    )


# ── 빔 그룹에서 음높이 판정 ──────────────────────────────────────────

def test_beamed_notes_pitch_detected_correctly():
    """
    빔으로 연결된 음표 그룹에서도 각 음표의 음높이가 정확히 판정되어야 함.

    빔 분리(beam_splitter) 후 head_y가 올바르게 추정되면
    head_y_to_pitch()도 정확히 작동한다.
    """
    import sys as _sys2
    _sys2.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
    from synthetic_score import SyntheticScoreSpec, NoteSpec, render_synthetic_staff
    from note_recognition.staff_removal import detect_staff_line_thickness
    from note_recognition.note_detector import detect_notes

    spec = SyntheticScoreSpec(notes=[
        NoteSpec(x=200, staff_step=2, duration="eighth", beam_to_next=True),  # G4
        NoteSpec(x=320, staff_step=4, duration="eighth", beam_to_next=True),  # B4
        NoteSpec(x=440, staff_step=6, duration="eighth"),                       # D5
    ])
    img, gt, _ = render_synthetic_staff(spec)
    top_y = spec.staff_top
    bot_y = spec.staff_top + 4 * spec.staff_gap
    t = detect_staff_line_thickness(img, [(top_y, bot_y)])
    result = detect_notes(img, top_y, bot_y, staff_gap=spec.staff_gap, line_thickness=t)

    assert len(result.notes) == 3, f"빔 3개 분리 실패: {len(result.notes)}개"
    expected = ["G4", "B4", "D5"]
    for i, (note, exp) in enumerate(zip(result.notes, expected)):
        p = head_y_to_pitch(note.head_y, top_y, spec.staff_gap)
        assert p.name_with_octave == exp, (
            f"빔 음표[{i}]: 기대={exp}, 검출={p.name_with_octave} "
            f"(head_y={note.head_y})"
        )
