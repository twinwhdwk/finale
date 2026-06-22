"""
note_recognition.key_signature 단위 테스트.

조표 적용(get_accidental_map, apply_key_signature)의 정확성을
Circle of Fifths 기준으로 검증한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from note_recognition.key_signature import (  # noqa: E402
    get_accidental_map,
    apply_key_signature,
    apply_key_signature_to_pitches,
    MeasureAccidentalState,
)
from note_recognition.note_pitcher import Pitch  # noqa: E402


def _p(step, octave, acc="") -> Pitch:
    """테스트용 Pitch 헬퍼."""
    return Pitch(step=step, octave=octave, staff_step=0, accidental=acc)


def _p(step: str, octave: int = 4, acc: str = "") -> Pitch:
    return Pitch(step=step, octave=octave, staff_step=0, accidental=acc)


# ── get_accidental_map ───────────────────────────────────────────────

def test_c_major_has_no_accidentals():
    assert get_accidental_map(0) == {}


def test_g_major_has_f_sharp():
    m = get_accidental_map(1)
    assert m == {"F": "#"}


def test_d_major_has_f_c_sharp():
    m = get_accidental_map(2)
    assert m == {"F": "#", "C": "#"}


def test_f_major_has_b_flat():
    m = get_accidental_map(-1)
    assert m == {"B": "b"}


def test_bb_major_has_b_e_flat():
    m = get_accidental_map(-2)
    assert m == {"B": "b", "E": "b"}


def test_seven_sharps():
    m = get_accidental_map(7)
    assert set(m.keys()) == {"F", "C", "G", "D", "A", "E", "B"}
    assert all(v == "#" for v in m.values())


def test_seven_flats():
    m = get_accidental_map(-7)
    assert set(m.keys()) == {"B", "E", "A", "D", "G", "C", "F"}
    assert all(v == "b" for v in m.values())


# ── apply_key_signature ──────────────────────────────────────────────

def test_f_becomes_f_sharp_in_g_major():
    """G장조(#1)에서 F4 → F#4."""
    p = apply_key_signature(_p("F", 4), key_sig=1)
    assert p.step == "F"
    assert p.accidental == "#"
    assert p.name_with_octave == "F#4"


def test_non_affected_note_unchanged_in_g_major():
    """G장조에서 F 이외 음은 변경 없음."""
    for step in ["C", "D", "E", "G", "A", "B"]:
        p = apply_key_signature(_p(step, 4), key_sig=1)
        assert p.accidental == "", f"{step}: accidental 변경됨"


def test_existing_accidental_not_overwritten():
    """이미 임시표가 있는 Pitch는 조표로 덮어쓰지 않음 (임시표 우선)."""
    p_with_acc = _p("F", 4, acc="b")  # Fb (임시표)
    result = apply_key_signature(p_with_acc, key_sig=1)  # G장조여도
    assert result.accidental == "b", "임시표가 조표에 덮어써짐"


def test_c_major_leaves_all_unchanged():
    for step in ["C", "D", "E", "F", "G", "A", "B"]:
        p = apply_key_signature(_p(step), key_sig=0)
        assert p.accidental == ""


def test_b_flat_in_f_major():
    """F장조(-1)에서 B4 → Bb4."""
    p = apply_key_signature(_p("B", 4), key_sig=-1)
    assert p.accidental == "b"
    assert p.name_with_octave == "Bb4"


def test_apply_to_list():
    pitches = [_p("F"), _p("C"), _p("G")]
    result = apply_key_signature_to_pitches(pitches, key_sig=2)  # D장조: F#, C#
    assert result[0].accidental == "#"   # F → F#
    assert result[1].accidental == "#"   # C → C#
    assert result[2].accidental == ""    # G → G (변경 없음)


# ── xml_builder 연동 검증 ─────────────────────────────────────────────

def test_key_sig_applied_in_musicxml(tmp_path):
    """G장조에서 저장된 MusicXML의 F음이 F#으로 나와야 함."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
    from synthetic_score import SyntheticScoreSpec, NoteSpec, render_synthetic_staff
    from note_recognition.staff_removal import detect_staff_line_thickness
    from note_recognition.note_detector import detect_notes
    from note_recognition.xml_builder import save_musicxml
    from music21 import converter

    # staff_step=1 = F4 (높은음자리표 기준)
    spec = SyntheticScoreSpec(notes=[NoteSpec(x=200, staff_step=1, duration="quarter")])
    img, gt, _ = render_synthetic_staff(spec)
    top_y = spec.staff_top
    bot_y = spec.staff_top + 4 * spec.staff_gap
    t = detect_staff_line_thickness(img, [(top_y, bot_y)])
    result = detect_notes(img, top_y, bot_y, staff_gap=spec.staff_gap, line_thickness=t)

    out = str(tmp_path / "g_major.musicxml")
    save_musicxml(result, out, key_sig=1)

    reloaded = converter.parse(out)
    ns = list(reloaded.flatten().notes)
    assert len(ns) == 1
    assert ns[0].nameWithOctave == "F#4", (
        f"G장조에서 F가 F#으로 변환되지 않음: {ns[0].nameWithOctave}"
    )


# ── MeasureAccidentalState 테스트 ─────────────────────────────────────

def test_state_applies_key_sig_when_no_in_measure_accidental():
    """조표가 있고 마디 내 임시표가 없으면 조표가 적용되어야 함 (G장조 → F#)."""
    state = MeasureAccidentalState(key_sig=1)  # G장조: F#
    result = state.apply(_p("F", 4))
    assert result.step == "F"
    assert result.accidental == "#", f"F가 F#이 되어야 함: {result.name_with_octave}"


def test_state_resets_on_new_measure():
    """마디 경계(reset)에서 마디 내 임시표 기억이 지워져야 함."""
    state = MeasureAccidentalState(key_sig=0)  # C장조
    # 마디 1: F#이 명시적으로 등장
    state.apply(_p("F", 4, acc="#"))
    # 마디 2 시작 → 상태 초기화
    state.reset()
    # 이제 F는 자연음이어야 함
    result = state.apply(_p("F", 4))
    assert result.accidental == "", (
        f"마디 경계 후 F가 여전히 #{result.accidental}으로 나옴 - reset 실패"
    )


def test_in_measure_accidental_propagates_within_measure():
    """
    마디 내에서 임시표가 붙은 음표 이후, 같은 음이름의 음표에
    임시표가 자동으로 전파되어야 함.

    예: C장조에서 C#이 한 번 나오면 그 뒤 C도 C#이어야 함.
    """
    state = MeasureAccidentalState(key_sig=0)
    # C#이 명시적으로 등장 → 상태에 기록
    first = state.apply(_p("C", 5, acc="#"))
    assert first.accidental == "#"
    # 같은 마디 내 C (임시표 없이) → 앞선 C#의 영향을 받아야 함
    second = state.apply(_p("C", 5))
    assert second.accidental == "#", (
        f"마디 내 임시표 전파 실패: C가 C#이어야 하는데 accidental='{second.accidental}'"
    )


def test_in_measure_overrides_key_sig():
    """
    마디 내 임시표(#/b)가 조표보다 우선해야 함.

    현재 한계: 제자리표(♮)는 Pitch.accidental=""로 표현되는데,
    ""는 "임시표 없음"과 구분이 안 되어 상태 기계에 기록되지 않음.
    따라서 제자리표로 조표를 무효화하는 케이스는 현재 미지원.
    이 테스트는 지원 가능한 케이스(#/b가 조표를 덮어쓰는 것)만 검증.
    """
    state = MeasureAccidentalState(key_sig=1)  # G장조: F#
    state.reset()
    # 마디 내 Fb(임시표) 등장 → 상태에 "b" 기록
    state.apply(_p("F", 4, acc="b"))
    # 이후 F는 Fb여야 함 (조표 F#보다 마디 내 임시표 Fb가 우선)
    result = state.apply(_p("F", 4))
    assert result.accidental == "b", (
        f"마디 내 Fb 이후 F가 조표(F#)로 돌아감: accidental='{result.accidental}'"
    )


def test_different_notes_independent_in_measure():
    """마디 내 임시표 상태는 음이름별로 독립적이어야 함."""
    state = MeasureAccidentalState(key_sig=0)
    state.apply(_p("C", 5, acc="#"))  # C# 등장
    # F는 영향 없어야 함
    result_f = state.apply(_p("F", 4))
    assert result_f.accidental == "", f"C#이 F에 영향을 줌: {result_f.accidental}"


def test_state_with_key_sig_and_in_measure():
    """조표 + 마디 내 임시표가 함께 있을 때 우선순위가 맞아야 함."""
    state = MeasureAccidentalState(key_sig=2)  # D장조: F#, C#
    # 조표 확인
    assert state.apply(_p("F", 4)).accidental == "#"
    assert state.apply(_p("C", 5)).accidental == "#"
    # G는 조표 없음
    assert state.apply(_p("G", 4)).accidental == ""
    # 마디 내 Gb 등장
    state.apply(_p("G", 4, acc="b"))
    # 이후 G는 Gb여야 함
    result = state.apply(_p("G", 4))
    assert result.accidental == "b", f"마디 내 Gb 이후 G: accidental='{result.accidental}'"


if __name__ == "__main__":
    tests = [
        test_c_major_has_no_accidentals,
        test_g_major_has_f_sharp,
        test_d_major_has_f_c_sharp,
        test_f_major_has_b_flat,
        test_bb_major_has_b_e_flat,
        test_seven_sharps,
        test_seven_flats,
        test_f_becomes_f_sharp_in_g_major,
        test_non_affected_note_unchanged_in_g_major,
        test_existing_accidental_not_overwritten,
        test_c_major_leaves_all_unchanged,
        test_b_flat_in_f_major,
        test_apply_to_list,
    ]
    import tempfile
    tests_with_tmp = [test_key_sig_applied_in_musicxml]

    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    for t in tests_with_tmp:
        with tempfile.TemporaryDirectory() as d:
            try:
                t(Path(d))
                print(f"PASS  {t.__name__}")
                passed += 1
            except AssertionError as e:
                print(f"FAIL  {t.__name__}: {e}")
                failed += 1

    print(f"\n{passed}개 통과, {failed}개 실패")
    import sys as _sys
    _sys.exit(1 if failed else 0)
