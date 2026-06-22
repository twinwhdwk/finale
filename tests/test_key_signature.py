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
)
from note_recognition.note_pitcher import Pitch  # noqa: E402


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
