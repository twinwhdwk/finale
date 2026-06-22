"""
note_recognition.xml_builder 단위 테스트 + 전체 파이프라인 end-to-end 검증.

합성 악보 이미지 → 오선 제거 → 음표 검출 → 음높이 판정 → MusicXML 저장
→ music21로 다시 파싱 → 음이름/음가 확인의 전체 왕복을 검증한다.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

from music21 import converter  # noqa: E402

from synthetic_score import SyntheticScoreSpec, NoteSpec, render_synthetic_staff  # noqa: E402
from note_recognition.staff_removal import detect_staff_line_thickness  # noqa: E402
from note_recognition.note_detector import detect_notes, NoteDetectionResult  # noqa: E402
from note_recognition.xml_builder import notes_to_score, save_musicxml  # noqa: E402


def _detect(notes_spec: list[NoteSpec]) -> tuple[NoteDetectionResult, SyntheticScoreSpec]:
    spec = SyntheticScoreSpec(notes=notes_spec)
    img, _ = render_synthetic_staff(spec)
    top_y = spec.staff_top
    bot_y = spec.staff_top + 4 * spec.staff_gap
    t = detect_staff_line_thickness(img, [(top_y, bot_y)])
    result = detect_notes(img, top_y, bot_y, staff_gap=spec.staff_gap, line_thickness=t)
    return result, spec


# ── notes_to_score 기본 동작 ──────────────────────────────────────────

def test_score_has_one_part():
    """생성된 Score에 파트가 1개 있어야 함."""
    result, spec = _detect([NoteSpec(x=200, staff_step=4, duration="quarter")])
    score = notes_to_score(result)
    assert len(score.parts) == 1


def test_score_note_count_matches_detection():
    """Score의 음표 수가 검출된 음표 수와 일치해야 함."""
    notes_spec = [
        NoteSpec(x=200, staff_step=4, duration="quarter"),
        NoteSpec(x=350, staff_step=6, duration="quarter"),
        NoteSpec(x=500, staff_step=2, duration="half"),
    ]
    result, spec = _detect(notes_spec)
    score = notes_to_score(result)
    score_notes = list(score.flatten().notes)
    assert len(score_notes) == len(result.notes), (
        f"Score 음표 수({len(score_notes)}) != 검출 음표 수({len(result.notes)})"
    )


def test_duration_mapping_quarter():
    """4분음표가 quarterLength=1.0으로 변환되어야 함."""
    result, _ = _detect([NoteSpec(x=200, staff_step=4, duration="quarter")])
    score = notes_to_score(result)
    n = list(score.flatten().notes)[0]
    assert n.quarterLength == 1.0


def test_duration_mapping_all_types():
    """5종 음가가 모두 올바른 quarterLength로 변환되어야 함."""
    expected_ql = {
        "whole": 4.0, "half": 2.0, "quarter": 1.0,
        "eighth": 0.5, "sixteenth": 0.25,
    }
    for dur, ql in expected_ql.items():
        result, _ = _detect([NoteSpec(x=300, staff_step=4, duration=dur)])
        score = notes_to_score(result)
        score_notes = list(score.flatten().notes)
        assert len(score_notes) >= 1, f"{dur}: 음표 미검출"
        assert score_notes[0].quarterLength == ql, (
            f"{dur}: 기대 {ql}, 실제 {score_notes[0].quarterLength}"
        )


def test_treble_clef_pitch_e4():
    """오선 맨 아래줄(step=0) 음표는 E4여야 함."""
    result, _ = _detect([NoteSpec(x=200, staff_step=0, duration="quarter")])
    score = notes_to_score(result, clef_type="treble")
    n = list(score.flatten().notes)[0]
    assert n.nameWithOctave == "E4", f"step=0: 기대 E4, 실제 {n.nameWithOctave}"


def test_treble_clef_pitch_b4():
    """step=4 (가운데 줄)은 B4여야 함."""
    result, _ = _detect([NoteSpec(x=200, staff_step=4, duration="quarter")])
    score = notes_to_score(result, clef_type="treble")
    n = list(score.flatten().notes)[0]
    assert n.nameWithOctave == "B4", f"step=4: 기대 B4, 실제 {n.nameWithOctave}"


def test_measure_split_on_4_4():
    """4/4박자에서 4박이 넘으면 자동으로 새 마디를 생성해야 함."""
    # 4분음표 6개 → 마디 1에 4개, 마디 2에 2개
    notes_spec = [
        NoteSpec(x=100 + i * 130, staff_step=4, duration="quarter")
        for i in range(6)
    ]
    result, _ = _detect(notes_spec)
    score = notes_to_score(result, time_sig="4/4")
    measures = [m for m in score.parts[0].getElementsByClass("Measure") if m.notes]
    assert len(measures) >= 2, (
        f"6개 4분음표에서 마디가 {len(measures)}개 (최소 2개 기대)"
    )


# ── save_musicxml + 파일 왕복 ─────────────────────────────────────────

def test_save_and_reload_preserves_note_count(tmp_path: Path):
    """저장 후 다시 파싱했을 때 음표 수가 같아야 함."""
    notes_spec = [
        NoteSpec(x=200, staff_step=4, duration="quarter"),
        NoteSpec(x=350, staff_step=2, duration="half"),
        NoteSpec(x=550, staff_step=6, duration="quarter"),
    ]
    result, _ = _detect(notes_spec)
    out = str(tmp_path / "test.musicxml")
    save_musicxml(result, out)

    reloaded = converter.parse(out)
    reloaded_notes = list(reloaded.flatten().notes)
    assert len(reloaded_notes) == len(result.notes)


def test_save_and_reload_pitches_correct(tmp_path: Path):
    """저장 후 파싱 시 음이름이 정확해야 함."""
    notes_spec = [
        NoteSpec(x=200, staff_step=0,  duration="quarter"),   # E4
        NoteSpec(x=350, staff_step=4,  duration="quarter"),   # B4
        NoteSpec(x=500, staff_step=6,  duration="quarter"),   # D5
    ]
    result, _ = _detect(notes_spec)
    out = str(tmp_path / "test.musicxml")
    save_musicxml(result, out)

    reloaded = converter.parse(out)
    reloaded_notes = list(reloaded.flatten().notes)
    expected = ["E4", "B4", "D5"]
    for i, (n, exp) in enumerate(zip(reloaded_notes, expected)):
        assert n.nameWithOctave == exp, (
            f"음표[{i}]: 기대={exp}, 실제={n.nameWithOctave}"
        )


def test_full_pipeline_end_to_end(tmp_path: Path):
    """
    합성 이미지 → 검출 → MusicXML → 파싱 전체 파이프라인 end-to-end 검증.

    4분음표 E4/B4, 8분음표 D5, 2분음표 G4 순으로 구성.
    MusicXML로 저장 후 다시 파싱해서 음이름/음가가 모두 일치해야 함.
    """
    notes_spec = [
        NoteSpec(x=150, staff_step=0, duration="quarter"),    # E4
        NoteSpec(x=300, staff_step=4, duration="quarter"),    # B4
        NoteSpec(x=450, staff_step=6, duration="eighth"),     # D5
        NoteSpec(x=570, staff_step=2, duration="half"),       # G4 (E4에서 2칸 위)
    ]
    result, spec = _detect(notes_spec)
    out = str(tmp_path / "full_pipeline.musicxml")
    save_musicxml(result, out, time_sig="4/4")

    reloaded = converter.parse(out)
    reloaded_notes = list(reloaded.flatten().notes)

    assert len(reloaded_notes) == len(result.notes), (
        f"음표 수: {len(reloaded_notes)} != {len(result.notes)}"
    )

    expected_ql = [1.0, 1.0, 0.5, 2.0]
    for i, (n, exp_ql) in enumerate(zip(reloaded_notes, expected_ql)):
        assert n.quarterLength == exp_ql, (
            f"음표[{i}] 음가: 기대={exp_ql}, 실제={n.quarterLength}"
        )


def test_dotted_note_detected(tmp_path: Path):
    """점4분음표(is_dotted=True)가 올바르게 탐지되어야 함."""
    result, _ = _detect([NoteSpec(x=300, staff_step=4, duration="quarter", dotted=True)])
    assert len(result.notes) == 1
    assert result.notes[0].is_dotted is True, "점음표가 탐지되지 않음"


def test_non_dotted_note_not_marked(tmp_path: Path):
    """일반 4분음표는 is_dotted=False이어야 함 (false positive 방지)."""
    result, _ = _detect([NoteSpec(x=300, staff_step=4, duration="quarter", dotted=False)])
    assert len(result.notes) == 1
    assert result.notes[0].is_dotted is False


def test_dotted_quarter_saves_as_1_5_quarter_length(tmp_path: Path):
    """점4분음표는 MusicXML에서 quarterLength=1.5로 저장되어야 함."""
    result, _ = _detect([NoteSpec(x=300, staff_step=4, duration="quarter", dotted=True)])
    out = str(tmp_path / "dotted.musicxml")
    save_musicxml(result, out)
    reloaded = converter.parse(out)
    ns = list(reloaded.flatten().notes)
    assert len(ns) == 1
    assert ns[0].quarterLength == 1.5, f"점4분음표 quarterLength={ns[0].quarterLength} (기대:1.5)"


if __name__ == "__main__":
    import tempfile as _tmp
    tests_no_tmp = [
        test_score_has_one_part,
        test_score_note_count_matches_detection,
        test_duration_mapping_quarter,
        test_duration_mapping_all_types,
        test_treble_clef_pitch_e4,
        test_treble_clef_pitch_b4,
        test_measure_split_on_4_4,
    ]
    tests_with_tmp = [
        test_save_and_reload_preserves_note_count,
        test_save_and_reload_pitches_correct,
        test_full_pipeline_end_to_end,
    ]
    passed, failed = 0, 0
    for t in tests_no_tmp:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    for t in tests_with_tmp:
        with _tmp.TemporaryDirectory() as d:
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
