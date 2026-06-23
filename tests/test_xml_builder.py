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
    img, _, _ = render_synthetic_staff(spec)
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


def test_barline_based_measure_split(tmp_path: Path):
    """
    barlines 파라미터가 주어지면 박자 누산 대신 마디선 x좌표로 마디를 분리해야 함.

    4분음표 6개, x=470에 마디선 → 마디1에 3개(x<470), 마디2에 3개(x>470).
    """
    notes_spec = [
        NoteSpec(x=150, staff_step=4, duration="quarter"),
        NoteSpec(x=270, staff_step=6, duration="quarter"),
        NoteSpec(x=390, staff_step=2, duration="quarter"),
        NoteSpec(x=550, staff_step=4, duration="quarter"),
        NoteSpec(x=670, staff_step=0, duration="quarter"),
        NoteSpec(x=790, staff_step=6, duration="quarter"),
    ]
    result, _ = _detect(notes_spec)
    out = str(tmp_path / "barline.musicxml")
    save_musicxml(result, out, barlines=[470])

    reloaded = converter.parse(out)
    measures = [m for m in reloaded.parts[0].getElementsByClass("Measure")
                if list(m.flatten().notes)]
    assert len(measures) == 2, f"마디 수: {len(measures)} (기대: 2)"
    m1_notes = list(measures[0].flatten().notes)
    m2_notes = list(measures[1].flatten().notes)
    assert len(m1_notes) == 3, f"마디1 음표 수: {len(m1_notes)} (기대: 3)"
    assert len(m2_notes) == 3, f"마디2 음표 수: {len(m2_notes)} (기대: 3)"


def test_rest_and_note_integrated_by_x_order(tmp_path: Path):
    """
    음표와 쉼표가 x좌표 기준으로 통합 정렬되어 MusicXML에 올바른 순서로 들어가야 함.
    """
    from synthetic_score import RestSpec
    spec = SyntheticScoreSpec(
        notes=[NoteSpec(x=200, staff_step=4, duration="quarter"),
               NoteSpec(x=600, staff_step=4, duration="quarter")],
        rests=[RestSpec(x=400, duration="whole")],
    )
    img, gt, rest_gt = render_synthetic_staff(spec)
    top_y = spec.staff_top
    bot_y = spec.staff_top + 4 * spec.staff_gap
    t = detect_staff_line_thickness(img, [(top_y, bot_y)])
    result = detect_notes(img, top_y, bot_y, staff_gap=spec.staff_gap, line_thickness=t)

    # 쉼표(x=400)가 음표(x=200)와 음표(x=600) 사이에 있어야 함
    from note_recognition.xml_builder import _make_events
    events = _make_events(result)
    xs = [e.x for e in events]
    assert xs == sorted(xs), f"x 순서 오류: {xs}"


def test_key_signature_applied(tmp_path: Path):
    """key_sig 파라미터가 Score에 반영되어야 함 (G장조=샵1개)."""
    result, _ = _detect([NoteSpec(x=200, staff_step=4, duration="quarter")])
    out = str(tmp_path / "keysig.musicxml")
    save_musicxml(result, out, key_sig=1)  # G장조

    reloaded = converter.parse(out)
    key_sigs = list(reloaded.flatten().getElementsByClass("KeySignature"))
    assert len(key_sigs) >= 1, "조표가 MusicXML에 없음"
    assert key_sigs[0].sharps == 1, f"조표 샵 수: {key_sigs[0].sharps} (기대: 1)"


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


# ── barlines 기반 마디 분리 테스트 ──────────────────────────────────

def test_barlines_split_notes_into_correct_measures(tmp_path: Path):
    """
    barlines 좌표를 넘기면 x좌표 기준으로 마디가 정확히 분리되어야 함.

    x < 500 → 마디1, 500 ≤ x < 1000 → 마디2, x ≥ 1000 → 마디3
    """
    notes_spec = [
        NoteSpec(x=200, staff_step=4, duration="quarter"),   # 마디1
        NoteSpec(x=350, staff_step=4, duration="quarter"),   # 마디1
        NoteSpec(x=600, staff_step=4, duration="quarter"),   # 마디2
        NoteSpec(x=800, staff_step=4, duration="half"),      # 마디2
        NoteSpec(x=1100, staff_step=4, duration="whole"),   # 마디3
    ]
    result, spec = _detect(notes_spec)
    out = str(tmp_path / "barlines.musicxml")
    save_musicxml(result, out, time_sig="4/4", barlines=[500, 1000])

    reloaded = converter.parse(out)
    measures = [m for m in reloaded.parts[0].getElementsByClass("Measure") if m.notes]
    assert len(measures) == 3, f"마디 3개 기대, 실제 {len(measures)}개"
    assert len(list(measures[0].notes)) == 2, "마디1에 음표 2개여야 함"
    assert len(list(measures[1].notes)) == 2, "마디2에 음표 2개여야 함"
    assert len(list(measures[2].notes)) == 1, "마디3에 음표 1개여야 함"


def test_beat_fallback_without_barlines(tmp_path: Path):
    """
    barlines 없이 박자 누산 방식으로도 마디가 올바르게 분리되어야 함.
    4분음표 8개 → 4/4박자 기준 2개 마디.
    """
    notes_spec = [
        NoteSpec(x=100 + i * 150, staff_step=4, duration="quarter")
        for i in range(8)
    ]
    result, spec = _detect(notes_spec)
    out = str(tmp_path / "beat_fallback.musicxml")
    save_musicxml(result, out, time_sig="4/4", barlines=None)

    reloaded = converter.parse(out)
    measures = [m for m in reloaded.parts[0].getElementsByClass("Measure") if m.notes]
    assert len(measures) == 2, f"마디 2개 기대, 실제 {len(measures)}개"
    for m in measures:
        total_ql = sum(n.quarterLength for n in m.notes)
        assert abs(total_ql - 4.0) < 0.01, f"마디 길이 4박 기대: {total_ql}"


def test_barlines_vs_beat_consistency(tmp_path: Path):
    """
    마디선 좌표가 박자와 정확히 일치할 때 두 방식의 결과가 같아야 함.
    """
    notes_spec = [
        NoteSpec(x=200, staff_step=4, duration="quarter"),
        NoteSpec(x=350, staff_step=6, duration="quarter"),
        NoteSpec(x=500, staff_step=2, duration="half"),      # 여기까지 마디1 = 4박
        NoteSpec(x=750, staff_step=4, duration="whole"),    # 마디2 = 4박
    ]
    result, spec = _detect(notes_spec)

    out_bar = str(tmp_path / "barlines.musicxml")
    out_beat = str(tmp_path / "beat.musicxml")

    # barlines 기반 (x=620 에서 마디 경계)
    save_musicxml(result, out_bar, time_sig="4/4",
                  barlines=[int(n.head_x) for n in result.notes
                             if n.duration == "whole"])

    # beat 기반
    save_musicxml(result, out_beat, time_sig="4/4", barlines=None)

    reloaded_bar = converter.parse(out_bar)
    reloaded_beat = converter.parse(out_beat)

    notes_bar = [n.nameWithOctave for n in reloaded_bar.flatten().notes]
    notes_beat = [n.nameWithOctave for n in reloaded_beat.flatten().notes]

    assert notes_bar == notes_beat, (
        f"두 방식의 음이름 순서가 다름: barlines={notes_bar}, beat={notes_beat}"
    )


# ── 조표(key signature) + MusicXML 통합 ─────────────────────────────

def test_key_sig_applied_correctly_in_score(tmp_path: Path):
    """
    G장조(key_sig=1)에서 F 음표가 F#으로 변환되어 MusicXML에 저장되어야 함.
    staff_step=1 → F4, G장조 조표에 의해 F#4가 되어야 한다.
    """
    notes_spec = [
        NoteSpec(x=200, staff_step=4, duration="quarter"),   # B4 (조표 무관)
        NoteSpec(x=350, staff_step=1, duration="quarter"),   # F4 → F#4 (G장조)
        NoteSpec(x=500, staff_step=2, duration="half"),      # G4 (조표 무관)
    ]
    result, _ = _detect(notes_spec)
    out = str(tmp_path / "g_major.musicxml")
    save_musicxml(result, out, key_sig=1)  # G장조

    reloaded = converter.parse(out)
    notes = list(reloaded.flatten().notes)
    assert len(notes) == 3
    assert notes[1].nameWithOctave == "F#4", (
        f"G장조에서 F4가 F#4이어야 함: {notes[1].nameWithOctave}"
    )
    assert notes[0].nameWithOctave == "B4"   # 조표 영향 없는 B
    assert notes[2].nameWithOctave == "G4"   # 조표 영향 없는 G


def test_in_measure_accidental_propagates_in_score(tmp_path: Path):
    """
    같은 마디에서 임시표가 붙은 음이 나온 후 같은 음이름이 다시 나오면
    임시표가 유지되어야 함 (MeasureAccidentalState 통합).

    C장조에서 C#이 나온 후 같은 마디의 C가 C#이 되어야 함.
    현재 파이프라인은 Pitch.accidental을 직접 설정하는 경로가 없으므로
    조표 경로(key_sig)를 통한 전파를 검증한다.
    """
    notes_spec = [
        NoteSpec(x=200, staff_step=0, duration="quarter"),   # E4 (G장조: F# 조표와 무관)
        NoteSpec(x=350, staff_step=1, duration="quarter"),   # F4 → F#4 (G장조)
        NoteSpec(x=500, staff_step=1, duration="quarter"),   # F4 → 같은 마디에서 F#4 유지
        NoteSpec(x=650, staff_step=3, duration="quarter"),   # A4 (조표 무관)
    ]
    result, _ = _detect(notes_spec)
    out = str(tmp_path / "accidental_propagation.musicxml")
    save_musicxml(result, out, key_sig=1, time_sig="4/4")

    reloaded = converter.parse(out)
    notes = list(reloaded.flatten().notes)
    assert len(notes) == 4
    # 두 번째와 세 번째 F 음표 모두 F#이어야 함
    assert notes[1].nameWithOctave == "F#4", f"두 번째: {notes[1].nameWithOctave}"
    assert notes[2].nameWithOctave == "F#4", f"세 번째(전파): {notes[2].nameWithOctave}"


def test_dotted_note_in_beam_group(tmp_path: Path):
    """
    빔 그룹 안의 점음표가 음가(dotted eighth = 0.75박)로 저장되어야 함.
    """
    notes_spec = [
        NoteSpec(x=200, staff_step=4, duration="eighth", dotted=True, beam_to_next=True),
        NoteSpec(x=350, staff_step=6, duration="eighth"),
    ]
    result, _ = _detect(notes_spec)
    out = str(tmp_path / "dotted_beam.musicxml")
    save_musicxml(result, out, time_sig="4/4")

    reloaded = converter.parse(out)
    notes = list(reloaded.flatten().notes)
    assert len(notes) == 2, f"음표 2개 기대: {len(notes)}"
    # 첫 번째: 점8분음표 = 0.75박
    assert abs(notes[0].quarterLength - 0.75) < 0.01, (
        f"점8분음표 0.75박 기대: {notes[0].quarterLength}"
    )
    # 두 번째: 8분음표 = 0.5박
    assert abs(notes[1].quarterLength - 0.5) < 0.01, (
        f"8분음표 0.5박 기대: {notes[1].quarterLength}"
    )
