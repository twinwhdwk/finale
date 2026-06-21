"""
xml_comparator의 타이(붙임줄) 비교 로직 단위 테스트.

music21로 메모리 상에서 작은 악보를 만들어 임시 .musicxml로 저장한 뒤
compare()를 직접 호출해 검증한다. (Audiveris/homr 등 외부 OMR 엔진
없이도 xml_comparator 로직만 독립적으로 검증 가능)

실행:
    python -m pytest tests/test_tie_comparison.py -v
    또는
    python tests/test_tie_comparison.py   (pytest 없이 직접 실행)
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from music21 import stream, note, chord, meter, tie  # noqa: E402
from xml_comparator import compare  # noqa: E402


def _write(score, tmpdir: Path, name: str) -> str:
    path = tmpdir / f"{name}.musicxml"
    score.write("musicxml", fp=str(path))
    return str(path)


def _measure_with(*elements, time_sig="4/4") -> stream.Score:
    s = stream.Score()
    p = stream.Part()
    m = stream.Measure(number=1)
    m.append(meter.TimeSignature(time_sig))
    for el in elements:
        m.append(el)
    p.append(m)
    s.append(p)
    return s


# ── 픽스처 빌더 ──────────────────────────────────────────────────────

def _orig_tied_half_notes():
    """원본: C4 2분음표 + C4 2분음표, 타이로 연결 (실질 온음표 길이)"""
    n1 = note.Note("C4", quarterLength=2.0)
    n1.tie = tie.Tie("start")
    n2 = note.Note("C4", quarterLength=2.0)
    n2.tie = tie.Tie("stop")
    return _measure_with(n1, n2)


def _pdf_split_into_quarters():
    """PDF 추출본: 같은 마디를 C4 4분음표 4개로 쪼개 인식 (타이 완전 누락)"""
    notes = [note.Note("C4", quarterLength=1.0) for _ in range(4)]
    return _measure_with(*notes)


def _pdf_tie_attr_missing_only():
    """PDF 추출본: 길이는 원본과 동일(2분음표x2)하지만 tie 속성만 빠짐"""
    n1 = note.Note("C4", quarterLength=2.0)
    n2 = note.Note("C4", quarterLength=2.0)
    return _measure_with(n1, n2)


def _orig_no_tie_two_pitches():
    """원본: 타이 없는 C4, D4 (서로 다른 음높이, 각 2분음표)"""
    return _measure_with(
        note.Note("C4", quarterLength=2.0),
        note.Note("D4", quarterLength=2.0),
    )


def _pdf_extra_tie():
    """PDF 추출본: 원본엔 없는 타이를 C4에 잘못 붙임"""
    n1 = note.Note("C4", quarterLength=2.0)
    n1.tie = tie.Tie("start")
    return _measure_with(n1, note.Note("D4", quarterLength=2.0))


def _simple_pitch_mismatch_orig():
    return _measure_with(
        note.Note("C4", quarterLength=1.0),
        note.Note("E4", quarterLength=1.0),
        chord.Chord(["C4", "E4", "G4"], quarterLength=2.0),
    )


def _simple_pitch_mismatch_pdf():
    return _measure_with(
        note.Note("C4", quarterLength=1.0),
        note.Note("F4", quarterLength=1.0),  # E4 -> F4 오인식
        chord.Chord(["C4", "E4", "G4"], quarterLength=2.0),
    )


# ── 테스트 케이스 ────────────────────────────────────────────────────

def test_split_into_quarters_detects_missing_and_split(tmp_path: Path):
    """타이 그룹이 4개 음표로 쪼개진 경우: tie_missing + tie_suspect(분할) 둘 다 검출"""
    orig = _write(_orig_tied_half_notes(), tmp_path, "orig")
    pdf  = _write(_pdf_split_into_quarters(), tmp_path, "pdf")

    result = compare(pdf, orig, part_index=0)
    kinds = [d.kind for d in result.discrepancies]

    assert "tie_missing" in kinds, "타이 누락이 검출되어야 함"
    assert "tie_suspect" in kinds, "분할 인식 의심이 검출되어야 함"
    split_msgs = [d.message for d in result.discrepancies if d.kind == "tie_suspect" and "분리 인식" in d.message]
    assert any("4개" in m for m in split_msgs), "4개로 분리됐다는 메시지가 있어야 함"


def test_tie_attribute_missing_only(tmp_path: Path):
    """길이는 같고 tie 속성만 빠진 경우: tie_missing이 명확히 잡혀야 함"""
    orig = _write(_orig_tied_half_notes(), tmp_path, "orig")
    pdf  = _write(_pdf_tie_attr_missing_only(), tmp_path, "pdf")

    result = compare(pdf, orig, part_index=0)
    kinds = [d.kind for d in result.discrepancies]

    assert "tie_missing" in kinds
    assert result.tie_missing_count == 1


def test_identical_score_has_no_false_positive(tmp_path: Path):
    """원본을 그대로 자기 자신과 비교하면 불일치가 0건이어야 함 (회귀 방지)"""
    orig = _write(_orig_tied_half_notes(), tmp_path, "orig")

    result = compare(orig, orig, part_index=0)
    assert len(result.discrepancies) == 0, (
        f"동일 파일 비교에서 false positive 발생: "
        f"{[(d.kind, d.message) for d in result.discrepancies]}"
    )


def test_extra_tie_detected(tmp_path: Path):
    """PDF측에만 잘못된 타이가 붙은 경우: tie_extra가 잡혀야 함"""
    orig = _write(_orig_no_tie_two_pitches(), tmp_path, "orig")
    pdf  = _write(_pdf_extra_tie(), tmp_path, "pdf")

    result = compare(pdf, orig, part_index=0)
    assert result.tie_extra_count == 1


def test_simple_pitch_mismatch_unaffected(tmp_path: Path):
    """타이 비교 로직 추가가 기존 단순 음높이 비교 기능을 깨지 않아야 함"""
    orig = _write(_simple_pitch_mismatch_orig(), tmp_path, "orig")
    pdf  = _write(_simple_pitch_mismatch_pdf(), tmp_path, "pdf")

    result = compare(pdf, orig, part_index=0)
    assert len(result.discrepancies) == 1
    assert result.discrepancies[0].kind == "pitch"


# ── pytest 없이 직접 실행 가능하게 ───────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_split_into_quarters_detects_missing_and_split,
        test_tie_attribute_missing_only,
        test_identical_score_has_no_false_positive,
        test_extra_tie_detected,
        test_simple_pitch_mismatch_unaffected,
    ]
    passed, failed = 0, 0
    for t in tests:
        with tempfile.TemporaryDirectory() as d:
            try:
                t(Path(d))
                print(f"PASS  {t.__name__}")
                passed += 1
            except AssertionError as e:
                print(f"FAIL  {t.__name__}: {e}")
                failed += 1
    print(f"\n{passed}개 통과, {failed}개 실패")
    sys.exit(1 if failed else 0)
