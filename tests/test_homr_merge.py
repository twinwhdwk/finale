"""
homr_runner.merge_page_musicxmls() 단위 테스트.

homr는 PDF를 페이지별 이미지로 쪼개 각각 .musicxml을 출력하므로(각
페이지마다 마디 번호가 1부터 다시 시작), 원본 Finale XML(곡 전체가
보통 한 파일)과 비교하려면 여러 페이지 결과를 마디 번호 1부터 다시
이어붙인 합본으로 만들어야 한다. 이 로직을 모델 다운로드 없이 검증한다.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from music21 import stream, note, meter, converter  # noqa: E402
from homr_runner import merge_page_musicxmls  # noqa: E402


def _make_page(tmpdir: Path, name: str, n_measures: int, start_midi: int) -> str:
    """homr 출력 스타일의 가짜 페이지 XML 생성 (마디 번호 항상 1부터 시작)."""
    s = stream.Score()
    p = stream.Part()
    for i in range(n_measures):
        m = stream.Measure(number=i + 1)
        if i == 0:
            m.append(meter.TimeSignature("4/4"))
        m.append(note.Note(start_midi + i, quarterLength=4.0))
        p.append(m)
    s.append(p)
    path = tmpdir / f"{name}.musicxml"
    s.write("musicxml", fp=str(path))
    return str(path)


def test_merge_renumbers_measures_sequentially(tmp_path: Path):
    """2페이지(각 1번부터 시작하는 마디 번호)를 병합하면 1..N으로 이어져야 함"""
    page1 = _make_page(tmp_path, "page1", n_measures=2, start_midi=60)  # 마디 1,2
    page2 = _make_page(tmp_path, "page2", n_measures=3, start_midi=70)  # 마디 1,2,3

    out_path = str(tmp_path / "merged.musicxml")
    merge_page_musicxmls([page1, page2], out_path)

    merged = converter.parse(out_path)
    measures = list(merged.parts[0].getElementsByClass("Measure"))

    assert len(measures) == 5, "두 페이지(2+3마디)가 5마디로 합쳐져야 함"
    assert [m.number for m in measures] == [1, 2, 3, 4, 5], (
        "마디 번호가 1..5로 순차 재부여되어야 함 (페이지 경계에서 충돌 없이)"
    )


def test_merge_preserves_note_order(tmp_path: Path):
    """병합 후에도 음표 내용 자체는 원래 페이지 순서를 그대로 유지해야 함"""
    page1 = _make_page(tmp_path, "page1", n_measures=1, start_midi=60)  # C4
    page2 = _make_page(tmp_path, "page2", n_measures=1, start_midi=62)  # D4

    out_path = str(tmp_path / "merged.musicxml")
    merge_page_musicxmls([page1, page2], out_path)

    merged = converter.parse(out_path)
    measures = list(merged.parts[0].getElementsByClass("Measure"))
    pitches = [list(m.getElementsByClass("Note"))[0].nameWithOctave for m in measures]

    assert pitches == ["C4", "D4"], f"음표 순서가 페이지 순서를 따라야 함, 실제: {pitches}"


def test_merge_single_page_still_works(tmp_path: Path):
    """페이지가 1개뿐이어도 정상적으로 (재번호 매기기만 거쳐) 동작해야 함"""
    page1 = _make_page(tmp_path, "page1", n_measures=2, start_midi=60)

    out_path = str(tmp_path / "merged.musicxml")
    merge_page_musicxmls([page1], out_path)

    merged = converter.parse(out_path)
    measures = list(merged.parts[0].getElementsByClass("Measure"))
    assert len(measures) == 2
    assert [m.number for m in measures] == [1, 2]


def test_merge_empty_list_raises(tmp_path: Path):
    """빈 목록을 넘기면 명확한 에러를 내야 함 (조용히 빈 파일을 만들지 않음)"""
    out_path = str(tmp_path / "merged.musicxml")
    try:
        merge_page_musicxmls([], out_path)
        raise AssertionError("ValueError가 발생해야 하는데 발생하지 않음")
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_merge_renumbers_measures_sequentially,
        test_merge_preserves_note_order,
        test_merge_single_page_still_works,
        test_merge_empty_list_raises,
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
