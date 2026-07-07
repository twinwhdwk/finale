"""step_diff 단위 테스트 — 정렬/오류탐지 로직 (PDF 불필요)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from step_diff import (
    _align, compare_steps, PdfNote, MxlNote,
)


def _pdf(seq):
    return [PdfNote(step=s, page=1, system=1, x=i * 50) for i, s in enumerate(seq)]


def _mxl(seq_with_measures):
    return [MxlNote(step=s, measure=m) for s, m in seq_with_measures]


def test_align_identical():
    pairs = _align([1, 2, 3], [1, 2, 3])
    assert all(a is not None and b is not None for a, b in pairs)


def test_align_with_gap():
    # PDF에 노이즈 하나 삽입
    pairs = _align([1, 9, 2, 3], [1, 2, 3])
    pdf_only = [a for a, b in pairs if b is None]
    assert len(pdf_only) == 1


def test_compare_perfect_match():
    r = compare_steps(_pdf([0, 2, 4]), _mxl([(0, 1), (2, 1), (4, 2)]))
    assert r.match == 3 and not r.suspects


def test_compare_detects_injected_error():
    # 마디 2의 음이 MXL과 다름 → 의심 마디 2
    pdf = _pdf([0, 2, 6, 4])
    mxl = _mxl([(0, 1), (2, 1), (4, 2), (4, 2)])
    r = compare_steps(pdf, mxl, try_offsets=False)
    assert any(s.measure == 2 for s in r.suspects)


def test_compare_detects_missing_note():
    pdf = _pdf([0, 2])
    mxl = _mxl([(0, 1), (2, 1), (4, 2)])
    r = compare_steps(pdf, mxl, try_offsets=False)
    assert any(s.measure == 2 and s.missing >= 1 for s in r.suspects)


def test_offset_transposition_absorbed():
    # 교과서가 3도 위로 조옮김된 경우: offset 탐색으로 완전 일치
    pdf = _pdf([2, 4, 6, 3, 5])
    mxl = _mxl([(0, 1), (2, 1), (4, 1), (1, 2), (3, 2)])
    r = compare_steps(pdf, mxl, try_offsets=True)
    assert r.match == 5 and not r.suspects


def test_noise_does_not_create_suspects():
    # PDF 잉여(노이즈)는 pdf_only로만 집계, 의심 마디 아님
    pdf = _pdf([0, 9, 2, 9, 4])
    mxl = _mxl([(0, 1), (2, 1), (4, 2)])
    r = compare_steps(pdf, mxl, try_offsets=False)
    assert r.pdf_only == 2 and not r.suspects
