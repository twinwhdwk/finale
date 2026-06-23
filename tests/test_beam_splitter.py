"""
note_recognition.beam_splitter 직접 단위 테스트.

note_detector 통합 테스트에서만 간접 검증되던 내부 알고리즘을
numpy 배열만으로 직접 테스트한다 (합성 이미지 불필요).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from note_recognition.beam_splitter import (  # noqa: E402
    is_beam_component,
    _vertical_projection,
    _find_stem_peaks,
    _find_split_positions,
    split_beam_component,
)


# ── is_beam_component ─────────────────────────────────────────────────

def test_wide_component_identified_as_beam():
    """단일 음표보다 4배 이상 넓은 컴포넌트는 빔으로 판별되어야 함."""
    notehead_radius = 11
    # 단일 음표 최대 폭 ≈ notehead_radius * 4 = 44px
    # 빔으로 묶인 2음표 너비 ≈ 80~120px
    assert is_beam_component(w=90, h=80, notehead_radius=notehead_radius) is True


def test_single_note_not_identified_as_beam():
    """단일 음표 크기(w < notehead_radius*4)는 빔 아님."""
    notehead_radius = 11
    assert is_beam_component(w=30, h=80, notehead_radius=notehead_radius) is False


def test_wide_but_short_not_beam():
    """가로는 넓어도 세로가 너무 낮으면(기둥 없음) 빔 아님."""
    notehead_radius = 11
    # h < notehead_radius * 2 이면 기둥이 없는 온음표 같은 것
    assert is_beam_component(w=90, h=10, notehead_radius=notehead_radius) is False


# ── _vertical_projection ──────────────────────────────────────────────

def test_vertical_projection_counts_nonzero_per_column():
    """각 x 위치에서 0이 아닌 픽셀 수를 반환해야 함."""
    region = np.array([
        [0,   255, 0],
        [255, 255, 0],
        [0,   255, 255],
    ], dtype=np.uint8)
    proj = _vertical_projection(region)
    assert list(proj) == [1, 3, 1]


def test_vertical_projection_empty_region():
    """빈 영역(모두 0)은 전체 0 배열 반환."""
    region = np.zeros((10, 10), dtype=np.uint8)
    proj = _vertical_projection(region)
    assert proj.sum() == 0


# ── _find_stem_peaks ──────────────────────────────────────────────────

def test_find_stem_peaks_two_stems():
    """두 기둥 위치에서 피크가 2개 나와야 함."""
    # x=5와 x=15에서 피크, 사이와 가장자리는 낮음
    vproj = np.array([5, 5, 5, 5, 5, 80, 80, 80, 5, 5, 5, 5, 5, 5, 5, 80, 80, 80, 5, 5], dtype=np.int32)
    peaks = _find_stem_peaks(vproj)
    assert len(peaks) == 2, f"피크 2개 기대, 실제: {peaks}"
    assert peaks[0] < peaks[1]


def test_find_stem_peaks_single_stem():
    """기둥 1개면 피크 1개."""
    vproj = np.array([2, 2, 2, 80, 80, 80, 2, 2, 2], dtype=np.int32)
    peaks = _find_stem_peaks(vproj)
    assert len(peaks) == 1


def test_find_stem_peaks_empty():
    """모두 0이면 피크 없음."""
    vproj = np.zeros(20, dtype=np.int32)
    peaks = _find_stem_peaks(vproj)
    assert len(peaks) == 0


def test_find_stem_peaks_threshold():
    """최댓값의 30% 미만은 피크로 잡히지 않아야 함."""
    # 최댓값 100, 30% = 30. 값이 25인 것은 피크 아님
    vproj = np.array([25, 25, 100, 100, 25, 25], dtype=np.int32)
    peaks = _find_stem_peaks(vproj)
    assert len(peaks) == 1  # 100짜리만 피크


# ── _find_split_positions ─────────────────────────────────────────────

def test_find_split_positions_between_two_peaks():
    """두 피크 사이 최솟값 위치가 분할점이어야 함."""
    # peak at 2, peak at 8, 사이 최솟값은 index 5
    vproj = np.array([50, 80, 80, 50, 10, 5, 10, 50, 80, 80, 50], dtype=np.int32)
    peaks = [2, 8]
    splits = _find_split_positions(vproj, peaks)
    assert len(splits) == 1
    # 최솟값 위치(index 5)가 분할점
    assert splits[0] == 5


def test_find_split_positions_no_peaks():
    """피크가 1개 이하면 분할점 없음."""
    vproj = np.array([10, 80, 80, 10], dtype=np.int32)
    assert _find_split_positions(vproj, []) == []
    assert _find_split_positions(vproj, [1]) == []


# ── split_beam_component ──────────────────────────────────────────────

def _make_beam_binary(stem1_x: int, stem2_x: int, width: int = 150, height: int = 80) -> np.ndarray:
    """두 기둥이 빔으로 연결된 합성 이진 이미지를 생성."""
    img = np.zeros((height, width), dtype=np.uint8)
    # 기둥 1 (세로선)
    img[:, stem1_x:stem1_x + 3] = 255
    # 기둥 2 (세로선)
    img[:, stem2_x:stem2_x + 3] = 255
    # 빔 (가로선, 위쪽)
    img[0:5, stem1_x:stem2_x + 3] = 255
    # 음표머리 근사 (타원 대신 사각형으로)
    img[60:75, stem1_x - 5:stem1_x + 8] = 255
    img[60:75, stem2_x - 5:stem2_x + 8] = 255
    return img


def test_split_beam_component_splits_two_stems():
    """두 기둥이 있는 빔 컴포넌트가 2개로 분리되어야 함."""
    binary = _make_beam_binary(stem1_x=20, stem2_x=110, width=150, height=80)
    bbox = (0, 0, 150, 80)
    result = split_beam_component(binary, bbox, notehead_radius=11)
    assert result is not None, "분할 실패 (None 반환)"
    assert len(result) == 2, f"2개 서브bbox 기대, 실제: {len(result)}"


def test_split_beam_returns_stem_up_hint():
    """split_beam_component는 stem_up 힌트를 반환해야 함."""
    binary = _make_beam_binary(stem1_x=20, stem2_x=110, width=150, height=80)
    bbox = (0, 0, 150, 80)
    result = split_beam_component(binary, bbox, notehead_radius=11)
    assert result is not None
    for item in result:
        assert "stem_up" in item
        assert "stem_x" in item
        assert "bbox" in item


def test_split_single_stem_returns_none():
    """기둥이 1개뿐이면 분할 불가로 None 반환해야 함."""
    # 기둥 1개만 있는 단순한 컴포넌트
    binary = np.zeros((80, 30), dtype=np.uint8)
    binary[:, 13:16] = 255  # 단일 기둥
    binary[60:75, 8:22] = 255  # 음표머리
    bbox = (0, 0, 30, 80)
    result = split_beam_component(binary, bbox, notehead_radius=11)
    assert result is None, f"단일 기둥에서 None 기대, 실제: {result}"


def test_split_positions_are_between_stems():
    """분할선이 두 기둥 사이에 위치해야 함."""
    stem1_x, stem2_x = 20, 110
    binary = _make_beam_binary(stem1_x=stem1_x, stem2_x=stem2_x, width=150, height=80)
    bbox = (0, 0, 150, 80)
    result = split_beam_component(binary, bbox, notehead_radius=11)
    assert result is not None
    # 첫 번째 서브bbox x+w ≤ 두 번째 서브bbox x (겹치지 않아야 함)
    bx1, _, bw1, _ = result[0]["bbox"]
    bx2, _, _, _ = result[1]["bbox"]
    assert bx1 + bw1 <= bx2 + 5, (
        f"서브bbox 1({bx1}+{bw1}={bx1+bw1})이 서브bbox 2({bx2})와 겹침"
    )
