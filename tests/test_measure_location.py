"""
pdf_parser의 마디 위치(bbox) 계산 로직 단위 테스트.

HTML 리포트에서 오류 클릭 시 PDF 원본 위치를 하이라이트하는 기능
(CLAUDE.md TODO)의 기반이 되는 좌표 변환을 검증한다. OCR/이미지
처리 없이 StaffZone/PageParseResult를 직접 생성해 순수 좌표 계산
로직만 테스트한다 (cv2/easyocr 등은 import되지만 실제로 호출하지 않음).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf_parser import StaffZone, PageParseResult, build_measure_location_map  # noqa: E402


def _make_zone(index, top_y, bot_y, barlines):
    staff_h = bot_y - top_y
    return StaffZone(index=index, top_y=top_y, bot_y=bot_y, staff_h=staff_h, barlines=barlines)


def test_measure_to_x_range_first_measure_starts_at_zero():
    """첫 마디는 barlines[0] 이전 구간이므로 x_start는 항상 0이어야 함"""
    zone = _make_zone(1, top_y=100, bot_y=180, barlines=[500, 1000, 1500])
    x0, x1 = zone.measure_to_x_range(1, staff_width=2000)
    assert x0 == 0
    assert x1 == 500


def test_measure_to_x_range_last_measure_ends_at_staff_width():
    """마지막 마디는 barlines 개수+1번째이므로 x_end는 staff_width여야 함"""
    zone = _make_zone(1, top_y=100, bot_y=180, barlines=[500, 1000, 1500])
    # barlines 3개 -> 마디 4개 (measure_count == 4)
    assert zone.measure_count == 4
    x0, x1 = zone.measure_to_x_range(4, staff_width=2000)
    assert x0 == 1500
    assert x1 == 2000


def test_measure_to_x_range_middle_measures():
    """중간 마디는 barlines[i-2]에서 barlines[i-1] 사이여야 함"""
    zone = _make_zone(1, top_y=100, bot_y=180, barlines=[500, 1000, 1500])
    x0, x1 = zone.measure_to_x_range(2, staff_width=2000)
    assert (x0, x1) == (500, 1000)
    x0, x1 = zone.measure_to_x_range(3, staff_width=2000)
    assert (x0, x1) == (1000, 1500)


def test_measure_to_x_range_matches_x_to_measure_inverse():
    """x_to_measure()의 역함수로서 일관성이 있어야 함: 범위 내 임의의 x를
    x_to_measure로 넣으면 같은 마디 번호가 나와야 함"""
    zone = _make_zone(1, top_y=100, bot_y=180, barlines=[500, 1000, 1500])
    staff_width = 2000

    for m in range(1, zone.measure_count + 1):
        x0, x1 = zone.measure_to_x_range(m, staff_width)
        mid_x = (x0 + x1) // 2
        assert zone.x_to_measure(mid_x) == m, (
            f"마디 {m}의 중간 x={mid_x}가 x_to_measure에서 {zone.x_to_measure(mid_x)}로 나옴"
        )


def test_measure_bbox_includes_chord_and_lyric_zones():
    """bbox의 y범위가 코드(오선 위)와 가사(오선 아래) 영역을 포함해야 함"""
    zone = _make_zone(1, top_y=200, bot_y=280, barlines=[500])  # staff_h = 80
    x0, y0, x1, y1 = zone.measure_bbox(1, staff_width=1000)

    assert y0 == 200 - 80  # top_y - staff_h (코드 영역 포함)
    assert y1 == 280 + int(80 * 2.5)  # bot_y + staff_h*2.5 (가사 영역 포함)
    assert x0 == 0
    assert x1 == 500


def test_measure_bbox_y0_never_negative():
    """오선이 페이지 맨 위에 가까우면 y0가 음수가 되지 않도록 클램프해야 함"""
    zone = _make_zone(1, top_y=30, bot_y=100, barlines=[])  # staff_h=70, top_y-staff_h = -40
    x0, y0, x1, y1 = zone.measure_bbox(1, staff_width=1000)
    assert y0 == 0, "음수 y좌표는 0으로 클램프되어야 함"


def test_build_measure_location_map_single_page_single_staff():
    """단일 페이지, 단일 오선: 절대 마디 번호가 1부터 순차 부여되어야 함"""
    zone = _make_zone(1, top_y=100, bot_y=180, barlines=[500, 1000])  # 마디 3개
    page = PageParseResult(pdf_path="dummy.pdf", page_num=0, staff_count=1, zones=[zone])

    loc_map = build_measure_location_map([page], page_width=1500)

    assert set(loc_map.keys()) == {1, 2, 3}
    assert loc_map[1].page_num == 0
    assert loc_map[1].staff_index == 1
    assert loc_map[2].bbox[0] == 500  # 두 번째 마디 x_start


def test_build_measure_location_map_multi_page_continues_numbering():
    """여러 페이지에 걸쳐 절대 마디 번호가 끊기지 않고 이어져야 함"""
    zone1 = _make_zone(1, top_y=100, bot_y=180, barlines=[500])  # 마디 2개 (1,2)
    page1 = PageParseResult(pdf_path="dummy.pdf", page_num=0, staff_count=1, zones=[zone1])

    zone2 = _make_zone(1, top_y=100, bot_y=180, barlines=[500, 1000])  # 마디 3개 (3,4,5)
    page2 = PageParseResult(pdf_path="dummy.pdf", page_num=1, staff_count=1, zones=[zone2])

    loc_map = build_measure_location_map([page1, page2], page_width=1500)

    assert set(loc_map.keys()) == {1, 2, 3, 4, 5}
    assert loc_map[1].page_num == 0
    assert loc_map[2].page_num == 0
    assert loc_map[3].page_num == 1, "두 번째 페이지 첫 마디는 절대 번호 3이어야 함"
    assert loc_map[5].page_num == 1


def test_build_measure_location_map_multi_staff_per_page():
    """한 페이지 안에 여러 오선이 있으면 오선 순서대로 마디 번호가 이어져야 함"""
    zone1 = _make_zone(1, top_y=100, bot_y=180, barlines=[])     # 마디 1개 (1)
    zone2 = _make_zone(2, top_y=400, bot_y=480, barlines=[500])  # 마디 2개 (2,3)
    page = PageParseResult(pdf_path="dummy.pdf", page_num=0, staff_count=2, zones=[zone1, zone2])

    loc_map = build_measure_location_map([page], page_width=1500)

    assert set(loc_map.keys()) == {1, 2, 3}
    assert loc_map[1].staff_index == 1
    assert loc_map[2].staff_index == 2
    assert loc_map[3].staff_index == 2


if __name__ == "__main__":
    tests = [
        test_measure_to_x_range_first_measure_starts_at_zero,
        test_measure_to_x_range_last_measure_ends_at_staff_width,
        test_measure_to_x_range_middle_measures,
        test_measure_to_x_range_matches_x_to_measure_inverse,
        test_measure_bbox_includes_chord_and_lyric_zones,
        test_measure_bbox_y0_never_negative,
        test_build_measure_location_map_single_page_single_staff,
        test_build_measure_location_map_multi_page_continues_numbering,
        test_build_measure_location_map_multi_staff_per_page,
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
    sys.exit(1 if failed else 0)
