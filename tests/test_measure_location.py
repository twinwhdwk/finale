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

from pdf_parser import StaffZone, PageParseResult, build_measure_location_map, iter_absolute_measures  # noqa: E402


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


def test_iter_zones_with_start_measure_matches_iter_absolute_measures():
    """
    iter_zones_with_start_measure()(zone 단위 시작 번호)와
    iter_absolute_measures()(마디 단위 절대 번호)가 같은 누산 규칙을
    쓰는지 교차 검증. main._extract_pdf_data()(전자 사용)와
    build_measure_location_map()(후자 사용)가 서로 다른 마디 번호
    기준을 쓰게 되는 리팩터링 회귀를 막기 위함.
    """
    from pdf_parser import iter_zones_with_start_measure

    zone1 = _make_zone(1, top_y=100, bot_y=180, barlines=[500])           # 마디 2개
    zone2 = _make_zone(1, top_y=100, bot_y=180, barlines=[500, 1000])     # 마디 3개
    page1 = PageParseResult(pdf_path="d", page_num=0, staff_count=1, zones=[zone1])
    page2 = PageParseResult(pdf_path="d", page_num=1, staff_count=1, zones=[zone2])
    pages = [page1, page2]

    # iter_absolute_measures가 보는 각 zone의 "첫 마디(m_in_staff=1)" 절대 번호
    first_measure_from_absolute = {}
    for abs_num, _page, zone, m_in_staff in iter_absolute_measures(pages):
        if m_in_staff == 1:
            first_measure_from_absolute[id(zone)] = abs_num

    # iter_zones_with_start_measure가 주는 zone 시작 번호
    start_from_zones = {}
    for zone_start, _page, zone in iter_zones_with_start_measure(pages):
        start_from_zones[id(zone)] = zone_start

    assert first_measure_from_absolute == start_from_zones, (
        "두 이터레이터가 같은 zone에 대해 다른 절대 마디 번호를 줌 - "
        "main.py와 pdf_parser.py가 서로 다른 기준을 쓰게 될 위험"
    )


def test_extract_pdf_data_style_chord_numbering_matches_location_map():
    """
    main._extract_pdf_data()와 동일한 방식(zone.chords를 절대 마디
    번호로 변환)으로 계산한 마디 번호가, build_measure_location_map()이
    매핑한 마디 번호와 정확히 일치하는지 end-to-end 검증.
    """
    from pdf_parser import iter_zones_with_start_measure

    zone = _make_zone(1, top_y=100, bot_y=180, barlines=[500, 1000])  # 마디 3개
    zone.chords = [(1, 50, "C", 0.9), (2, 600, "G", 0.9), (3, 1100, "Am", 0.9)]
    page = PageParseResult(pdf_path="d", page_num=0, staff_count=1, zones=[zone])

    # _extract_pdf_data 방식
    pdf_chords = []
    for zone_start, _p, z in iter_zones_with_start_measure([page]):
        for m, _x, ch, _cf in z.chords:
            pdf_chords.append((zone_start + m - 1, ch))

    # build_measure_location_map 방식
    loc_map = build_measure_location_map([page], page_width=1500)

    chord_measure_nums = {num for num, _ch in pdf_chords}
    assert chord_measure_nums == set(loc_map.keys()), (
        f"코드 추출이 쓰는 마디 번호 {chord_measure_nums}와 "
        f"위치 매핑의 마디 번호 {set(loc_map.keys())}가 달라야 할 이유가 없음"
    )
    assert pdf_chords == [(1, "C"), (2, "G"), (3, "Am")]


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
        test_build_measure_location_map_bbox_y1_clamped_by_next_staff,
        test_iter_zones_with_start_measure_matches_iter_absolute_measures,
        test_extract_pdf_data_style_chord_numbering_matches_location_map,
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


def test_build_measure_location_map_bbox_y1_clamped_by_next_staff():
    """
    bbox y1이 다음 오선의 top_y - staff_h로 클램프되어야 함.
    인접 오선의 가사/코드 영역과 겹치지 않도록 보정.
    """
    # zone1: top_y=100, bot_y=180, staff_h=80
    # 클램프 없으면 y1 = 180 + int(80*2.5) = 380
    # zone2: top_y=250 → 클램프 한계 = 250 - 80 = 170
    zone1 = _make_zone(1, top_y=100, bot_y=180, barlines=[])
    zone2 = _make_zone(2, top_y=250, bot_y=330, barlines=[])
    page = PageParseResult(pdf_path="dummy.pdf", page_num=0, staff_count=2, zones=[zone1, zone2])

    loc_map = build_measure_location_map([page], page_width=1000)

    # zone1 마디의 y1은 클램프되어야 함
    y1_zone1 = loc_map[1].bbox[3]
    clamp_limit = zone2.top_y - zone1.staff_h  # 250 - 80 = 170
    assert y1_zone1 <= clamp_limit, (
        f"zone1 bbox y1={y1_zone1}이 next_top 클램프({clamp_limit})를 초과"
    )

    # zone2(마지막 오선)는 클램프 없음
    y1_zone2 = loc_map[2].bbox[3]
    expected_y1 = zone2.bot_y + int(zone2.staff_h * 2.5)
    assert y1_zone2 == expected_y1, (
        f"마지막 오선 bbox y1={y1_zone2}이 기대값({expected_y1})과 다름"
    )
