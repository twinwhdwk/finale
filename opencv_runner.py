"""
OpenCV 기반 자체 OMR 파이프라인 진입점.

note_recognition/ 패키지의 5단계 파이프라인을 PDF 파일에서 실행해
MusicXML을 생성한다. pdf_to_xml.py(Audiveris), homr_runner.py(homr)와
동일한 인터페이스를 제공해 main.py에서 --engine opencv로 선택 가능.

## 파이프라인 흐름

  PDF
    ↓ PyMuPDF(fitz) - 페이지별 이미지 렌더링
  np.ndarray (그레이스케일 이미지)
    ↓ pdf_parser._detect_staves() - 오선 위치/간격 검출
  StaffZone 목록 (top_y, bot_y, staff_gap, barlines)
    ↓ staff_removal.detect_staff_line_thickness()
    ↓ note_detector.detect_notes()
    ↓ beam_splitter (내부 호출)
  NoteDetectionResult (DetectedNote 목록 + pitch 판정 준비)
    ↓ xml_builder.save_musicxml()
  .musicxml 파일

## 현재 한계 (로컬 실측 전)

- 파라미터(HEAD_FILL_THRESHOLD, _NOTEHEAD_RADIUS_RATIO 등)가 합성 이미지
  기준으로 설정돼 있음. 실제 교과서 PDF 폰트/해상도에서 튜닝 필요.
- 임시표/쉼표/점음표/코드 미처리.
- 이성부 악보는 오선 1개씩 개별 처리 후 단일 Part로 합침.
- 음자리표(treble/bass) 자동 감지 미구현 → 기본 treble 사용.
"""

import sys
from pathlib import Path


def _staff_gap_from_zone(zone) -> int:
    """StaffZone 또는 (top_y, bot_y) 튜플에서 오선 간격을 반환한다."""
    if isinstance(zone, tuple):
        top_y, bot_y = zone
    else:
        top_y, bot_y = zone.top_y, zone.bot_y
    return max(1, (bot_y - top_y) // 4)


def _group_zones_into_systems(zones: list, img_gray=None) -> list[list]:
    """오선 목록을 시스템(그랜드 스태프 포함) 단위로 묶는다.

    적응형 y-gap 임계값:
    시스템 내 gap과 시스템 간 gap이 절대값으로는 비슷할 수 있어
    (꿈꾸지 않으면 F: 2.1~2.3h vs 2.6~2.7h) 고정 임계(2.0h)가 실패함.
    → gap 목록을 정렬해 가장 큰 상대 점프 지점을 임계로 사용.
      점프가 작으면(< 15%) 모든 오선이 독립 시스템(단선율 악보).
    """
    if not zones:
        return []

    def _top(z):  return z[0] if isinstance(z, tuple) else z.top_y
    def _bot(z):  return z[1] if isinstance(z, tuple) else z.bot_y

    if len(zones) == 1:
        return [[zones[0]]]

    # 인접 오선 간 y_gap을 staff_h 배수로 정규화
    ratios = []
    for i in range(1, len(zones)):
        staff_h = max(1, _bot(zones[i-1]) - _top(zones[i-1]))
        ratios.append((_top(zones[i]) - _bot(zones[i-1])) / staff_h)

    # 적응형 임계값: 정렬된 ratio에서 최대 상대 점프 지점
    threshold = 2.0  # fallback (기존 동작)
    if len(ratios) >= 3:
        s = sorted(ratios)
        best_jump, best_i = 0.0, -1
        for i in range(len(s) - 1):
            jump = s[i+1] - s[i]
            if jump > best_jump:
                best_jump, best_i = jump, i
        if best_i >= 0 and best_jump > 0.2:
            threshold = (s[best_i] + s[best_i + 1]) / 2.0

    def _split(thr):
        gs: list[list] = [[zones[0]]]
        for i, zone in enumerate(zones[1:]):
            if ratios[i] < thr:
                gs[-1].append(zone)
            else:
                gs.append([zone])
        return gs

    groups = _split(threshold)

    # ── 균일성 검증 ──
    # 진짜 다단(2단 합창/그랜드 스태프) 악보는 페이지 내 모든 시스템이
    # 같은 단수로 균일하다. 그룹 크기가 섞이면(예: [2,1,1,...]) 임계값이
    # 우연히 걸린 것 → 전부 1단으로 flatten.
    # (태양 F: 단선율인데 첫 gap=1.99h로 [2,1,...]이 되던 문제 해결)
    sizes = {len(g) for g in groups}
    if len(sizes) > 1 and max(sizes) >= 2:
        groups = [[z] for z in zones]
    return groups


def convert_pdf_to_xml(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    time_sig: str | None = None,
    clef_type: str | None = None,
    key_sig: int | None = None,
) -> list[str]:
    """
    PDF → OpenCV 파이프라인 → .musicxml 변환.

    pdf_to_xml.convert_pdf_to_xml()과 동일한 반환 형식.

    Args:
        pdf_path:   원본 PDF 경로
        output_dir: 결과 .musicxml 저장 폴더
        dpi:        PDF 렌더링 해상도 (기본 300 - 합성 이미지 기준, 실측 후 조정 필요)
        time_sig:   박자표. None이면 config.ini [opencv] time_sig 사용 (기본 "4/4").
        clef_type:  음자리표. None이면 config.ini [opencv] clef_type 사용 (기본 "treble").
        key_sig:    조표 (샵 개수: 양수, 플랫: 음수, C장조: 0).
                    None이면 config.ini [opencv] key_sig 사용 (기본 0).

    Returns:
        생성된 .musicxml 파일 경로 목록 (페이지당 1개)
    """
    try:
        from config_loader import get_opencv_params
        params = get_opencv_params()
        if time_sig is None:
            time_sig = params["time_sig"]
        if clef_type is None:
            clef_type = params["clef_type"]
        if key_sig is None:
            key_sig = params["key_sig"]
    except ImportError:
        time_sig  = time_sig  or "4/4"
        clef_type = clef_type or "treble"
        key_sig   = key_sig   if key_sig is not None else 0
    import fitz
    from pdf_parser import _pdf_page_to_np, _detect_staves
    from note_recognition.staff_removal import (
        detect_staff_line_thickness, remove_staff_lines,
    )
    from note_recognition.note_detector import detect_notes, NoteDetectionResult, DetectedNote
    from note_recognition.note_pitcher import head_y_to_pitch
    from note_recognition.xml_builder import save_musicxml

    pdf_path = Path(pdf_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일 없음: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)
    doc.close()

    try:
        from config_loader import get_part_index
        part_index = get_part_index()
    except ImportError:
        part_index = 0

    print(f"[OpenCV OMR 시작] {pdf_path.name} ({n_pages}페이지)")
    result_paths: list[str] = []

    for page_num in range(n_pages):
        print(f"  페이지 {page_num + 1}/{n_pages} 처리 중...")
        try:
            xml_path = _process_page(
                pdf_path=str(pdf_path),
                page_num=page_num,
                output_dir=output_dir,
                dpi=dpi,
                time_sig=time_sig,
                clef_type=clef_type,
                key_sig=key_sig,
                part_index=part_index,
            )
            result_paths.append(xml_path)
        except Exception as e:
            print(f"    오류: {e}", file=sys.stderr)

    print(f"[OpenCV OMR 완료] {len(result_paths)}/{n_pages}페이지 성공")
    return result_paths


def _process_page(
    pdf_path: str,
    page_num: int,
    output_dir: Path,
    dpi: int,
    time_sig: str,
    clef_type: str,
    key_sig: int = 0,
    part_index: int = 0,
) -> str:
    """단일 페이지를 처리해 .musicxml로 저장하고 경로를 반환한다."""
    from pdf_parser import _pdf_page_to_np, _detect_staves
    from note_recognition.staff_removal import (
        detect_staff_line_thickness, remove_staff_lines,
    )
    from note_recognition.note_detector import detect_notes, NoteDetectionResult, DetectedNote
    from note_recognition.xml_builder import save_musicxml

    import cv2 as _cv2
    img_rgb = _pdf_page_to_np(pdf_path, page_num=page_num, dpi=dpi)
    img_gray = _cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2GRAY) if img_rgb.ndim == 3 else img_rgb

    # ── 오선 위치 검출 ──
    all_zones = _detect_staves(img_gray)
    if not all_zones:
        raise RuntimeError(f"페이지 {page_num + 1}: 오선을 감지하지 못했습니다")

    # 주 gap(최빈값)과 크게 다른 작은 오선 제외.
    # 교과서 페이지 상단의 참고 악보(발성 연습 등)는 본 악보보다 작게 인쇄됨
    # → MXL에 없는 음표를 대량 검출하는 노이즈원 (오 나의 태양 D: 69건).
    if len(all_zones) >= 3:
        import collections as _coll
        gaps = [(zb - zt) // 4 for zt, zb in all_zones]
        main_gap = _coll.Counter(gaps).most_common(1)[0][0]
        kept = [z for z, g in zip(all_zones, gaps)
                if main_gap * 0.85 <= g <= main_gap * 1.5]
        if len(kept) >= 2 and len(kept) < len(all_zones):
            print(f"    비정상 크기 오선 {len(all_zones) - len(kept)}개 제외 "
                  f"(주 gap={main_gap}px 기준)")
            all_zones = kept

    # 그랜드 스태프(트레블+베이스 2단 등) 시스템 감지 후 원하는 파트만 추출
    # 바라인 x좌표 공유 여부로 같은 시스템 판단
    systems = _group_zones_into_systems(all_zones, img_gray=img_gray)
    staves_per_sys = max(len(s) for s in systems)
    zones = [s[part_index] for s in systems if part_index < len(s)]
    if staves_per_sys > 1:
        print(f"    {len(all_zones)}개 오선 감지 → {len(systems)}개 시스템 "
              f"({staves_per_sys}단/시스템), 파트 {part_index} 처리")
    else:
        print(f"    {len(zones)}개 오선 시스템 감지")

    from pdf_parser import _detect_barlines
    from note_recognition.xml_builder import notes_to_score
    from music21 import stream as m21stream

    first_top, first_bot = zones[0]
    staff_gap_0 = _staff_gap_from_zone(zones[0])
    line_thickness = detect_staff_line_thickness(
        img_gray, [(first_top, first_bot)]
    )
    print(f"    오선 두께={line_thickness}px, 간격≈{staff_gap_0}px")

    # ── 헤더 자동 감지 (음자리표·조표·박자표) ──────────────────────────
    # config.ini 값을 기본값으로 쓰되, 이미지에서 더 구체적인 정보가 감지되면 덮어씀.
    from note_recognition.header_detector import detect_header
    _h = first_bot - first_top
    staff_ys_est = [first_top + round(_h * i / 4) for i in range(5)]
    try:
        hdr = detect_header(img_gray, staff_ys_est)
        detected_clef   = 'treble' if hdr.clef == 'G' else 'bass'
        detected_key    = hdr.key_sig
        detected_tsig   = hdr.time_sig
        print(f"    헤더 감지: {hdr}")
        # config.ini 값이 명시적으로 넘어온 경우(기본값과 다른 경우)에만 우선 적용
        if clef_type == 'treble':   clef_type = detected_clef
        if key_sig   == 0:          key_sig   = detected_key
        if time_sig  == '4/4':      time_sig  = detected_tsig
    except Exception as e:
        print(f"    [경고] 헤더 자동 감지 실패 ({e}), config.ini 값 사용")

    # ── 오선별 독립 처리 (각 오선 = 악보의 연속 구간) ──
    # 오선마다 바라인을 독립적으로 감지하고 notes_to_score로 마디를 생성한 뒤
    # 마디 번호를 이어붙여 전체 페이지를 하나의 Part로 조립한다.
    combined_part = m21stream.Part(id="Part 1")
    measure_offset = 0
    total_notes = 0
    total_arcs = 0

    for zi, zone in enumerate(zones):
        top_y, bot_y = zone
        staff_gap = _staff_gap_from_zone(zone)
        zone_barlines = _detect_barlines(img_gray, top_y, bot_y)
        if zone_barlines:
            x_start = max(0, zone_barlines[0] - staff_gap * 3)
        else:
            x_start = img_gray.shape[1] // 10

        # 다음 오선 top_y (ROI 하단 클램프용 — 가사/코드 기호 제외)
        next_top = zones[zi + 1][0] if zi + 1 < len(zones) else None

        result = detect_notes(
            img_gray,
            staff_top_y=top_y,
            staff_bot_y=bot_y,
            staff_gap=staff_gap,
            line_thickness=line_thickness,
            x_start=x_start,
            next_staff_top_y=next_top,
        )
        if not result.notes:
            continue
        total_notes += len(result.notes)
        total_arcs += len(result.arcs)

        # 이 오선의 마디 생성 (마디 번호는 1부터)
        zone_score = notes_to_score(
            result,
            time_sig=time_sig,
            clef_type=clef_type,
            part_name="Part 1",
            barlines=zone_barlines if zone_barlines else None,
            key_sig=key_sig,
        )
        zone_part = zone_score.parts[0] if zone_score.parts else None
        if zone_part is None:
            continue

        zone_measures = list(zone_part.getElementsByClass(m21stream.Measure))
        for m in zone_measures:
            m.number = m.number + measure_offset
            combined_part.append(m)
        measure_offset += len(zone_measures)

    if total_notes == 0:
        raise RuntimeError(f"페이지 {page_num + 1}: 음표를 검출하지 못했습니다")

    print(f"    {total_notes}개 음표 / {total_arcs}개 호(arc) / {measure_offset}마디")

    # ── MusicXML 저장 ──
    final_score = m21stream.Score()
    final_score.append(combined_part)
    stem = Path(pdf_path).stem
    out_path = str(output_dir / f"{stem}_p{page_num + 1:03d}_opencv.musicxml")
    final_score.write("musicxml", fp=out_path)
    print(f"    저장: {out_path}")
    return out_path
