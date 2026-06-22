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
    """StaffZone에서 오선 간격을 반환한다."""
    # StaffZone은 5줄이고 top_y~bot_y가 4*staff_gap
    return max(1, (zone.bot_y - zone.top_y) // 4)


def convert_pdf_to_xml(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    time_sig: str = "4/4",
    clef_type: str = "treble",
) -> list[str]:
    """
    PDF → OpenCV 파이프라인 → .musicxml 변환.

    pdf_to_xml.convert_pdf_to_xml()과 동일한 반환 형식.

    Args:
        pdf_path:   원본 PDF 경로
        output_dir: 결과 .musicxml 저장 폴더
        dpi:        PDF 렌더링 해상도 (기본 300 - 합성 이미지 기준, 실측 후 조정 필요)
        time_sig:   박자표 (자동 감지 미구현 - 수동 지정 필요)
        clef_type:  음자리표 ("treble" | "bass", 자동 감지 미구현)

    Returns:
        생성된 .musicxml 파일 경로 목록 (페이지당 1개)
    """
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
) -> str:
    """단일 페이지를 처리해 .musicxml로 저장하고 경로를 반환한다."""
    from pdf_parser import _pdf_page_to_np, _detect_staves
    from note_recognition.staff_removal import (
        detect_staff_line_thickness, remove_staff_lines,
    )
    from note_recognition.note_detector import detect_notes, NoteDetectionResult, DetectedNote
    from note_recognition.xml_builder import save_musicxml

    img_gray = _pdf_page_to_np(pdf_path, page_num=page_num, dpi=dpi)

    # ── 오선 위치 검출 ──
    zones = _detect_staves(img_gray)
    if not zones:
        raise RuntimeError(f"페이지 {page_num + 1}: 오선을 감지하지 못했습니다")

    print(f"    {len(zones)}개 오선 시스템 감지")

    # ── 오선별 음표 검출 ──
    all_detected_notes = []
    # 첫 번째 오선에서 두께 측정 (전 페이지 동일하다고 가정)
    first_zone = zones[0]
    staff_gap_0 = _staff_gap_from_zone(first_zone)
    line_thickness = detect_staff_line_thickness(
        img_gray, [(first_zone.top_y, first_zone.bot_y)]
    )
    print(f"    오선 두께={line_thickness}px, 간격≈{staff_gap_0}px")

    for zone in zones:
        staff_gap = _staff_gap_from_zone(zone)
        result = detect_notes(
            img_gray,
            staff_top_y=zone.top_y,
            staff_bot_y=zone.bot_y,
            staff_gap=staff_gap,
            line_thickness=line_thickness,
        )
        all_detected_notes.extend(result.notes)

    if not all_detected_notes:
        raise RuntimeError(f"페이지 {page_num + 1}: 음표를 검출하지 못했습니다")

    print(f"    {len(all_detected_notes)}개 음표 검출")

    # ── 전체 페이지 NoteDetectionResult 조립 ──
    # 여러 오선의 결과를 하나로 합치기 위해 대표 오선 파라미터 사용
    # (향후: 오선별로 별도 Part 생성하는 이성부 분리로 확장 가능)
    page_result = NoteDetectionResult(
        notes=all_detected_notes,
        staff_top_y=zones[0].top_y,
        staff_bot_y=zones[0].bot_y,
        line_thickness=line_thickness,
        staff_gap=staff_gap_0,
    )

    # ── MusicXML 저장 (마디선 정보 연결) ──
    # 첫 번째 오선의 barlines를 대표 마디선으로 사용.
    # 여러 오선이 있으면 각 오선의 마디선 수가 같다고 가정 (표준 악보).
    barlines = zones[0].barlines if zones else []
    if barlines:
        print(f"    마디선 {len(barlines)}개 감지 → {len(barlines)+1}마디")

    stem = Path(pdf_path).stem
    out_path = str(output_dir / f"{stem}_p{page_num + 1:03d}_opencv.musicxml")
    save_musicxml(page_result, out_path, time_sig=time_sig,
                  clef_type=clef_type, barlines=barlines if barlines else None)
    print(f"    저장: {out_path}")
    return out_path
