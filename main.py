"""
악보 검수 자동화 메인 실행 파일

사용법:
  1. config.ini 설정 후 한 번에 실행 (권장):
       python main.py run

  2. XML 두 파일 직접 비교:
       python main.py compare --pdf pdf_extracted.xml --orig finale_original.xml

  3. PDF → XML 변환 후 비교:
       python main.py full --pdf score.pdf --orig finale_original.xml

  4. 폴더 일괄 변환만:
       python main.py batch-convert

  5. 현재 config 확인:
       python main.py config
"""

import argparse
import sys
import io
from pathlib import Path

# Windows 콘솔 UTF-8 강제 설정
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import config_loader
from xml_comparator import compare
from report_generator import print_console, save_html


def _build_measure_map_and_save_images(
    pdf_path: str,
    report_dir: Path,
    stem: str,
    dpi: int = 150,
) -> dict | None:
    """
    PDF 각 페이지를 이미지로 저장하고 마디 위치 매핑을 반환한다.

    HTML 리포트의 '오류 클릭 → PDF 위치 하이라이트' 기능을 위한 준비 작업.
    이미지는 report_dir/images/page_{n}.png 형태로 저장된다.
    페이지 폭(px)은 build_measure_location_map()에 필요하므로 첫 페이지 렌더링
    결과에서 직접 측정한다.

    Args:
        pdf_path:   원본 PDF 경로
        report_dir: 리포트 저장 폴더 (images/ 하위 폴더가 여기에 생성됨)
        stem:       파일 스템 (페이지 이미지 파일명에 쓰이지 않고 현재 로그용)
        dpi:        페이지 이미지 렌더링 해상도.
                    150dpi면 A4 기준 ≈1240×1754px. 파일 크기와 선명도의 균형.

    Returns:
        {절대마디번호: MeasureLocation} 또는 실패 시 None
    """
    try:
        import fitz  # PyMuPDF
        import cv2
        import numpy as np
        from pdf_parser import parse_all_pages, build_measure_location_map

        img_dir = report_dir / "images"
        img_dir.mkdir(exist_ok=True)

        doc = fitz.open(pdf_path)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        page_width = None

        print(f"  페이지 이미지 저장 중 ({len(doc)}페이지, {dpi}dpi)...")
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img_path = str(img_dir / f"page_{i + 1}.png")
            pix.save(img_path)
            if page_width is None:
                page_width = pix.width
        doc.close()

        if page_width is None:
            return None

        print(f"  마디 위치 매핑 계산 중...")
        pages = parse_all_pages(pdf_path, dpi=dpi)
        loc_map = build_measure_location_map(pages, page_width)
        print(f"  총 {len(loc_map)}개 마디 위치 매핑 완료")
        return loc_map

    except Exception as e:
        print(f"  [경고] 마디 위치 매핑 실패 ({e}) — 하이라이트 기능 비활성화")
        return None


def _convert_and_resolve_single_xml(pdf_path: str, conv_dir: str, engine: str) -> str | None:
    """
    PDF를 지정 엔진으로 변환하고, xml_comparator.compare()가 바로 쓸 수 있는
    "단일 XML 경로" 하나를 반환합니다.

    audiveris: 보통 결과가 1개라 그대로 반환.
    homr:      페이지별로 여러 .musicxml이 나오므로, 2개 이상이면
               homr_runner.merge_page_musicxmls()로 병합한 합본을 반환.
               dpi/gpu는 config.ini [homr] 섹션 값을 사용.
    """
    if engine == "homr":
        from homr_runner import convert_pdf_to_xml
        dpi = config_loader.get_homr_dpi()
        gpu = config_loader.get_homr_gpu()
        xml_paths = convert_pdf_to_xml(pdf_path, conv_dir, dpi=dpi, gpu=gpu)
    elif engine == "opencv":
        from opencv_runner import convert_pdf_to_xml
        xml_paths = convert_pdf_to_xml(pdf_path, conv_dir)
    else:
        from pdf_to_xml import convert_pdf_to_xml
        xml_paths = convert_pdf_to_xml(pdf_path, conv_dir)

    if not xml_paths:
        return None
    if len(xml_paths) == 1:
        return xml_paths[0]

    if engine == "homr":
        from homr_runner import merge_page_musicxmls
        merged_path = str(Path(conv_dir) / (Path(pdf_path).stem + "_merged.musicxml"))
        print(f"  homr 결과 {len(xml_paths)}페이지를 병합합니다 -> {merged_path}")
        return merge_page_musicxmls(xml_paths, merged_path)

    return xml_paths[0]


def _extract_pdf_data(pdf_path: str):
    """
    pdf_parser로 코드 기호·가사를 추출합니다.

    마디선 감지 결과를 이용해 절대 마디 번호를 정확히 계산합니다.
    실패 시 (None, None) 반환.
    """
    try:
        from pdf_parser import parse_all_pages, iter_zones_with_start_measure
        print("  PDF OCR 추출 중 (코드 기호 / 가사)...")
        pages = parse_all_pages(pdf_path)

        pdf_chords: list[tuple[int, str]] = []
        pdf_lyrics: list[tuple[int, str]] = []
        total_measures = 0

        # 절대 마디 번호 누산 규칙은 pdf_parser.iter_zones_with_start_measure()
        # (= build_measure_location_map()이 쓰는 것과 동일한 단일 진실 공급원)를
        # 공유해, 코드/가사 추출과 마디 위치 매핑이 항상 같은 번호 기준을 쓰도록 함.
        for zone_start, _page, zone in iter_zones_with_start_measure(pages):
            for m_in_staff, _x, ch, _cf in zone.chords:
                pdf_chords.append((zone_start + m_in_staff - 1, ch))
            for m_in_staff, text in zone.lyrics:
                pdf_lyrics.append((zone_start + m_in_staff - 1, text))
            total_measures = zone_start + zone.measure_count - 1

        print(
            f"    코드 기호 {len(pdf_chords)}건 / 가사 {len(pdf_lyrics)}건 추출 "
            f"(총 {total_measures}마디 분량 처리)"
        )
        return pdf_chords, pdf_lyrics
    except Exception as e:
        print(f"  PDF OCR 건너뜀 ({e})")
        return None, None


# ── run (config.ini 기반 전체 실행) ──────────────────────────────────

def cmd_run(args):
    """config.ini에 지정된 폴더를 기준으로 변환 + 비교를 일괄 실행합니다."""
    from pdf_to_xml import convert_pdf_to_xml

    paths = config_loader.get_paths()
    part  = config_loader.get_part_index()

    pdf_dir           = Path(paths["pdf_dir"])
    xml_dir           = Path(paths["xml_dir"])
    converted_xml_dir = Path(paths["converted_xml_dir"])
    report_dir        = Path(paths["report_dir"])

    _check_dir(pdf_dir, "pdf_dir")
    _check_dir(xml_dir, "xml_dir")
    converted_xml_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"PDF 파일이 없습니다: {pdf_dir}")
        sys.exit(1)

    print(f"\n총 {len(pdf_files)}개 PDF 처리 시작\n")

    for pdf in pdf_files:
        stem = pdf.stem
        print(f"{'─'*50}")
        print(f"[{stem}]")

        # 1. PDF → XML 변환
        print("  1단계: PDF → XML 변환")
        try:
            xml_paths = convert_pdf_to_xml(str(pdf), str(converted_xml_dir))
        except RuntimeError as e:
            print(f"  변환 실패: {e}")
            continue

        if not xml_paths:
            print("  변환된 XML을 찾을 수 없습니다. 건너뜁니다.")
            continue

        pdf_xml = xml_paths[0]
        print(f"  변환 완료: {pdf_xml}")

        # 2. 대응하는 원본 XML 찾기 (파일명이 같다고 가정)
        orig_xml = _find_orig_xml(xml_dir, stem)
        if orig_xml is None:
            print(f"  원본 XML 없음: {xml_dir / stem}.xml (또는 .mxl) - 건너뜁니다.")
            continue
        print(f"  원본 XML: {orig_xml}")

        # 3. 비교 (PDF OCR 코드/가사 포함)
        print("  2단계: XML 비교")
        pdf_chords, pdf_lyrics = _extract_pdf_data(str(pdf))
        result = compare(pdf_xml, str(orig_xml), part_index=part,
                         pdf_chords=pdf_chords, pdf_lyrics=pdf_lyrics)
        print_console(result)

        # 4. HTML 리포트 저장
        html_path = report_dir / f"{stem}_report.html"
        save_html(result, str(html_path))


# ── compare ──────────────────────────────────────────────────────────

def cmd_compare(args):
    paths  = config_loader.get_paths()
    part   = args.part if args.part is not None else config_loader.get_part_index()
    report_dir = Path(paths["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    # OCR은 원본 PDF가 명시된 경우에만 실행
    pdf_source = getattr(args, "pdf_source", None)
    if pdf_source:
        pdf_chords, pdf_lyrics = _extract_pdf_data(pdf_source)
    else:
        pdf_chords, pdf_lyrics = None, None

    voice = getattr(args, "voice", None)
    result = compare(args.pdf, args.orig, part_index=part, voice_index=voice,
                     pdf_chords=pdf_chords, pdf_lyrics=pdf_lyrics)
    print_console(result)

    html_path = args.html or str(report_dir / (Path(args.pdf).stem + "_report.html"))
    save_html(result, html_path)


# ── full (단일 PDF 변환 + 비교) ───────────────────────────────────────

def cmd_full(args):
    paths         = config_loader.get_paths()
    part          = args.part if args.part is not None else config_loader.get_part_index()
    conv_dir      = args.output_dir or paths["converted_xml_dir"]
    report_dir    = Path(paths["report_dir"])

    Path(conv_dir).mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1단계] PDF → XML 변환 (엔진: {args.engine})")
    pdf_xml = _convert_and_resolve_single_xml(args.pdf, conv_dir, args.engine)

    if not pdf_xml:
        print("변환된 XML 파일을 찾을 수 없습니다.")
        sys.exit(1)

    print(f"  변환 결과: {pdf_xml}\n")

    print("[2단계] XML 비교")
    pdf_chords, pdf_lyrics = _extract_pdf_data(args.pdf)
    result = compare(pdf_xml, args.orig, part_index=part,
                     pdf_chords=pdf_chords, pdf_lyrics=pdf_lyrics)
    print_console(result)

    suffix = f"_{args.engine}" if args.engine != "audiveris" else ""
    html_path = args.html or str(report_dir / (Path(args.pdf).stem + suffix + "_report.html"))

    # 마디 위치 매핑 + 페이지 이미지 저장 (PDF 하이라이트 기능)
    measure_location_map = _build_measure_map_and_save_images(
        args.pdf, report_dir, stem=Path(args.pdf).stem
    )
    save_html(result, html_path, measure_location_map=measure_location_map)


# ── convert (단일 PDF) ────────────────────────────────────────────────

def cmd_convert(args):
    """PDF 파일 하나를 XML로 변환합니다."""
    from pdf_to_xml import convert_pdf_to_xml

    paths   = config_loader.get_paths()
    out_dir = args.output_dir or paths["converted_xml_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n[변환] {args.pdf}")
    xml_paths = convert_pdf_to_xml(args.pdf, out_dir)

    if xml_paths:
        print("\n변환 성공 - 저장 위치:")
        for p in xml_paths:
            print(f"  {p}")
    else:
        print("\n변환된 XML 파일을 찾지 못했습니다. Audiveris 로그를 확인하세요.")


# ── batch-convert ─────────────────────────────────────────────────────

def cmd_batch_convert(args):
    from pdf_to_xml import batch_convert

    paths = config_loader.get_paths()
    pdf_dir   = args.pdf_dir   or paths["pdf_dir"]
    out_dir   = args.output_dir or paths["converted_xml_dir"]

    _check_dir(Path(pdf_dir), "pdf_dir")
    batch_convert(pdf_dir, out_dir)


# ── visual (단 단위 시각 비교) ────────────────────────────────────────

def cmd_visual(args):
    """교과서 PDF + Finale(XML 또는 PDF)를 단(System) 단위로 시각 비교합니다."""
    from system_slicer import (
        pair_systems,
        save_slices_to_disk,
        slice_pdf_to_systems,
        slice_pngs_to_systems,
    )
    from visual_report import save_visual_html

    paths   = config_loader.get_paths()
    dpi     = args.dpi
    out_dir = Path(args.output_dir or paths["report_dir"]) / "visual"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.pdf).stem

    # 1. Finale 소스 처리: XML(MuseScore 변환) 또는 PDF 직접 사용
    if args.xml:
        from xml_to_systems import xml_to_systems
        print(f"\n[2/3] 단(System) 슬라이싱")
    else:
        print(f"\n[1/3] Finale PDF 직접 사용")
        print(f"\n[2/3] 단(System) 슬라이싱")
        finale = slice_pdf_to_systems(args.finale_pdf, dpi=dpi)
        print(f"  Finale PDF: {finale.total_systems}단, 경고 {len(finale.warnings)}건")

    print(f"  교과서 PDF: {args.pdf}")
    textbook = slice_pdf_to_systems(args.pdf, dpi=dpi)
    mps = textbook.measures_per_system
    print(f"  교과서: {textbook.total_systems}단, 단별 마디 수: {mps}")

    if args.xml:
        print(f"\n[1/3] Finale XML → 단별 SVG 렌더링 (PDF 레이아웃 강제)")
        try:
            finale = xml_to_systems(args.xml, measures_per_system=mps)
        except RuntimeError as e:
            print(f"  오류: {e}")
            sys.exit(1)

    if args.dump_slices:
        save_slices_to_disk(textbook, out_dir / f"{stem}_tb_slices")
        save_slices_to_disk(finale,   out_dir / f"{stem}_fn_slices")

    # 3. HTML 생성
    print(f"\n[3/3] HTML 뷰어 생성")
    pairs     = pair_systems(textbook, finale)
    html_path = out_dir / f"{stem}_visual.html"
    save_visual_html(pairs, textbook, finale, str(html_path))


# ── compare-engines (Audiveris vs homr 비교) ───────────────────────────

def cmd_compare_engines(args):
    """동일 PDF를 Audiveris와 homr 양쪽으로 변환해 원본과 비교, 결과를 나란히 출력합니다.

    이음줄(tie_suspect)을 포함한 트랙별 오류 건수를 엔진별로 대조하기 위한 용도.
    """
    paths      = config_loader.get_paths()
    part       = args.part if args.part is not None else config_loader.get_part_index()
    report_dir = Path(paths["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    conv_dir = Path(args.output_dir or paths["converted_xml_dir"])
    conv_dir.mkdir(parents=True, exist_ok=True)

    pdf_chords, pdf_lyrics = _extract_pdf_data(args.pdf)

    results = {}
    for engine in ("audiveris", "homr", "opencv"):
        print(f"\n{'='*60}\n[엔진: {engine}]\n{'='*60}")
        try:
            pdf_xml = _convert_and_resolve_single_xml(args.pdf, str(conv_dir / engine), engine)
        except RuntimeError as e:
            print(f"  변환 실패 ({engine}): {e}")
            results[engine] = None
            continue

        if not pdf_xml:
            print(f"  변환된 XML 없음 ({engine})")
            results[engine] = None
            continue

        result = compare(pdf_xml, args.orig, part_index=part,
                         pdf_chords=pdf_chords, pdf_lyrics=pdf_lyrics)
        print_console(result)
        results[engine] = result

        html_path = report_dir / f"{Path(args.pdf).stem}_{engine}_report.html"
        save_html(result, str(html_path))

    # ── 비교 요약 ──
    print(f"\n{'='*72}\n[엔진 비교 요약]\n{'='*72}")
    header = f"{'항목':<20}{'audiveris':>14}{'homr':>14}{'opencv':>14}"
    print(header)
    print("-" * len(header))

    def _row(label, fn):
        a = fn(results["audiveris"]) if results.get("audiveris") else "-"
        h = fn(results["homr"])      if results.get("homr")      else "-"
        o = fn(results["opencv"])    if results.get("opencv")    else "-"
        print(f"{label:<20}{str(a):>14}{str(h):>14}{str(o):>14}")

    _row("총 불일치", lambda r: len(r.discrepancies))
    _row("음표 오류",   lambda r: r.note_errors)
    _row("  타이 누락", lambda r: r.tie_missing_count)
    _row("  타이 오인식", lambda r: r.tie_extra_count)
    _row("  타이 의심", lambda r: r.tie_suspect_count)
    _row("  OMR 누락",  lambda r: r.missing_count)
    _row("  OMR 노이즈", lambda r: r.noise_count)
    _row("코드 오류",   lambda r: r.chord_errors)
    _row("가사 오류",   lambda r: r.lyric_errors)

    if results.get("homr") is not None:
        print(
            "\n[참고] homr(현재 0.6.2)는 슬러/타이 인식 결과를 MusicXML에 "
            "출력하지 않으므로 '타이' 항목은 비교 대상 외.\n"
            "음높이/리듬 정확도 비교 용도로만 활용하세요."
        )
    if results.get("opencv") is not None:
        print(
            "\n[참고] opencv 엔진은 자체 파이프라인(note_recognition/)으로 "
            "합성 이미지 기준으로 파라미터가 설정돼 있습니다.\n"
            "실제 교과서 PDF에서 정확도 차이가 클 수 있으며, 로컬 실측 후 "
            "임계값 튜닝이 권장됩니다."
        )


# ── config 확인 ───────────────────────────────────────────────────────

def cmd_config(_args):
    config_loader.print_config()


# ── 헬퍼 ──────────────────────────────────────────────────────────────

def _check_dir(path: Path, name: str):
    if not path.exists():
        print(f"폴더가 존재하지 않습니다 ({name}): {path}")
        print("config.ini 경로를 확인하세요.")
        sys.exit(1)


def _find_orig_xml(xml_dir: Path, stem: str):
    for ext in (".xml", ".mxl", ".musicxml"):
        candidate = xml_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


# ── CLI 정의 ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PDF 악보 ↔ 피날레 XML 검수 자동화 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="config.ini에서 폴더 경로를 설정한 뒤 'python main.py run'으로 실행하세요.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="config.ini 설정으로 전체 일괄 실행 (변환 + 비교 + 리포트)")
    p_run.set_defaults(func=cmd_run)

    # compare
    p_compare = sub.add_parser("compare", help="XML 두 파일 직접 비교")
    p_compare.add_argument("--pdf",        required=True, help="PDF OMR 변환 XML 경로")
    p_compare.add_argument("--orig",       required=True, help="피날레 원본 XML 경로")
    p_compare.add_argument("--pdf-source", help="OCR용 원본 PDF 경로 (코드/가사 추출, 생략 시 OCR 건너뜀)")
    p_compare.add_argument("--part",  type=int, default=None, help="파트 인덱스 (기본값: config.ini)")
    p_compare.add_argument("--voice", type=int, default=None, help="성부 인덱스 (0=소프라노, 1=알토, 생략=전체)")
    p_compare.add_argument("--html", help="HTML 리포트 저장 경로 (기본값: config.ini report_dir)")
    p_compare.set_defaults(func=cmd_compare)

    # full
    p_full = sub.add_parser("full", help="단일 PDF 변환 + 비교")
    p_full.add_argument("--pdf",        required=True, help="원본 PDF 경로")
    p_full.add_argument("--orig",       required=True, help="피날레 원본 XML 경로")
    p_full.add_argument("--output-dir", help="변환 XML 저장 폴더 (기본값: config.ini converted_xml_dir)")
    p_full.add_argument("--part", type=int, default=None, help="파트 인덱스 (기본값: config.ini)")
    p_full.add_argument("--html", help="HTML 리포트 저장 경로 (기본값: config.ini report_dir)")
    p_full.add_argument("--engine", choices=["audiveris", "homr", "opencv"], default="audiveris",
                        help="OMR 변환 엔진 선택 (기본값: audiveris)")
    p_full.set_defaults(func=cmd_full)

    # convert (단일)
    p_conv = sub.add_parser("convert", help="PDF 파일 하나를 XML로 변환")
    p_conv.add_argument("--pdf", required=True, help="변환할 PDF 파일 경로")
    p_conv.add_argument("--output-dir", help="저장 폴더 (기본값: config.ini converted_xml_dir)")
    p_conv.set_defaults(func=cmd_convert)

    # batch-convert
    p_batch = sub.add_parser("batch-convert", help="PDF 폴더 일괄 XML 변환")
    p_batch.add_argument("--pdf-dir",    help="PDF 폴더 (기본값: config.ini pdf_dir)")
    p_batch.add_argument("--output-dir", help="저장 폴더 (기본값: config.ini converted_xml_dir)")
    p_batch.set_defaults(func=cmd_batch_convert)

    # compare-engines
    p_cmp_eng = sub.add_parser(
        "compare-engines",
        help="동일 PDF를 Audiveris와 homr 양쪽으로 변환해 원본과 비교 (엔진별 정확도 대조)"
    )
    p_cmp_eng.add_argument("--pdf",        required=True, help="원본 PDF 경로")
    p_cmp_eng.add_argument("--orig",       required=True, help="피날레 원본 XML 경로")
    p_cmp_eng.add_argument("--output-dir", help="변환 XML 저장 폴더 (기본값: config.ini converted_xml_dir)")
    p_cmp_eng.add_argument("--part", type=int, default=None, help="파트 인덱스 (기본값: config.ini)")
    p_cmp_eng.set_defaults(func=cmd_compare_engines)

    # config
    p_cfg = sub.add_parser("config", help="현재 config.ini 설정 확인")
    p_cfg.set_defaults(func=cmd_config)

    # visual
    p_vis = sub.add_parser(
        "visual",
        help="교과서 PDF + Finale XML을 단(System) 단위로 시각 비교 (HTML 뷰어)"
    )
    p_vis.add_argument("--pdf",         required=True,
                       help="교과서 스캔본 PDF 경로")
    # Finale 소스: XML(MuseScore 필요) 또는 PDF 직접 지정
    fin_grp = p_vis.add_mutually_exclusive_group(required=True)
    fin_grp.add_argument("--xml",        dest="xml",
                         help="Finale MusicXML 경로 (.xml/.mxl) — MuseScore 필요")
    fin_grp.add_argument("--finale-pdf", dest="finale_pdf",
                         help="Finale에서 직접 Export한 PDF 경로")
    p_vis.add_argument("--output-dir",  default=None,
                       help="출력 폴더 (기본: report_dir/visual/)")
    p_vis.add_argument("--dpi",         type=int, default=600,
                       help="렌더 해상도 (기본 600)")
    p_vis.add_argument("--dump-slices", action="store_true",
                       help="단 PNG를 개별 파일로도 저장 (디버깅)")
    p_vis.set_defaults(func=cmd_visual)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
