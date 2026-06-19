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


def _extract_pdf_data(pdf_path: str):
    """
    pdf_parser로 코드 기호·가사를 추출합니다.

    마디선 감지 결과를 이용해 절대 마디 번호를 정확히 계산합니다.
    실패 시 (None, None) 반환.
    """
    try:
        from pdf_parser import parse_all_pages
        print("  PDF OCR 추출 중 (코드 기호 / 가사)...")
        pages = parse_all_pages(pdf_path)

        pdf_chords: list[tuple[int, str]] = []
        pdf_lyrics: list[tuple[int, str]] = []
        abs_measure = 1  # 전체 악보 기준 절대 마디 번호 누산기

        for page in pages:
            for zone in page.zones:
                # 오선 내 (measure_in_staff) → 절대 마디 번호
                for m_in_staff, _x, ch, _cf in zone.chords:
                    pdf_chords.append((abs_measure + m_in_staff - 1, ch))
                for m_in_staff, text in zone.lyrics:
                    pdf_lyrics.append((abs_measure + m_in_staff - 1, text))
                # 마디선 감지 결과로 다음 오선의 시작 마디 번호 계산
                abs_measure += zone.measure_count

        print(
            f"    코드 기호 {len(pdf_chords)}건 / 가사 {len(pdf_lyrics)}건 추출 "
            f"(총 {abs_measure - 1}마디 분량 처리)"
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
    from pdf_to_xml import convert_pdf_to_xml

    paths         = config_loader.get_paths()
    part          = args.part if args.part is not None else config_loader.get_part_index()
    conv_dir      = args.output_dir or paths["converted_xml_dir"]
    report_dir    = Path(paths["report_dir"])

    Path(conv_dir).mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1단계] PDF → XML 변환")
    xml_paths = convert_pdf_to_xml(args.pdf, conv_dir)

    if not xml_paths:
        print("변환된 XML 파일을 찾을 수 없습니다.")
        sys.exit(1)

    pdf_xml = xml_paths[0]
    print(f"  변환 결과: {pdf_xml}\n")

    print("[2단계] XML 비교")
    pdf_chords, pdf_lyrics = _extract_pdf_data(args.pdf)
    result = compare(pdf_xml, args.orig, part_index=part,
                     pdf_chords=pdf_chords, pdf_lyrics=pdf_lyrics)
    print_console(result)

    html_path = args.html or str(report_dir / (Path(args.pdf).stem + "_report.html"))
    save_html(result, html_path)


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
