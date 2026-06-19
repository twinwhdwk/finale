"""
PDF 악보 → MusicXML 변환 모듈 (Audiveris OMR 엔진 연동)

Audiveris 설치 필요: https://github.com/Audiveris/audiveris/releases
- Java 17+ 필수
- 환경변수 AUDIVERIS_PATH 또는 config에 경로 지정
"""

import subprocess
import os
import glob
from pathlib import Path

from config_loader import get_audiveris_path


def _audiveris_cmd() -> str:
    return os.environ.get("AUDIVERIS_PATH") or get_audiveris_path()


def convert_pdf_to_xml(pdf_path: str, output_dir: str) -> list[str]:
    """
    단일 PDF 파일을 MusicXML로 변환합니다.

    Returns:
        변환된 .mxl 또는 .xml 파일 경로 목록
    """
    pdf_path = Path(pdf_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일 없음: {pdf_path}")

    print(f"[변환 시작] {pdf_path.name}")

    cmd = [
        _audiveris_cmd(),
        "-batch",
        "-export",
        "-output", str(output_dir),
        str(pdf_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        print(f"[변환 완료] {pdf_path.name}")
        if result.stdout:
            print(result.stdout[-500:])  # 마지막 500자만 표시
    except subprocess.CalledProcessError as e:
        print(f"[변환 실패] {pdf_path.name}")
        print(e.stderr[-500:] if e.stderr else "")
        raise RuntimeError(f"Audiveris 변환 실패: {pdf_path.name}") from e
    except FileNotFoundError:
        raise RuntimeError(
            "Audiveris를 찾을 수 없습니다.\n"
            "1. https://github.com/Audiveris/audiveris/releases 에서 설치\n"
            "2. config.ini 의 [audiveris] path 값을 실제 경로로 수정하세요."
        )

    # 변환된 XML/MXL 파일 찾기
    stem = pdf_path.stem
    found = (
        glob.glob(str(output_dir / f"{stem}*.mxl")) +
        glob.glob(str(output_dir / f"{stem}*.xml")) +
        glob.glob(str(output_dir / stem / "*.mxl")) +
        glob.glob(str(output_dir / stem / "*.xml"))
    )

    # 불필요한 부산물 삭제 (.omr, .log)
    for ext in ("*.omr", "*.log"):
        for f in glob.glob(str(output_dir / ext)):
            Path(f).unlink()

    return found


def batch_convert(pdf_dir: str, output_dir: str) -> dict[str, list[str]]:
    """
    폴더 내 모든 PDF를 일괄 변환합니다.

    Returns:
        {pdf_파일명: [변환된 xml 경로, ...]}
    """
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"PDF 파일 없음: {pdf_dir}")
        return {}

    print(f"총 {len(pdf_files)}개 PDF 변환 시작\n")
    results = {}

    for pdf in pdf_files:
        try:
            xml_paths = convert_pdf_to_xml(str(pdf), output_dir)
            results[pdf.name] = xml_paths
        except RuntimeError as e:
            print(f"  오류: {e}")
            results[pdf.name] = []

    success = sum(1 for v in results.values() if v)
    print(f"\n변환 완료: {success}/{len(pdf_files)} 성공")
    return results
