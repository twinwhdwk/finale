"""
MuseScore CLI 연동 — MusicXML → PNG 변환 모듈

MuseScore 설치 필요: https://musescore.org/
config.ini 의 [musescore] path 값을 실제 경로로 수정하세요.
"""

import glob
import os
import subprocess
from pathlib import Path

from config_loader import get_musescore_dpi, get_musescore_path


def _musescore_cmd() -> str:
    return os.environ.get("MUSESCORE_PATH") or get_musescore_path()


def convert_xml_to_pngs(
    xml_path: str,
    output_dir: str,
    dpi: int | None = None,
) -> list[str]:
    """
    MusicXML 파일 하나를 페이지별 PNG로 변환합니다.

    MuseScore는 다음 규칙으로 파일을 저장합니다.
      - 1페이지: song.png  또는  song-1.png
      - 다페이지: song-1.png, song-2.png, ...

    Returns:
        페이지 순서로 정렬된 PNG 파일 경로 목록
    """
    if dpi is None:
        dpi = get_musescore_dpi()

    xml_path = Path(xml_path).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML 파일 없음: {xml_path}")

    stem = xml_path.stem
    out_png = out / f"{stem}.png"

    cmd = [
        _musescore_cmd(),
        "-r", str(dpi),
        "-o", str(out_png),
        str(xml_path),
    ]

    print(f"  [MuseScore] {xml_path.name} 변환 중...")
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"MuseScore 변환 실패: {xml_path.name}\n"
            + (e.stderr[-500:] if e.stderr else "")
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "MuseScore를 찾을 수 없습니다.\n"
            "1. https://musescore.org 에서 설치\n"
            "2. config.ini 의 [musescore] path 값을 실제 경로로 수정하세요."
        )

    # 출력 파일 탐색: song-1.png, song-2.png, ... 또는 song.png
    numbered = sorted(
        out.glob(f"{stem}-*.png"),
        key=lambda p: int(p.stem.rsplit("-", 1)[1]),
    )
    if numbered:
        found = [str(p) for p in numbered]
    elif out_png.exists():
        found = [str(out_png)]
    else:
        found = []

    if not found:
        raise RuntimeError(
            f"MuseScore 변환 후 PNG를 찾을 수 없습니다: {out}\n"
            "MuseScore 버전 또는 경로를 확인하세요."
        )

    print(f"  [MuseScore] {len(found)}페이지 PNG 저장: {out}")
    return found


def batch_convert_xmls(
    xml_dir: str,
    output_dir: str,
    dpi: int | None = None,
) -> dict[str, list[str]]:
    """
    폴더 내 모든 .xml / .mxl / .musicxml 파일을 일괄 변환합니다.

    Returns:
        {파일명: [PNG 경로, ...]}
    """
    if dpi is None:
        dpi = get_musescore_dpi()

    xml_dir = Path(xml_dir)
    patterns = ["*.xml", "*.mxl", "*.musicxml"]
    xml_files = []
    for pat in patterns:
        xml_files.extend(xml_dir.glob(pat))
    xml_files = sorted(set(xml_files))

    if not xml_files:
        print(f"XML 파일 없음: {xml_dir}")
        return {}

    print(f"총 {len(xml_files)}개 XML 변환 시작\n")
    results: dict[str, list[str]] = {}

    for xml in xml_files:
        try:
            pngs = convert_xml_to_pngs(str(xml), output_dir, dpi=dpi)
            results[xml.name] = pngs
        except RuntimeError as e:
            print(f"  오류: {e}")
            results[xml.name] = []

    success = sum(1 for v in results.values() if v)
    print(f"\n변환 완료: {success}/{len(xml_files)} 성공")
    return results
