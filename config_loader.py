"""
config.ini 로더

config.ini 파일을 읽어 경로 및 옵션을 제공합니다.
파일이 없으면 기본값을 사용합니다.
"""

import configparser
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.ini"


def load() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if _CONFIG_PATH.exists():
        cfg.read(_CONFIG_PATH, encoding="utf-8")
    return cfg


def get_paths() -> dict:
    cfg = load()
    return {
        "pdf_dir":           cfg.get("paths", "pdf_dir",           fallback=""),
        "xml_dir":           cfg.get("paths", "xml_dir",           fallback=""),
        "musx":              cfg.get("paths", "musx",              fallback=""),
        "converted_xml_dir": cfg.get("paths", "converted_xml_dir", fallback="output/converted"),
        "report_dir":        cfg.get("paths", "report_dir",        fallback="output/reports"),
    }


def get_musescore_path() -> str:
    cfg = load()
    return cfg.get(
        "musescore", "path",
        fallback=r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
    )


def get_musescore_dpi() -> int:
    cfg = load()
    return cfg.getint("musescore", "dpi", fallback=300)


def get_audiveris_path() -> str:
    cfg = load()
    return cfg.get(
        "audiveris", "path",
        fallback=r"C:\Program Files\Audiveris\Audiveris.exe"
    )


def get_part_index() -> int:
    cfg = load()
    return cfg.getint("options", "part_index", fallback=0)


def print_config() -> None:
    """현재 설정값을 출력합니다."""
    paths = get_paths()
    print("=" * 50)
    print("현재 config.ini 설정")
    print("=" * 50)
    print(f"  PDF 폴더:       {paths['pdf_dir']}")
    print(f"  XML 폴더:       {paths['xml_dir']}")
    print(f"  musx 폴더:      {paths['musx']}")
    print(f"  변환 저장 폴더: {paths['converted_xml_dir']}")
    print(f"  리포트 폴더:    {paths['report_dir']}")
    print(f"  Audiveris:      {get_audiveris_path()}")
    print(f"  파트 인덱스:    {get_part_index()}")
    print("=" * 50)
