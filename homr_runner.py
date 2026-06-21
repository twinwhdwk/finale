"""
PDF 악보 → MusicXML 변환 모듈 (homr OMR 엔진 연동)

Audiveris(pdf_to_xml.py)의 대안/병행 비교용 엔진.
딥러닝 기반(UNet 분할 + Transformer 시퀀스 인식)이라 이음줄/슬러 등
아티큘레이션 인식에서 Audiveris와 다른 특성을 보일 수 있다.

설치 필요:
    pip install homr
    homr --init   # ONNX 모델 최초 1회 다운로드 (인터넷 필요, ~수백MB)

homr은 PDF를 직접 받지 못하고 "이미지 1장 = 악보 1장(또는 1시스템)"을
입력으로 받는다. 따라서 이 모듈은:
  1. PyMuPDF로 PDF 각 페이지를 PNG로 변환
  2. 페이지별로 homr CLI 호출 → {page}.musicxml 생성
  3. 생성된 .musicxml 경로 목록을 반환

PDF 1개가 여러 .musicxml로 쪼개지므로, xml_comparator와 비교하려면
페이지별로 따로 비교하거나 별도 병합 로직이 필요하다 (TODO).
"""

import subprocess
import shutil
import sys
from pathlib import Path


def _homr_cmd() -> str:
    found = shutil.which("homr")
    if not found:
        raise RuntimeError(
            "homr를 찾을 수 없습니다.\n"
            "  pip install homr\n"
            "  homr --init   (모델 최초 다운로드, 인터넷 필요)"
        )
    return found


def ensure_models_downloaded() -> None:
    """homr 모델(ONNX 가중치)이 없으면 다운로드. 최초 1회만 필요."""
    cmd = [_homr_cmd(), "--init"]
    print("[homr] 모델 확인/다운로드 중 (최초 1회만 시간 소요)...")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"homr 모델 다운로드 실패:\n{result.stderr[-800:]}")
    print("[homr] 모델 준비 완료")


def _pdf_to_page_images(pdf_path: str, output_dir: Path, dpi: int = 300) -> list[Path]:
    """PDF 각 페이지를 PNG로 변환. (pdf_parser._pdf_page_to_np와 동일한 fitz 사용)"""
    import fitz  # PyMuPDF

    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    page_images: list[Path] = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        out_path = output_dir / f"{pdf_path.stem}_p{i+1:03d}.png"
        pix.save(str(out_path))
        page_images.append(out_path)
    doc.close()
    return page_images


def _run_homr_on_image(image_path: Path, gpu: str = "auto", extra_args: list[str] | None = None) -> Path:
    """단일 이미지에 homr 실행 → {image_stem}.musicxml 경로 반환."""
    cmd = [_homr_cmd(), str(image_path)]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    expected_out = image_path.with_suffix(".musicxml")

    if result.returncode != 0 or not expected_out.exists():
        print(f"  [homr 실패] {image_path.name}")
        if result.stderr:
            print(result.stderr[-500:])
        raise RuntimeError(f"homr 변환 실패: {image_path.name}")

    return expected_out


def convert_pdf_to_xml(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    gpu: str = "auto",
) -> list[str]:
    """
    PDF 파일을 페이지별로 homr를 통해 MusicXML로 변환합니다.

    pdf_to_xml.convert_pdf_to_xml()과 동일한 시그니처/반환 형태를 유지해
    main.py에서 엔진을 교체 가능하도록 맞춤.

    Args:
        pdf_path:   원본 PDF 경로
        output_dir: 변환 결과(.musicxml) 및 중간 PNG 저장 폴더
        dpi:        PDF → PNG 렌더링 해상도 (homr는 사진 기반이라
                    Audiveris(600dpi 권장)보다 낮아도 무방, 기본 300)
        gpu:        "auto" | "force" | "off" (homr --gpu 옵션과 동일)

    Returns:
        변환된 .musicxml 파일 경로 목록 (페이지 순서대로)
    """
    pdf_path = Path(pdf_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일 없음: {pdf_path}")

    print(f"[homr 변환 시작] {pdf_path.name}")

    print("  1단계: PDF → 페이지 이미지 변환")
    page_images = _pdf_to_page_images(pdf_path, output_dir, dpi=dpi)
    print(f"    {len(page_images)}개 페이지 추출")

    extra_args = []
    if gpu != "auto":
        extra_args += ["--gpu", gpu]

    print("  2단계: homr OMR 실행 (페이지별)")
    results: list[str] = []
    for i, img in enumerate(page_images):
        print(f"    페이지 {i+1}/{len(page_images)}: {img.name}")
        try:
            xml_path = _run_homr_on_image(img, gpu=gpu, extra_args=extra_args)
            results.append(str(xml_path))
        except RuntimeError as e:
            print(f"    건너뜀: {e}")

    print(f"[homr 변환 완료] {pdf_path.name} - {len(results)}/{len(page_images)} 페이지 성공")
    return results


def batch_convert(pdf_dir: str, output_dir: str, dpi: int = 300, gpu: str = "auto") -> dict[str, list[str]]:
    """폴더 내 모든 PDF를 homr로 일괄 변환합니다. (pdf_to_xml.batch_convert와 동일 패턴)"""
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"PDF 파일 없음: {pdf_dir}")
        return {}

    print(f"총 {len(pdf_files)}개 PDF 변환 시작 (엔진: homr)\n")
    results = {}
    for pdf in pdf_files:
        try:
            results[pdf.name] = convert_pdf_to_xml(str(pdf), output_dir, dpi=dpi, gpu=gpu)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"  오류: {e}")
            results[pdf.name] = []

    success = sum(1 for v in results.values() if v)
    print(f"\n변환 완료: {success}/{len(pdf_files)} 성공")
    return results


if __name__ == "__main__":
    # 단독 실행 테스트: python homr_runner.py <pdf경로> [output_dir]
    if len(sys.argv) < 2:
        print("사용법: python homr_runner.py <pdf경로> [output_dir]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "./homr_test_output"
    ensure_models_downloaded()
    paths = convert_pdf_to_xml(sys.argv[1], out)
    print("\n결과:")
    for p in paths:
        print(f"  {p}")
