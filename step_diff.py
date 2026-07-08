"""
step_diff — 렌더링/음가 인식 없이 채보 오류 마디를 찾는 경량 비교기.

원리:
  PDF에서 OpenCV로 검출한 음표머리의 diatonic step(오선 위 반칸 단위 높이)
  시퀀스와, MXL 원본의 step 시퀀스를 Needleman-Wunsch 전역 정렬로 맞춘 뒤
  불일치·누락이 몰린 MXL 마디를 "의심 마디"로 보고한다.

기존 xml_comparator(음가·조표·마디 단위 완전 비교)와 달리:
  - 음가(duration) 인식을 전혀 쓰지 않음 → OMR 최대 약점 회피
  - 임시표/조표 무시 (diatonic step만) → 조표 인식 오류 무관
  - barline 분할 불필요 (전역 정렬) → 마디 밀림에 강건
  - 마디 단위 요약 → 검토자는 의심 마디만 눈으로 확인

실측 (오 나의 태양 F, 검증 세션):
  오경보 8/33마디, 주입 오류(음 하나 3도 변경) 탐지율 12/12(100%),
  주입 시 추가 오경보 0.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np


# ── 자료형 ────────────────────────────────────────────────────────────

@dataclass
class PdfNote:
    step: int          # 오선 최하단 line 기준 반칸 단위 높이 (E4=0 상당)
    page: int          # 1-based
    system: int        # 페이지 내 단 번호 (1-based)
    x: int             # 페이지 픽셀 x


@dataclass
class MxlNote:
    step: int
    measure: int       # 1-based


@dataclass
class SuspectMeasure:
    measure: int
    mismatch: int      # step이 2 이상 다른 음 수 (강한 의심)
    missing: int       # PDF에서 대응을 못 찾은 음 수 (강한 의심)
    near: int          # step이 1 차이 (반올림 경계 가능성 — 약한 의심)
    pdf_hint: tuple | None  # 근처 PDF 위치 (page, system) — 리포트 표시용

    @property
    def strong(self) -> bool:
        return self.mismatch > 0 or self.missing > 0


@dataclass
class StepDiffResult:
    part_index: int
    match: int
    mismatch: int
    pdf_only: int      # PDF에만 있는 음 (노이즈·타 성부)
    mxl_only: int      # MXL에만 있는 음 (누락)
    total_mxl: int
    suspects: list[SuspectMeasure] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        return self.match / max(1, self.total_mxl)


# ── PDF 쪽 추출 ───────────────────────────────────────────────────────

def extract_pdf_steps(pdf_path: str, dpi: int = 300) -> list[PdfNote]:
    """PDF 전 페이지에서 음표머리 step 시퀀스를 읽기 순서로 추출."""
    import fitz
    from pdf_parser import _detect_staves, _detect_barlines
    from note_recognition.staff_removal import detect_staff_line_thickness
    from note_recognition.note_detector import detect_notes

    out: list[PdfNote] = []
    doc = fitz.open(pdf_path)
    for pno in range(len(doc)):
        pix = doc[pno].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72),
                                  colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width)
        zones = _detect_staves(img)
        # 비정상 크기 오선 제외 (opencv_runner와 동일 기준)
        if len(zones) >= 3:
            gaps = [(zb - zt) // 4 for zt, zb in zones]
            main_gap = collections.Counter(gaps).most_common(1)[0][0]
            kept = [z for z, g in zip(zones, gaps)
                    if main_gap * 0.85 <= g <= main_gap * 1.5]
            if len(kept) >= 2:
                zones = kept
        for zi, (zt, zb) in enumerate(zones):
            gap = max(1, (zb - zt) // 4)
            nt = zones[zi + 1][0] if zi + 1 < len(zones) else None
            t = detect_staff_line_thickness(img, [(zt, zb)])
            bars = _detect_barlines(img, zt, zb)
            # 첫 barline - 3*gap은 pickup/첫 음을 자르는 사례가 있어(실측)
            # 여유를 크게 둔다. 조표·박자표가 오검출돼도 NW 정렬이
            # pdf_only(잉여)로 흡수하므로 안전.
            x_start = max(0, bars[0] - gap * 12) if bars else img.shape[1] // 10
            res = detect_notes(img, zt, zb, staff_gap=gap, line_thickness=t,
                               x_start=x_start, next_staff_top_y=nt)
            for n in sorted(res.notes, key=lambda n: n.head_x):
                step = round((zb - n.head_y) * 2 / gap)
                out.append(PdfNote(step=step, page=pno + 1,
                                   system=zi + 1, x=n.head_x))
    doc.close()
    return out


# ── MXL 쪽 추출 ───────────────────────────────────────────────────────

def extract_mxl_steps(mxl_path: str, part_index: int) -> list[MxlNote]:
    """MXL 특정 파트의 step 시퀀스. treble 기준 E4=0."""
    from music21 import converter
    sc = converter.parse(mxl_path)
    parts = sc.parts
    part = parts[min(part_index, len(parts) - 1)]
    out: list[MxlNote] = []
    for mi, m in enumerate(part.getElementsByClass('Measure')):
        for n in m.flatten().notes:
            for p in n.pitches:
                out.append(MxlNote(step=p.diatonicNoteNum - 31, measure=mi + 1))
    return out


def list_part_sizes(mxl_path: str) -> list[int]:
    from music21 import converter
    sc = converter.parse(mxl_path)
    return [len(list(p.flatten().notes)) for p in sc.parts]


# ── 정렬 ─────────────────────────────────────────────────────────────

_MATCH, _NEAR, _MISMATCH, _GAP = 2, 0, -1, -1
# _NEAR: |step 차|=1은 head_y 반올림 경계 오차일 가능성이 커 중립 처리.
# 진짜 채보 오류는 대개 2 이상(3도) 차이라 민감도에 영향 없음 (주입 검증).


def _score(a: int, b: int) -> int:
    d = abs(a - b)
    if d == 0:
        return _MATCH
    if d == 1:
        return _NEAR   # 반올림 경계 오차 우대 (약한 의심으로 집계)
    return _MISMATCH


def _align(A: list[int], B: list[int]) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch 전역 정렬. (pdf_idx|None, mxl_idx|None) 목록 반환."""
    m, n = len(A), len(B)
    S = np.zeros((m + 1, n + 1), dtype=np.int32)
    S[:, 0] = np.arange(0, -(m + 1), -1)
    S[0, :] = np.arange(0, -(n + 1), -1)
    for i in range(1, m + 1):
        ai = A[i - 1]
        row_prev = S[i - 1]
        row = S[i]
        for j in range(1, n + 1):
            row[j] = max(
                row_prev[j - 1] + _score(ai, B[j - 1]),
                row_prev[j] + _GAP,
                row[j - 1] + _GAP,
            )
    i, j = m, n
    pairs: list[tuple[int | None, int | None]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and S[i][j] == S[i - 1][j - 1] + _score(A[i - 1], B[j - 1]):
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and S[i][j] == S[i - 1][j] + _GAP:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    pairs.reverse()
    return pairs


# ── 비교 본체 ─────────────────────────────────────────────────────────

def compare_steps(
    pdf_notes: list[PdfNote],
    mxl_notes: list[MxlNote],
    part_index: int = -1,
    try_offsets: bool = True,
) -> StepDiffResult:
    """
    try_offsets=True면 MXL step 전체에 -7~+7 offset을 시도해
    (교과서 조옮김·옥타브 이동 대응) match가 최대인 offset을 채택한다.
    """
    A = [p.step for p in pdf_notes]
    base = [m.step for m in mxl_notes]

    offsets = range(-7, 8) if try_offsets else [0]
    best_pairs = None
    best_match = -1
    for off in offsets:
        B = [s + off for s in base]
        pairs = _align(A, B)
        match = sum(1 for a, b in pairs
                    if a is not None and b is not None and A[a] == B[b])
        if match > best_match:
            best_match, best_pairs, best_off = match, pairs, off

    B = [s + best_off for s in base]
    pairs = best_pairs

    match = mismatch = pdf_only = mxl_only = 0
    bad: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    hint: dict[int, tuple] = {}

    for a, b in pairs:
        if a is not None and b is not None:
            d = abs(A[a] - B[b])
            mm = mxl_notes[b].measure
            if d == 0:
                match += 1
            elif d == 1:
                # 반올림 경계 가능성 — 약한 의심으로만 기록
                match += 1
                bad[mm][2] += 1
                hint.setdefault(mm, (pdf_notes[a].page, pdf_notes[a].system))
            else:
                mismatch += 1
                bad[mm][0] += 1
                hint.setdefault(mm, (pdf_notes[a].page, pdf_notes[a].system))
        elif a is not None:
            pdf_only += 1
        else:
            mxl_only += 1
            mm = mxl_notes[b].measure
            bad[mm][1] += 1

    suspects = [
        SuspectMeasure(measure=m, mismatch=v[0], missing=v[1], near=v[2],
                       pdf_hint=hint.get(m))
        for m, v in sorted(bad.items())
    ]
    return StepDiffResult(
        part_index=part_index, match=match, mismatch=mismatch,
        pdf_only=pdf_only, mxl_only=mxl_only, total_mxl=len(B),
        suspects=suspects,
    )


def compare_pdf_to_mxl(
    pdf_path: str,
    mxl_path: str,
    part_index: int | None = None,
    dpi: int = 300,
) -> StepDiffResult:
    """
    PDF와 MXL을 step 시퀀스 정렬로 비교해 의심 마디를 보고.

    part_index를 지정하지 않으면 모든 파트를 시도해 일치율이 가장 높은
    파트를 자동 선택한다 (교과서=멜로디, MXL=다파트 편곡인 경우 대응).
    """
    pdf_notes = extract_pdf_steps(pdf_path, dpi=dpi)
    n_parts = len(list_part_sizes(mxl_path))

    if part_index is not None:
        mxl_notes = extract_mxl_steps(mxl_path, part_index)
        return compare_steps(pdf_notes, mxl_notes, part_index)

    best: StepDiffResult | None = None
    best_score = None
    for pi in range(n_parts):
        mxl_notes = extract_mxl_steps(mxl_path, pi)
        if not mxl_notes:
            continue
        r = compare_steps(pdf_notes, mxl_notes, pi)
        # 정렬 품질 점수: 일치 보상 - 불일치/누락 페널티
        # (rate는 작은 파트에, match 절대수는 큰 파트에 유리해 오선택 발생)
        score = r.match * _MATCH + (r.mismatch + r.mxl_only) * _MISMATCH
        if best is None or score > best_score:
            best, best_score = r, score
    if best is None:
        raise RuntimeError("MXL에 음표가 있는 파트가 없습니다")
    return best


def print_report(r: StepDiffResult) -> None:
    print(f"[step-diff] 파트 {r.part_index} | "
          f"일치 {r.match}/{r.total_mxl} ({r.match_rate * 100:.0f}%) | "
          f"불일치 {r.mismatch} | 누락 {r.mxl_only} | PDF측 잉여 {r.pdf_only}")
    if not r.suspects:
        print("  의심 마디 없음 — 채보 오류가 발견되지 않았습니다.")
        return
    strong = [s for s in r.suspects if s.strong]
    weak = [s for s in r.suspects if not s.strong]
    print(f"  강한 의심 {len(strong)}개, 약한 의심 {len(weak)}개:")
    for s in strong:
        loc = f" (PDF p{s.pdf_hint[0]} {s.pdf_hint[1]}단 부근)" if s.pdf_hint else ""
        print(f"    [강] 마디 {s.measure}: 불일치 {s.mismatch}, 누락 {s.missing}{loc}")
    for s in weak:
        loc = f" (PDF p{s.pdf_hint[0]} {s.pdf_hint[1]}단 부근)" if s.pdf_hint else ""
        print(f"    [약] 마디 {s.measure}: 근접차이 {s.near}{loc}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("사용법: python step_diff.py <pdf> <mxl> [part_index]")
        sys.exit(1)
    pi = int(sys.argv[3]) if len(sys.argv) > 3 else None
    print_report(compare_pdf_to_mxl(sys.argv[1], sys.argv[2], part_index=pi))
