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
    pdf_hint: tuple | None  # (page, system, x_min, x_max) — 리포트/크롭용

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
    hint: dict[int, list] = {}

    for a, b in pairs:
        if a is not None and b is not None:
            d = abs(A[a] - B[b])
            mm = mxl_notes[b].measure
            if d == 0:
                match += 1
                # 정상 매칭도 위치 힌트에 누적 (의심 마디 크롭 범위 계산용)
                hint.setdefault(mm, []).append(
                    (pdf_notes[a].page, pdf_notes[a].system, pdf_notes[a].x))
            elif d == 1:
                # 반올림 경계 가능성 — 약한 의심으로만 기록
                match += 1
                bad[mm][2] += 1
                hint.setdefault(mm, []).append(
                    (pdf_notes[a].page, pdf_notes[a].system, pdf_notes[a].x))
            else:
                mismatch += 1
                bad[mm][0] += 1
                hint.setdefault(mm, []).append(
                    (pdf_notes[a].page, pdf_notes[a].system, pdf_notes[a].x))
        elif a is not None:
            pdf_only += 1
        else:
            mxl_only += 1
            mm = mxl_notes[b].measure
            bad[mm][1] += 1

    def _hint_of(m):
        pts = hint.get(m)
        if not pts:
            return None
        # 최빈 (page, system)의 x 범위
        ps = collections.Counter((p, s) for p, s, _ in pts).most_common(1)[0][0]
        xs = [x for p, s, x in pts if (p, s) == ps]
        return (ps[0], ps[1], min(xs), max(xs))

    suspects = [
        SuspectMeasure(measure=m, mismatch=v[0], missing=v[1], near=v[2],
                       pdf_hint=_hint_of(m))
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




# ── 시각 리포트 ───────────────────────────────────────────────────────

def _render_mxl_measures(mxl_path: str, part_index: int,
                         measures: set[int]) -> dict[int, str]:
    """MXL 특정 파트의 지정 마디들을 verovio로 마디당 1단 렌더해
    {마디번호: base64 PNG}로 반환. 실패 시 빈 dict (리포트는 계속 진행)."""
    try:
        import base64
        import tempfile
        import cairosvg
        from music21 import converter, stream
        from xml_to_systems import xml_to_systems

        sc = converter.parse(mxl_path)
        part = sc.parts[min(part_index, len(sc.parts) - 1)]
        n_meas = len(part.getElementsByClass('Measure'))
        new_sc = stream.Score()
        new_sc.append(part)
        with tempfile.NamedTemporaryFile(suffix='.musicxml',
                                         delete=False) as tf:
            tmp = tf.name
        new_sc.write('musicxml', tmp)
        sliced = xml_to_systems(tmp, measures_per_system=[1] * n_meas)
        out: dict[int, str] = {}
        for m in measures:
            if 1 <= m <= sliced.total_systems:
                svg = sliced.systems[m - 1].png_bytes
                png = cairosvg.svg2png(bytestring=svg, scale=1.6,
                                       background_color='white')
                out[m] = base64.b64encode(png).decode()
        return out
    except Exception as e:
        print(f"  (원본 마디 렌더 생략: {e})")
        return {}


def save_visual_report(pdf_path: str, result: StepDiffResult,
                       out_html: str, mxl_path: str | None = None,
                       dpi: int = 300) -> str:
    """
    의심 마디마다 PDF 해당 구간(빨간 박스)과 — mxl_path가 주어지면 —
    원본(MXL) 마디의 verovio 렌더를 나란히 담은 HTML 저장.
    검토자는 이 파일만 열어 두 이미지를 눈으로 대조하면 된다.
    """
    import base64
    import html as _html
    import fitz
    import cv2
    from pathlib import Path
    from pdf_parser import _detect_staves

    # 페이지 이미지 캐시
    doc = fitz.open(pdf_path)
    pages: dict[int, np.ndarray] = {}
    zones_by_page: dict[int, list] = {}
    for pno in range(len(doc)):
        pix = doc[pno].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72),
                                  colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width)
        pages[pno + 1] = img
        zones_by_page[pno + 1] = _detect_staves(img)
    doc.close()

    # 원본 마디 렌더 (의심 마디만)
    orig_imgs: dict[int, str] = {}
    if mxl_path:
        orig_imgs = _render_mxl_measures(
            mxl_path, result.part_index,
            {s.measure for s in result.suspects})

    def crop_b64(page: int, system: int, x0: int, x1: int) -> str | None:
        img = pages.get(page)
        zones = zones_by_page.get(page)
        if img is None or not zones or system > len(zones):
            return None
        zt, zb = zones[system - 1]
        gap = max(1, (zb - zt) // 4)
        y0 = max(0, zt - gap * 4)
        y1 = min(img.shape[0], zb + gap * 4)
        pad = gap * 4
        cx0 = max(0, x0 - pad)
        cx1 = min(img.shape[1], x1 + pad)
        crop = cv2.cvtColor(img[y0:y1, cx0:cx1], cv2.COLOR_GRAY2BGR)
        cv2.rectangle(crop, (x0 - cx0 - gap, 0),
                      (x1 - cx0 + gap, crop.shape[0] - 1), (0, 0, 230), 3)
        ok, buf = cv2.imencode('.png', crop)
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode()

    n_strong = sum(1 for s in result.suspects if s.strong)
    n_weak = len(result.suspects) - n_strong

    rows = []
    for s in sorted(result.suspects, key=lambda s: (not s.strong, s.measure)):
        grade_cls = "strong" if s.strong else "weak"
        grade_txt = "강한 의심" if s.strong else "약한 의심"
        details = []
        if s.mismatch:
            details.append(f"음높이 불일치 {s.mismatch}건")
        if s.missing:
            details.append(f"PDF에서 못 찾은 음 {s.missing}건")
        if s.near:
            details.append(f"반칸 차이(검출 오차 가능) {s.near}건")
        detail = " · ".join(details) or "-"

        pdf_img = orig_img = ""
        loc = "위치 미상"
        if s.pdf_hint:
            p, sysn, x0, x1 = s.pdf_hint
            loc = f"교과서 PDF {p}페이지 {sysn}단"
            b64 = crop_b64(p, sysn, x0, x1)
            if b64:
                pdf_img = (f'<figure><figcaption>교과서 PDF (빨간 박스 부근)'
                           f'</figcaption>'
                           f'<img src="data:image/png;base64,{b64}"></figure>')
        if s.measure in orig_imgs:
            orig_img = (f'<figure><figcaption>Finale 원본 (마디 {s.measure})'
                        f'</figcaption><img src="data:image/png;base64,'
                        f'{orig_imgs[s.measure]}"></figure>')

        rows.append(f"""
<section class="item {grade_cls}">
  <header><span class="badge">{grade_txt}</span>
    <h3>마디 {s.measure}</h3><span class="loc">{loc}</span></header>
  <p class="detail">{_html.escape(detail)}</p>
  <div class="pair">{orig_img}{pdf_img}</div>
</section>""")

    score_name = _html.escape(Path(pdf_path).stem)
    html_doc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>채보 검토 리포트 — {score_name}</title><style>
:root {{ --strong:#d63031; --weak:#e17055; --bg:#fafafa; }}
* {{ box-sizing:border-box; }}
body {{ font-family:'Apple SD Gothic Neo','Malgun Gothic',sans-serif;
  max-width:1200px; margin:0 auto; padding:24px; background:var(--bg);
  color:#2d3436; }}
h1 {{ font-size:1.5em; margin:0 0 4px; }}
.sub {{ color:#636e72; margin-bottom:16px; }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:24px; }}
.card {{ background:#fff; border-radius:10px; padding:14px 20px;
  box-shadow:0 1px 4px rgba(0,0,0,.08); min-width:130px; }}
.card b {{ display:block; font-size:1.6em; }}
.card.red b {{ color:var(--strong); }}
.card.orange b {{ color:var(--weak); }}
.item {{ background:#fff; border-radius:10px; padding:16px 20px;
  margin:16px 0; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
.item.strong {{ border-left:6px solid var(--strong); }}
.item.weak {{ border-left:6px solid var(--weak); }}
.item header {{ display:flex; align-items:center; gap:10px; }}
.item h3 {{ margin:0; font-size:1.15em; }}
.badge {{ font-size:.75em; color:#fff; padding:3px 10px;
  border-radius:12px; background:var(--strong); }}
.weak .badge {{ background:var(--weak); }}
.loc {{ color:#636e72; font-size:.85em; margin-left:auto; }}
.detail {{ color:#636e72; margin:6px 0 12px; font-size:.9em; }}
.pair {{ display:flex; gap:16px; flex-wrap:wrap; }}
.pair figure {{ flex:1 1 320px; margin:0; }}
.pair figcaption {{ font-size:.8em; color:#636e72; margin-bottom:4px; }}
.pair img {{ width:100%; border:1px solid #dfe6e9; border-radius:6px;
  background:#fff; }}
.help {{ background:#fff; border-radius:10px; padding:12px 20px;
  font-size:.85em; color:#636e72; }}
</style></head><body>
<h1>채보 검토 리포트</h1>
<div class="sub">{score_name} — MXL 파트 {result.part_index}과 비교</div>
<div class="cards">
  <div class="card"><b>{result.match_rate*100:.0f}%</b>일치율
    ({result.match}/{result.total_mxl}음)</div>
  <div class="card red"><b>{n_strong}</b>강한 의심 마디</div>
  <div class="card orange"><b>{n_weak}</b>약한 의심 마디</div>
</div>
<div class="help"><b>보는 법</b> — 강한 의심(빨강)부터 확인하세요.
왼쪽이 Finale 원본, 오른쪽이 교과서 PDF입니다. 두 이미지의 음이 같으면
검출 오차(무시), 다르면 채보 오류입니다. 약한 의심(주황)은 한 칸 차이라
대부분 검출 반올림 오차입니다.</div>
{''.join(rows) if rows else '<div class="item"><h3>의심 마디 없음 🎉</h3></div>'}
</body></html>"""

    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_doc)
    return out_html


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("사용법: python step_diff.py <pdf> <mxl> [part_index]")
        sys.exit(1)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    html_out = None
    for a in sys.argv[1:]:
        if a.startswith("--html="):
            html_out = a.split("=", 1)[1]
    pi = int(args[2]) if len(args) > 2 else None
    result = compare_pdf_to_mxl(args[0], args[1], part_index=pi)
    print_report(result)
    if html_out:
        path = save_visual_report(args[0], result, html_out, mxl_path=args[1])
        print(f"  시각 리포트 저장: {path}")
