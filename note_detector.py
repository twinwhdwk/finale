"""
PDF 악보 이미지에서 음표 머리(notehead)를 감지하고 음높이로 변환.
붙임줄(tie) / 이음줄(slur) 호 감지 및 분류 포함.
"""

from __future__ import annotations

import cv2
import numpy as np

# ── 음이름 테이블 ────────────────────────────────────────────────────

_NOTE_NAMES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
_NOTE_IDX   = {n: i for i, n in enumerate(_NOTE_NAMES)}

_CLEF_BASE = {
    'G': ('E', 4),   # 높은음자리표 최하단 줄 E4
    'F': ('G', 2),   # 낮은음자리표 최하단 줄 G2
    'C': ('F', 3),   # 가온음자리표 최하단 줄 F3
}


# ── 마디선 감지 (슬라이스 이미지용) ────────────────────────────────────

def _detect_barlines_local(
    img_gray: np.ndarray,
    top_y: int,
    bot_y: int,
) -> list[int]:
    """
    단(System) 크롭 이미지에서 마디선 x좌표 목록 반환.
    pdf_parser._detect_barlines와 동일한 알고리즘, 독립 구현.
    """
    h, w   = img_gray.shape
    staff_h = bot_y - top_y
    roi    = img_gray[top_y:bot_y + 1, :]
    _, bw  = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)
    col_sum = bw.sum(axis=0).astype(float)

    thresh = (bot_y - top_y) * 255 * 0.95
    candidates = [x for x in range(w) if col_sum[x] >= thresh]
    if not candidates:
        return [0, w]

    # NMS 100px
    clusters, cl = [], [candidates[0]]
    for x in candidates[1:]:
        if x - cl[-1] <= 100:
            cl.append(x)
        else:
            clusters.append(int(np.mean(cl)))
            cl = [x]
    clusters.append(int(np.mean(cl)))

    # 좌측 15% / 우측 89% 필터
    left_cut  = int(w * 0.15)
    right_cut = int(w * 0.89)
    filtered  = [x for x in clusters if left_cut <= x <= right_cut]

    if not filtered:
        return [0, w]

    # 첫 바라인이 25% 미만이면 클레프 영역 추가 제거
    if filtered[0] < w * 0.25:
        filtered = filtered[1:]
    if not filtered:
        return [0, w]

    # 이상치 제거: 인접 간격 중앙값의 45% 미만인 바라인 삭제
    if len(filtered) >= 2:
        gaps   = [filtered[i+1] - filtered[i] for i in range(len(filtered)-1)]
        med    = float(np.median(gaps))
        result = [filtered[0]]
        for i, g in enumerate(gaps):
            if g >= med * 0.45:
                result.append(filtered[i+1])
        filtered = result

    # 격자형 기보 필터
    if len(filtered) >= 4:
        gaps  = [filtered[i+1] - filtered[i] for i in range(len(filtered)-1)]
        med_g = float(np.median(gaps))
        if med_g < staff_h * 2:
            return []

    return filtered if filtered else [0, w]


# ── 오선 5줄 y좌표 감지 ──────────────────────────────────────────────

def _detect_staff_lines(img_gray: np.ndarray) -> list[list[int]]:
    h, w = img_gray.shape
    _, binary = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 6, 1))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    rows = np.where(mask.sum(axis=1) > w * 0.3)[0]

    if not len(rows):
        return []

    merged, cluster = [], [int(rows[0])]
    for r in rows[1:]:
        if r - cluster[-1] <= 20:
            cluster.append(int(r))
        else:
            merged.append(int(np.mean(cluster)))
            cluster = [int(r)]
    merged.append(int(np.mean(cluster)))

    staves = []
    i = 0
    while i + 4 < len(merged):
        five = merged[i:i+5]
        gaps = [five[j+1] - five[j] for j in range(4)]
        avg  = np.mean(gaps)
        if max(gaps) < avg * 1.8:
            staves.append([int(y) for y in five])
            i += 5
        else:
            i += 1
    return staves


# ── 오선 제거 ────────────────────────────────────────────────────────

def _remove_staff_lines(
    binary: np.ndarray,
    line_ys: list[int],
    spacing: float,
) -> np.ndarray:
    """2단계 수평 Opening으로 오선과 빔·레저선을 제거."""
    result = binary.copy()
    h, w   = result.shape

    # 1단계: 전폭 오선 제거
    kw1 = max(int(w * 0.55), 60)
    k1  = cv2.getStructuringElement(cv2.MORPH_RECT, (kw1, 1))
    horiz1 = cv2.morphologyEx(result, cv2.MORPH_OPEN, k1)
    result = cv2.subtract(result, horiz1)

    # 2단계: 짧은 빔·레저선 제거 (spacing*0.95 너비)
    kw2 = max(int(spacing * 0.95), 20)
    k2  = cv2.getStructuringElement(cv2.MORPH_RECT, (kw2, 1))
    horiz2 = cv2.morphologyEx(result, cv2.MORPH_OPEN, k2)
    result = cv2.subtract(result, horiz2)

    return result


# ── y좌표 → 음높이 변환 ──────────────────────────────────────────────

def _y_to_step(y: float, line_ys: list[int]) -> int:
    """픽셀 y좌표 → 오선 기준 스텝(아래가 낮은음, 위가 높은음)."""
    if len(line_ys) < 2:
        return 0
    spacing = (line_ys[-1] - line_ys[0]) / (len(line_ys) - 1)
    if spacing <= 0:
        return 0
    bottom = line_ys[-1]
    step = round((bottom - y) * 2 / spacing)
    return int(step)


def _step_to_pitch(step: int, clef: str) -> str:
    """스텝 번호 → 음이름+옥타브 (예: 'G4')."""
    base_note, base_oct = _CLEF_BASE.get(clef, ('E', 4))
    base_idx = _NOTE_IDX[base_note]
    abs_idx  = base_idx + step
    note_idx = abs_idx % 7
    octave   = base_oct + abs_idx // 7
    return f"{_NOTE_NAMES[note_idx]}{octave}"


# ── 음표 머리 감지 ───────────────────────────────────────────────────

def _detect_noteheads(
    binary_clean: np.ndarray,
    spacing: float,
) -> tuple[list[tuple[int, int, bool]], np.ndarray]:
    """
    오선 제거 후 이미지에서 음표 머리 (x, y, is_hollow) 목록 반환.
    반환: (noteheads, filled_image)
      - noteheads: [(cx, cy, is_hollow), ...]
      - filled_image: 빈머리 채운 후 이미지 (Opening용)
    """
    # 반음표·온음표 링 채우기: CLOSE 후 RETR_CCOMP로 내부 구멍만 fill
    kh = max(8, int(spacing * 0.20))
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    closed = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kc)

    raw_closed = closed.copy()  # fill 이전 복사본 (hollow 판정용)

    cnts, hier = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = closed.copy()
    if hier is not None:
        for i, h in enumerate(hier[0]):
            parent = h[3]
            if parent >= 0:
                x, y, bw, bh = cv2.boundingRect(cnts[i])
                area = bw * bh
                if area <= 2 * spacing ** 2:
                    cv2.drawContours(filled, cnts, i, 255, cv2.FILLED)

    # Opening으로 기둥·꼬리 제거, 음표 머리만 남김
    radius = max(int(spacing * 0.40), 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2, radius * 2))
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)

    cnts2, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    noteheads: list[tuple[int, int, bool]] = []
    for cnt in cnts2:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < (spacing * 0.3) ** 2:
            continue
        # 화음 합성 블롭 제거
        if bh > spacing * 1.3:
            continue

        cx = x + bw // 2
        cy = y + bh // 2

        # hollow 판정: raw_closed 기준 fill_ratio
        roi = raw_closed[y:y+bh, x:x+bw]
        fill_ratio = roi.sum() / (255 * bw * bh) if bw * bh > 0 else 1.0
        is_hollow = fill_ratio < 0.50

        noteheads.append((cx, cy, is_hollow))

    return noteheads, filled


# ── 마디 배정 ────────────────────────────────────────────────────────

def _assign_to_measures(
    noteheads: list[tuple[int, int, bool]],
    barlines: list[int],
    left_margin: int,
    spacing: float,
) -> tuple[list[list[str]], list[list[bool]]]:
    """음표 머리를 마디 번호별로 분배. 첫 마디(i==0) excluded."""
    n = len(barlines)
    if n == 0:
        return [], []

    pitches_per = [[] for _ in range(n)]
    hollows_per = [[] for _ in range(n)]

    for (cx, cy, is_hollow) in noteheads:
        if cx < left_margin:
            continue
        m_idx = 0
        for bx in barlines[:-1]:
            if cx >= bx:
                m_idx += 1
        m_idx = min(m_idx, n - 1)
        pitches_per[m_idx].append((cx, cy, is_hollow))
        hollows_per[m_idx].append(is_hollow)

    return pitches_per, hollows_per


# ── 호(arc) 감지 ─────────────────────────────────────────────────────

def _detect_arcs(
    binary_clean: np.ndarray,
    spacing: float,
    left_margin: int = 0,
    y_range: tuple[int, int] | None = None,
) -> list[dict]:
    """
    오선 제거 후 binary_clean에서 붙임줄/슬러 호 후보 검출.
    Opening 이전 단계에서 호출해야 함 — Opening 후에는 호가 소멸.
    y_range: (y_min, y_max) — 가사·코드 영역 제외용.
    """
    min_arc_w = max(int(spacing * 1.2), 40)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (min_arc_w, 1))
    horiz = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, hk)

    cnts, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arcs = []
    for cnt in cnts:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < spacing * 1.4:
            continue
        if bh > spacing * 0.9:
            continue
        if bw / max(bh, 1) < 2.5:
            continue
        if x < left_margin:
            continue
        if y_range is not None:
            cy_arc = y + bh // 2
            if cy_arc < y_range[0] or cy_arc > y_range[1]:
                continue

        pts      = cnt.reshape(-1, 2)
        apex_y   = float(pts[:, 1].min())
        base_y   = float(pts[:, 1].max())
        left_y   = float(pts[pts[:, 0].argmin()][1])
        right_y  = float(pts[pts[:, 0].argmax()][1])
        end_avg  = (left_y + right_y) / 2.0
        up_ness  = end_avg - apex_y
        down_ness = base_y - end_avg

        # 직선(빔·아티팩트) 제거: 최소 곡률 2px
        if max(up_ness, down_ness) < 2.0:
            continue

        convex = 'up' if up_ness > down_ness else 'down'
        arcs.append({
            'x0': x, 'x1': x + bw, 'y0': y, 'y1': y + bh,
            'cx': x + bw // 2, 'cy': y + bh // 2,
            'width': bw, 'convex': convex,
            'curve': round(max(up_ness, down_ness), 1),
        })
    return arcs


# ── 호 tie/slur 분류 ─────────────────────────────────────────────────

def _classify_arcs(
    arcs: list[dict],
    noteheads: list[tuple[int, int, bool]],
    line_ys: list[int],
    spacing: float,
    clef: str,
) -> list[dict]:
    """
    각 호의 양끝에서 nearest notehead 두 개를 찾아 tie/slur 분류.
    snap = spacing*1.0 (엄격 — 완화 시 슬러 오분류 증가).
    """
    snap = spacing * 1.0

    for a in arcs:
        def nearest(px: float, cy: float):
            best, bestd = None, 1e9
            for (nx, ny, _) in noteheads:
                d = abs(nx - px) + 0.3 * abs(ny - cy)
                if d < bestd:
                    bestd, best = d, (nx, ny)
            return best, bestd

        left_nh,  dl = nearest(a['x0'], a['cy'])
        right_nh, dr = nearest(a['x1'], a['cy'])

        if left_nh is None or right_nh is None or dl > snap or dr > snap:
            a['cls'] = 'unknown'
            continue

        if abs(left_nh[0] - right_nh[0]) < spacing * 0.8:
            a['cls'] = 'unknown'
            continue

        p_left  = _step_to_pitch(_y_to_step(left_nh[1],  line_ys), clef)
        p_right = _step_to_pitch(_y_to_step(right_nh[1], line_ys), clef)
        a['cls'] = 'tie' if p_left == p_right else 'slur'

    return arcs


def _arc_to_measure_ties(
    arcs: list[dict],
    barlines: list[int],
) -> dict[int, dict]:
    """tie 분류된 호를 마디별 이벤트로 변환. {m_idx(0-based): {start,stop,internal}}"""
    n = len(barlines)
    if n == 0:
        return {}

    def x_to_midx(x: int) -> int:
        idx = 0
        for bx in barlines[:-1]:
            if x >= bx:
                idx += 1
        return min(idx, n - 1)

    events: dict[int, dict] = {i: {'start': 0, 'stop': 0, 'internal': 0}
                                for i in range(n)}
    for a in arcs:
        if a.get('cls') != 'tie':
            continue
        m_left  = x_to_midx(a['x0'])
        m_right = x_to_midx(a['x1'])
        if m_left != m_right:
            events[m_left]['start']   += 1
            events[m_right]['stop']   += 1
        else:
            events[m_left]['internal'] += 1
    return events


def compare_ties(
    arc_events: dict[int, dict],
    xml_ties: dict[int, list[str]],
    start_m: int,
) -> dict[int, str]:
    """
    PDF 감지 tie 이벤트 vs XML tie 정보 마디 단위 비교.
    Returns: {m_num: 'pdf_extra'|'xml_extra'} — 불일치 마디만.
    """
    issues: dict[int, str] = {}
    for m_idx, ev in arc_events.items():
        m_num    = start_m + m_idx
        xml_list = xml_ties.get(m_num, [])
        xml_start = sum(1 for t in xml_list if t in ('start', 'both'))
        xml_stop  = sum(1 for t in xml_list if t in ('stop',  'both'))

        pdf_start = ev['start'] + ev['internal']
        pdf_stop  = ev['stop']  + ev['internal']

        if pdf_start > xml_start:
            issues[m_num] = 'pdf_extra'
        elif xml_start > pdf_start:
            issues[m_num] = 'xml_extra'
    return issues


# ── 핵심 감지 루틴 ───────────────────────────────────────────────────

def _run_detection(
    img_gray: np.ndarray,
    clef: str,
    barlines_hint: list[int] | None = None,
) -> tuple[list[list[str]], list[list[bool]], dict[int, dict]]:
    """
    단(System) PNG 한 장에서 음표·hollow·붙임줄 이벤트를 모두 감지.
    Returns: (pitches_per_measure, hollows_per_measure, arc_events)
    """
    h, w = img_gray.shape
    _, binary = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)

    staves = _detect_staff_lines(img_gray)
    if not staves:
        return [], [], {}

    line_ys = staves[0]
    spacing = (line_ys[-1] - line_ys[0]) / 4.0
    top_y   = line_ys[0]
    bot_y   = line_ys[-1]

    # 오선 제거
    binary_clean = _remove_staff_lines(binary, line_ys, spacing)

    # 마디선
    if barlines_hint:
        barlines = barlines_hint
    else:
        barlines = _detect_barlines_local(img_gray, top_y, bot_y)
    if not barlines:
        barlines = [0, w]

    left_margin = min(int(barlines[0] * 0.60), int(spacing * 10))

    # 호 감지 (Opening 이전)
    arc_y_min = int(top_y - 2.5 * spacing)
    arc_y_max = int(bot_y + 2.5 * spacing)
    arcs = _detect_arcs(
        binary_clean, spacing,
        left_margin=left_margin,
        y_range=(arc_y_min, arc_y_max),
    )

    # 음표 머리 감지
    noteheads, _ = _detect_noteheads(binary_clean, spacing)

    # 음표 → 마디 배정
    pitches_buckets, hollows_buckets = _assign_to_measures(
        noteheads, barlines, left_margin, spacing
    )

    # 음높이 변환 + x 근접 중복 제거
    pitches_per: list[list[str]] = []
    hollows_per: list[list[bool]] = []

    for m_noteheads, m_hollows in zip(pitches_buckets, hollows_buckets):
        if not m_noteheads:
            pitches_per.append([])
            hollows_per.append([])
            continue

        raw = sorted(m_noteheads, key=lambda t: t[0])
        merged: list[tuple[int, int, bool]] = []
        for (cx, cy, hol) in raw:
            if merged and abs(cx - merged[-1][0]) < spacing / 2 and abs(cy - merged[-1][1]) < spacing / 2:
                ox, oy, oh = merged[-1]
                merged[-1] = ((ox + cx) // 2, (oy + cy) // 2, oh or hol)
            else:
                merged.append((cx, cy, hol))

        p_list: list[str] = []
        h_list: list[bool] = []
        prev_pitch = None
        prev_x     = None
        for (cx, cy, hol) in merged:
            pitch = _step_to_pitch(_y_to_step(cy, line_ys), clef)
            if prev_pitch == pitch and prev_x is not None and abs(cx - prev_x) < spacing * 1.5:
                continue  # 반음표·온음표 이중감지 제거
            p_list.append(pitch)
            h_list.append(hol)
            prev_pitch = pitch
            prev_x     = cx

        pitches_per.append(p_list)
        hollows_per.append(h_list)

    # 호 분류 및 tie 이벤트
    all_noteheads = [nh for bucket in pitches_buckets for nh in bucket]
    arcs = _classify_arcs(arcs, all_noteheads, line_ys, spacing, clef)
    arc_events = _arc_to_measure_ties(arcs, barlines)

    return pitches_per, hollows_per, arc_events


# ── 공개 API ─────────────────────────────────────────────────────────

def detect_notes_and_ties_from_png(
    png_bytes: bytes,
    clef: str = 'G',
    barlines_hint: list[int] | None = None,
) -> tuple[list[list[str]], list[list[bool]], dict[int, dict]]:
    """
    단(System) PNG → (pitches_per_measure, hollows_per_measure, arc_events).
    arc_events = {m_idx(0-based): {'start':n,'stop':n,'internal':n}}
    """
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], [], {}
    return _run_detection(img, clef, barlines_hint)


def detect_notes_from_png(
    png_bytes: bytes,
    clef: str = 'G',
    barlines_hint: list[int] | None = None,
) -> tuple[list[list[str]], list[list[bool]]]:
    """하위 호환 — arc_events 버림."""
    pitches, hollows, _ = detect_notes_and_ties_from_png(png_bytes, clef, barlines_hint)
    return pitches, hollows


def detect_pitches_from_png(
    png_bytes: bytes,
    clef: str = 'G',
    barlines_hint: list[int] | None = None,
) -> list[list[str]]:
    """하위 호환 — pitches만 반환."""
    pitches, _ = detect_notes_from_png(png_bytes, clef, barlines_hint)
    return pitches


def detect_arcs_debug_png(
    png_bytes: bytes,
    clef: str = 'G',
    barlines_hint: list[int] | None = None,
) -> bytes:
    """호 감지 결과를 컬러 오버레이한 PNG bytes 반환 (--dump-slices 디버깅용)."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return png_bytes

    h, w = img.shape
    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)

    staves = _detect_staff_lines(img)
    if not staves:
        return png_bytes

    line_ys = staves[0]
    spacing = (line_ys[-1] - line_ys[0]) / 4.0
    top_y   = line_ys[0]
    bot_y   = line_ys[-1]

    binary_clean = _remove_staff_lines(binary, line_ys, spacing)

    if barlines_hint:
        barlines = barlines_hint
    else:
        barlines = _detect_barlines_local(img, top_y, bot_y)
    if not barlines:
        barlines = [0, w]

    left_margin = min(int(barlines[0] * 0.60), int(spacing * 10))
    arc_y_range = (int(top_y - 2.5 * spacing), int(bot_y + 2.5 * spacing))
    arcs = _detect_arcs(binary_clean, spacing, left_margin=left_margin, y_range=arc_y_range)

    noteheads, _ = _detect_noteheads(binary_clean, spacing)

    arcs = _classify_arcs(arcs, noteheads, line_ys, spacing, clef)

    # BGR 오버레이
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 바라인
    for bx in barlines:
        cv2.line(overlay, (bx, 0), (bx, h), (200, 200, 200), 1)

    # 음표 머리
    for (cx, cy, is_hollow) in noteheads:
        color = (255, 140, 0) if is_hollow else (0, 180, 0)
        cv2.circle(overlay, (cx, cy), max(4, int(spacing * 0.3)), color, 2)

    # 호
    for a in arcs:
        cls = a.get('cls', 'unknown')
        color = (0, 0, 220) if cls == 'tie' else (180, 0, 180) if cls == 'slur' else (120, 120, 120)
        cv2.rectangle(overlay, (a['x0'], a['y0']), (a['x1'], a['y1']), color, 1)
        label = f"{cls[0].upper()} w{a['width']} c{a['curve']}"
        cv2.putText(overlay, label, (a['x0'], a['y0'] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    ok, buf = cv2.imencode('.png', overlay)
    return bytes(buf) if ok else png_bytes
