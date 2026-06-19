"""
검수 결과 리포트 생성 모듈

- print_console(): 콘솔 출력 (3-트랙 요약)
- save_html(): 인터랙티브 HTML 리포트 저장
  * 유형별 탭 필터 (전체 / 음표 / 코드 / 가사)
  * 심각도별 색상 코딩
  * 검수 액션 버튼 ([무시] / [수정완료] / [재확인]) — localStorage 저장
"""

from datetime import datetime
from pathlib import Path
from xml_comparator import CompareResult, Discrepancy


# ── 콘솔 출력 ──────────────────────────────────────────────────────────

def print_console(result: CompareResult) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("[악보 검수 결과 리포트]")
    print(sep)
    print(result.summary())
    print(sep)

    if result.is_perfect:
        print("\n두 악보가 완벽하게 일치합니다!\n")
        return

    by_measure: dict[int, list[Discrepancy]] = {}
    for d in result.discrepancies:
        by_measure.setdefault(d.measure, []).append(d)

    track_labels = {"note": "[음표]", "chord": "[코드]", "lyric": "[가사]"}

    for m_num in sorted(by_measure):
        items = by_measure[m_num]
        print(f"\n-- 마디 {m_num} ({len(items)}건) --")
        for d in items:
            tk = track_labels.get(d.track, "")
            print(f"  {tk} {d}")

    print(f"\n{sep}")
    total = len(result.discrepancies)
    print(
        f"총 {total}건 발견  |  "
        f"음표 오류: {result.note_errors}  "
        f"(누락 {result.missing_count} / 노이즈 {result.noise_count})  |  "
        f"코드 오류: {result.chord_errors}  |  "
        f"가사 오류: {result.lyric_errors}"
    )
    print(sep)


# ── HTML 리포트 ────────────────────────────────────────────────────────

_KIND_META = {
    # kind: (track, severity, display_label, color)
    "pitch":        ("note",  "high",   "음높이 오류",  "#e53935"),
    "duration":     ("note",  "medium", "음길이 오류",  "#fb8c00"),
    "type":         ("note",  "medium", "형식 오류",    "#f4511e"),
    "tie_suspect":  ("note",  "low",    "붙임줄 의심",  "#8e24aa"),
    "measure_miss": ("note",  "high",   "마디 누락",    "#b71c1c"),
    "missing":      ("note",  "medium", "OMR 누락",     "#ef6c00"),
    "noise":        ("note",  "low",    "OMR 노이즈",   "#9e9e9e"),
    "chord_miss":   ("chord", "medium", "코드 누락",    "#1565c0"),
    "chord_diff":   ("chord", "high",   "코드 불일치",  "#0d47a1"),
    "lyric_miss":   ("lyric", "low",    "가사 누락",    "#2e7d32"),
    "lyric_diff":   ("lyric", "medium", "가사 불일치",  "#1b5e20"),
}

_SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


def _badge(kind: str) -> str:
    meta = _KIND_META.get(kind, ("note", "low", kind, "#777"))
    _, sev, label, color = meta
    return f'<span class="badge" style="background:{color}">{label}</span>'


def _track_chip(track: str) -> str:
    colors = {"note": "#455a64", "chord": "#1565c0", "lyric": "#2e7d32"}
    labels = {"note": "음표", "chord": "코드", "lyric": "가사"}
    c = colors.get(track, "#555")
    lbl = labels.get(track, track)
    return f'<span class="chip" style="background:{c}">{lbl}</span>'


def save_html(result: CompareResult, output_path: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf_name  = Path(result.pdf_xml).name
    orig_name = Path(result.orig_xml).name

    # 행 데이터 생성
    rows_html = []
    for i, d in enumerate(sorted(
        result.discrepancies,
        key=lambda x: (_SEV_ORDER.get(_KIND_META.get(x.kind, ("","low","",""))[1], 9), x.measure, x.offset)
    )):
        meta  = _KIND_META.get(d.kind, (d.track, "low", d.kind, "#777"))
        track = meta[0]
        sev   = meta[1]
        pos   = f"{d.measure}마디" if d.kind == "measure_miss" else f"{d.measure}마디 {d.offset:.2f}박"

        row = (
            f'<tr class="row" data-id="{i}" data-track="{track}" data-sev="{sev}" data-status="pending">'
            f'<td class="col-pos">{pos}</td>'
            f'<td class="col-type">{_track_chip(track)} {_badge(d.kind)}</td>'
            f'<td class="col-msg">{d.message}</td>'
            f'<td class="col-action">'
            f'<button class="btn-action" onclick="setStatus({i},\'ignore\')">무시</button>'
            f'<button class="btn-action" onclick="setStatus({i},\'done\')">수정완료</button>'
            f'<button class="btn-action" onclick="setStatus({i},\'later\')">재확인</button>'
            f'</td>'
            f'</tr>'
        )
        rows_html.append(row)

    table_body = "\n".join(rows_html) if rows_html else (
        '<tr><td colspan="4" style="text-align:center;padding:24px;color:#2e7d32">'
        '불일치 없음 - 두 악보가 완벽하게 일치합니다!</td></tr>'
    )

    total      = len(result.discrepancies)
    note_total = result.note_total      # 음표 트랙 전체 (measure_miss 포함)
    note_err   = result.note_errors
    missing    = result.missing_count
    noise      = result.noise_count
    chord_err  = result.chord_errors
    lyric_err  = result.lyric_errors

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>악보 검수 결과 - {pdf_name}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
       background: #f0f2f5; color: #222; font-size: 14px; }}
header {{ background: #1a237e; color: #fff; padding: 20px 32px; }}
header h1 {{ font-size: 1.4em; font-weight: 700; }}
header .sub {{ font-size: .82em; opacity: .75; margin-top: 4px; }}
.container {{ max-width: 1200px; margin: 24px auto; padding: 0 16px; }}

/* 요약 카드 */
.cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
.card {{ flex: 1; min-width: 140px; background: #fff; border-radius: 10px;
         padding: 16px 20px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
.card .num {{ font-size: 2em; font-weight: 700; line-height: 1; }}
.card .lbl {{ font-size: .78em; color: #666; margin-top: 4px; }}
.card.red   .num {{ color: #e53935; }}
.card.orange .num {{ color: #fb8c00; }}
.card.gray  .num {{ color: #9e9e9e; }}
.card.blue  .num {{ color: #1565c0; }}
.card.green .num {{ color: #2e7d32; }}

/* 파일 정보 */
.file-info {{ background: #fff; border-radius: 10px; padding: 14px 20px;
             margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
             font-size: .88em; color: #444; }}
.file-info span {{ font-weight: 600; color: #222; }}

/* 탭 필터 */
.tabs {{ display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }}
.tab {{ padding: 7px 18px; border-radius: 20px; border: 1.5px solid #ccc;
        background: #fff; cursor: pointer; font-size: .85em; font-weight: 600;
        transition: all .15s; }}
.tab.active {{ background: #1a237e; color: #fff; border-color: #1a237e; }}
.tab:hover:not(.active) {{ border-color: #1a237e; color: #1a237e; }}

/* 심각도 토글 */
.sev-filter {{ display: flex; gap: 8px; margin-bottom: 16px; align-items: center; }}
.sev-filter label {{ font-size: .82em; color: #555; }}
.sev-btn {{ padding: 4px 14px; border-radius: 12px; border: 1.5px solid #ccc;
           background: #fff; cursor: pointer; font-size: .8em; }}
.sev-btn.active {{ color: #fff; border-color: transparent; }}
.sev-btn[data-sev="high"].active   {{ background: #e53935; }}
.sev-btn[data-sev="medium"].active {{ background: #fb8c00; }}
.sev-btn[data-sev="low"].active    {{ background: #9e9e9e; }}

/* 상태 필터 */
.status-filter {{ display: flex; gap: 8px; margin-bottom: 16px; align-items: center; }}
.status-filter label {{ font-size: .82em; color: #555; }}
.st-btn {{ padding: 4px 12px; border-radius: 12px; border: 1.5px solid #ccc;
          background: #fff; cursor: pointer; font-size: .8em; }}
.st-btn.active {{ background: #1a237e; color: #fff; border-color: #1a237e; }}

/* 테이블 */
.tbl-wrap {{ background: #fff; border-radius: 10px; overflow: hidden;
             box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
table {{ border-collapse: collapse; width: 100%; }}
thead th {{ background: #283593; color: #fff; padding: 11px 14px;
            text-align: left; font-size: .85em; font-weight: 600; }}
tbody td {{ padding: 9px 14px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }}
tbody tr:last-child td {{ border-bottom: none; }}
tbody tr:hover td {{ background: #f5f7ff; }}

/* 상태별 행 스타일 */
tr.status-ignore {{ opacity: .35; }}
tr.status-done   {{ background: #f1f8e9 !important; }}
tr.status-later  {{ background: #fff8e1 !important; }}

/* 배지 / 칩 */
.badge {{ display: inline-block; font-size: .75em; padding: 2px 8px;
          border-radius: 4px; color: #fff; white-space: nowrap; }}
.chip  {{ display: inline-block; font-size: .72em; padding: 2px 7px;
          border-radius: 10px; color: #fff; white-space: nowrap; margin-right: 4px; }}

/* 액션 버튼 */
.btn-action {{ padding: 3px 10px; border-radius: 4px; border: 1px solid #ccc;
               background: #fafafa; cursor: pointer; font-size: .78em;
               margin-right: 4px; transition: all .12s; white-space: nowrap; }}
.btn-action:hover {{ background: #1a237e; color: #fff; border-color: #1a237e; }}
.col-pos    {{ width: 110px; color: #555; font-size: .85em; white-space: nowrap; }}
.col-type   {{ width: 200px; }}
.col-msg    {{ }}
.col-action {{ width: 200px; white-space: nowrap; }}

/* 카운터 */
.counter {{ float: right; font-size: .82em; color: #888; padding-top: 6px; }}
#visible-count {{ font-weight: 700; color: #1a237e; }}
</style>
</head>
<body>

<header>
  <h1>악보 검수 결과 리포트</h1>
  <div class="sub">생성: {now} &nbsp;|&nbsp; 총 {total}건 발견</div>
</header>

<div class="container">

  <!-- 파일 정보 -->
  <div class="file-info">
    <b>PDF 추출본:</b> <span>{result.pdf_xml}</span><br>
    <b>피날레 원본:</b> <span>{result.orig_xml}</span>&nbsp;&nbsp;
    <b>총 마디:</b> <span>{result.total_measures}마디</span>
  </div>

  <!-- 요약 카드 -->
  <div class="cards">
    <div class="card red">
      <div class="num">{note_err}</div>
      <div class="lbl">음표 오류<br>(누락 {missing} / 노이즈 {noise})</div>
    </div>
    <div class="card blue">
      <div class="num">{chord_err}</div>
      <div class="lbl">코드 기호 오류</div>
    </div>
    <div class="card green">
      <div class="num">{lyric_err}</div>
      <div class="lbl">가사 오류</div>
    </div>
    <div class="card orange">
      <div class="num">{total}</div>
      <div class="lbl">전체 불일치</div>
    </div>
  </div>

  <!-- 탭 필터 (트랙별) -->
  <div class="tabs">
    <button class="tab active" onclick="filterTrack('all', this)">전체 ({total})</button>
    <button class="tab" onclick="filterTrack('note', this)">음표 ({note_total})</button>
    <button class="tab" onclick="filterTrack('chord', this)">코드 ({chord_err})</button>
    <button class="tab" onclick="filterTrack('lyric', this)">가사 ({lyric_err})</button>
  </div>

  <!-- 심각도 필터 -->
  <div class="sev-filter">
    <label>심각도:</label>
    <button class="sev-btn active" data-sev="high"   onclick="toggleSev('high',   this)">높음</button>
    <button class="sev-btn active" data-sev="medium" onclick="toggleSev('medium', this)">보통</button>
    <button class="sev-btn active" data-sev="low"    onclick="toggleSev('low',    this)">낮음(OMR 노이즈 포함)</button>
  </div>

  <!-- 검수 상태 필터 -->
  <div class="status-filter">
    <label>검수 상태:</label>
    <button class="st-btn active" data-st="pending" onclick="toggleStatus('pending', this)">미검토</button>
    <button class="st-btn active" data-st="later"   onclick="toggleStatus('later',   this)">재확인</button>
    <button class="st-btn active" data-st="done"    onclick="toggleStatus('done',    this)">수정완료</button>
    <button class="st-btn active" data-st="ignore"  onclick="toggleStatus('ignore',  this)">무시됨</button>
    &nbsp;
    <button class="st-btn" style="border-color:#e53935;color:#e53935"
            onclick="resetAll()">전체 초기화</button>
  </div>

  <!-- 테이블 -->
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>위치</th>
          <th>유형</th>
          <th>내용 <span class="counter">표시: <b id="visible-count">{total}</b>건</span></th>
          <th>검수 액션</th>
        </tr>
      </thead>
      <tbody id="tbody">
{table_body}
      </tbody>
    </table>
  </div>

</div><!-- /container -->

<script>
// ── 저장 키 ──────────────────────────────────────────────────────────
const STORE_KEY = 'review_{Path(result.pdf_xml).stem}';

// ── 필터 상태 ─────────────────────────────────────────────────────────
let activeTrack = 'all';
let activeSev   = new Set(['high', 'medium', 'low']);
let activeSt    = new Set(['pending', 'later', 'done', 'ignore']);

// ── 검수 상태 로드/저장 ───────────────────────────────────────────────
function loadStatuses() {{
  try {{ return JSON.parse(localStorage.getItem(STORE_KEY) || '{{}}'); }}
  catch {{ return {{}}; }}
}}
function saveStatuses(obj) {{
  localStorage.setItem(STORE_KEY, JSON.stringify(obj));
}}
function setStatus(id, status) {{
  const statuses = loadStatuses();
  statuses[id] = status;
  saveStatuses(statuses);
  const row = document.querySelector(`tr[data-id="${{id}}"]`);
  if (row) {{
    row.dataset.status = status;
    row.className = 'row status-' + status;
  }}
  applyFilters();
}}
function resetAll() {{
  if (!confirm('모든 검수 상태를 초기화할까요?')) return;
  localStorage.removeItem(STORE_KEY);
  document.querySelectorAll('tr.row').forEach(r => {{
    r.dataset.status = 'pending';
    r.className = 'row';
  }});
  applyFilters();
}}

// 페이지 로드 시 저장된 상태 복원
(function initStatuses() {{
  const statuses = loadStatuses();
  document.querySelectorAll('tr.row').forEach(r => {{
    const id = r.dataset.id;
    if (statuses[id]) {{
      r.dataset.status = statuses[id];
      r.className = 'row status-' + statuses[id];
    }}
  }});
  applyFilters();
}})();

// ── 트랙 필터 ─────────────────────────────────────────────────────────
function filterTrack(track, btn) {{
  activeTrack = track;
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}}

// ── 심각도 토글 ───────────────────────────────────────────────────────
function toggleSev(sev, btn) {{
  if (activeSev.has(sev)) {{ activeSev.delete(sev); btn.classList.remove('active'); }}
  else                     {{ activeSev.add(sev);    btn.classList.add('active');    }}
  applyFilters();
}}

// ── 검수 상태 토글 ────────────────────────────────────────────────────
function toggleStatus(st, btn) {{
  if (activeSt.has(st)) {{ activeSt.delete(st); btn.classList.remove('active'); }}
  else                   {{ activeSt.add(st);    btn.classList.add('active');    }}
  applyFilters();
}}

// ── 필터 적용 ─────────────────────────────────────────────────────────
function applyFilters() {{
  let visible = 0;
  document.querySelectorAll('tr.row').forEach(r => {{
    const track  = r.dataset.track;
    const sev    = r.dataset.sev;
    const status = r.dataset.status || 'pending';
    const show   = (activeTrack === 'all' || activeTrack === track)
                && activeSev.has(sev)
                && activeSt.has(status);
    r.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  document.getElementById('visible-count').textContent = visible;
}}
</script>
</body>
</html>
"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"\n[리포트 저장] {output_path}")
