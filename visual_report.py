"""
단(System) 단위 시각 비교 HTML 뷰어 생성 모듈.

Space = 통과 / Enter = 수정 필요 / ←→ = 이전·다음 / R = 초기화
결과는 JSON으로 다운로드 가능.
마디별 음표 비교 테이블 포함 (sys_measure_data 전달 시).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from system_slicer import PairedSystem, SlicedScore


def save_visual_html(
    pairs:           list[PairedSystem],
    textbook:        SlicedScore,
    finale:          SlicedScore,
    output_path:     str,
    title:           str  = "악보 단(System) 시각 비교",
    sys_measure_data: list | None = None,
    xml_total:       int  = 0,
    mps_total:       int  = 0,
    is_acc_xml:      bool = False,
) -> None:
    """
    sys_measure_data: 단(System)별 마디 리스트.
      각 마디 = {
        'num': int,           마디 번호 (1-based)
        'cls': str,           CSS 클래스 (det-ok, det-err, ...)
        'chord': str,         코드 문자열 (여러 개면 공백 구분)
        'lyric': str,         가사 문자열
        'xml_notes': list,    XML 음높이 리스트
        'det_notes': list,    감지 음높이 리스트
        'xml_hollow': list,   XML hollow 여부 리스트
        'det_hollow': list,   감지 hollow 여부 리스트
        'extra': list,        원본에 없는 추가 감지 음
        'missing': list,      감지 못한 누락 음
        'tie_issue': str,     '' | 'pdf_extra' | 'xml_extra'
        'excluded': bool,     첫 마디 제외 여부
      }
    """
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stem = Path(output_path).stem

    cards = []
    for i, p in enumerate(pairs):
        sys_measures = sys_measure_data[i] if sys_measure_data and i < len(sys_measure_data) else []
        cards.append({
            "n":        p.abs_system,
            "tb":       p.textbook.base64_src if p.textbook else "",
            "fn":       p.finale.base64_src   if p.finale   else "",
            "tb_pos":   (f"p{p.textbook.source_page + 1} s{p.textbook.staff_idx}")
                        if p.textbook else "없음",
            "fn_pos":   (f"p{p.finale.source_page + 1} s{p.finale.staff_idx}")
                        if p.finale else "없음",
            "measures": sys_measures,
        })

    warn_items = (
        [f"<li>[교과서] {w}</li>" for w in textbook.warnings]
        + [f"<li>[Finale] {w}</li>" for w in finale.warnings]
    )
    warn_html = ""
    if warn_items:
        warn_html = (
            f"<details class='warn-box'>"
            f"<summary>감지 경고 ({len(warn_items)}건)</summary>"
            f"<ul>{''.join(warn_items)}</ul></details>"
        )

    # 악보 불일치 배너
    mismatch_banner = ""
    if xml_total and mps_total and xml_total != mps_total:
        mismatch_banner = (
            f"<div class='banner banner-warn'>"
            f"마디 수 불일치: PDF={mps_total} / XML={xml_total} — "
            f"비교 결과 신뢰도 낮을 수 있음</div>"
        )

    acc_banner = ""
    if is_acc_xml:
        acc_banner = (
            "<div class='banner banner-acc'>"
            "반주 전용 XML — 이미지에 멜로디가 포함되어 있어 "
            "주황(superset)·노랑(imbalance)은 정상 패턴</div>"
        )

    has_measures = bool(sys_measure_data)
    cards_js = json.dumps(cards, ensure_ascii=False)
    total    = len(cards)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
       background: #f0f2f5; color: #222; }}

header {{ background: #1a237e; color: #fff; padding: 14px 28px;
         display: flex; justify-content: space-between; align-items: center; }}
header h1 {{ font-size: 1.15em; font-weight: 700; }}
header .sub {{ font-size: .78em; opacity: .75; margin-top: 3px; }}
.toolbar {{ display: flex; gap: 10px; align-items: center; font-size: .88em; flex-shrink: 0; }}
.toolbar button {{ padding: 6px 16px; border: none; border-radius: 6px;
                   background: rgba(255,255,255,.15); color: #fff;
                   cursor: pointer; font-weight: 600; transition: background .1s; }}
.toolbar button:hover {{ background: rgba(255,255,255,.28); }}
.toolbar .counter {{ font-size: .95em; min-width: 70px; text-align: center; }}
.toolbar input[type=range] {{ width: 80px; accent-color: #ffc107; cursor: pointer; }}
.toolbar label {{ font-size: .78em; white-space: nowrap; }}

.prog-wrap {{ background: #283593; height: 4px; }}
.prog-bar  {{ background: #ffc107; height: 100%; width: 0%; transition: width .12s; }}

.viewer {{ max-width: 1360px; margin: 20px auto; padding: 0 16px; }}

.banner {{ padding: 10px 18px; border-radius: 8px; margin-bottom: 14px;
          font-size: .85em; font-weight: 600; }}
.banner-warn {{ background: #fff3e0; border-left: 4px solid #ff9800; color: #e65100; }}
.banner-acc  {{ background: #e8f5e9; border-left: 4px solid #4caf50; color: #1b5e20; }}

.card {{ background: #fff; border-radius: 12px; padding: 22px 28px;
         box-shadow: 0 2px 12px rgba(0,0,0,.08); }}
.card-header {{ display: flex; justify-content: space-between; align-items: center;
               margin-bottom: 16px; flex-wrap: wrap; gap: 8px; }}
.sys-num {{ font-size: 1.8em; font-weight: 700; color: #1a237e; }}
.pos-info {{ font-size: .8em; color: #888; }}
.verdict-badge {{ font-size: .82em; padding: 4px 14px; border-radius: 20px; font-weight: 600; }}
.verdict-badge.pass    {{ background: #c8e6c9; color: #1b5e20; }}
.verdict-badge.fail    {{ background: #ffcdd2; color: #b71c1c; }}
.verdict-badge.pending {{ background: #eceff1; color: #546e7a; }}

.panel {{ border: 1.5px solid #e0e0e0; border-radius: 8px; overflow: hidden; margin-bottom: 14px; }}
.panel-label {{ background: #283593; color: #fff; padding: 6px 16px;
               font-size: .8em; font-weight: 600;
               display: flex; justify-content: space-between; }}
.panel img {{ display: block; width: 100%;
             height: var(--panel-h, 280px); object-fit: contain;
             background: #fafafa; padding: 6px 0; }}
.panel .missing {{ padding: 40px; text-align: center; background: #ffebee;
                  color: #c62828; font-style: italic; font-size: .9em; }}

/* 마디 비교 테이블 */
.m-table-wrap {{ overflow-x: auto; margin-bottom: 14px; }}
.m-table {{ border-collapse: collapse; font-size: .78em; width: max-content; }}
.m-table td, .m-table th {{
  border: 1px solid #ddd; padding: 3px 6px; vertical-align: top;
  white-space: nowrap; text-align: center; min-width: 52px;
}}
.m-table th {{ background: #37474f; color: #fff; font-size: .75em; padding: 4px 6px; }}
.m-row-label {{ background: #eceff1 !important; color: #37474f;
               font-weight: 600; font-size: .72em; text-align: right !important; }}

/* 마디 배경 */
.det-ok        {{ background: #e8f5e9; }}
.det-err       {{ background: #ffebee; }}
.det-superset  {{ background: #fff3e0; }}
.det-imbalance {{ background: #fffde7; }}
.det-rhythm    {{ background: #f3e5f5; }}
.det-miss      {{ background: #f5f5f5; }}
.det-excl      {{ background: #fafafa; opacity: .55; }}

/* 음표 강조 */
.note-extra   {{ color: #c62828; font-weight: 700; }}
.note-missing {{ color: #1565c0; font-style: italic; }}
.note-ok      {{ color: #555; }}
.note-hollow  {{ text-decoration: underline; }}

.tie-pdf {{ font-size: .7em; color: #e53935; font-weight: 700; }}
.tie-xml {{ font-size: .7em; color: #1565c0; font-weight: 700; }}

.hints {{ text-align: center; color: #888; font-size: .82em; margin-top: 6px; }}
.hints kbd {{ background: #eeeeee; padding: 2px 8px; border-radius: 4px;
             font-family: monospace; margin: 0 2px; }}

/* 범례 */
.legend {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; font-size: .74em; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; }}
.legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid #ccc; flex-shrink: 0; }}

/* 요약 화면 */
.summary {{ display: none; text-align: center; padding: 64px 20px; }}
.summary h2 {{ font-size: 2em; color: #1a237e; margin-bottom: 20px; }}
.summary .stats {{ display: flex; justify-content: center; gap: 48px; margin: 24px 0; }}
.summary .stat .n {{ font-size: 3.2em; font-weight: 700; line-height: 1; }}
.summary .stat .lbl {{ font-size: .82em; color: #777; margin-top: 4px; }}
.summary .stat.pass .n {{ color: #2e7d32; }}
.summary .stat.fail .n {{ color: #e53935; }}
.summary .stat.pend .n {{ color: #546e7a; }}
.summary .btns {{ margin-top: 28px; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }}
.summary .btns button {{ padding: 12px 28px; border: none; border-radius: 8px;
                         font-size: 1em; cursor: pointer; font-weight: 600; }}
.summary .btns .btn-dl {{ background: #1a237e; color: #fff; }}
.summary .btns .btn-back {{ background: #78909c; color: #fff; }}

.warn-box {{ background: #fff8e1; border-left: 4px solid #ffc107;
            padding: 12px 16px; border-radius: 6px; margin-top: 20px;
            font-size: .82em; }}
.warn-box summary {{ cursor: pointer; font-weight: 600; color: #555; }}
.warn-box ul {{ margin-top: 8px; padding-left: 18px; color: #666; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>{title}</h1>
    <div class="sub">생성: {now} &nbsp;|&nbsp; 총 {total}단</div>
  </div>
  <div class="toolbar">
    <button onclick="prev()">◀ 이전</button>
    <span class="counter" id="counter">1 / {total}</span>
    <button onclick="next()">다음 ▶</button>
    <label>높이 <input type="range" id="hslider" min="120" max="600" value="280"
           oninput="setH(this.value)"> <span id="hlabel">280px</span></label>
    <button onclick="exportJson()">JSON 저장</button>
  </div>
</header>

<div class="prog-wrap"><div class="prog-bar" id="bar"></div></div>

<div class="viewer">

  {mismatch_banner}
  {acc_banner}

  <!-- 범례 -->
  {'<div class="legend">'
    '<div class="legend-item"><div class="legend-swatch det-ok" style="border-color:#81c784"></div>일치</div>'
    '<div class="legend-item"><div class="legend-swatch det-err" style="border-color:#e57373"></div>오류</div>'
    '<div class="legend-item"><div class="legend-swatch det-superset" style="border-color:#ffb74d"></div>XML⊆감지(반주)</div>'
    '<div class="legend-item"><div class="legend-swatch det-imbalance" style="border-color:#fff176"></div>개수 불균형</div>'
    '<div class="legend-item"><div class="legend-swatch det-rhythm" style="border-color:#ce93d8"></div>음가 불일치</div>'
    '<div class="legend-item"><div class="legend-swatch det-miss" style="border-color:#bdbdbd"></div>감지 실패</div>'
    '<div class="legend-item"><div class="legend-swatch det-excl"></div>제외(첫마디)</div>'
    '</div>'
  if has_measures else ''}

  <!-- 카드 뷰 -->
  <div class="card" id="card">
    <div class="card-header">
      <div class="sys-num">제 <span id="sys-n">1</span> 단</div>
      <div class="pos-info" id="pos"></div>
      <div class="verdict-badge pending" id="badge">미검토</div>
    </div>

    <div class="panel">
      <div class="panel-label">
        <span>교과서 원본 (PDF)</span>
        <span id="tb-pos"></span>
      </div>
      <div id="tb-slot"></div>
    </div>

    <div class="panel">
      <div class="panel-label">
        <span>Finale (verovio 렌더)</span>
        <span id="fn-pos"></span>
      </div>
      <div id="fn-slot"></div>
    </div>

    <!-- 마디 비교 테이블 -->
    <div id="m-table-area"></div>

    <div class="hints">
      <kbd>Space</kbd> 통과 &nbsp;
      <kbd>Enter</kbd> 수정필요 &nbsp;
      <kbd>←</kbd> 이전 &nbsp;
      <kbd>→</kbd> 다음 &nbsp;
      <kbd>R</kbd> 전체 초기화
    </div>
  </div>

  <!-- 완료 요약 -->
  <div class="summary" id="summary">
    <h2>검수 완료</h2>
    <div class="stats">
      <div class="stat pass"><div class="n" id="n-pass">0</div><div class="lbl">통과</div></div>
      <div class="stat fail"><div class="n" id="n-fail">0</div><div class="lbl">수정필요</div></div>
      <div class="stat pend"><div class="n" id="n-pend">0</div><div class="lbl">미검토</div></div>
    </div>
    <div class="btns">
      <button class="btn-dl"   onclick="exportJson()">결과 JSON 다운로드</button>
      <button class="btn-back" onclick="restart()">처음으로 돌아가기</button>
    </div>
  </div>

  {warn_html}
</div>

<script>
const CARDS = {cards_js};
const KEY   = 'visual_{stem}';
const TOTAL = {total};

let idx      = 0;
let verdicts = load();

function setH(v) {{
  document.documentElement.style.setProperty('--panel-h', v + 'px');
  document.getElementById('hlabel').textContent = v + 'px';
}}

function load() {{
  try {{ return JSON.parse(localStorage.getItem(KEY) || '{{}}'); }}
  catch {{ return {{}}; }}
}}
function save() {{ localStorage.setItem(KEY, JSON.stringify(verdicts)); }}

function noteSpan(n, isExtra, isMissing, isHollow) {{
  const cls = isExtra ? 'note-extra' : isMissing ? 'note-missing' : 'note-ok';
  const hcls = isHollow ? ' note-hollow' : '';
  return `<span class="${{cls}}${{hcls}}">${{n}}</span>`;
}}

function renderMeasureTable(measures) {{
  if (!measures || measures.length === 0) return '';

  let html = '<div class="m-table-wrap"><table class="m-table"><thead><tr>';
  html += '<th class="m-row-label"></th>';
  for (const m of measures) {{
    const excl = m.excluded ? ' style="opacity:.5"' : '';
    html += `<th class="${{m.cls}}"${{excl}}>마디 ${{m.num}}</th>`;
  }}
  html += '</tr></thead><tbody>';

  // 코드 행
  const hasChord = measures.some(m => m.chord);
  if (hasChord) {{
    html += '<tr><td class="m-row-label">코드</td>';
    for (const m of measures) {{
      html += `<td class="${{m.cls}}">${{m.chord || ''}}</td>`;
    }}
    html += '</tr>';
  }}

  // 가사 행
  const hasLyric = measures.some(m => m.lyric);
  if (hasLyric) {{
    html += '<tr><td class="m-row-label">가사</td>';
    for (const m of measures) {{
      html += `<td class="${{m.cls}}">${{m.lyric || ''}}</td>`;
    }}
    html += '</tr>';
  }}

  // 원본 음표 행 (XML)
  html += '<tr><td class="m-row-label">원본</td>';
  for (const m of measures) {{
    if (m.excluded) {{
      html += `<td class="${{m.cls}}"><span class="note-ok">—</span></td>`;
    }} else {{
      const xmlNotes = (m.xml_notes || []).map((n, i) => {{
        const isMissing = (m.missing || []).includes(n);
        const isHollow  = m.xml_hollow && m.xml_hollow[i];
        return noteSpan(n, false, isMissing, isHollow);
      }});
      html += `<td class="${{m.cls}}">${{xmlNotes.join(' ') || '—'}}</td>`;
    }}
  }}
  html += '</tr>';

  // 감지 음표 행 (Det)
  html += '<tr><td class="m-row-label">감지</td>';
  for (const m of measures) {{
    if (m.excluded) {{
      html += `<td class="${{m.cls}}"><span class="note-ok">제외</span></td>`;
    }} else {{
      const detNotes = (m.det_notes || []).map((n, i) => {{
        const isExtra  = (m.extra || []).includes(n);
        const isHollow = m.det_hollow && m.det_hollow[i];
        return noteSpan(n, isExtra, false, isHollow);
      }});
      let tieHtml = '';
      if (m.tie_issue === 'pdf_extra') tieHtml = ' <span class="tie-pdf">⌒+</span>';
      if (m.tie_issue === 'xml_extra') tieHtml = ' <span class="tie-xml">⌒?</span>';
      html += `<td class="${{m.cls}}">${{detNotes.join(' ') || (m.cls === 'det-miss' ? '<em style="color:#999">없음</em>' : '—')}}${{tieHtml}}</td>`;
    }}
  }}
  html += '</tr>';

  html += '</tbody></table></div>';
  return html;
}}

function render() {{
  if (idx >= CARDS.length) {{ showSummary(); return; }}
  document.getElementById('card').style.display    = 'block';
  document.getElementById('summary').style.display = 'none';

  const c = CARDS[idx];
  document.getElementById('sys-n').textContent   = c.n;
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + TOTAL;
  document.getElementById('pos').textContent     = `교과서 ${{c.tb_pos}} | Finale ${{c.fn_pos}}`;
  document.getElementById('tb-pos').textContent  = c.tb_pos;
  document.getElementById('fn-pos').textContent  = c.fn_pos;

  document.getElementById('tb-slot').innerHTML = c.tb
    ? `<img src="${{c.tb}}" alt="교과서 ${{c.n}}단">`
    : `<div class="missing">교과서 단 ${{c.n}} — 오선 감지 실패</div>`;

  document.getElementById('fn-slot').innerHTML = c.fn
    ? `<img src="${{c.fn}}" alt="Finale ${{c.n}}단">`
    : `<div class="missing">Finale 단 ${{c.n}} — 오선 감지 실패</div>`;

  document.getElementById('m-table-area').innerHTML = renderMeasureTable(c.measures);

  const v = verdicts[c.n] || 'pending';
  const badge = document.getElementById('badge');
  badge.className = 'verdict-badge ' + v;
  badge.textContent = {{pass:'통과', fail:'수정필요', pending:'미검토'}}[v];

  document.getElementById('bar').style.width = ((idx + 1) / TOTAL * 100) + '%';
}}

function setVerdict(v) {{
  if (idx >= CARDS.length) return;
  verdicts[CARDS[idx].n] = v;
  save();
  next();
}}
function next() {{ if (idx < CARDS.length) {{ idx++; render(); }} }}
function prev() {{ if (idx > 0)             {{ idx--; render(); }} }}
function restart() {{ idx = 0; render(); }}

function showSummary() {{
  document.getElementById('card').style.display    = 'none';
  document.getElementById('summary').style.display = 'block';
  let p = 0, f = 0, pe = 0;
  for (const c of CARDS) {{
    const v = verdicts[c.n] || 'pending';
    if (v === 'pass') p++;
    else if (v === 'fail') f++;
    else pe++;
  }}
  document.getElementById('n-pass').textContent = p;
  document.getElementById('n-fail').textContent = f;
  document.getElementById('n-pend').textContent = pe;
  document.getElementById('counter').textContent = TOTAL + ' / ' + TOTAL;
  document.getElementById('bar').style.width = '100%';
}}

function exportJson() {{
  const out = {{
    generated: '{now}',
    total: TOTAL,
    results: CARDS.map(c => ({{ system: c.n, verdict: verdicts[c.n] || 'pending' }})),
  }};
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type: 'application/json'}});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  a.download = 'visual_{stem}_results.json';
  a.click();
  URL.revokeObjectURL(url);
}}

document.addEventListener('keydown', e => {{
  if (e.key === ' ')           {{ e.preventDefault(); setVerdict('pass'); }}
  else if (e.key === 'Enter')  {{ e.preventDefault(); setVerdict('fail'); }}
  else if (e.key === 'ArrowLeft')  prev();
  else if (e.key === 'ArrowRight') next();
  else if (e.key.toLowerCase() === 'r') {{
    if (confirm('모든 판정을 초기화할까요?')) {{ verdicts = {{}}; save(); render(); }}
  }}
}});

render();
</script>
</body>
</html>
"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"\n[시각 비교 뷰어 저장] {output_path}")


def save_visual_index_html(
    entries: list[dict],
    output_path: str,
    title: str = "악보 시각 비교 인덱스",
) -> None:
    """
    batch-visual 결과 인덱스 HTML을 생성합니다.

    entries 항목:
      stem    str   — 파일 스템(곡명)
      html    str|None — 생성된 HTML 파일명 (None이면 오류/스킵)
      systems int|str  — 단 수
      error   str|None — 오류 메시지 (None이면 성공/스킵)
    """
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(entries)
    ok    = sum(1 for e in entries if e["html"] and not e["error"])
    err   = sum(1 for e in entries if e["error"] and e["error"] != "XML 없음")
    skip  = sum(1 for e in entries if e["error"] == "XML 없음")

    rows = []
    for i, e in enumerate(entries, 1):
        stem    = e["stem"]
        html    = e.get("html")
        systems = e.get("systems", 0)
        error   = e.get("error")

        if error == "XML 없음":
            status = '<span class="st-skip">XML 없음</span>'
            link   = "—"
        elif error:
            status = f'<span class="st-err" title="{error}">오류</span>'
            link   = "—"
        elif html:
            status = '<span class="st-ok">완료</span>'
            link   = f'<a href="{html}" target="_blank">열기 ↗</a>'
        else:
            status = '<span class="st-skip">스킵</span>'
            link   = "—"

        sys_disp = str(systems) if systems else "—"
        rows.append(
            f'<tr data-stem="{stem.lower()}">'
            f'<td class="td-num">{i}</td>'
            f'<td class="td-name">{stem}</td>'
            f'<td class="td-sys">{sys_disp}</td>'
            f'<td>{status}</td>'
            f'<td>{link}</td>'
            f'</tr>'
        )

    rows_html = "\n".join(rows)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
       background: #f0f2f5; color: #222; }}
header {{ background: #1a237e; color: #fff; padding: 14px 28px; }}
header h1 {{ font-size: 1.2em; font-weight: 700; }}
header .sub {{ font-size: .78em; opacity: .75; margin-top: 3px; }}

.stats {{ display: flex; gap: 28px; padding: 18px 28px;
         background: #fff; border-bottom: 1px solid #e0e0e0; }}
.stat-item {{ text-align: center; }}
.stat-item .n {{ font-size: 2em; font-weight: 700; line-height: 1.1; }}
.stat-item .l {{ font-size: .75em; color: #777; }}
.n-ok   {{ color: #2e7d32; }}
.n-err  {{ color: #c62828; }}
.n-skip {{ color: #9e9e9e; }}
.n-total {{ color: #1a237e; }}

.filter-wrap {{ padding: 14px 28px; background: #fff; border-bottom: 1px solid #e0e0e0; }}
.filter-wrap input {{ width: 100%; max-width: 480px; padding: 8px 14px;
                      border: 1.5px solid #c5cae9; border-radius: 6px;
                      font-size: .9em; outline: none; }}
.filter-wrap input:focus {{ border-color: #3949ab; }}

.table-wrap {{ padding: 20px 28px; }}
table {{ border-collapse: collapse; width: 100%; background: #fff;
        border-radius: 10px; overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,.07); }}
th {{ background: #283593; color: #fff; padding: 10px 14px;
     font-size: .8em; text-align: left; }}
td {{ padding: 8px 14px; border-bottom: 1px solid #f0f0f0;
     font-size: .82em; vertical-align: middle; }}
tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background: #f5f7ff; }}
tr[style*="display:none"] {{ display: none !important; }}

.td-num  {{ color: #9e9e9e; width: 48px; text-align: right; }}
.td-name {{ font-weight: 500; }}
.td-sys  {{ width: 60px; text-align: center; color: #555; }}

.st-ok   {{ color: #2e7d32; font-weight: 600; }}
.st-err  {{ color: #c62828; font-weight: 600; cursor: help; }}
.st-skip {{ color: #9e9e9e; }}

a {{ color: #1a237e; font-weight: 600; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>

<header>
  <h1>{title}</h1>
  <div class="sub">생성: {now} &nbsp;|&nbsp; 총 {total}곡</div>
</header>

<div class="stats">
  <div class="stat-item"><div class="n n-total">{total}</div><div class="l">전체</div></div>
  <div class="stat-item"><div class="n n-ok">{ok}</div><div class="l">완료</div></div>
  <div class="stat-item"><div class="n n-err">{err}</div><div class="l">오류</div></div>
  <div class="stat-item"><div class="n n-skip">{skip}</div><div class="l">XML 없음</div></div>
</div>

<div class="filter-wrap">
  <input type="text" id="filter" placeholder="곡명 검색…" oninput="filterRows(this.value)">
</div>

<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th class="td-num">#</th>
        <th>곡명</th>
        <th class="td-sys">단</th>
        <th style="width:80px">상태</th>
        <th style="width:80px">링크</th>
      </tr>
    </thead>
    <tbody id="tbody">
{rows_html}
    </tbody>
  </table>
</div>

<script>
function filterRows(q) {{
  q = q.toLowerCase();
  document.querySelectorAll('#tbody tr').forEach(tr => {{
    tr.style.display = tr.dataset.stem.includes(q) ? '' : 'none';
  }});
}}
</script>
</body>
</html>
"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"[인덱스 저장] {output_path}")
