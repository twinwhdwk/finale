"""
단(System) 단위 시각 비교 HTML 뷰어 생성 모듈.

Space = 통과 / Enter = 수정 필요 / ←→ = 이전·다음 / R = 초기화
결과는 JSON으로 다운로드 가능.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from system_slicer import PairedSystem, SlicedScore


def save_visual_html(
    pairs:       list[PairedSystem],
    textbook:    SlicedScore,
    finale:      SlicedScore,
    output_path: str,
    title:       str = "악보 단(System) 시각 비교",
) -> None:
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stem = Path(output_path).stem

    cards = [
        {
            "n":       p.abs_system,
            "tb":      p.textbook.base64_src if p.textbook else "",
            "fn":      p.finale.base64_src   if p.finale   else "",
            "tb_pos":  (f"p{p.textbook.source_page + 1} s{p.textbook.staff_idx}")
                       if p.textbook else "없음",
            "fn_pos":  (f"p{p.finale.source_page + 1} s{p.finale.staff_idx}")
                       if p.finale else "없음",
        }
        for p in pairs
    ]

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

.prog-wrap {{ background: #283593; height: 4px; }}
.prog-bar  {{ background: #ffc107; height: 100%; width: 0%; transition: width .12s; }}

.viewer {{ max-width: 1280px; margin: 20px auto; padding: 0 16px; }}

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
.panel img {{ display: block; width: 100%; background: #fafafa;
             padding: 10px 0; }}
.panel .missing {{ padding: 40px; text-align: center; background: #ffebee;
                  color: #c62828; font-style: italic; font-size: .9em; }}

.hints {{ text-align: center; color: #888; font-size: .82em; margin-top: 6px; }}
.hints kbd {{ background: #eeeeee; padding: 2px 8px; border-radius: 4px;
             font-family: monospace; margin: 0 2px; }}

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
    <button onclick="exportJson()">JSON 저장</button>
  </div>
</header>

<div class="prog-wrap"><div class="prog-bar" id="bar"></div></div>

<div class="viewer">

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
        <span>Finale (MuseScore 렌더)</span>
        <span id="fn-pos"></span>
      </div>
      <div id="fn-slot"></div>
    </div>

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

function load() {{
  try {{ return JSON.parse(localStorage.getItem(KEY) || '{{}}'); }}
  catch {{ return {{}}; }}
}}
function save() {{ localStorage.setItem(KEY, JSON.stringify(verdicts)); }}

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
