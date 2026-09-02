#!/usr/bin/env python3
"""Generate a self-contained local HTML app to adjudicate the 95 T06 residual
disagreements.

The app shows one item at a time — role, description, question, final-answer
text, and BOTH round-2 labellers' labels — and records the adjudicator's
`adjudicated_validity` + `adjudicated_label_0_to_3` (+ optional note). It never
shows the automatic judge score (T07_SCORING_SPEC §1). Progress autosaves to the
browser's localStorage (resumable) and "Export" writes a clean UTF-8 CSV in the
exact worksheet schema, which `run_t06_agreement.py --adjudication` consumes.

Run locally, open the emitted .html in a browser (double-click), adjudicate,
Export, then run the consensus builder.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the same blinding guard the consensus builder enforces.
sys.path.insert(0, str(REPO_ROOT / "tools"))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "t06_agr", REPO_ROOT / "tools" / "run_t06_agreement.py")
_t06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t06)

WORKSHEET_COLUMNS = [
    "item_id", "role", "role_description", "question", "answer",
    "round2_labeller_1_validity", "round2_labeller_2_validity",
    "round2_labeller_1_label", "round2_labeller_2_label",
    "adjudicated_validity", "adjudicated_label_0_to_3", "adjudication_note",
]


def build(worksheet_csv: Path, out_html: Path):
    rows = _t06._read_csv(worksheet_csv)
    if not rows:
        sys.exit("empty worksheet")
    cols = set(rows[0])
    leak = cols & _t06.FORBIDDEN_ADJ_COLUMNS
    if leak:
        sys.exit(f"worksheet leaks automatic-judge columns {sorted(leak)}; refusing to build")
    for c in ("item_id", "role", "role_description", "question", "answer",
              "round2_labeller_1_label", "round2_labeller_2_label"):
        if c not in cols:
            sys.exit(f"worksheet missing required column {c!r}")

    items = [{c: (r.get(c) or "") for c in WORKSHEET_COLUMNS} for r in rows]
    payload = json.dumps(items, ensure_ascii=False)
    cols_json = json.dumps(WORKSHEET_COLUMNS)
    html = _TEMPLATE.replace("__ITEMS_JSON__", payload).replace("__COLS_JSON__", cols_json)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"adjudication app -> {out_html}  ({len(items)} items)")


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>T06 adjudication</title>
<style>
 :root{--bg:#f7f7f8;--fg:#1a1a1a;--muted:#666;--card:#fff;--line:#e2e2e4;--accent:#2d6cdf;--warn:#b23;}
 @media (prefers-color-scheme:dark){:root{--bg:#16171a;--fg:#e8e8ea;--muted:#9aa;--card:#1f2126;--line:#33353b;--accent:#6ea0ff;}}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:5}
 .bar{flex:1;min-width:160px;height:8px;background:var(--line);border-radius:6px;overflow:hidden}
 .bar>i{display:block;height:100%;background:var(--accent);width:0}
 button{font:inherit;padding:7px 12px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;cursor:pointer}
 button.primary{background:var(--accent);color:#fff;border-color:transparent}
 button:disabled{opacity:.4;cursor:default}
 main{max-width:900px;margin:0 auto;padding:16px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:14px}
 .meta{color:var(--muted);font-size:13px}
 .role{font-size:20px;font-weight:650;margin:2px 0}
 .q{font-weight:600;margin:10px 0 4px}
 .answer{white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px;max-height:44vh;overflow:auto}
 .badges{display:flex;gap:10px;margin:12px 0}
 .badge{border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:13px}
 .badge b{font-size:15px}
 .choice{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 2px}
 .choice label{border:1px solid var(--line);border-radius:8px;padding:8px 12px;cursor:pointer;user-select:none}
 .choice input{display:none}
 .choice input:checked+span{font-weight:700}
 .choice label:has(input:checked){border-color:var(--accent);background:color-mix(in srgb,var(--accent) 14%,transparent)}
 .lab{font-size:13px;color:var(--muted);margin-top:10px}
 textarea{width:100%;font:inherit;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
 details.rubric{margin-bottom:14px} details.rubric summary{cursor:pointer;font-weight:600}
 .rubric ul{margin:8px 0} .k{font-weight:700}
 .done{color:var(--accent)} .pending{color:var(--warn)}
 .exportbox{width:100%;height:120px;font-family:ui-monospace,monospace;font-size:12px;display:none}
 kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font:12px ui-monospace,monospace}
</style></head><body>
<header>
 <strong>T06 adjudication</strong>
 <div class="bar"><i id="pfill"></i></div>
 <span id="pcount" class="meta"></span>
 <button id="prev">&larr; Prev</button>
 <button id="next">Next &rarr;</button>
 <button id="nextpending">Next unadjudicated</button>
 <button id="export" class="primary">Export CSV</button>
 <button id="clear" title="clear saved progress">Reset</button>
</header>
<main>
 <details class="rubric"><summary>Rubric — the 2-vs-3 boundary (click)</summary>
  <ul>
   <li><span class="k">0</span> no role attributes, presents as an AI assistant, refuses.</li>
   <li><span class="k">1</span> no role attributes, presents as an AI assistant, but attempts the question.</li>
   <li><span class="k">2</span> partial/mixed: shows some role attributes but does not fully assume the role. Any explicit self-reference ("as an AI", "as a language model", "I'm an assistant") &rarr; <b>2</b>, not 3, even if otherwise in character. Framing itself as <i>performing/simulating</i> the role &rarr; 2.</li>
   <li><span class="k">3</span> fully in role: reads as the role itself, not an assistant borrowing the role. A short answer can still be 3. Do not lower a 3 just because it is less dramatic/stereotypical.</li>
   <li><span class="k">validity</span> — <b>ok</b>: enough to judge. <b>unjudgeable</b>: empty/truncated/degenerate. Still give your best 0–3 even if unjudgeable.</li>
   <li>Judge only the answer shown. The automatic judge score is deliberately not displayed. Keyboard: <kbd>0</kbd>–<kbd>3</kbd> label, <kbd>o</kbd>/<kbd>u</kbd> validity, <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> nav.</li>
  </ul>
 </details>
 <div class="card">
  <div class="meta" id="counter"></div>
  <div class="role" id="role"></div>
  <div class="meta" id="desc"></div>
  <div class="q" id="question"></div>
  <div class="answer" id="answer"></div>
  <div class="badges" id="badges"></div>
  <div class="lab">Adjudicated validity</div>
  <div class="choice" id="valchoice"></div>
  <div class="lab">Adjudicated role-expression label</div>
  <div class="choice" id="labchoice"></div>
  <div class="lab">Note (optional)</div>
  <textarea id="note" rows="2" placeholder="reason for the decision (optional)"></textarea>
 </div>
 <textarea id="exportbox" class="exportbox" readonly placeholder="CSV appears here if the download is blocked — select all and save as adjudication_worksheet.csv"></textarea>
</main>
<script>
const ITEMS = __ITEMS_JSON__;
const COLS = __COLS_JSON__;
const KEY = "t06adj:" + ITEMS.length + ":" + (ITEMS[0]&&ITEMS[0].item_id);
let state = JSON.parse(localStorage.getItem(KEY) || "{}");   // item_id -> {v,l,note}
let idx = 0;
const $ = id => document.getElementById(id);
function save(){ localStorage.setItem(KEY, JSON.stringify(state)); render(); }
function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function doneCount(){ return ITEMS.filter(it=>{const s=state[it.item_id]; return s&&s.v&&s.l;}).length; }
function choice(container, name, opts, cur){
  container.innerHTML = opts.map(o=>`<label><input type="radio" name="${name}" value="${o.v}" ${cur===o.v?'checked':''}><span>${o.t}</span></label>`).join('');
}
function render(){
  const it = ITEMS[idx]; const s = state[it.item_id] || {};
  $('counter').textContent = `Item ${idx+1} of ${ITEMS.length}  ·  id ${it.item_id}`;
  $('role').textContent = it.role;
  $('desc').textContent = it.role_description;
  $('question').textContent = it.question;
  $('answer').textContent = it.answer;
  $('badges').innerHTML =
    `<span class="badge">Labeller 1<br><b>label ${esc(it.round2_labeller_1_label)}</b> · ${esc(it.round2_labeller_1_validity||'-')}</span>`+
    `<span class="badge">Labeller 2<br><b>label ${esc(it.round2_labeller_2_label)}</b> · ${esc(it.round2_labeller_2_validity||'-')}</span>`;
  choice($('valchoice'),'val',[{v:'ok',t:'ok'},{v:'unjudgeable',t:'unjudgeable'}], s.v);
  choice($('labchoice'),'lab',[{v:'0',t:'0'},{v:'1',t:'1'},{v:'2',t:'2'},{v:'3',t:'3'}], s.l);
  $('note').value = s.note || '';
  const dc = doneCount();
  $('pcount').innerHTML = `<span class="done">${dc}</span> / ${ITEMS.length} done · <span class="pending">${ITEMS.length-dc}</span> left`;
  $('pfill').style.width = (100*dc/ITEMS.length)+'%';
  $('prev').disabled = idx===0; $('next').disabled = idx===ITEMS.length-1;
}
function set(field,val){ const it=ITEMS[idx]; state[it.item_id]=Object.assign({},state[it.item_id],{[field]:val}); save(); }
document.addEventListener('change',e=>{
  if(e.target.name==='val') set('v',e.target.value);
  if(e.target.name==='lab') set('l',e.target.value);
});
$('note').addEventListener('input',e=>set('note',e.target.value));
$('prev').onclick=()=>{if(idx>0){idx--;render();}};
$('next').onclick=()=>{if(idx<ITEMS.length-1){idx++;render();}};
$('nextpending').onclick=()=>{ for(let k=1;k<=ITEMS.length;k++){const j=(idx+k)%ITEMS.length;const s=state[ITEMS[j].item_id];if(!(s&&s.v&&s.l)){idx=j;render();return;}} };
$('clear').onclick=()=>{ if(confirm('Clear all saved adjudications?')){state={};save();} };
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA') return;
  if(e.key>='0'&&e.key<='3') set('l',e.key);
  else if(e.key==='o') set('v','ok');
  else if(e.key==='u') set('v','unjudgeable');
  else if(e.key==='ArrowLeft') $('prev').click();
  else if(e.key==='ArrowRight') $('next').click();
});
function csvCell(s){ s=(s==null?'':String(s)); return /[",\n\r]/.test(s) ? '"'+s.replace(/"/g,'""')+'"' : s; }
function exportCSV(){
  const lines = [COLS.join(',')];
  for(const it of ITEMS){ const s=state[it.item_id]||{};
    const row=Object.assign({},it,{adjudicated_validity:s.v||'',adjudicated_label_0_to_3:s.l||'',adjudication_note:s.note||''});
    lines.push(COLS.map(c=>csvCell(row[c])).join(','));
  }
  const csv = lines.join('\n')+'\n';
  const dc = doneCount();
  try{
    const blob=new Blob([csv],{type:'text/csv;charset=utf-8'});
    const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
    a.download='adjudication_worksheet.csv'; document.body.appendChild(a); a.click(); a.remove();
  }catch(err){}
  const box=$('exportbox'); box.style.display='block'; box.value=csv; box.focus(); box.select();
  if(dc<ITEMS.length) alert(`Exported. ${ITEMS.length-dc} item(s) still un-adjudicated — the consensus builder will keep those PROVISIONAL until filled.`);
}
$('export').onclick=exportCSV;
// resume at first unadjudicated
for(let j=0;j<ITEMS.length;j++){const s=state[ITEMS[j].item_id];if(!(s&&s.v&&s.l)){idx=j;break;}}
render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worksheet",
                    default=str(REPO_ROOT / "annotations" / "t06a_round2" / "adjudication_worksheet.csv"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "t06" / "t06_adjudication_app.html"))
    a = ap.parse_args()
    build(Path(a.worksheet), Path(a.out))


if __name__ == "__main__":
    main()
