"""Offline, self-contained HTML evaluation summaries."""

from __future__ import annotations

import html
import json
from collections.abc import Iterable
from pathlib import Path

from agenteval.core.schema import EvalResult


def render_html_report(results: Iterable[EvalResult], title: str = "AgentEval-Bench report") -> str:
    rows = []
    chart_data = []
    for result in results:
        details = result.details
        case_id = str(details.get("case_id", "case"))
        rows.append(
            "<details class='case'><summary class='{status}'>{case} — {status} · accuracy {score:.2f}</summary>"
            "<pre>{body}</pre></details>".format(
                status="pass" if result.passed else "fail",
                case=html.escape(case_id),
                score=result.tool_accuracy_score,
                body=html.escape(json.dumps(result.model_dump(), indent=2, ensure_ascii=False)),
            )
        )
        chart_data.append(
            {
                "id": case_id,
                "passed": result.passed,
                "accuracy": result.semantic_intent_score
                if result.semantic_intent_score is not None
                else result.deterministic_accuracy_score,
                "deterministic": result.deterministic_accuracy_score,
                "semantic": result.semantic_intent_score,
                "latency": details.get("latency_ms", 0),
                "tokens": details.get("token_count", 0),
                "cost": result.total_cost_usd,
            }
        )
    payload = (
        json.dumps(chart_data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    )
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title><style>
:root{{--bg:#0b1020;--panel:#141c2f;--line:#26334b;--text:#e8eef8;--muted:#9baac2;--green:#37d39a;--red:#ff7185;--blue:#8ab4ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,Segoe UI,sans-serif;padding:28px}}
main{{max-width:1100px;margin:auto}}h1{{margin:0 0 5px;font-size:26px}}.muted{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:20px 0}}
.card,.case{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px}}canvas{{width:100%;height:210px;display:block;margin-top:10px}}.cases{{display:grid;gap:9px}}.case summary{{cursor:pointer;font-weight:650}}.pass{{color:var(--green)}}.fail{{color:var(--red)}}pre{{white-space:pre-wrap;overflow:auto;color:#c6d2e4;font-size:12px}}
@media(max-width:760px){{body{{padding:15px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>{safe_title}</h1><div class="muted">Self-contained regression snapshot · <span id="summary"></span></div>
<section class="grid"><div class="card"><b>Accuracy trend</b><canvas id="accuracy-chart"></canvas></div><div class="card"><b>Latency distribution (ms)</b><canvas id="latency-chart"></canvas></div><div class="card"><b>Token usage</b><canvas id="token-chart"></canvas></div></section>
<h2>Cases</h2><section class="cases">{"".join(rows)}</section></main>
<script>
const data={payload};
const passed=data.filter(x=>x.passed).length;
document.getElementById('summary').textContent=`${{passed}} / ${{data.length}} passed · ${{data.reduce((a,x)=>a+x.tokens,0)}} tokens · $${{data.reduce((a,x)=>a+x.cost,0).toFixed(4)}}`;
function chart(id,key,color,maxOverride){{const c=document.getElementById(id),ctx=c.getContext('2d');const dpr=window.devicePixelRatio||1;c.width=c.clientWidth*dpr;c.height=210*dpr;ctx.scale(dpr,dpr);const w=c.clientWidth,h=210,p=26;ctx.strokeStyle='#26334b';ctx.fillStyle='#9baac2';ctx.font='11px system-ui';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p,10);ctx.lineTo(p,h-p);ctx.lineTo(w-5,h-p);ctx.stroke();const max=maxOverride||Math.max(1,...data.map(x=>Number(x[key])||0));const bw=Math.max(5,(w-p-15)/Math.max(1,data.length)*.58);data.forEach((x,i)=>{{const slot=(w-p-15)/Math.max(1,data.length),v=Number(x[key])||0,bh=(h-2*p)*v/max,xp=p+i*slot+(slot-bw)/2,yp=h-p-bh;ctx.fillStyle=x.passed?color:'#ff7185';ctx.fillRect(xp,yp,bw,bh);ctx.fillStyle='#9baac2';ctx.save();ctx.translate(xp+bw/2,h-7);ctx.rotate(-.35);ctx.textAlign='right';ctx.fillText(x.id.slice(0,13),0,0);ctx.restore();}});}}
function draw(){{chart('accuracy-chart','accuracy','#37d39a',1);chart('latency-chart','latency','#8ab4ff');chart('token-chart','tokens','#c39cff');}}
draw();window.addEventListener('resize',draw);
</script></body></html>"""


def write_html_report(
    path: str | Path, results: Iterable[EvalResult], title: str = "AgentEval-Bench report"
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html_report(results, title), encoding="utf-8")
    return target
