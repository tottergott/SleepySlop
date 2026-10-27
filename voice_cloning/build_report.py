"""Render results/benchmark.json into a standalone voice_arena.html.

Audio is embedded as base64 MP3 (ffmpeg) so the single HTML file works offline
and can be sent around; --link keeps <audio src="outputs/..."> paths instead.
"""
import argparse
import base64
import html
import json
import subprocess
from pathlib import Path

import numpy as np

from voice_arena.characters import PHRASES, PROFILES
from voice_arena.paths import BIN, RESULTS, ROOT

MODE_TITLE = {"f5": "F5-TTS Zero-Shot", "rvc": "Silero + RVC", "native": "Native / DSP"}
MODE_ORDER = {"f5": 0, "rvc": 1, "native": 2}
PHRASE_TITLE = {"ping": "1 · Пинг", "emotion": "2 · Эмоция", "digits": "3 · Цифры", "lore": "4 · Лор"}


def audio_src(path: Path, embed: bool, cache: dict) -> str:
    if not embed:
        return path.relative_to(ROOT).as_posix()
    if path not in cache:
        mp3 = subprocess.run([str(BIN / "ffmpeg.exe"), "-v", "error", "-i", str(path), "-ac", "1", "-b:a", "96k",
                              "-f", "mp3", "-"], capture_output=True, check=True).stdout
        cache[path] = "data:audio/mpeg;base64," + base64.b64encode(mp3).decode()
    return cache[path]


def ttfa_class(ms):
    if ms is None:
        return ""
    return "good" if ms < 350 else "warn" if ms < 1000 else "bad"


def cer_class(c):
    if c is None:
        return ""
    return "good" if c <= 10 else "warn" if c <= 25 else "bad"


def sim_class(s):
    if s is None:
        return ""
    return "good" if s >= 0.85 else "warn" if s >= 0.7 else "bad"


def fmt(x, nd=0, suffix=""):
    if x is None:
        return "—"
    return f"{x:.{nd}f}{suffix}"


def summarize(records):
    groups = {}
    for r in records:
        groups.setdefault((r["voice"], r["variant"]), []).append(r)
    rows = []
    for (voice, variant), rs in groups.items():
        ok = [r for r in rs if "error" not in r]
        first = rs[0]
        if not ok:
            rows.append({"voice": voice, "variant": variant, "mode": first["mode"], "label": first["label"],
                         "error": first.get("error"), "runs": {}})
            continue

        def mean(k):
            vals = [r[k] for r in ok if r.get(k) is not None]
            return float(np.mean(vals)) if vals else None

        rows.append({
            "voice": voice, "variant": variant, "mode": first["mode"], "label": first["label"],
            "source": first.get("source"), "source_lang": first.get("source_lang"), "note": first.get("note"),
            "ttfa": mean("ttfa_ms"), "ttfa_ping": next((r["ttfa_ms"] for r in ok if r["phrase"] == "ping"), None),
            "total": mean("total_ms"), "rtf": mean("rtf"), "vram": max((r["vram_mb"] for r in ok), default=None),
            "wer": mean("wer"), "cer": mean("cer"), "sim": mean("sim"), "switch": mean("switch_ms"),
            "runs": {r["phrase"]: r for r in ok},
        })
    rows.sort(key=lambda r: (list(PROFILES).index(r["voice"]), MODE_ORDER[r["mode"]], r["variant"]))
    return rows


def best_of(rows, voice):
    from voice_assistant_router import pick_variant

    stats = {r["variant"]: {"ttfa": r["ttfa"], "cer": r["cer"], "sim": r["sim"] or 0} for r in rows
             if r["voice"] == voice and r.get("cer") is not None}
    return {p: pick_variant(stats, p) for p in ("speed", "balanced", "quality")}


CSS = """
:root{--bg:#f4f5f7;--panel:#ffffff;--ink:#16181d;--muted:#5d6472;--line:#e3e6eb;--chip:#eef0f4;
--good:#1a7f37;--good-bg:#dcf5e3;--warn:#9a6700;--warn-bg:#fff4d1;--bad:#b42318;--bad-bg:#fde4e1;--accent:#ff7a1a;
--code-bg:#0f1117;--code-ink:#e6e9ef}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0f14;--panel:#151922;--ink:#e8ebf1;--muted:#98a1b3;
--line:#262c38;--chip:#1d2330;--good:#4ade80;--good-bg:#12301e;--warn:#fbbf24;--warn-bg:#33280c;--bad:#f87171;--bad-bg:#3a1515;
--code-bg:#0a0c10;--code-ink:#e6e9ef}}
:root[data-theme="dark"]{--bg:#0d0f14;--panel:#151922;--ink:#e8ebf1;--muted:#98a1b3;--line:#262c38;--chip:#1d2330;
--good:#4ade80;--good-bg:#12301e;--warn:#fbbf24;--warn-bg:#33280c;--bad:#f87171;--bad-bg:#3a1515;--code-bg:#0a0c10;--code-ink:#e6e9ef}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 Inter,Segoe UI,system-ui,-apple-system,sans-serif}
.wrap{max-width:1320px;margin:0 auto;padding:28px 16px 80px}
header h1{margin:0 0 4px;font-size:30px;letter-spacing:-.02em}header p{margin:0;color:var(--muted)}
.hw{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 4px}.hw span{background:var(--chip);border:1px solid var(--line);
border-radius:999px;padding:3px 11px;font-size:13px;color:var(--muted)}.hw b{color:var(--ink);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--muted);margin:10px 0 0}
.pill{display:inline-block;border-radius:6px;padding:1px 7px;font-variant-numeric:tabular-nums;font-weight:600;font-size:13px}
.pill.good{color:var(--good);background:var(--good-bg)}.pill.warn{color:var(--warn);background:var(--warn-bg)}
.pill.bad{color:var(--bad);background:var(--bad-bg)}
section.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;margin:18px 0}
h2{font-size:19px;margin:0 0 12px}h3{font-size:15px;margin:18px 0 8px;color:var(--muted);font-weight:600}
.tabs{display:flex;flex-wrap:wrap;gap:6px;position:sticky;top:0;z-index:5;background:var(--bg);padding:12px 0}
.tab{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:999px;padding:7px 14px;cursor:pointer;
font:inherit;font-size:14px;display:flex;align-items:center;gap:8px}
.tab .dot{width:9px;height:9px;border-radius:50%}.tab[aria-selected="true"]{border-color:var(--ink);box-shadow:0 0 0 1px var(--ink)}
.panel{display:none}.panel.active{display:block}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600;font-size:12.5px;white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.mode{font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;font-weight:700;color:var(--muted)}
.vname{font-weight:600}.src{color:var(--muted);font-size:12.5px;word-break:break-word}
.badge{display:inline-block;font-size:11px;border-radius:5px;padding:1px 6px;margin:2px 4px 0 0;background:var(--chip);
color:var(--muted);border:1px solid var(--line)}.badge.pick{background:var(--accent);color:#fff;border-color:transparent}
audio{width:210px;height:34px;display:block}
.cell-meta{font-size:12px;color:var(--muted);margin-top:4px;font-variant-numeric:tabular-nums}
.asr{font-size:12px;color:var(--muted);margin-top:2px;max-width:230px}
.ref{display:flex;flex-wrap:wrap;gap:18px;align-items:center}.ref .meta{font-size:13.5px;color:var(--muted);max-width:760px}
.warnbox{border-left:3px solid var(--warn);background:var(--warn-bg);padding:8px 12px;border-radius:6px;font-size:13.5px;margin:10px 0}
.grid-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
pre{background:var(--code-bg);color:var(--code-ink);padding:16px;border-radius:10px;overflow:auto;font-size:13px;line-height:1.45}
button.copy{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:5px 12px;cursor:pointer;font:inherit}
.bars{display:grid;grid-template-columns:150px 1fr 70px;gap:6px 10px;align-items:center;font-size:13px}
.bar{height:12px;border-radius:3px;background:var(--accent)}
footer{color:var(--muted);font-size:12.5px;margin-top:30px}
.themebtn{float:right;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:5px 12px;cursor:pointer;font:inherit}
@media (max-width:700px){header h1{font-size:23px}audio{width:170px}.bars{grid-template-columns:110px 1fr 56px}}
"""

JS = """
const tabs=[...document.querySelectorAll('.tab')];
function show(id){tabs.forEach(t=>t.setAttribute('aria-selected',t.dataset.id===id));
document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id===id));
try{localStorage.setItem('arena-tab',id)}catch(e){}}
tabs.forEach(t=>t.addEventListener('click',()=>show(t.dataset.id)));
let saved=null;try{saved=localStorage.getItem('arena-tab')}catch(e){}
show(saved&&document.getElementById(saved)?saved:tabs[0].dataset.id);
document.addEventListener('play',e=>{document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true);
document.querySelectorAll('button.copy').forEach(b=>b.addEventListener('click',()=>{
const t=document.getElementById(b.dataset.target).innerText;navigator.clipboard.writeText(t).then(()=>{b.textContent='Скопировано';
setTimeout(()=>b.textContent='Копировать',1500)})}));
const tb=document.querySelector('.themebtn');tb.addEventListener('click',()=>{const r=document.documentElement;
const dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
r.dataset.theme=dark?'light':'dark'});
"""


def render(data, embed):
    cache = {}
    records = data["records"]
    rows = summarize(records)
    hw = data["hardware"]
    refs = data.get("references", {})
    tpath = RESULTS / "voice_tuning.json"
    tuning = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else {}
    repeats =next((r.get("repeats", 1) for r in records if "error" not in r), 1)
    e = html.escape
    out = []
    a = out.append
    a(f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
      f"<title>Voice Arena</title><style>{CSS}</style></head><body><div class='wrap'>")
    a("<header><button class='themebtn' title='Светлая / тёмная тема'>◐</button><h1>Voice Arena · 7 игровых голосов на русском</h1>"
      "<p>F5-TTS Zero-Shot vs Silero → RVC v2 vs Native/DSP. Одинаковые 4 фразы, 1 прогрев, затем "
      f"{repeats} замера на фразу (медиана), стриминг по предложениям. Сгенерировано {e(data['generated'])}.</p>")
    chips = [("GPU", f"{hw.get('gpu')} · {hw.get('vram_gb')} ГБ"), ("CPU", f"{hw.get('cpu')} · {hw.get('cpu_threads')} потоков"),
             ("RAM", f"{hw.get('ram_gb')} ГБ"), ("PyTorch", f"{hw.get('torch')} / CUDA {hw.get('cuda')}")]
    if "gpu_util_pct" in hw:
        chips.append(("Фон GPU до старта", f"{hw['gpu_util_pct']}% · {hw['gpu_mem_used_mb']} МБ (другие приложения)"))
    a("<div class='hw'>" + "".join(f"<span>{k} <b>{e(str(v))}</b></span>" for k, v in chips) + "</div>")
    a("<div class='legend'><span><span class='pill good'>TTFA &lt; 350 мс</span> готово к диалогу</span>"
      "<span><span class='pill warn'>350–1000 мс</span> заметная пауза</span><span><span class='pill bad'>&gt; 1 с</span> медленно</span>"
      "<span>CER/WER — faster-whisper small (ru) · SIM — WavLM-SV, косинус к среднему голосу реальных реплик персонажа</span></div></header>")

    # ---------- overview by architecture
    a("<section class='card'><h2>Итог по архитектурам</h2><div class='scroll'><table><tr><th>Архитектура</th>"
      "<th class='num'>TTFA, мс</th><th class='num'>TTFA пинга</th><th class='num'>Total, мс</th><th class='num'>RTF</th>"
      "<th class='num'>Peak VRAM, МБ</th><th class='num'>WER %</th><th class='num'>CER %</th><th class='num'>SIM</th>"
      "<th class='num'>Вариантов</th></tr>")
    groups = [("F5-TTS Zero-Shot (NFE 32)", lambda r: r["variant"] == "f5:nfe32"),
              ("F5-TTS Zero-Shot (NFE 16)", lambda r: r["variant"] == "f5:nfe16"),
              ("Silero + RVC v2", lambda r: r["mode"] == "rvc"),
              ("Native / DSP", lambda r: r["mode"] == "native")]
    for name, match in groups:
        rs = [r for r in rows if match(r) and "error" not in r]
        if not rs:
            continue

        def m(k):
            vals = [r[k] for r in rs if r.get(k) is not None]
            return float(np.mean(vals)) if vals else None

        a(f"<tr><td class='vname'>{e(name)}</td><td class='num'><span class='pill {ttfa_class(m('ttfa'))}'>{fmt(m('ttfa'))}</span></td>"
          f"<td class='num'><span class='pill {ttfa_class(m('ttfa_ping'))}'>{fmt(m('ttfa_ping'))}</span></td>"
          f"<td class='num'>{fmt(m('total'))}</td><td class='num'>{fmt(m('rtf'), 3)}</td>"
          f"<td class='num'>{fmt(max(r['vram'] for r in rs))}</td><td class='num'>{fmt(m('wer'), 1)}</td>"
          f"<td class='num'><span class='pill {cer_class(m('cer'))}'>{fmt(m('cer'), 1)}</span></td>"
          f"<td class='num'><span class='pill {sim_class(m('sim'))}'>{fmt(m('sim'), 3)}</span></td>"
          f"<td class='num'>{len(rs)}</td></tr>")
    a("</table></div>")
    # TTFA bars per voice (best balanced pick)
    a("<h3>TTFA выбранного роутером варианта (policy=balanced)</h3><div class='bars'>")
    picks = {}
    for key, prof in PROFILES.items():
        picks[key] = best_of(rows, key)
        r = next((r for r in rows if r["voice"] == key and r["variant"] == picks[key]["balanced"]), None)
        if r and r.get("ttfa") is not None:
            w = min(100, r["ttfa"] / 25)
            a(f"<div>{e(prof.title)}</div><div><div class='bar' style='width:{max(w, 1):.1f}%;background:{prof.color}'></div></div>"
              f"<div class='num'><span class='pill {ttfa_class(r['ttfa'])}'>{fmt(r['ttfa'])}</span></div>")
    a("</div><p class='src'>Шкала: полная ширина = 2,5 с. Пиковая VRAM — torch.cuda.max_memory_allocated за прогон, включая веса, "
      "загруженные этой архитектурой (архитектуры гоняются по очереди, между фазами модели выгружаются). ONNX-модели (TeraTTS, Piper) "
      "работают на CPU и VRAM не занимают. F5: фразы короче 25 символов склеиваются со следующими и генерируются с запасом "
      "длительности 10% (speed 0.9) — иначе F5 обрезает окончания (CER падал в 3–6 раз); поэтому TTFA пинга у F5 — это время "
      "всей фразы. Все референсы и цели SIM — настоящие русские реплики из игр.</p></section>")

    # ---------- tabs
    a("<nav class='tabs' role='tablist'>")
    a("<button class='tab' role='tab' data-id='all'><span class='dot' style='background:var(--accent)'></span>Все 7 голосов</button>")
    for key, prof in PROFILES.items():
        a(f"<button class='tab' role='tab' data-id='{key}'><span class='dot' style='background:{prof.color}'></span>{e(prof.title)}</button>")
    a("</nav>")

    # all voices side by side (balanced pick)
    a("<div class='panel' id='all'><section class='card'><div class='grid-head'><h2>Все голоса рядом · лучший вариант по balanced</h2>"
      "<span class='src'>Строка — персонаж, столбец — фраза</span></div><div class='scroll'><table><tr><th>Персонаж</th><th>Вариант</th>")
    for pid, _ in PHRASES:
        a(f"<th>{PHRASE_TITLE[pid]}</th>")
    a("</tr>")
    for key, prof in PROFILES.items():
        r = next((r for r in rows if r["voice"] == key and r["variant"] == picks[key]["balanced"]), None)
        if not r:
            continue
        a(f"<tr><td class='vname'><span class='dot' style='display:inline-block;width:9px;height:9px;border-radius:50%;"
          f"background:{prof.color};margin-right:6px'></span>{e(prof.title)}<div class='src'>{e(prof.game)}</div></td>"
          f"<td><span class='mode'>{MODE_TITLE[r['mode']]}</span><div class='src'>{e(r['variant'])}</div></td>")
        for pid, _ in PHRASES:
            run = r["runs"].get(pid)
            if run:
                a(f"<td><audio controls preload='none' src='{audio_src(ROOT / run['wav'], embed, cache)}'></audio>"
                  f"<div class='cell-meta'><span class='pill {ttfa_class(run['ttfa_ms'])}'>{fmt(run['ttfa_ms'])} мс</span> "
                  f"CER {fmt(run.get('cer'), 0)}%</div></td>")
            else:
                a("<td>—</td>")
        a("</tr>")
    a("</table></div></section></div>")

    for key, prof in PROFILES.items():
        vrows = [r for r in rows if r["voice"] == key]
        ref = refs.get(key)
        a(f"<div class='panel' id='{key}'><section class='card' style='border-top:3px solid {prof.color}'>"
          f"<div class='grid-head'><h2>{e(prof.title)} <span class='src'>· {e(prof.game)}</span></h2>"
          f"<span class='src'>speed → {e(str(picks[key]['speed']))} · balanced → {e(str(picks[key]['balanced']))} · "
          f"quality → {e(str(picks[key]['quality']))}</span></div>")
        if ref:
            a("<h3>7-секундный референс (Zero-Shot) — настоящие реплики из игры</h3><div class='ref'>"
              f"<audio controls preload='none' src='{audio_src(Path(ref['wav']), embed, cache)}'></audio>"
              f"<div class='meta'>«{e(ref['text'])}»<br>{e(ref['source'])} · {ref['duration_s']} с · язык {ref['lang']}"
              + (f" · Whisper CER {ref['whisper_cer']}%" if 'whisper_cer' in ref else "")
              + f" · SIM считается к среднему голосу {len(ref.get('sim_set', []))} реальных реплик</div></div>")
        else:
            a("<div class='warnbox'>Настоящих русских записей этого персонажа нет на Hugging Face, а синтетический референс "
              "был бы нечестным — поэтому F5 Zero-Shot для него пока не запускается и SIM не считается. "
              f"Положите реплики из игры (любой формат и длина) в <code>references/user/{key}/</code> и выполните "
              "<code>prepare_references.py</code> и <code>run_benchmark.py --voices " + key + "</code>.</div>")
        tuned = tuning.get("voices", {}).get(key)
        if tuned:
            real = tuning["trials"][key]["real"]
            rows_t = tuning["trials"][key]["rows"]
            eqs = ", ".join(tuned.get("eq", {})) or "нигде (не улучшал)"
            nat = tuned.get("native_fx", {})
            a("<h3>Подбор под настоящий голос (tune_voices.py)</h3><div class='src'>"
              f"Настоящий голос: медиана F0 <b>{real['f0_med']:.0f} Гц</b>, разброс интонации {real['f0_range']:.1f} пт. "
              f"Выбрано: база Silero <b>{e(tuned.get('silero_speaker', prof.silero_speaker))}</b>, сдвиг RVC "
              f"<b>{tuned.get('rvc_pitch', prof.rvc_pitch):+d} пт</b>, matched EQ на: {e(eqs)}"
              + (" · TeraTTS без процедурного DSP (так ближе к оригиналу)" if nat.get("bypass") else "")
              + f". Проверено {len(rows_t)} конфигураций по разрыву SIM, спектру (LTAS), F0 и CER.</div>")
        if prof.post_fx:
            a("<div class='warnbox' style='border-color:#7fd3ff'>Эффект Ледяной Скорби (−2 пт, суб-октава −12 пт 15%, Chorus, тёмный Reverb) "
              "применяется во всех режимах — это часть профиля. Время DSP входит в задержку.</div>")

        a("<h3>Сравнение архитектур</h3><div class='scroll'><table><tr><th>Архитектура / вариант</th><th class='num'>TTFA</th>"
          "<th class='num'>Total</th><th class='num'>RTF</th><th class='num'>VRAM</th><th class='num'>WER</th><th class='num'>CER</th>"
          "<th class='num'>SIM</th><th class='num'>Смена голоса</th></tr>")
        for r in vrows:
            marks = "".join(f"<span class='badge pick'>{p}</span>" for p in ("speed", "balanced", "quality")
                            if picks[key][p] == r["variant"])
            if "error" in r:
                a(f"<tr><td><span class='mode'>{MODE_TITLE[r['mode']]}</span><div class='vname'>{e(r['label'])}</div></td>"
                  f"<td colspan='8' class='src'>Ошибка: {e(r['error'])}</td></tr>")
                continue
            a(f"<tr><td><span class='mode'>{MODE_TITLE[r['mode']]}</span><div class='vname'>{e(r['label'])}</div>"
              f"<div class='src'>{e(str(r.get('source') or ''))}{' · ' + e(r['note']) if r.get('note') else ''}</div>{marks}</td>"
              f"<td class='num'><span class='pill {ttfa_class(r['ttfa'])}'>{fmt(r['ttfa'])} мс</span></td>"
              f"<td class='num'>{fmt(r['total'])} мс</td><td class='num'>{fmt(r['rtf'], 3)}</td><td class='num'>{fmt(r['vram'])} МБ</td>"
              f"<td class='num'>{fmt(r['wer'], 1)}%</td><td class='num'><span class='pill {cer_class(r['cer'])}'>{fmt(r['cer'], 1)}%</span></td>"
              f"<td class='num'><span class='pill {sim_class(r['sim'])}'>{fmt(r['sim'], 3)}</span></td>"
              f"<td class='num'>{fmt(r['switch'])} мс</td></tr>")
        a("</table></div>")

        a("<h3>Прослушать: фразы × архитектуры</h3><div class='scroll'><table><tr><th>Вариант</th>")
        for pid, _ in PHRASES:
            a(f"<th>{PHRASE_TITLE[pid]}</th>")
        a("</tr><tr><td class='src'>Текст</td>")
        for pid, _ in PHRASES:
            text = prof.lore if pid == "lore" else dict(PHRASES)[pid]
            a(f"<td class='src'>{e(text)}</td>")
        a("</tr>")
        for r in vrows:
            if "error" in r:
                continue
            a(f"<tr><td><span class='mode'>{MODE_TITLE[r['mode']]}</span><div class='vname'>{e(r['variant'])}</div></td>")
            for pid, _ in PHRASES:
                run = r["runs"].get(pid)
                if not run:
                    a("<td>—</td>")
                    continue
                a(f"<td><audio controls preload='none' src='{audio_src(ROOT / run['wav'], embed, cache)}'></audio>"
                  f"<div class='cell-meta'><span class='pill {ttfa_class(run['ttfa_ms'])}'>{fmt(run['ttfa_ms'])} мс</span> "
                  f"RTF {fmt(run['rtf'], 2)} · CER {fmt(run.get('cer'), 0)}% · SIM {fmt(run.get('sim'), 2)}</div>"
                  f"<div class='asr'>ASR: {e(run.get('asr') or '')}</div></td>")
            a("</tr>")
        a("</table></div></section></div>")

    # ---------- sources
    a("<section class='card'><h2>Источники и статус загрузки</h2><div class='scroll'><table><tr><th></th><th>Источник</th><th>Примечание</th></tr>")
    for s in data.get("manifest_status", []):
        a(f"<tr><td>{'✅' if s['ok'] else '⚠️'}</td><td>{e(s['what'])}</td><td class='src'>{e(s['note'])}</td></tr>")
    a("</table></div></section>")

    # ---------- router
    code = (ROOT / "voice_assistant_router.py").read_text(encoding="utf-8")
    a("<section class='card'><div class='grid-head'><h2>VoiceAssistantRouter</h2><button class='copy' data-target='router-code'>Копировать</button></div>"
      "<p class='src'>Лежит в <code>voice_assistant_router.py</code>, использует пакет <code>voice_arena/</code> и результаты этого бенчмарка "
      "для выбора варианта каждого голоса. Пример: <code>assistant.set_voice(\"arthas_deathknight\")</code>.</p>"
      f"<pre id='router-code'>{e(code)}</pre></section>")
    a("<footer>Модели принадлежат их авторам на Hugging Face; голоса персонажей — Valve и Blizzard. Только для личного некоммерческого использования "
      "(F5-TTS_RUSSIAN — CC BY-NC 4.0).</footer>")
    a(f"</div><script>{JS}</script></body></html>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", action="store_true", help="reference outputs/*.wav instead of embedding audio")
    ap.add_argument("--out", default=str(ROOT / "voice_arena.html"))
    args = ap.parse_args()
    data = json.loads((RESULTS / "benchmark.json").read_text(encoding="utf-8"))
    page = render(data, embed=not args.link)
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"{args.out}  ({len(page.encode()) / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
