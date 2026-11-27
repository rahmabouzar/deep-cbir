"""ÉTAPE 6 -- Site HTML multi-pages (hors-ligne) : tous les résultats + explications détaillées.

Usage : python etape6_site_html.py      ->  results/site/index.html  (ouvrir avec le navigateur)
Prérequis : étapes 1, 2 et 4 déjà exécutées.
"""
import argparse, hashlib, html, itertools, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from PIL import Image

import config as C
import analyses as A
from common import load_manifest, load_features, class_names, top_k, pairwise, load_rgb, relabel_G, relabel_ST

SITE = C.RESULTS / "site"
for d in ("assets", "thumbs", "img"):
    (SITE / d).mkdir(parents=True, exist_ok=True)
(SITE / "assets" / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")

def _exiger(*noms):
    manq = [n for n in noms if not (C.RESULTS / n).exists()]
    if manq:
        raise SystemExit("\n[ERREUR] Fichier(s) manquant(s) dans results\\ : " + ", ".join(manq) +
                         "\n   -> lancer d'abord, dans l'ordre :  python etape1_preparation.py  |  python etape2_extraction.py  |  python etape4_evaluation.py\n")


_exiger("metrics_global.csv", "metrics_par_classe.csv", "pr_curves.json", "etape1_stats.json")
R = A.load_results()
G = relabel_G(pd.read_csv(C.RESULTS / "metrics_global.csv"))
PC = pd.read_csv(C.RESULTS / "metrics_par_classe.csv", dtype={"classe": str})
PR = json.loads((C.RESULTS / "pr_curves.json").read_text(encoding="utf-8"))
ST = relabel_ST(json.loads((C.RESULTS / "etape1_stats.json").read_text(encoding="utf-8")))
DSS = [d for d in C.DATASETS if d in set(G.dataset)]
MODS = [m for m in C.MODELS if m in set(G.model)]
nd = lambda d: C.DATASETS[d]["label"]
nm = lambda m: C.MODELS[m][0]
_ids = itertools.count(1)
MAN = {d: load_manifest(d) for d in DSS}
PRESET = next(iter(ST.values()))["preset"] if ST else "?"

NAV = [("index", "Accueil"), ("etape1", "① Données"), ("etape2", "② Extraction"), ("etape3", "③ Recherche"),
       ("Étape 4 ▾", [("resultats", "Résultats globaux"), ("courbes", "Courbes P-R"), ("classes", "Par classe"), ("confusion", "Confusions")]),
       ("etape5", "⑤ Rapport"), ("comparaisons", "Comparaisons"), ("conclusion", "Conclusion"), ("methode", "Méthodologie"), ("modeles", "Modèles")]
PAGES = [(n, t) for n, t in NAV if not isinstance(t, list)] + [x for n, t in NAV if isinstance(t, list) for x in t]

# ----------------------------------------------------------------- design (CSS/JS/page)
CSS = """
:root{--bg:#f4f6fb;--card:#fff;--tx:#1b2030;--mut:#667085;--ac:#3b5bdb;--ac2:#7048e8;--ok:#12805c;--ko:#c0392b;--bd:#e3e7ef;--sh:0 1px 2px rgba(16,24,40,.06),0 4px 14px rgba(16,24,40,.05);--side:268px}
:root[data-theme=dark]{--bg:#0f1218;--card:#181c25;--tx:#e8ebf2;--mut:#98a2b8;--ac:#8aa4ff;--ac2:#b197fc;--ok:#4cc38a;--ko:#ff7b6b;--bd:#2a3040;--sh:0 1px 2px rgba(0,0,0,.4)}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:70px}
body{margin:0;font:15px/1.65 system-ui,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--tx)}
header.top{position:sticky;top:0;z-index:50;height:56px;display:flex;align-items:center;gap:12px;padding:0 18px;background:linear-gradient(90deg,var(--ac),var(--ac2));color:#fff;box-shadow:var(--sh)}
header.top .brand{color:#fff;text-decoration:none;font-weight:700;font-size:18px;letter-spacing:.2px}header.top .brand small{font-weight:400;opacity:.85;margin-left:8px;font-size:13px}
header.top .sp{flex:1}header.top .btn{background:rgba(255,255,255,.18);color:#fff;border:1px solid rgba(255,255,255,.35)}header.top .btn:hover{background:rgba(255,255,255,.3)}
#burger{display:none}
.shell{display:flex;align-items:flex-start}
aside#side{position:sticky;top:56px;width:var(--side);flex:none;height:calc(100vh - 56px);overflow:auto;padding:14px 12px 30px;background:var(--card);border-right:1px solid var(--bd)}
aside h4{margin:16px 8px 6px;font-size:11.5px;letter-spacing:.9px;text-transform:uppercase;color:var(--mut)}
aside a.nv{display:block;padding:7px 11px;margin:1px 0;border-radius:8px;color:var(--tx);text-decoration:none;font-size:14px}
aside a.nv.sub{margin-left:16px;font-size:13px;color:var(--mut)}aside a.nv:hover{background:var(--bd)}
aside a.nv.on{background:linear-gradient(90deg,var(--ac),var(--ac2));color:#fff}
aside a.dl{display:block;padding:5px 11px;font-size:13px;color:var(--ac);text-decoration:none}aside a.dl:hover{text-decoration:underline}
#toc a{display:block;padding:3px 10px;font-size:12.5px;color:var(--mut);text-decoration:none;border-left:2px solid var(--bd)}#toc a:hover{color:var(--ac);border-color:var(--ac)}
main{flex:1;min-width:0;max-width:1180px;padding:26px 28px 60px}
h1{font-size:28px;margin:.1em 0 .5em;line-height:1.25}h2{margin:2em 0 .6em;padding-bottom:.25em;border-bottom:2px solid var(--bd);font-size:21px}h3{margin-top:1.5em}
.card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:14px 18px;margin:12px 0;box-shadow:var(--sh)}
a.card:hover{border-color:var(--ac);transform:translateY(-1px);transition:.15s}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.kpi b{display:block;font-size:27px;color:var(--ac)}.kpi span{color:var(--mut);font-size:13px}
.box{border-left:5px solid var(--ac);background:var(--card);padding:11px 16px;margin:12px 0;border-radius:0 12px 12px 0;box-shadow:var(--sh)}
.box.note{border-color:#e0a100}.box.res{border-color:var(--ok)}.box b.t{display:block;margin-bottom:2px}
.tw{overflow:auto;margin:10px 0;border:1px solid var(--bd);border-radius:12px;background:var(--card);max-height:640px;box-shadow:var(--sh)}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:7px 11px;border-bottom:1px solid var(--bd);text-align:left;white-space:nowrap}
tbody tr:nth-child(even){background:rgba(127,127,160,.06)}tbody tr:hover{background:rgba(59,91,219,.09)}
th{position:sticky;top:0;background:var(--card);cursor:pointer;font-size:12.5px;color:var(--mut)}th:hover{color:var(--ac)}td.best{background:rgba(18,128,92,.2)!important;font-weight:600}
input.filter{padding:8px 11px;border:1px solid var(--bd);border-radius:9px;background:var(--card);color:var(--tx);width:260px;margin:6px 0}
.tabs .bar{display:flex;gap:6px;margin:8px 0}.tabbtn{padding:6px 14px;border:1px solid var(--bd);border-radius:9px;background:var(--card);color:var(--tx);cursor:pointer;text-decoration:none;font-size:14px}
.tabbtn.on{background:var(--ac);color:#fff;border-color:var(--ac)}.tabpane{display:none}.tabpane:first-of-type{display:block}
.btn{display:inline-flex;align-items:center;gap:6px;padding:7px 14px;border-radius:9px;border:1px solid var(--bd);background:var(--card);color:var(--tx);text-decoration:none;font-size:13.5px;cursor:pointer;font-family:inherit}
.btn:hover{border-color:var(--ac);color:var(--ac)}.btn.pri{background:var(--ac);color:#fff;border-color:var(--ac)}.btn.pri:hover{filter:brightness(1.1);color:#fff}.btn.sm{padding:3px 10px;font-size:12px}
.dlbar{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 14px}
.plot{background:#fff;border:1px solid var(--bd);border-radius:12px;margin:10px 0;padding:4px;box-shadow:var(--sh)}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}.gcard{background:var(--card);border:1px solid var(--bd);border-radius:14px;overflow:hidden;box-shadow:var(--sh)}
.gcard a.im{display:block;background:#fff;padding:6px}.gcard img{width:100%;height:auto;display:block}.gcard .cap{padding:10px 14px 12px}.gcard .cap b{display:block;font-size:14px}.gcard .cap span{display:block;color:var(--mut);font-size:12.5px;margin:2px 0 8px}
img.fig{max-width:100%;border-radius:10px;border:1px solid var(--bd);background:#fff;margin:6px 0}code,.f{background:var(--bd);padding:1px 6px;border-radius:5px;font-family:Consolas,monospace;font-size:.92em}
.ex{margin:14px 0;padding:10px;border:1px solid var(--bd);border-radius:14px;background:var(--card);box-shadow:var(--sh)}
.rowx{display:flex;gap:8px;align-items:flex-start;margin:6px 0;overflow-x:auto}.lab{min-width:105px;font-size:13px}
figure{margin:0;text-align:center;font-size:11px;min-width:108px;flex:none}figure img{height:96px;border:4px solid var(--bd);border-radius:6px;display:block;margin:auto}
figure.ok img{border-color:var(--ok)}figure.ko img{border-color:var(--ko)}figure.q img{border-color:var(--ac)}
.tag{display:inline-block;padding:1px 9px;border-radius:10px;background:var(--bd);font-size:12px}
.stepper{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 10px}.stepper .st{padding:7px 14px;border-radius:22px;border:1px solid var(--bd);background:var(--card);color:var(--tx);text-decoration:none;font-size:14px}
.stepper .st b{display:inline-block;width:22px;height:22px;line-height:22px;text-align:center;border-radius:50%;background:var(--bd);margin-right:4px}.stepper .st.on{background:var(--ac);color:#fff;border-color:var(--ac)}.stepper .st.on b{background:rgba(255,255,255,.3)}
.subnav{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 10px}.subnav a{padding:4px 12px;border-radius:9px;background:var(--bd);color:var(--tx);text-decoration:none;font-size:13px}.subnav a.on{background:var(--ac);color:#fff}
.pos{color:var(--ok);font-weight:600}.neg{color:var(--ko);font-weight:600}
.pn{display:flex;justify-content:space-between;gap:10px;margin-top:44px}.pn a{flex:1;max-width:48%}.pn .n{text-align:right}.pn small{display:block;color:var(--mut)}
#totop{position:fixed;right:18px;bottom:18px;display:none;z-index:40}
@media(max-width:900px){#burger{display:inline-flex}aside#side{position:fixed;left:-300px;top:56px;z-index:45;transition:left .2s;box-shadow:var(--sh)}aside#side.open{left:0}main{padding:18px 14px 50px}}
@media print{header.top,aside#side,.pn,#totop,.dlbar{display:none}main{max-width:none;padding:0}}
"""
JS = """
try{if(localStorage.getItem('theme'))document.documentElement.dataset.theme=localStorage.getItem('theme')}catch(e){}
document.querySelectorAll('table.sortable th').forEach(th=>th.addEventListener('click',()=>{const t=th.closest('table'),i=[...th.parentNode.children].indexOf(th),asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
const rows=[...t.tBodies[0].rows];rows.sort((a,b)=>{const x=a.cells[i].dataset.v,y=b.cells[i].dataset.v,nx=parseFloat(x),ny=parseFloat(y);const r=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc?r:-r});rows.forEach(r=>t.tBodies[0].appendChild(r))}));
document.querySelectorAll('input.filter').forEach(inp=>inp.addEventListener('input',()=>{const t=document.getElementById(inp.dataset.t),q=inp.value.toLowerCase();[...t.tBodies[0].rows].forEach(r=>r.style.display=r.innerText.toLowerCase().includes(q)?'':'none')}));
document.querySelectorAll('.tabs').forEach(tb=>{const bs=tb.querySelectorAll(':scope>.bar>.tabbtn'),ps=tb.querySelectorAll(':scope>.tabpane');bs.forEach((b,i)=>b.addEventListener('click',()=>{bs.forEach(x=>x.classList.remove('on'));ps.forEach(x=>x.style.display='none');b.classList.add('on');ps[i].style.display='block';window.dispatchEvent(new Event('resize'))}))});
(function(){const toc=document.getElementById('toc');if(toc){document.querySelectorAll('main h2').forEach((h,i)=>{h.id=h.id||'s'+i;const a=document.createElement('a');a.href='#'+h.id;a.textContent=h.textContent;toc.appendChild(a)})}
const b=document.getElementById('burger'),s=document.getElementById('side');if(b)b.onclick=()=>s.classList.toggle('open');
const th=document.getElementById('theme');if(th)th.onclick=()=>{const d=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=d;try{localStorage.setItem('theme',d)}catch(e){}};
const up=document.getElementById('totop');if(up){window.addEventListener('scroll',()=>up.style.display=scrollY>500?'inline-flex':'none');up.onclick=()=>scrollTo({top:0})}})();
"""
SIDE = [("Vue d'ensemble", [("index", "Accueil", 0)]),
        ("Pipeline", [("etape1", "1 · Préparation des données", 0), ("etape2", "2 · Extraction des vecteurs", 0), ("etape3", "3 · Recherche et similarité", 0),
                      ("resultats", "4 · Évaluation – résultats", 0), ("courbes", "↳ Courbes Précision-Rappel", 1), ("classes", "↳ Comparaison par classe", 1),
                      ("confusion", "↳ Matrices de confusion", 1), ("etape5", "5 · Rapport", 0)]),
        ("Synthèse", [("comparaisons", "Comparaisons croisées", 0), ("conclusion", "Conclusion", 0)]),
        ("Annexes", [("methode", "Méthodologie", 0), ("modeles", "Modèles", 0)])]
ORDER = ["index", "etape1", "etape2", "etape3", "resultats", "courbes", "classes", "confusion", "etape5", "comparaisons", "conclusion", "methode", "modeles"]
LABEL = {n: t.replace("↳ ", "") for _, items in SIDE for n, t, _ in items}
STAGE_OF = {"etape1": 1, "etape2": 2, "etape3": 3, "resultats": 4, "courbes": 4, "classes": 4, "confusion": 4}
ART = dict(art={}, pdf={})


def _copy_img(name):
    src = C.FIGS / name
    if src.exists(): shutil.copy(src, SITE / "img" / name)
    return f"img/{name}"


def pdf_links():
    d = {}
    for k, p in ART["pdf"].items():
        if Path(p).exists():
            (SITE / "pdf").mkdir(exist_ok=True); shutil.copy(p, SITE / "pdf" / Path(p).name); d[k] = f"pdf/{Path(p).name}"
    return d


def dl_bar(name):
    pl = pdf_links(); st = STAGE_OF.get(name); b = ""
    if "rapport_complet" in pl:
        if st and f"etape{st}" in pl: b += f'<a class="btn pri" href="{pl[f"etape{st}"]}" download>PDF de l\'étape {st}</a>'
        b += f'<a class="btn{"" if b else " pri"}" href="{pl["rapport_complet"]}" download>Rapport PDF complet</a>'
    return f'<div class="dlbar">{b}</div>' if b else ""


def gallery(name):
    items = ART["art"].get(name, [])
    if not items: return ""
    import zipfile
    zp = SITE / "img" / f"images_{name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for _, f, _ in items:
            if (C.FIGS / f).exists(): z.write(C.FIGS / f, f)
    cards = "".join(f'<div class="gcard"><a class="im" href="{_copy_img(f)}" target="_blank"><img loading="lazy" src="{_copy_img(f)}" alt="{esc(t)}"></a>'
                    f'<div class="cap"><b>{esc(t)}</b><span>{esc(c)}</span><a class="btn sm" href="img/{f}" download>PNG</a></div></div>' for t, f, c in items)
    return (f'<h2>Images des résultats</h2><div class="dlbar"><a class="btn" href="img/images_{name}.zip" download>Toutes les images de cette page (ZIP)</a></div>'
            f'<div class="gal">{cards}</div>')


def page(name, title, body):
    side = ""
    for grp, items in SIDE:
        side += f"<h4>{grp}</h4>" + "".join(f'<a class="nv{" sub" if lv else ""}{" on" if n == name else ""}" href="{n}.html">{t}</a>' for n, t, lv in items)
    pl = pdf_links()
    if pl:
        side += "<h4>Téléchargements</h4>" + (f'<a class="dl" href="{pl["rapport_complet"]}" download>Rapport PDF complet</a>' if "rapport_complet" in pl else "")
        side += "".join(f'<a class="dl" href="{pl[f"etape{i}"]}" download>PDF étape {i}</a>' for i in range(1, 5) if f"etape{i}" in pl)
    side += '<h4>Sur cette page</h4><div id="toc"></div>'
    i = ORDER.index(name) if name in ORDER else -1
    pn = ""
    if i >= 0:
        pv = f'<a class="card" href="{ORDER[i - 1]}.html"><small>← Précédent</small><b>{LABEL[ORDER[i - 1]]}</b></a>' if i > 0 else "<span></span>"
        nx = f'<a class="card n" href="{ORDER[i + 1]}.html"><small>Suivant →</small><b>{LABEL[ORDER[i + 1]]}</b></a>' if i < len(ORDER) - 1 else ""
        pn = f'<div class="pn">{pv}{nx}</div>'
    top = dl_bar(name) if name in STAGE_OF else ""
    if name == "index" and pl: top = f'<div class="dlbar"><a class="btn pri" href="{pl["rapport_complet"]}" download>Rapport PDF complet</a></div>'
    (SITE / f"{name}.html").write_text(
        f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{esc(title)} – DeepCBIR</title><script>try{{var t=localStorage.getItem("theme");if(t)document.documentElement.dataset.theme=t}}catch(e){{}}</script>'
        f'<style>{CSS}</style><script src="assets/plotly.min.js"></script></head><body>'
        f'<header class="top"><button class="btn" id="burger"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" style="vertical-align:-3px;margin:0"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button><a class="brand" href="index.html"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" style="vertical-align:-3px;margin-right:6px"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></svg>DeepCBIR<small>CBIR par réseaux profonds</small></a><span class="sp"></span>'
        f'<button class="btn" id="theme" title="Mode clair / sombre">◐ Thème</button></header>'
        f'<div class="shell"><aside id="side">{side}</aside><main><h1>{esc(title)}</h1>{top}{body}{gallery(name)}{pn}</main></div>'
        f'<button class="btn pri" id="totop">↑ Haut</button><script>{JS}</script></body></html>', encoding="utf-8")



# ----------------------------------------------------------------- outils HTML
def esc(x): return html.escape(str(x))


def box(text, kind="", titre=None):
    t = f'<b class="t">{titre}</b>' if titre else ""
    return f'<div class="box {kind}">{t}{text}</div>'


def fig_html(fig, h=420):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=45, r=20, t=55, b=45), font=dict(size=12))
    return '<div class="plot">' + fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True}) + "</div>"


def table(df, best_max=(), best_min=(), fmt=None, filt=False):
    tid = f"t{next(_ids)}"; fmt = fmt or {}
    bm = {c: df[c].max() for c in best_max if c in df}; bn = {c: df[c].min() for c in best_min if c in df}
    th = "".join(f"<th>{esc(c)}</th>" for c in df.columns); rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]; cls = ' class="best"' if (c in bm and v == bm[c]) or (c in bn and v == bn[c]) else ""
            txt = (fmt[c].format(v) if c in fmt else f"{v:.4f}") if isinstance(v, (float, np.floating)) else esc(v)
            tds.append(f'<td{cls} data-v="{esc(v)}">{txt}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    f = f'<input class="filter" placeholder="Filtrer…" data-t="{tid}">' if filt else ""
    return f'{f}<div class="tw"><table id="{tid}" class="sortable"><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def tabs(items):
    bar = "".join(f'<button class="tabbtn{" on" if i == 0 else ""}">{esc(t)}</button>' for i, (t, _) in enumerate(items))
    panes = "".join(f'<div class="tabpane">{h}</div>' for _, h in items)
    return f'<div class="tabs"><div class="bar">{bar}</div>{panes}</div>'


def img(name):
    src = C.FIGS / name
    if not src.exists(): return ""
    shutil.copy(src, SITE / "img" / name)
    return f'<img class="fig" src="img/{name}" alt="{esc(name)}">'


def thumb(path):
    out = SITE / "thumbs" / (hashlib.md5(path.encode()).hexdigest()[:12] + ".jpg")
    if not out.exists():
        try:
            im = load_rgb(path, 160); w, h = im.size
            if max(w, h) < 96:
                s = 96 / max(w, h); im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
            else:
                im.thumbnail((160, 160))
            im.save(out, quality=85)
        except Exception:
            return ""
    return f"thumbs/{out.name}"


def best_row(ds, metric=None):
    s = G[G.dataset == ds] if metric is None else G[(G.dataset == ds) & (G.metrique == metric)]
    return s.sort_values("mAP", ascending=False).iloc[0]


def mean_map(metric="cosinus"):
    return G[G.metrique == metric].groupby("model").mAP.mean()


# ================================================================= PAGES
def p_index():
    kp = [(len(DSS), "bases de données"), (sum(len(MAN[d]) for d in DSS), "images indexées"), (len(MODS), "réseaux CNN comparés"),
          (len(G), "configurations évaluées")]
    b = '<div class="grid">' + "".join(f'<div class="card kpi"><b>{v}</b><span>{l}</span></div>' for v, l in kp) + "</div>"
    b += box("Ce site présente <b>tous les résultats</b> du mini-projet CBIR (Content-Based Image Retrieval) : des réseaux profonds "
             "pré-entraînés sur ImageNet servent d'extracteurs de descripteurs ; on retrouve les images les plus proches d'une image "
             "requête, puis on mesure la qualité de la recherche (Précision-Rappel, mAP). Chaque page contient les chiffres, les "
             "graphiques interactifs et une explication de lecture.", titre="Objectif")
    b += "<h2>Parcours par étape</h2><div class='grid'>"
    for n, t, d, o in [("etape1", "① Préparation des données", "Bases, classes, types de fichiers, résolutions, couverture, prétraitement 224×224.", "manifestes + statistiques"),
                       ("etape2", "② Extraction des vecteurs", "Un vecteur par image et par réseau : dimension, temps, qualité des descripteurs, projection 2D.", "index de vecteurs (.npy)"),
                       ("etape3", "③ Recherche et similarité", "Distances cosinus/euclidienne, Top-K, coût d'une recherche, distribution des distances, exemples.", "listes Top-K"),
                       ("resultats", "④ Évaluation", "mAP, P@K, R@K, courbes Précision-Rappel, par classe, confusions.", "tableaux + courbes"),
                       ("etape5", "⑤ Rapport", "Rapport rédigé complet généré automatiquement.", "rapport.html")]:
        b += f'<a class="card" href="{n}.html" style="text-decoration:none;color:inherit"><b>{t}</b><br>{d}<br><span class="tag">sortie : {o}</span></a>'
    b += "</div><h2>Meilleure configuration par base</h2>"
    rows = [best_row(d) for d in DSS]
    df = pd.DataFrame([{"Base": r.dataset_nom, "Modèle": r.modele, "Métrique": r.metrique, "mAP": r.mAP, "P@1": r["P@1"],
                        "P@10": r["P@10"], "R@50": r["R@50"], "Dimension": int(r.dim)} for r in rows])
    cov = pd.DataFrame([{"Base": nd(d), "Images sur disque": ST[d].get("n_disponibles", ST[d]["n_images"]), "Images utilisées": ST[d]["n_images"],
                         "Couverture (%)": ST[d].get("couverture_pct", 100.0), "Classes": ST[d]["n_classes"]} for d in DSS])
    b += table(df, best_max=["mAP"])
    b += "<h2>Couverture des données</h2>" + table(cov, fmt={"Couverture (%)": "{:.1f}"})
    b += box(f"Mode d'exécution des étapes : <code>{esc(PRESET)}</code> "
             "(<i>full</i> = toutes les images ; <i>quick</i> ou <i>capN</i> = au plus N images par classe, pour accélérer le calcul).", "note")
    b += "<h2>Annexes et synthèses</h2><div class='grid'>"
    for n, t in [("comparaisons", "Comparaisons croisées"), ("conclusion", "Conclusion"), ("methode", "Méthodologie"), ("modeles", "Modèles")]:
        b += f'<a class="card" href="{n}.html" style="text-decoration:none;color:inherit"><b>{t}</b></a>'
    page("index", "CBIR par réseaux profonds – Résultats détaillés", b + "</div>")


DESC = {
    "corel1k": "Base historique : 10 catégories (africains, plages, bâtiments, bus, dinosaures, éléphants, fleurs, nourriture, chevaux, montagnes), 100 images chacune. Classes visuellement très distinctes → base « facile ».",
    "corel5k": "Version intermédiaire de Corel : 50 classes de 100 images (JPEG). Plus de classes que Corel-1K → distinguer des catégories voisines devient plus difficile.",
    "corel10k": "Version étendue de Corel : 100 classes de 100 images. Plus de classes proches → plus difficile que Corel-1K.",
    "cifar100": "100 classes fines (pomme, requin, bus, orchidée…), images minuscules 32×32 en PNG. Agrandies à 224×224 pour le réseau : très flou et très éloigné d'ImageNet → base « difficile » pour la recherche.",
    "paris6k": "Photos de monuments parisiens (Eiffel, Louvre, Notre-Dame, Sacré-Cœur…). Grandes images JPEG, forts changements de point de vue/éclairage. La classe « general » (distracteurs) reste dans la base mais n'est jamais une requête.",
}


def p_donnees():
    b = box("Chaque base est parcourue récursivement ; la <b>classe</b> est le nom du dossier (Corel, CIFAR) ou lue dans le nom du fichier "
            "<code>paris_&lt;classe&gt;_NNNNNN.jpg</code> (Paris). Les images illisibles sont ignorées. Les statistiques ci-dessous sont mesurées sur les fichiers réels.")
    allr = []
    for d in DSS:
        s = ST[d]; df = MAN[d]
        b += f"<h2>{nd(d)}</h2><p>{DESC.get(d, '')}</p>"
        kv = pd.DataFrame([("Images trouvées sur le disque", s.get("n_disponibles", s["n_images"])), ("Images utilisées dans l'évaluation", s["n_images"]),
                           ("Couverture", f"{s.get('couverture_pct', 100)} %"), ("Nombre de classes", s["n_classes"]),
                           ("Images / classe (min–max)", f"{s['images_par_classe_min']}–{s['images_par_classe_max']}"),
                           ("Extensions", ", ".join(s["extensions"])), ("Formats", ", ".join(s["formats"])),
                           ("Résolution moyenne", f"{s['largeur_moy']:.0f}×{s['hauteur_moy']:.0f}"),
                           ("Largeur min–max", f"{s['largeur_min']}–{s['largeur_max']}"),
                           ("Taille moyenne d'un fichier", f"{s['taille_moy_ko']} Ko"), ("Taille totale", f"{s['taille_totale_mo']} Mo"),
                           ("Images illisibles ignorées", s["illisibles"])], columns=["Propriété", "Valeur"])
        b += table(kv) + img(f"echantillon_{d}.png")
        pv = df.head(8).assign(chemin=lambda x: x.path.str.split("[\\\\/]").str[-2:].str.join("/"))[["chemin", "class_name", "label", "ext", "width", "height", "size_kb"]]
        pv.columns = ["Fichier", "Classe", "Label", "Ext.", "Largeur", "Hauteur", "Taille (Ko)"]
        b += "<h3>Aperçu du manifeste (workdir/manifests/" + d + ".csv)</h3>" + table(pv, fmt={"Taille (Ko)": "{:.1f}"})
        b += "<h3>Prétraitement appliqué</h3>" + img(f"pretraitement_{d}.png")
        cnt = df.groupby("class_name").size().reset_index(name="images")
        b += fig_html(px.bar(cnt, x="class_name", y="images", title=f"Nombre d'images par classe – {nd(d)}",
                             labels={"class_name": "Classe"}), 320)
        h = make_subplots(rows=1, cols=2, subplot_titles=("Largeur des images (px)", "Taille des fichiers (Ko)"))
        h.add_histogram(x=df.width, nbinsx=30, showlegend=False, row=1, col=1)
        h.add_histogram(x=df.size_kb, nbinsx=30, showlegend=False, row=1, col=2)
        b += fig_html(h, 300)
    b += box("Chaque image est redimensionnée en <b>224×224</b> (sans recadrage) puis normalisée avec la moyenne (0.485, 0.456, 0.406) et "
             "l'écart-type (0.229, 0.224, 0.225) d'ImageNet : c'est exactement le prétraitement utilisé à l'entraînement des réseaux.", "note", "Pourquoi ce prétraitement ?")
    ban = banner(1, "Repérer les images de chaque base, en déduire la <b>classe</b>, vérifier les fichiers (format, résolution, lisibilité) et définir le <b>prétraitement</b> commun.",
                 "etape1_preparation.py", "dossiers d'images (Corel, CIFAR-100, Paris)", "<code>workdir/manifests/*.csv</code>, <code>results/etape1_stats.json</code>, figures d'échantillon")
    cov = pd.DataFrame([{"Base": nd(d), "Images sur disque": ST[d].get("n_disponibles", ST[d]["n_images"]), "Images utilisées": ST[d]["n_images"],
                         "Couverture (%)": ST[d].get("couverture_pct", 100.0), "Classes": ST[d]["n_classes"]} for d in DSS])
    page("etape1", "Étape 1 · Préparation des données", ban + "<h2>Couverture</h2>" + table(cov, fmt={"Couverture (%)": "{:.1f}"}) + b)


MDESC = {
    "squeezenet1_1": ("Modules « Fire » (squeeze 1×1 puis expand 1×1/3×3) : très peu de paramètres. Vecteur = Global Average Pooling de la dernière couche convolutive (512 canaux).", "GAP conv. finale"),
    "mobilenet_v3_small": ("Convolutions séparables en profondeur + blocs squeeze-excitation : très rapide sur CPU. Vecteur = GAP avant le classifieur (576).", "GAP avant classifieur"),
    "resnet18": ("18 couches à connexions résiduelles. Référence légère du CBIR. Vecteur = GAP (512).", "GAP (avgpool)"),
    "resnet50": ("50 couches, blocs « bottleneck » : référence absolue en CBIR. Vecteur = GAP (2048).", "GAP (avgpool)"),
    "vgg16": ("13 convolutions 3×3 empilées + 3 couches denses : lourd (≈ 138 M de paramètres) mais descripteurs très riches. Vecteur = couche fc7 (4096).", "fc7 après ReLU"),
}


def p_modeles():
    mm = G[G.metrique == "cosinus"].groupby("model").agg(dim=("dim", "first"), params=("params_M", "first"),
                                                         ms=("ms_extraction_img", "mean"), mAP=("mAP", "mean")).reindex(MODS).reset_index()
    t = pd.DataFrame({"Modèle": mm.model.map(nm), "Famille": mm.model.map(lambda m: C.MODELS[m][1]), "Couche utilisée": mm.model.map(lambda m: MDESC[m][1]),
                      "Dimension": mm.dim, "Paramètres (M)": mm.params, "Extraction (ms/image)": mm.ms, "mAP moyen (cosinus)": mm.mAP})
    b = box("Tous les réseaux sont <b>pré-entraînés sur ImageNet</b> (torchvision, poids par défaut) et utilisés <b>sans fine-tuning</b> : on retire la tête de classification "
            "et on garde le vecteur de la couche indiquée comme descripteur de l'image.")
    b += table(t, best_max=["mAP moyen (cosinus)"], best_min=["Extraction (ms/image)", "Paramètres (M)"], fmt={"Paramètres (M)": "{:.2f}", "Extraction (ms/image)": "{:.2f}"})
    b += "<h2>Description des architectures</h2>"
    for m in MODS:
        b += f'<div class="card"><b>{nm(m)}</b> <span class="tag">{C.MODELS[m][1]}</span><br>{MDESC[m][0]}</div>'
    c = make_subplots(rows=1, cols=3, subplot_titles=("Dimension du vecteur", "Paramètres (millions)", "Extraction (ms/image, log)"))
    for i, col in enumerate(["dim", "params", "ms"], 1):
        c.add_bar(x=[nm(m) for m in mm.model], y=mm[col], showlegend=False, row=1, col=i)
    c.update_yaxes(type="log", row=1, col=3)
    b += "<h2>Coût de chaque modèle</h2>" + fig_html(c, 380)
    b += box("Le temps d'extraction est mesuré sur <i>votre machine</i> (CPU ou GPU) : seuls les rapports entre modèles sont comparables d'une machine à l'autre. "
             "Une dimension plus grande = indexation plus lourde en mémoire (N × d × 4 octets) et recherche plus lente.", "note")
    page("modeles", "Modèles de réseaux profonds", b)


def worked_example():
    ds = DSS[0]; r = best_row(ds, "cosinus"); df = MAN[ds]; F = load_features(ds, r.model); y = df.label.values
    excl = C.DATASETS[ds].get("query_exclude", [])
    qi = int(df.index[~df.class_name.isin(excl)][0])
    d = pairwise(F[qi:qi + 1], F, "cosinus")[0]; d[qi] = np.inf
    order = np.argsort(d)[:-1]; rel = (y[order] == y[qi]); nrel = int(rel.sum()); cs = np.cumsum(rel)
    prec = cs / np.arange(1, len(rel) + 1); ap = (prec * rel).sum() / nrel
    rows = [{"Rang": i + 1, "Image": f'<img src="{thumb(df.path[order[i]])}" height="42">', "Classe": df.class_name[order[i]],
             "Pertinente": "oui" if rel[i] else "non", "Précision@rang": round(float(prec[i]), 3), "Rappel@rang": round(float(cs[i] / nrel), 3)} for i in range(10)]
    t = pd.DataFrame(rows)
    th = "".join(f"<th>{c}</th>" for c in t.columns)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r_.values()) + "</tr>" for r_ in rows)
    return (f"<h2>Exemple chiffré (calcul de l'AP)</h2><p>Requête : image #{qi} de <b>{nd(ds)}</b>, classe <b>{esc(df.class_name[qi])}</b>, "
            f"modèle <b>{nm(r.model)}</b>, distance cosinus. La base contient {nrel} images pertinentes (même classe, hors requête).</p>"
            f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'
            f"<p>AP sur le classement complet = moyenne des précisions aux rangs où une image pertinente apparaît = <b>{ap:.4f}</b>.</p>")


def p_methode():
    b = "<h2>Distances de similarité</h2><div class='grid'>"
    b += '<div class="card"><b>Cosinus</b><br><span class="f">d(q,x) = 1 − (q·x) / (‖q‖·‖x‖)</span><br>Compare la <i>direction</i> des vecteurs, insensible à leur norme. Les vecteurs sont normalisés L2.</div>'
    b += '<div class="card"><b>Euclidienne</b><br><span class="f">d(q,x) = √ Σ (qᵢ − xᵢ)²</span><br>Compare la distance géométrique brute : sensible à la norme, qui varie avec le contraste et le contenu de l\'image.</div></div>'
    b += "<h2>Protocole d'évaluation</h2>"
    b += box("<ul><li>Chaque image de la base est utilisée tour à tour comme <b>requête</b> (leave-one-out : elle est exclue de son propre classement).</li>"
             "<li>Une image retournée est <b>pertinente</b> si elle appartient à la même classe que la requête.</li>"
             "<li>Paris6k : la classe « general » (distracteurs) reste dans la base mais n'est jamais requête.</li>"
             "<li>Les images sont classées par distance croissante sur <b>toute</b> la base (classement complet).</li></ul>")
    b += "<h2>Mesures</h2>"
    for t, f, e in [("Précision@K", "P@K = (nb pertinentes dans le Top-K) / K", "Parmi les K images renvoyées, quelle proportion est bonne ? Mesure ce que voit l'utilisateur."),
                    ("Rappel@K", "R@K = (nb pertinentes dans le Top-K) / (nb pertinentes totales)", "Quelle part des images de la classe a-t-on retrouvée ?"),
                    ("Average Precision (AP)", "AP = (1/R) · Σₖ P(k)·rel(k)", "Moyenne des précisions aux rangs où l'on trouve une image pertinente ; récompense les bonnes images placées tôt."),
                    ("mean AP (mAP)", "mAP = moyenne des AP sur toutes les requêtes", "Note globale d'un système (0 à 1). Correspond à l'aire sous la courbe Précision-Rappel."),
                    ("Courbe Précision-Rappel", "précision interpolée à 11 niveaux de rappel (0, 0.1 … 1)", "Courbe haute et étendue = bon système : il garde une précision élevée même en cherchant toutes les images de la classe.")]:
        b += f'<div class="card"><b>{t}</b> — <span class="f">{f}</span><br>{e}</div>'
    b += worked_example()
    b += box("Un tirage aléatoire donnerait un mAP ≈ 1 / (nombre de classes) : environ " + ", ".join(f"{1 / ST[d]['n_classes']:.3f} pour {nd(d)}" for d in DSS) + ".", "note", "Repère de lecture")
    page("methode", "Méthodologie : similarité et mesures", b)


def p_resultats():
    b = "<h2>Vue d'ensemble : mAP</h2>"
    piv = G.pivot_table(index="modele", columns=["dataset_nom", "metrique"], values="mAP").reindex([nm(m) for m in MODS])
    piv.columns = [f"{a} – {c}" for a, c in piv.columns]
    pv = piv.reset_index().rename(columns={"modele": "Modèle"})
    b += table(pv, best_max=list(piv.columns))
    b += box("Cellule verte = meilleur modèle de la colonne. Cliquez sur un en-tête pour trier.", "note")
    for d in DSS:
        s = G[G.dataset == d]; br = best_row(d); worst = s.sort_values("mAP").iloc[0]
        b += f"<h2>{nd(d)}</h2>"
        b += box(f"Meilleure configuration : <b>{br.modele}</b> ({br.metrique}) avec un mAP de <b>{br.mAP:.4f}</b> et un P@10 de <b>{br['P@10']:.4f}</b> "
                 f"(≈ {br['P@10'] * 10:.1f} images pertinentes sur 10 retournées). La moins bonne, {worst.modele} ({worst.metrique}), obtient {worst.mAP:.4f} "
                 f"(écart de {br.mAP - worst.mAP:.4f}). Un tirage aléatoire donnerait ≈ {1 / br.n_classes:.3f}.", "res", "Constat")
        t = s.sort_values("mAP", ascending=False)[["modele", "metrique", "mAP"] + [f"P@{k}" for k in C.K_LIST] + ["R@10", "R@50", "dim", "ms_extraction_img", "ms_recherche_req", "n_requetes"]]
        t = t.rename(columns={"modele": "Modèle", "metrique": "Métrique", "dim": "Dim.", "ms_extraction_img": "Extraction ms/img", "ms_recherche_req": "Recherche ms/req.", "n_requetes": "Requêtes"})
        b += table(t, best_max=["mAP"] + [f"P@{k}" for k in C.K_LIST] + ["R@10", "R@50"], fmt={"Extraction ms/img": "{:.2f}", "Recherche ms/req.": "{:.2f}"})
        c1 = px.bar(s, x="modele", y="mAP", color="metrique", barmode="group", title=f"mAP – {nd(d)}", labels={"modele": "Modèle"})
        long = s.melt(id_vars=["modele", "metrique"], value_vars=[f"P@{k}" for k in C.K_LIST], var_name="K", value_name="Précision")
        long["Série"] = long.modele + " · " + long.metrique
        c2 = px.line(long, x="K", y="Précision", color="Série", markers=True, title=f"Précision@K – {nd(d)}")
        b += fig_html(c1, 380) + fig_html(c2, 440)
    b += box("<b>mAP</b> résume tout le classement ; <b>P@K</b> décrit la qualité des premiers résultats (ce que voit l'utilisateur) ; <b>R@K</b> la couverture de la classe. "
             "P@K baisse quand K augmente (on va chercher des images de plus en plus éloignées), R@K augmente.", "note", "Comment lire")
    page("resultats", "Étape 4 · Résultats globaux par base, modèle et métrique", B4("resultats") + b)


def p_courbes():
    b = box("La courbe donne, pour chaque niveau de rappel, la meilleure précision atteignable. Plus la courbe est <b>haute et à droite</b>, meilleur est le descripteur. "
            "L'aire sous la courbe correspond au mAP.", titre="Comment lire")
    lv = PR["niveaux_rappel"]
    for d in DSS:
        b += f"<h2>{nd(d)}</h2>"
        f = make_subplots(rows=1, cols=2, subplot_titles=C.METRICS, shared_yaxes=True)
        for j, met in enumerate(C.METRICS, 1):
            for m in MODS:
                k = f"{d}|{m}|{met}"
                if k in PR["courbes"]:
                    f.add_scatter(x=lv, y=PR["courbes"][k], mode="lines+markers", name=nm(m), legendgroup=m, showlegend=(j == 1), row=1, col=j)
        f.update_xaxes(title_text="Rappel"); f.update_yaxes(title_text="Précision", col=1)
        b += fig_html(f, 430)
        rows = []
        for m in MODS:
            for met in C.METRICS:
                k = f"{d}|{m}|{met}"
                if k in PR["courbes"]:
                    c = PR["courbes"][k]; rows.append({"Modèle": nm(m), "Métrique": met, "Précision @ rappel 0.1": c[1], "@ 0.3": c[3], "@ 0.5": c[5], "@ 0.8": c[8], "@ 1.0": c[10]})
        t = pd.DataFrame(rows); b += table(t, best_max=[c for c in t.columns if c.startswith(("Pré", "@"))])
        r5 = t.sort_values("@ 0.5", ascending=False).iloc[0]
        b += box(f"À rappel 0.5 (retrouver la moitié de la classe), la meilleure précision est <b>{r5['@ 0.5']:.3f}</b> avec <b>{r5['Modèle']}</b> ({r5['Métrique']}). "
                 f"À rappel 1.0 (toute la classe), la meilleure précision tombe à {t['@ 1.0'].max():.3f} : les derniers exemplaires de chaque classe sont les plus difficiles à retrouver.", "res", "Constat")
    page("courbes", "Étape 4 · Courbes Précision-Rappel", B4("courbes") + b)


def p_classes():
    b = box("Pour chaque classe, l'AP est moyennée sur toutes les images de cette classe prises comme requêtes. Cela montre quelles catégories sont "
            "faciles (visuellement homogènes) ou difficiles (variables, ou proches d'une autre classe).", titre="Principe")
    for d in DSS:
        b += f"<h2>{nd(d)}</h2>"; items = []
        for met in C.METRICS:
            sub = PC[(PC.dataset == d) & (PC.metrique == met)]
            piv = sub.pivot(index="classe", columns="model", values="AP")[[m for m in MODS if m in set(sub.model)]]
            piv["moyenne"] = piv.mean(axis=1); piv = piv.sort_values("moyenne", ascending=False)
            z = piv.drop(columns="moyenne")
            hm = go.Figure(go.Heatmap(z=z.values, x=[nm(c) for c in z.columns], y=list(z.index), colorscale="Viridis", zmin=0,
                                      text=np.round(z.values, 2) if len(z) <= 30 else None, texttemplate="%{text}" if len(z) <= 30 else None))
            hm.update_yaxes(autorange="reversed"); hm.update_layout(title=f"AP par classe – {nd(d)} ({met})")
            t = z.copy(); t.columns = [nm(c) for c in t.columns]; t["Moyenne"] = piv["moyenne"]; t["Meilleur modèle"] = z.idxmax(axis=1).map(nm)
            t = t.reset_index().rename(columns={"classe": "Classe"})
            ez = piv.moyenne
            txt = (f"Classes les plus faciles : " + ", ".join(f"<b>{esc(c)}</b> ({v:.3f})" for c, v in ez.head(3).items()) +
                   ". Classes les plus difficiles : " + ", ".join(f"<b>{esc(c)}</b> ({v:.3f})" for c, v in ez.tail(3).items()) +
                   f". Écart entre la meilleure et la pire classe : {ez.max() - ez.min():.3f}.")
            items.append((met.capitalize(), box(txt, "res", "Constat") + fig_html(hm, max(380, 22 * len(z) + 120)) +
                          table(t, best_max=[c for c in t.columns if c in [nm(m) for m in MODS] + ["Moyenne"]], filt=True)))
        b += tabs(items)
    page("classes", "Étape 4 · Comparaison entre classes", B4("classes") + b)


def top_pairs(ds, model, met, n=15):
    M = np.load(C.CONF_DIR / f"{ds}__{model}__{met}.npy").copy(); names = class_names(MAN[ds]); np.fill_diagonal(M, 0)
    ix = np.dstack(np.unravel_index(np.argsort(-M.ravel())[:n], M.shape))[0]
    return [(names[i], names[j], float(M[i, j])) for i, j in ix]


def p_confusion():
    b = box("Ligne = classe de la requête, colonne = classe des images retrouvées dans le <b>Top-10</b> (part moyenne). Une diagonale forte = recherche fiable ; "
            "une case claire hors diagonale = deux classes que le réseau confond. Utilisez les boutons pour changer de modèle/métrique.", titre="Comment lire")
    for d in DSS:
        names = class_names(MAN[d]); br = best_row(d); combos = [(m, met) for m in MODS for met in C.METRICS if (C.CONF_DIR / f"{d}__{m}__{met}.npy").exists()]
        fig = go.Figure(); buttons = []
        start = combos.index((br.model, br.metrique))
        for i, (m, met) in enumerate(combos):
            fig.add_heatmap(z=np.load(C.CONF_DIR / f"{d}__{m}__{met}.npy"), x=names, y=names, colorscale="Magma", visible=(i == start), showscale=(i == start))
            vis = [j == i for j in range(len(combos))]
            buttons.append(dict(label=f"{nm(m)} · {met}", method="update", args=[{"visible": vis, "showscale": vis}]))
        fig.update_layout(updatemenus=[dict(buttons=buttons, active=start, x=0, y=1.18, xanchor="left")], xaxis_title="Classe retrouvée",
                          yaxis_title="Classe requête", yaxis_autorange="reversed")
        b += f"<h2>{nd(d)}</h2>" + fig_html(fig, 720 if len(names) > 30 else 600)
        t = pd.DataFrame(top_pairs(d, br.model, br.metrique), columns=["Classe requête", "Classe retrouvée à tort", "Part du Top-10"])
        t["Part du Top-10"] = (t["Part du Top-10"] * 100).round(2).astype(str) + " %"
        a, c, v = top_pairs(d, br.model, br.metrique, 1)[0]
        b += box(f"Pour la meilleure configuration ({br.modele}, {br.metrique}), la confusion la plus forte est <b>{esc(a)} → {esc(c)}</b> ({v:.1%} du Top-10).", "res", "Constat")
        b += table(t, fmt={})
    page("confusion", "Étape 4 · Matrices de confusion (Top-10)", B4("confusion") + b)


def p_comparaisons():
    b = "<h2>1. Cosinus vs Euclidienne</h2>"
    d_ = G.pivot_table(index=["dataset_nom", "modele"], columns="metrique", values="mAP").reset_index()
    d_["Écart (cos − euc)"] = d_["cosinus"] - d_["euclidienne"]; d_["Gagnant"] = np.where(d_["Écart (cos − euc)"] > 0, "cosinus", "euclidienne")
    d_ = d_.rename(columns={"dataset_nom": "Base", "modele": "Modèle"})
    wins = int((d_["Écart (cos − euc)"] > 0).sum()); moy = d_["Écart (cos − euc)"].mean()
    b += box(f"Le cosinus gagne dans <b>{wins}/{len(d_)}</b> configurations ; écart moyen <b>{moy:+.4f}</b> mAP. " +
             ("La normalisation L2 supprime l'effet de la norme des vecteurs, qui varie avec le contenu et pénalise l'euclidienne brute." if moy > 0
              else "L'euclidienne brute fait au moins aussi bien ici : la norme des descripteurs porte une information utile."), "res", "Constat")
    b += table(d_, best_max=["cosinus", "euclidienne"]) + fig_html(px.bar(d_, x="Modèle", y="Écart (cos − euc)", color="Base", barmode="group", title="Écart de mAP (positif = cosinus meilleur)"), 380)
    b += "<h3>… classe par classe (meilleur modèle de chaque base)</h3>"
    for d in DSS:
        br = best_row(d); s = PC[(PC.dataset == d) & (PC.model == br.model)].pivot(index="classe", columns="metrique", values="AP").reset_index()
        f = px.scatter(s, x="cosinus", y="euclidienne", hover_name="classe", title=f"AP par classe – {nd(d)} ({br.modele})")
        mx = float(max(s.cosinus.max(), s.euclidienne.max())); f.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dash"))
        b += fig_html(f, 420)
    b += box("Chaque point est une classe ; au-dessus de la diagonale, l'euclidienne est meilleure pour cette classe, en dessous le cosinus.", "note")

    b += "<h2>2. Entre modèles : légers vs classiques</h2>"
    fam = G[G.metrique == "cosinus"].groupby(["dataset_nom", "famille"]).agg(mAP_moyen=("mAP", "mean"), ms_extraction=("ms_extraction_img", "mean"), dimension_moy=("dim", "mean")).reset_index()
    fam = fam.rename(columns={"dataset_nom": "Base", "famille": "Famille", "mAP_moyen": "mAP moyen", "ms_extraction": "Extraction ms/img", "dimension_moy": "Dimension moy."})
    b += table(fam, best_max=["mAP moyen"], best_min=["Extraction ms/img"], fmt={"Extraction ms/img": "{:.2f}", "Dimension moy.": "{:.0f}"})
    e = G[G.metrique == "cosinus"]
    b += fig_html(px.scatter(e, x="ms_extraction_img", y="mAP", color="dataset_nom", size="dim", text="modele", log_x=True, title="Compromis précision / vitesse (taille = dimension)",
                             labels={"ms_extraction_img": "Extraction (ms/image, log)"}), 480)
    hm = G[G.metrique == "cosinus"].pivot(index="modele", columns="dataset_nom", values="mAP").reindex([nm(m) for m in MODS])
    b += "<h3>mAP modèle × base (cosinus)</h3>" + fig_html(px.imshow(hm, text_auto=".3f", color_continuous_scale="Viridis", aspect="auto"), 380)
    rk = G[G.metrique == "cosinus"].assign(Rang=lambda x: x.groupby("dataset")["mAP"].rank(ascending=False).astype(int))
    rk = rk.pivot(index="modele", columns="dataset_nom", values="Rang").reindex([nm(m) for m in MODS]); rk["Rang moyen"] = rk.mean(axis=1)
    b += "<h3>Classement des modèles par base (1 = meilleur)</h3>" + table(rk.reset_index().rename(columns={"modele": "Modèle"}), best_min=list(rk.columns), fmt={"Rang moyen": "{:.2f}"})

    b += "<h2>3. Entre bases et types de fichiers</h2>"
    rows = []
    for d in DSS:
        s = ST[d]; g = G[(G.dataset == d) & (G.metrique == "cosinus")]
        rows.append({"Base": nd(d), "Extension": ", ".join(s["extensions"]), "Résolution moy.": f"{s['largeur_moy']:.0f}×{s['hauteur_moy']:.0f}", "Largeur (px)": s["largeur_moy"],
                     "Taille moy. (Ko)": s["taille_moy_ko"], "Classes": s["n_classes"], "mAP moyen": g.mAP.mean(), "mAP max": g.mAP.max(), "P@10 moyen": g["P@10"].mean()})
    t = pd.DataFrame(rows); b += table(t, best_max=["mAP moyen", "mAP max"], fmt={"Largeur (px)": "{:.0f}", "Taille moy. (Ko)": "{:.2f}"})
    easy, hard = t.sort_values("mAP moyen").iloc[-1], t.sort_values("mAP moyen").iloc[0]
    b += box(f"Base la plus facile : <b>{easy.Base}</b> (mAP moyen {easy['mAP moyen']:.3f}) ; la plus difficile : <b>{hard.Base}</b> ({hard['mAP moyen']:.3f}). "
             "Deux facteurs se mélangent : le <b>nombre de classes</b> (plus il est grand, plus les classes sont proches) et la <b>qualité des images</b> "
             "(PNG 32×32 agrandis = floues et hors distribution ImageNet ; JPEG ≈ 384×256 de Corel nets ; grandes photos Paris avec changements de point de vue).", "res", "Constat")
    b += fig_html(px.scatter(t, x="Largeur (px)", y="mAP moyen", color="Extension", size="Taille moy. (Ko)", text="Base", log_x=True, title="mAP selon la résolution et le format"), 400)
    box_df = PC[PC.metrique == "cosinus"].groupby(["dataset", "classe"]).AP.mean().reset_index(); box_df["Base"] = box_df.dataset.map(nd)
    b += fig_html(px.box(box_df, x="Base", y="AP", points="all", hover_name="classe", title="Dispersion de l'AP entre les classes de chaque base (cosinus, moyenne des modèles)"), 440)
    page("comparaisons", "Comparaisons croisées", b)


def examples_html():
    b = box("Pour chaque base : le meilleur modèle, deux requêtes tirées de la <b>classe la plus facile</b> et deux de la <b>classe la plus difficile</b>. "
            "Bordure <span class='pos'>verte</span> = même classe que la requête, <span class='neg'>rouge</span> = autre classe. d = distance à la requête.", titre="Comment lire")
    rng = np.random.default_rng(C.SEED)
    for d in DSS:
        br = best_row(d, "cosinus"); df = MAN[d]; F = load_features(d, br.model)
        pc = PC[(PC.dataset == d) & (PC.model == br.model) & (PC.metrique == "cosinus")].sort_values("AP", ascending=False)
        b += f"<h2>{nd(d)} – {br.modele}</h2>"
        for lab, cls in [("classe facile", pc.classe.iloc[0]), ("classe difficile", pc.classe.iloc[-1])]:
            ap_c = float(pc[pc.classe == cls].AP.iloc[0])
            b += f"<h3>{lab.capitalize()} : « {esc(cls)} » (AP = {ap_c:.3f})</h3>"
            pool = df.index[df.class_name == cls].values
            for qi in rng.choice(pool, size=min(2, len(pool)), replace=False):
                qi = int(qi); k = 8; ex = f'<div class="ex"><b>Requête #{qi}</b> – classe {esc(cls)}'
                for met in C.METRICS:
                    idx, dist = top_k(F, F[qi], met, k, exclude=qi); ok = df.label.values[idx] == df.label[qi]
                    cells = f'<figure class="q"><img src="{thumb(df.path[qi])}"><figcaption>requête</figcaption></figure>'
                    cells += "".join(f'<figure class="{"ok" if o else "ko"}"><img src="{thumb(df.path[i])}"><figcaption>{esc(df.class_name[i])}<br>d={dd:.3f}</figcaption></figure>'
                                     for i, dd, o in zip(idx, dist, ok))
                    ex += f'<div class="rowx"><div class="lab"><b>{met}</b><br>{int(ok.sum())}/{k} pertinentes</div>{cells}</div>'
                b += ex + "</div>"
    return b


def p_conclusion():
    L = A.conclusion_points(R)
    b = "<h2>Synthèse des résultats mesurés</h2><div class='card'><ul>" + "".join(f"<li>{x}</li>" for x in L) + "</ul></div>"
    b += "<h2>Réponses aux questions du cahier des charges</h2><div class='card'><ul>"
    b += "<li><b>Quelle famille de réseaux extrait les meilleurs descripteurs ?</b> Voir « Meilleur modèle en moyenne » et le classement par base (page Comparaisons).</li>"
    b += "<li><b>Cosinus ou euclidienne ?</b> Voir la synthèse ci-dessus et la comparaison classe par classe.</li>"
    b += "<li><b>Le coût (dimension, temps) se justifie-t-il ?</b> Voir le compromis précision/vitesse et le « meilleur compromis ».</li>"
    b += "<li><b>Comment varie la difficulté selon la base ?</b> Elle dépend du nombre de classes, de leur proximité et de la résolution/format des images.</li></ul></div>"
    b += box(f"Mode d'exécution : <code>{esc(PRESET)}</code>. En mode <i>quick</i>, CIFAR-100 et Paris6k sont sous-échantillonnés (nombre d'images par classe plafonné) : "
             "les valeurs absolues de mAP peuvent différer légèrement du protocole complet, mais les comparaisons entre modèles et métriques restent valides.<br>"
             "Autres limites : réseaux non fine-tunés ; pertinence = même classe (pas de bounding boxes ni de requêtes Oxford/Paris officielles) ; CIFAR-100 agrandi de 32 à 224 px.", "note", "Limites")
    b += box("Pistes : fine-tuning sur la base cible, descripteurs multi-échelles, réduction PCA/whitening, recherche approximative (FAISS) pour passer à grande échelle, "
             "re-classement des résultats (query expansion).", titre="Perspectives")
    page("conclusion", "Conclusion et recommandations", b)


# ================================================================= PAGES PAR ÉTAPE
GFLOPS = {"squeezenet1_1": 0.35, "mobilenet_v3_small": 0.06, "resnet18": 1.81, "resnet50": 4.09, "vgg16": 15.47}  # ≈ docs torchvision
STAGE_PAGE = {1: "etape1", 2: "etape2", 3: "etape3", 4: "resultats", 5: "etape5"}
STAGE_LAB = {1: "Préparation", 2: "Extraction", 3: "Recherche", 4: "Évaluation", 5: "Rapport"}
SUB4 = [("resultats", "Résultats globaux"), ("courbes", "Courbes P-R"), ("classes", "Par classe"), ("confusion", "Confusions")]


def banner(n, role, script, entree, sortie, sub=None):
    st = "".join(f'<a class="st{" on" if i == n else ""}" href="{STAGE_PAGE[i]}.html"><b>{i}</b> {STAGE_LAB[i]}</a>' for i in range(1, 6))
    s = f'<div class="stepper">{st}</div>'
    if sub:
        s += '<div class="subnav">' + "".join(f'<a href="{p}.html" class="{"on" if p == sub else ""}">{t}</a>' for p, t in SUB4) + "</div>"
    s += (f'<div class="card"><b>Rôle de l\'étape :</b> {role}<br><b>Script :</b> <code>{script}</code><br>'
          f'<b>Entrée :</b> {entree}<br><b>Sortie :</b> {sortie}</div>')
    return s


B4 = lambda sub: banner(4, "Mesurer la qualité de la recherche : chaque image sert de requête, on classe toute la base et on compare au résultat attendu (même classe).",
                        "etape4_evaluation.py", "vecteurs <code>workdir/features/*.npy</code> + manifestes", "<code>results/metrics_global.csv</code>, <code>metrics_par_classe.csv</code>, <code>pr_curves.json</code>, <code>confusion/*.npy</code>", sub)


def p_etape2():
    T, norms = A.extraction_table(R); rng = np.random.default_rng(C.SEED)
    b = banner(2, "Transformer chaque image en <b>un vecteur de nombres</b> (le descripteur) avec un réseau pré-entraîné ; cette indexation est faite <b>une seule fois</b>, hors-ligne.",
               "etape2_extraction.py", "manifestes <code>workdir/manifests/*.csv</code> (images 224×224 normalisées)",
               "un fichier <code>workdir/features/&lt;base&gt;__&lt;modèle&gt;.npy</code> (N × d) + un <code>.json</code> de métadonnées")
    b += box("Aucune image n'est stockée dans l'index : seulement ses N vecteurs. La recherche (étape 3) ne relit plus jamais les images. "
             "Le vecteur est la sortie de la couche <b>Global Average Pooling</b> (ou fc7 pour VGG-16) : la tête de classification ImageNet est retirée.", titre="Principe")
    tot_min = T["Temps total (min)"].sum(); slow, fast = T.groupby("Modèle")["ms/image"].mean().idxmax(), T.groupby("Modèle")["ms/image"].mean().idxmin()
    ratio = T.groupby("Modèle")["ms/image"].mean().max() / T.groupby("Modèle")["ms/image"].mean().min()
    kp = [(f"{int(T.drop_duplicates('dataset').Images.sum())}", "images indexées"), (f"{len(T)}", "index de vecteurs créés"), (f"{tot_min:.0f} min", "temps de calcul cumulé (réseau)"),
          (f"{T['Index (Mo)'].sum():.0f} Mo", "taille totale des index")]
    b += '<div class="grid">' + "".join(f'<div class="card kpi"><b>{v}</b><span>{l}</span></div>' for v, l in kp) + "</div>"
    b += box(f"<b>{slow}</b> est le plus lent ({T.groupby('Modèle')['ms/image'].mean().max():.1f} ms/image) et <b>{fast}</b> le plus rapide "
             f"({T.groupby('Modèle')['ms/image'].mean().min():.1f} ms/image) : un facteur <b>{ratio:.0f}×</b>. Le coût dépend surtout du nombre d'opérations du réseau "
             "(GFLOPs ci-dessous), pas de la base.", "res", "Constat")
    b += "<h2>Tableau détaillé de l'extraction</h2>" + table(T.drop(columns=["dataset", "model"]), best_max=["Séparabilité", "mAP (étape 4)"],
                                                           best_min=["ms/image", "Temps total (min)"],
                                                           fmt={"Index (Mo)": "{:.1f}", "Temps total (min)": "{:.1f}", "ms/image": "{:.2f}", "images/s": "{:.1f}", "Norme moy.": "{:.2f}",
                                                                "Var. des normes (%)": "{:.1f}", "Valeurs nulles (%)": "{:.1f}"}, filt=True)
    b += box("<b>Index (Mo)</b> = N × dimension × 4 octets. <b>Valeurs nulles</b> : les couches ReLU produisent beaucoup de zéros (vecteurs « creux »). "
             "<b>Var. des normes</b> : variation (écart-type / moyenne) de la longueur des vecteurs d'une image à l'autre : si elle est forte, la distance euclidienne "
             "est perturbée par la norme alors que le cosinus ne l'est pas.", "note", "Colonnes")
    g = pd.DataFrame({"Modèle": [nm(m) for m in MODS], "GFLOPs (224×224)": [GFLOPS[m] for m in MODS], "Paramètres (M)": [float(G[G.model == m].params_M.iloc[0]) for m in MODS],
                      "ms/image mesuré": [T[T.model == m]["ms/image"].mean() for m in MODS]})
    b += "<h2>Coût de calcul des réseaux</h2>" + table(g, fmt={"GFLOPs (224×224)": "{:.2f}", "Paramètres (M)": "{:.1f}", "ms/image mesuré": "{:.2f}"})
    c = make_subplots(rows=1, cols=2, subplot_titles=("Temps de calcul par image (ms)", "Temps total d'extraction par base (min)"))
    a = T.groupby("Modèle", sort=False)["ms/image"].mean()
    c.add_bar(x=list(a.index), y=list(a.values), showlegend=False, row=1, col=1)
    for mdl in T.Modèle.unique():
        s = T[T.Modèle == mdl]; c.add_bar(x=list(s.Base), y=list(s["Temps total (min)"]), name=mdl, row=1, col=2)
    c.update_layout(barmode="stack"); b += fig_html(c, 400)
    b += box("Le temps est proportionnel à GFLOPs : VGG-16 effectue ≈ 15 milliards d'opérations par image contre ≈ 0,06 pour MobileNetV3-Small. "
             "C'est pourquoi VGG-16 domine le temps total. Mesures réalisées sur <i>votre machine</i> (CPU ou GPU).", "note", "Pourquoi ces écarts ?")

    b += "<h2>Qualité des descripteurs, avant même la recherche</h2>"
    f = px.bar(T, x="Modèle", y="Séparabilité", color="Base", barmode="group", title="Séparabilité = cosinus moyen même classe − cosinus moyen classes différentes")
    b += fig_html(f, 400)
    sp = T[["Séparabilité", "mAP (étape 4)"]].corr(method="spearman").iloc[0, 1] if len(T) >= 4 else float("nan")
    bs = T.sort_values("Séparabilité", ascending=False).iloc[0]
    b += box(f"Plus l'écart entre « même classe » et « classes différentes » est grand, plus les images de même classe sont regroupées. Le meilleur écart est obtenu par "
             f"<b>{bs.Modèle}</b> sur {bs.Base} ({bs['Séparabilité']:.3f}). " +
             (f"Ce indicateur, calculé dès l'étape 2 sur un échantillon, est corrélé au mAP final de l'étape 4 (corrélation de Spearman ρ = <b>{sp:.2f}</b>) : il prédit donc bien le classement des configurations." if sp == sp else ""),
             "res", "Constat")
    b += "<h2>Longueur (norme) des vecteurs</h2>"
    nb = go.Figure()
    for m in MODS:
        if m in norms: nb.add_box(y=norms[m], name=nm(m), boxpoints=False)
    nb.update_layout(title=f"Distribution des normes – {nd(DSS[0])}", yaxis_title="Norme L2")
    b += fig_html(nb, 380) + box("La norme n'est pas constante d'une image à l'autre. Le <b>cosinus</b> l'ignore (les vecteurs sont normalisés), la distance <b>euclidienne</b> y est sensible : "
                                 "c'est l'explication de l'écart observé entre les deux métriques à l'étape 4.", "note", "Pourquoi c'est important")

    b += "<h2>Projection 2D des vecteurs (ACP)</h2>" + box("Chaque point est une image ; la couleur est sa classe. Les nuages bien séparés annoncent une recherche facile ; "
                                                         "les nuages mélangés une recherche difficile. (Échantillon de 2000 images, meilleur modèle de chaque base.)", titre="Comment lire")
    for d in DSS:
        bm = best_row(d, "cosinus").model; F = load_features(d, bm); df = MAN[d]
        s, P = A.pca2(F, rng)
        pdf = pd.DataFrame({"x": P[:, 0], "y": P[:, 1], "Classe": df.class_name.values[s]})
        fig = px.scatter(pdf, x="x", y="y", color="Classe", title=f"{nd(d)} – {nm(bm)}", render_mode="webgl")
        if df.class_name.nunique() > 20: fig.update_layout(showlegend=False)
        fig.update_traces(marker=dict(size=5, opacity=.75)); b += fig_html(fig, 480)
    page("etape2", "Étape 2 · Extraction des vecteurs caractéristiques", b)


def p_etape3():
    b = banner(3, "Pour une <b>image requête</b>, calculer sa distance à tous les vecteurs de la base et retourner les <b>K plus proches</b> (Top-K). C'est la phase « en ligne ».",
               "etape3_recherche.py (et fonctions de <code>common.py</code>)", "un vecteur requête + la base de vecteurs (étape 2)", "la liste ordonnée des K images les plus proches (figure <code>recherche_*.png</code>)")
    b += box("<ol><li>Extraire (ou relire) le vecteur <i>q</i> de la requête.</li><li>Calculer <span class='f'>d(q, xᵢ)</span> pour toutes les images de la base (une seule multiplication matricielle).</li>"
             "<li>Trier par distance croissante et garder les K premières.</li></ol>Recherche <b>exhaustive exacte</b> : coût ∝ N × dimension. "
             "Pour des bases beaucoup plus grandes, on utiliserait une recherche approximative (FAISS).", titre="Algorithme")
    b += "<h2>Qualité immédiate : les premiers résultats</h2><div class='grid'>"
    for d in DSS:
        r = best_row(d, "cosinus")
        b += (f'<div class="card"><b>{nd(d)}</b> · {r.modele}<br><span class="pos">1er résultat correct : {r["P@1"]:.1%}</span><br>'
              f'Top-5 correct : {r["P@5"]:.1%} · Top-10 correct : {r["P@10"]:.1%}</div>')
    b += "</div>" + box("« 1er résultat correct » = proportion de requêtes dont l'image la plus proche est de la bonne classe (P@1). Ce que voit l'utilisateur en premier.", "note")

    b += "<h2>Coût d'une recherche</h2>"
    lat = G.copy(); lat["Série"] = lat.modele + " · " + lat.metrique
    b += fig_html(px.bar(lat, x="dataset_nom", y="ms_recherche_req", color="Série", barmode="group",
                         labels={"dataset_nom": "Base", "ms_recherche_req": "ms / requête"}, title="Temps moyen d'une recherche exhaustive (ms)"), 400)
    t = lat.pivot_table(index=["dataset_nom", "modele"], columns="metrique", values="ms_recherche_req").reset_index().rename(columns={"dataset_nom": "Base", "modele": "Modèle"})
    b += table(t, best_min=["cosinus", "euclidienne"], fmt={"cosinus": "{:.2f}", "euclidienne": "{:.2f}"})
    b += box("Le temps croît avec la taille de la base et la dimension du descripteur (VGG-16 : 4096 valeurs contre 512). "
             "Il est mesuré en lot (plusieurs requêtes à la fois) ; une requête isolée est un peu plus lente.", "note", "Lecture")

    b += "<h2>Pourquoi la recherche réussit ou échoue : distribution des distances</h2>" + box(
        "Pour 200 requêtes, on compare les distances (cosinus) aux images <b>de même classe</b> et aux images <b>d'autres classes</b>. "
        "Si les deux histogrammes se séparent, les bonnes images sont plus proches : la recherche est facile. S'ils se chevauchent, la recherche est difficile.", titre="Comment lire")
    rng = np.random.default_rng(C.SEED)
    for d in DSS:
        bm = best_row(d, "cosinus").model; F = load_features(d, bm); y = MAN[d].label.values
        ds_, dd_ = A.dist_samples(F, y, MAN[d], d, rng)
        f = go.Figure(); f.add_histogram(x=ds_, name="même classe (pertinentes)", histnorm="probability density", opacity=.65, nbinsx=60)
        f.add_histogram(x=dd_, name="autres classes", histnorm="probability density", opacity=.65, nbinsx=60)
        f.update_layout(barmode="overlay", title=f"{nd(d)} – {nm(bm)} (cosinus)", xaxis_title="Distance cosinus", yaxis_title="Densité")
        b += fig_html(f, 360) + box(f"Distance moyenne : <b>{ds_.mean():.3f}</b> (même classe) contre <b>{dd_.mean():.3f}</b> (autres classes) ; écart <b>{dd_.mean() - ds_.mean():.3f}</b>. "
                                    "Plus cet écart est grand par rapport à la largeur des histogrammes, plus la recherche est fiable.", "res", "Constat")
    b += "<h2>Exemples de recherche (cosinus vs euclidienne)</h2>" + examples_html()
    page("etape3", "Étape 3 · Recherche et mesure de similarité", b)


def p_etape5():
    src = C.RESULTS / "rapport.html"; pl = pdf_links()
    b = banner(5, "Rassembler tous les résultats mesurés en un <b>rapport rédigé</b> (tableaux, figures, analyses) généré automatiquement, en PDF et en HTML.", "etape5_rapport.py (HTML) et etape7_pdf.py (PDF)",
               "<code>results/metrics_*.csv</code>, <code>pr_curves.json</code>, figures", "<code>results/pdf/rapport_complet.pdf</code> et <code>results/rapport.html</code>")
    items = []
    if "rapport_complet" in pl:
        bar = '<div class="dlbar"><a class="btn pri" href="%s" download>Rapport PDF complet</a>' % pl["rapport_complet"]
        bar += "".join(f'<a class="btn" href="{pl[f"etape{i}"]}" download>PDF étape {i}</a>' for i in range(1, 5) if f"etape{i}" in pl) + "</div>"
        items.append(("Rapport PDF", bar + f'<iframe src="{pl["rapport_complet"]}" style="width:100%;height:900px;border:1px solid var(--bd);border-radius:12px;background:#fff"></iframe>'))
    if src.exists():
        shutil.copy(src, SITE / "rapport.html")
        items.append(("Rapport HTML", '<div class="dlbar"><a class="btn" href="rapport.html" target="_blank">Ouvrir dans un nouvel onglet</a><a class="btn" href="rapport.html" download>Télécharger</a></div>'
                      '<iframe src="rapport.html" style="width:100%;height:900px;border:1px solid var(--bd);border-radius:12px;background:#fff"></iframe>'))
    b += tabs(items) if items else box("Aucun rapport généré : lancez <code>python etape6_site_html.py</code> (PDF) et <code>python etape5_rapport.py</code> (HTML).", "note")
    b += box("Le PDF contient : couverture, sommaire cliquable, puis pour chaque étape son rôle, ses résultats chiffrés, ses figures et ses explications ; "
             "suivent les comparaisons croisées et la conclusion. Un PDF séparé est aussi fourni pour chacune des étapes 1 à 4.", titre="Contenu")
    page("etape5", "Étape 5 · Rapport", b)


def main():
    global ART
    ap = argparse.ArgumentParser(); ap.add_argument("--sans-pdf", action="store_true", help="ne pas générer les images/PDF (plus rapide)")
    a = ap.parse_args()
    if not a.sans_pdf:
        print("  · étape 7 : images + PDF (quelques dizaines de secondes)…")
        import etape7_pdf as P7
        ART = P7.build_all()
    for f in (p_index, p_donnees, p_etape2, p_etape3, p_resultats, p_courbes, p_classes, p_confusion, p_etape5, p_comparaisons, p_modeles, p_methode, p_conclusion):
        print("  ·", f.__name__); f()
    print("\n[OK] Site généré :", SITE / "index.html")


if __name__ == "__main__":
    main()
