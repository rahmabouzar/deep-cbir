"""Habillage de l'application, inspiré du design system « Notion » (DESIGN-notion.md) :
papier chaud #f6f5f4, cartes blanches à bordure fine, un seul bleu d'action, pastilles « stickers » colorées pour les icônes.

Icônes : si un fichier assets/icons/<nom>.svg existe (par ex. exporté depuis Streamline Flex Pop avec votre licence), il est utilisé ;
sinon une icône simple de secours est affichée. Voir assets/icons/README.md.
"""
import base64
import re
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------ Jetons de design (DESIGN-notion.md)
PRIMARY, PRIMARY_ACTIVE, INDIGO = "#0075de", "#005bab", "#213183"
CANVAS_SOFT, HAIRLINE = "#f6f5f4", "#e6e6e6"
INK, INK_MUTED, INK_FAINT = "#0d0d0d", "#615d59", "#a39e98"
# palette « stickers » : décorative (icônes, séries de graphiques), jamais pour les actions
STICKERS = {"blue": ("#e7f2fc", PRIMARY), "purple": ("#f1e8fc", "#7a4cc2"), "orange": ("#fdeee3", "#dd5b00"),
            "teal": ("#e2f4f3", "#2a9d99"), "green": ("#e3f6e7", "#1aae39"), "pink": ("#ffe6f6", "#d6339c")}
COLORWAY = [PRIMARY, "#dd5b00", "#2a9d99", "#ff64c8", INDIGO, "#1aae39", "#62aef0"]
SHADOW = ("0 .175px 1.041px rgba(0,0,0,.01), 0 .8px 2.925px rgba(0,0,0,.02), "
          "0 2.025px 7.847px rgba(0,0,0,.027), 0 4px 18px rgba(0,0,0,.04)")

ICON_DIR = Path(__file__).parent / "assets" / "icons"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp input, .stApp textarea, .stApp button, .stApp [data-baseweb="select"] {{
    font-family: Inter, -apple-system, system-ui, "Segoe UI", Helvetica, Arial, sans-serif; font-feature-settings: "lnum"; }}
.stApp {{ background: {CANVAS_SOFT}; color: {INK}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
footer, #MainMenu {{ visibility: hidden; }}

/* ---- typographie */
.stApp h1 {{ font-weight: 700; letter-spacing: -1px; color: {INK}; }}
.stApp h2, .stApp h3 {{ font-weight: 700; letter-spacing: -.3px; color: {INK}; }}
.stApp p, .stApp li {{ line-height: 1.5; }}

/* ---- en-tête de page */
.ph {{ display: flex; gap: 1rem; align-items: center; margin: 0 0 1.6rem 0; }}
.ph .tile {{ flex: none; width: 3.2rem; height: 3.2rem; border-radius: 14px; display: flex; align-items: center; justify-content: center; }}
.ph .tile img {{ width: 1.9rem; height: 1.9rem; }}
.pill {{ display: inline-block; background: #fff; color: {PRIMARY}; border: 1px solid {HAIRLINE}; border-radius: 9999px;
        padding: 3px 10px; font-size: 12px; font-weight: 600; letter-spacing: .125px; }}
.ph h1 {{ margin: .35rem 0 .2rem 0; padding: 0; font-size: 2.1rem; line-height: 1.1; letter-spacing: -1px; }}
.ph p {{ margin: 0; color: {INK_MUTED}; font-size: 1rem; max-width: 60rem; }}

/* ---- bandeau d'accueil (le seul bloc sombre, comme le hero Notion) */
.hero {{ background: {INDIGO}; color: #fff; border-radius: 16px; padding: 2.6rem 2.8rem; margin: 0 0 1.6rem 0; }}
.hero h1 {{ color: #fff; font-size: 2.9rem; line-height: 1.03; letter-spacing: -1.8px; margin: .7rem 0 .8rem 0; padding: 0; }}
.hero p {{ color: #d5dbfa; font-size: 1.1rem; max-width: 46rem; margin: 0 0 1.2rem 0; }}
.hero .pill {{ background: rgba(255,255,255,.12); color: #fff; border-color: rgba(255,255,255,.22); margin-right: .35rem; }}

/* ---- étapes */
.step {{ display: flex; align-items: center; gap: .75rem; margin: 1.8rem 0 .7rem 0; }}
.step .n {{ width: 1.9rem; height: 1.9rem; border-radius: 9999px; background: {PRIMARY}; color: #fff; display: inline-flex;
           align-items: center; justify-content: center; font-weight: 700; font-size: .9rem; }}
.step .t {{ font-size: 1.35rem; font-weight: 700; letter-spacing: -.3px; color: {INK}; }}
.step .s {{ color: {INK_MUTED}; font-size: .92rem; }}

/* ---- cartes de statistiques */
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: .2rem 0 1.2rem 0; }}
.stat {{ background: #fff; border: 1px solid {HAIRLINE}; border-radius: 12px; padding: 18px 20px; display: flex; gap: 14px; align-items: center; }}
.stat .tile {{ flex: none; width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; }}
.stat .tile img {{ width: 26px; height: 26px; }}
.stat .lab {{ font-size: 12px; font-weight: 600; letter-spacing: .125px; color: {INK_MUTED}; text-transform: uppercase; }}
.stat .val {{ font-size: 1.55rem; font-weight: 700; letter-spacing: -.5px; line-height: 1.15; color: {INK}; }}
.stat .sub {{ font-size: 13px; color: {INK_FAINT}; }}

/* ---- surfaces : bordure fine + ombre très légère */
div[data-testid="stPlotlyChart"] {{ background: #fff; border: 1px solid {HAIRLINE}; border-radius: 12px; padding: .6rem; box-shadow: {SHADOW}; }}
div[data-testid="stExpander"] {{ background: #fff; border: 1px solid {HAIRLINE}; border-radius: 12px; }}
div[data-testid="stExpander"] details {{ border: none; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {HAIRLINE}; border-radius: 12px; overflow: hidden; background: #fff; }}
div[data-testid="stAlert"] {{ border-radius: 12px; border: 1px solid {HAIRLINE}; }}
div[data-testid="stImage"] img {{ border-radius: 8px; }}
div[data-testid="stMetric"] {{ background: #fff; border: 1px solid {HAIRLINE}; border-radius: 12px; padding: 14px 18px; }}

/* ---- boutons : primaire = pilule bleue, secondaire = utilitaire 8px à bordure fine */
.stButton > button, .stDownloadButton > button {{ font-weight: 500; transition: background .12s, transform .08s; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
    background: {PRIMARY}; color: #fff; border: none; border-radius: 9999px; padding: .45rem 1.3rem; }}
.stButton > button[kind="primary"]:hover {{ background: {PRIMARY_ACTIVE}; color: #fff; }}
.stButton > button[kind="primary"]:active {{ background: {PRIMARY_ACTIVE}; transform: scale(.97); }}
.stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"] {{
    background: #fff; color: {INK}; border: 1px solid {HAIRLINE}; border-radius: 8px; padding: .3rem .9rem; }}
.stButton > button[kind="secondary"]:hover, .stDownloadButton > button[kind="secondary"]:hover {{ border-color: {PRIMARY}; color: {PRIMARY}; }}

/* ---- champs : coins serrés (4px) */
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {{
    border-radius: 4px !important; background: #fff; border-color: #dddddd; }}
div[data-baseweb="input"] > div:focus-within, div[data-baseweb="select"] > div:focus-within {{ border-color: {PRIMARY}; box-shadow: {SHADOW}; }}

/* ---- barre latérale */
section[data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid {HAIRLINE}; }}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{ font-size: 1rem; letter-spacing: -.1px; }}
a[data-testid="stSidebarNavLink"] {{ border-radius: 5px; font-size: 15px; }}
a[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {CANVAS_SOFT}; box-shadow: inset 3px 0 0 {PRIMARY}; font-weight: 600; }}
[data-testid="stNavSectionHeader"] {{ font-size: 12px; font-weight: 600; letter-spacing: .125px; text-transform: uppercase; color: {INK_FAINT}; }}
.side-brand {{ display: flex; align-items: center; gap: .65rem; padding: .1rem 0 .9rem 0; }}
.side-brand .logo {{ width: 2.2rem; height: 2.2rem; border-radius: 10px; background: {INDIGO}; display: flex; align-items: center; justify-content: center; }}
.side-brand .logo img {{ width: 1.3rem; height: 1.3rem; }}
.side-brand b {{ display: block; color: {INK}; line-height: 1.1; letter-spacing: -.2px; }}
.side-brand small {{ color: {INK_MUTED}; }}

/* ---- onglets */
button[data-baseweb="tab"] {{ font-weight: 600; }}
</style>
"""

# ------------------------------------------------------------------ Icônes
# Icônes de secours (tracés simples, 24x24) utilisées tant qu'aucun SVG n'est déposé dans assets/icons/
_GLYPHS = {
    "home": '<path d="M3 11l9-7 9 7"/><path d="M5 10v10h14V10"/>',
    "experience": '<path d="M9 3h6"/><path d="M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3"/>',
    "search": '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/>',
    "bars": '<path d="M5 20V11M12 20V4M19 20v-6"/>',
    "curve": '<path d="M3 19c4 0 4-14 9-14s5 14 9 14"/>',
    "grid": '<rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" height="7" rx="1.5"/>'
            '<rect x="4" y="13" width="7" height="7" rx="1.5"/><rect x="13" y="13" width="7" height="7" rx="1.5"/>',
    "matrix": '<circle cx="6" cy="6" r="1.6"/><circle cx="12" cy="6" r="1.6"/><circle cx="18" cy="6" r="1.6"/><circle cx="6" cy="12" r="1.6"/>'
              '<circle cx="12" cy="12" r="1.6"/><circle cx="18" cy="12" r="1.6"/><circle cx="6" cy="18" r="1.6"/><circle cx="12" cy="18" r="1.6"/><circle cx="18" cy="18" r="1.6"/>',
    "compare": '<path d="M4 8h14l-3-3M20 16H6l3 3"/>',
    "database": '<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.5"/>',
    "image": '<rect x="3.5" y="5" width="17" height="14" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="M4 17l5-4 4 3 3-2 4 3"/>',
}


def _data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def icon_uri(name: str, color: str = PRIMARY) -> str:
    """URI de données d'une icône : fichier assets/icons/<name>.svg s'il existe, sinon icône de secours."""
    f = ICON_DIR / f"{name}.svg"
    if f.exists():
        svg = f.read_text(encoding="utf-8")
        svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
        if "xmlns=" not in svg:
            svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
        return _data_uri(svg)
    body = _GLYPHS.get(name, _GLYPHS["target"])
    return _data_uri(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
                     f'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


def _tile(icon, tint="blue", cls="tile"):
    bg, fg = STICKERS[tint]
    return f'<span class="{cls}" style="background:{bg}"><img src="{icon_uri(icon, fg)}" alt=""/></span>'


# ------------------------------------------------------------------ Composants
def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_brand():
    st.sidebar.markdown(f'<div class="side-brand"><span class="logo"><img src="{icon_uri("search", "#ffffff")}" alt=""/></span>'
                        '<span><b>DeepCBIR</b><small>Master 2 SID · USTO-MB</small></span></div>', unsafe_allow_html=True)


def page_header(title, subtitle="", eyebrow="CBIR · IRDM", icon="target", tint="blue"):
    st.markdown(f'<div class="ph">{_tile(icon, tint)}<div><span class="pill">{eyebrow}</span><h1>{title}</h1>'
                + (f"<p>{subtitle}</p>" if subtitle else "") + "</div></div>", unsafe_allow_html=True)


def hero(title, subtitle="", pills=()):
    st.markdown('<div class="hero"><span class="pill">Master 2 SID · Module IRDM</span>'
                f"<h1>{title}</h1><p>{subtitle}</p>" + "".join(f'<span class="pill">{p}</span>' for p in pills) + "</div>",
                unsafe_allow_html=True)


def step(n, title, hint=""):
    st.markdown(f'<div class="step"><span class="n">{n}</span><span class="t">{title}</span>'
                + (f'<span class="s">{hint}</span>' if hint else "") + "</div>", unsafe_allow_html=True)


def stats(items):
    """Cartes de statistiques. items = [dict(label, value, sub='', icon='bars', tint='blue'), ...]"""
    html = "".join(f'<div class="stat">{_tile(i.get("icon", "bars"), i.get("tint", "blue"))}<div><div class="lab">{i["label"]}</div>'
                   f'<div class="val">{i["value"]}</div>' + (f'<div class="sub">{i["sub"]}</div>' if i.get("sub") else "") + "</div></div>"
                   for i in items)
    st.markdown(f'<div class="stats">{html}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ Graphiques Plotly
def style(fig: go.Figure, height=None) -> go.Figure:
    """Style Plotly homogène : police Inter, palette Notion, légende À DROITE (ne chevauche plus les titres)."""
    fig.update_layout(
        template="plotly_white", colorway=COLORWAY, font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=78, b=10),
        title=dict(font=dict(size=17), x=0.0, xanchor="left", y=0.97, yanchor="top", pad=dict(l=4)),
        legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top", title_text="", bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=13))
    fig.update_xaxes(gridcolor="#efeeec", linecolor=HAIRLINE, zeroline=False)
    fig.update_yaxes(gridcolor="#efeeec", linecolor=HAIRLINE, zeroline=False)
    if height:
        fig.update_layout(height=height)
    return fig


def facets(fig: go.Figure, tickangle=-30) -> go.Figure:
    """Nettoie les graphiques à facettes : « dataset_nom=Corel-1K » -> « Corel-1K », un seul libellé d'axe, étiquettes inclinées."""
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=13, color=INK)))
    fig.update_xaxes(title_text="", tickangle=tickangle, tickfont=dict(size=11))
    return fig


def show(fig: go.Figure, height=None, **kw):
    st.plotly_chart(style(fig, height), width="stretch")


def category_axes(fig, x=None, y=None, big=40):
    """Axes catégoriels dans l'ordre donné (indispensable pour Corel-5K/10K dont les classes s'appellent '0', '1', ...)."""
    for axis, labels in (("x", x), ("y", y)):
        if labels is None:
            continue
        labels = [str(v) for v in labels]
        upd = dict(type="category", categoryorder="array", categoryarray=labels, tickfont=dict(size=10 if len(labels) > 25 else 12))
        if len(labels) > big:
            upd.update(tickmode="array", tickvals=labels[::5], ticktext=labels[::5])
        elif len(labels) > 1:
            upd.update(dtick=1)
        (fig.update_xaxes if axis == "x" else fig.update_yaxes)(**upd)
    return fig
