"""Habillage de l'application. Règles : SKILL.md (landing-page-design, partie B) + DESIGN-notion.md pour la palette.

Une seule source de vérité pour :
  - TEXT    : styles de texte nommés (heading-1 ... caption-5), tailles uniquement sur l'échelle Tailwind ;
  - SPACE   : jetons d'espacement (0, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96 px) ;
  - RADIUS  : rayons Tailwind (0, 2, 4, 6, 8, 12, 16, 24, 9999 px) ;
  - EASE    : courbe de mouvement cubic-bezier(0.22, 1, 0.36, 1).
Le script design_check.py vérifie automatiquement que le CSS généré respecte ces jetons.

Icônes : assets/icons/<nom>.svg (vos exports, p. ex. Streamline) > assets/phosphor/<nom>.svg (Phosphor, licence MIT) > icône de secours.
"""
import base64
import re
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from common import load_rgb

# ------------------------------------------------------------------ Jetons
SPACE = {"0": 0, "25": 2, "50": 4, "75": 8, "100": 12, "200": 16, "300": 24, "400": 32, "500": 40, "600": 48, "700": 64, "800": 80, "900": 96}
RADIUS = {"none": 0, "sm": 2, "base": 4, "md": 6, "lg": 8, "xl": 12, "2xl": 16, "3xl": 24, "full": 9999}
TAILWIND_SIZES = {12: 16, 14: 20, 16: 24, 18: 28, 20: 28, 24: 32, 30: 36, 36: 40, 48: 48, 60: 60, 72: 72, 96: 96, 128: 128}
# nom: (taille px, hauteur de ligne px, graisse) ; 400 regular, 500 medium, 600 semibold (jamais plus)
TEXT = {
    "heading-1": (60, 60, 600), "heading-2": (48, 48, 600), "heading-3": (36, 40, 600), "heading-4": (30, 36, 600),
    "heading-5": (24, 32, 600), "heading-6": (20, 28, 600),
    "title-1": (20, 28, 500), "title-2": (18, 28, 500), "title-3": (16, 24, 500), "title-4": (14, 20, 500), "title-5": (12, 16, 500),
    "body-1": (16, 24, 600), "body-2": (16, 24, 500), "body-3": (16, 24, 400),
    "body-4": (14, 20, 600), "body-5": (14, 20, 500), "body-6": (14, 20, 400),
    "button-lg": (18, 28, 600), "button-md": (16, 24, 600), "button-sm": (14, 20, 600), "button-xsm": (12, 16, 600),
    "caption-1": (14, 20, 500), "caption-2": (14, 20, 400), "caption-3": (12, 16, 600), "caption-4": (12, 16, 500), "caption-5": (12, 16, 400),
}
EASE = "cubic-bezier(0.22, 1, 0.36, 1)"

# ------------------------------------------------------------------ Couleurs (palette Notion)
PRIMARY, PRIMARY_ACTIVE = "#0075de", "#005bab"
CANVAS, SURFACE, HAIRLINE = "#f6f5f4", "#ffffff", "#e6e6e6"
INK, INK_MUTED, INK_FAINT = "#0d0d0d", "#615d59", "#a39e98"
STICKERS = {"blue": ("#e7f2fc", PRIMARY), "purple": ("#f1e8fc", "#7a4cc2"), "orange": ("#fdeee3", "#dd5b00"),
            "teal": ("#e2f4f3", "#2a9d99"), "green": ("#e3f6e7", "#1aae39"), "pink": ("#ffe6f6", "#d6339c")}
COLORWAY = [PRIMARY, "#dd5b00", "#2a9d99", "#ff64c8", "#213183", "#1aae39", "#62aef0"]
SHADOW = ("0 0.175px 1.041px rgba(0,0,0,.01), 0 0.8px 2.925px rgba(0,0,0,.02), "
          "0 2.025px 7.847px rgba(0,0,0,.027), 0 4px 18px rgba(0,0,0,.04)")

ROOT = Path(__file__).parent
ICON_DIRS = [ROOT / "assets" / "icons", ROOT / "assets" / "phosphor"]


def _t(name):                                   # déclaration CSS d'un style de texte nommé
    s, lh, w = TEXT[name]
    return f"font-size:{s}px;line-height:{lh}px;font-weight:{w};"


def build_css() -> str:
    s = {k: f"{v}px" for k, v in SPACE.items()}
    r = {k: f"{v}px" for k, v in RADIUS.items()}
    classes = "".join(f".{n}{{{_t(n)}}}" for n in TEXT)
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&display=swap');
:root {{ --ease: {EASE}; }}
{classes}

/* ---------- base : une seule famille, jamais d'italique, jamais plus que semibold */
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp input, .stApp textarea, .stApp button, .stApp [data-baseweb="select"] {{
    font-family: Geist, system-ui, sans-serif; }}
.stApp {{ background: {CANVAS}; color: {INK}; }}
.stApp em, .stApp i {{ font-style: normal; }}
.stApp p, .stApp li {{ {_t("body-3")} text-wrap: pretty; }}
.stApp h1 {{ {_t("heading-3")} letter-spacing: 0; color: {INK}; text-wrap: balance; padding: 0; }}
.stApp h2 {{ {_t("heading-4")} color: {INK}; text-wrap: balance; }}
.stApp h3 {{ {_t("heading-5")} color: {INK}; text-wrap: balance; }}
.stApp h4 {{ {_t("heading-6")} color: {INK}; }}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p {{ {_t("caption-2")} color: {INK_MUTED}; }}
.stApp [data-testid="stWidgetLabel"] p {{ {_t("body-5")} }}
.block-container {{ padding: {s["600"]} {s["400"]} {s["700"]} {s["400"]}; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
footer, #MainMenu {{ visibility: hidden; }}

/* ---------- en-tête de page */
.ph {{ display: flex; gap: {s["200"]}; align-items: center; margin: 0 0 {s["400"]} 0; }}
.ph .tile {{ flex: none; width: {s["600"]}; height: {s["600"]}; border-radius: {r["lg"]}; display: flex; align-items: center; justify-content: center; }}
.ph .tile img {{ width: {s["300"]}; height: {s["300"]}; }}
.ph h1 {{ margin: 0; }}
.ph p {{ margin: {s["50"]} 0 0 0; color: {INK_MUTED}; max-width: 680px; }}

/* ---------- accueil : bandeau clair, titre en dégradé de texte (seul dégradé autorisé), 680px max */
.hero {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: {r["2xl"]}; padding: {s["600"]}; margin: 0 0 {s["400"]} 0; box-shadow: {SHADOW}; }}
.hero h1 {{ {_t("heading-1")} max-width: 680px; margin: 0 0 {s["200"]} 0; padding: 0; text-wrap: balance;
           background: linear-gradient(90deg, #000000, #666666); -webkit-background-clip: text; background-clip: text;
           -webkit-text-fill-color: transparent; color: transparent; }}
.hero p {{ max-width: 680px; margin: 0 0 {s["300"]} 0; color: {INK_MUTED}; }}
.pills {{ display: flex; flex-wrap: wrap; gap: {s["75"]}; }}
.pill {{ {_t("caption-3")} display: inline-block; background: {CANVAS}; color: {INK}; border: 1px solid {HAIRLINE};
        border-radius: {r["full"]}; padding: {s["50"]} {s["75"]}; }}

/* ---------- étapes numérotées */
.step {{ display: flex; align-items: center; gap: {s["100"]}; margin: {s["400"]} 0 {s["100"]} 0; }}
.step .n {{ {_t("title-4")} width: {s["400"]}; height: {s["400"]}; border-radius: {r["full"]}; background: {PRIMARY}; color: #fff;
           display: inline-flex; align-items: center; justify-content: center; }}
.step .t {{ {_t("heading-5")} color: {INK}; }}
.step .s {{ {_t("caption-2")} color: {INK_MUTED}; }}

/* ---------- cartes de statistiques : rayon extérieur 16, marge 12 -> rayon intérieur 16 - 12 = 4 */
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: {s["200"]}; margin: 0 0 {s["300"]} 0; }}
.stat {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: {r["2xl"]}; padding: {s["100"]}; display: flex; gap: {s["100"]}; align-items: center; }}
.stat .tile {{ flex: none; width: {s["500"]}; height: {s["500"]}; border-radius: {r["base"]}; display: flex; align-items: center; justify-content: center; }}
.stat .tile img {{ width: {s["300"]}; height: {s["300"]}; }}
.stat .lab {{ {_t("caption-4")} color: {INK_MUTED}; }}
.stat .val {{ {_t("heading-5")} color: {INK}; overflow-wrap: anywhere; }}
.stat .sub {{ {_t("caption-5")} color: {INK_FAINT}; }}

/* ---------- état vide : vue de démarrage composée, jamais un panneau blanc */
.empty {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: {r["2xl"]}; padding: {s["400"]}; margin: {s["200"]} 0; }}
.empty h3 {{ margin: 0 0 {s["75"]} 0; }}
.empty > p {{ margin: 0 0 {s["300"]} 0; color: {INK_MUTED}; max-width: 680px; }}
.empty .how {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: {s["200"]}; }}
.empty .how div {{ background: {CANVAS}; border-radius: {r["xl"]}; padding: {s["200"]}; }}
.empty .how b {{ {_t("title-4")} display: block; color: {PRIMARY}; margin-bottom: {s["50"]}; }}
.empty .how span {{ {_t("body-6")} color: {INK}; }}

/* ---------- chargement : squelette, jamais de roue qui tourne (animation d'opacité uniquement) */
.skel {{ background: #e9e8e6; border-radius: {r["lg"]}; height: {s["300"]}; margin-bottom: {s["100"]}; animation: pulse 1600ms var(--ease) infinite; }}
.skel.big {{ height: {s["700"]}; }}
@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .45; }} }}

/* ---------- surfaces : bordure fine tout autour, jamais d'un seul côté */
div[data-testid="stPlotlyChart"] {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: {r["xl"]}; padding: {s["75"]}; box-shadow: {SHADOW}; }}
div[data-testid="stExpander"] {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: {r["xl"]}; }}
div[data-testid="stExpander"] details {{ border: none; }}
div[data-testid="stExpander"] summary p {{ {_t("title-3")} }}
div[data-testid="stDataFrame"] {{ border: 1px solid {HAIRLINE}; border-radius: {r["xl"]}; overflow: hidden; background: {SURFACE}; }}
div[data-testid="stAlert"] {{ border-radius: {r["xl"]}; border: 1px solid {HAIRLINE}; }}
div[data-testid="stImage"] img {{ border-radius: {r["lg"]}; }}

/* ---------- boutons : 8px / 12px de marge, états hover, active, focus ; courbe de mouvement unique */
.stButton > button, .stDownloadButton > button {{
    {_t("button-md")} padding: {s["75"]} {s["100"]}; border-radius: {r["lg"]};
    transition: background-color 300ms var(--ease), border-color 300ms var(--ease), color 300ms var(--ease), transform 200ms var(--ease); }}
.stButton > button p, .stDownloadButton > button p {{ {_t("button-md")} }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{ background: {PRIMARY}; color: #fff; border: 1px solid {PRIMARY}; }}
.stButton > button[kind="primary"]:hover {{ background: {PRIMARY_ACTIVE}; border-color: {PRIMARY_ACTIVE}; color: #fff; }}
.stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"] {{ background: {SURFACE}; color: {INK}; border: 1px solid {HAIRLINE}; }}
.stButton > button[kind="secondary"]:hover, .stDownloadButton > button[kind="secondary"]:hover {{ border-color: {PRIMARY}; color: {PRIMARY}; }}
.stButton > button:active, .stDownloadButton > button:active {{ transform: scale(0.98); }}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible, a:focus-visible, input:focus-visible,
[data-baseweb="select"]:focus-within, [role="radio"]:focus-visible, [role="checkbox"]:focus-visible {{ outline: 2px solid {PRIMARY}; outline-offset: 2px; }}

/* ---------- champs : rayon 4 */
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {{
    border-radius: {r["base"]} !important; background: {SURFACE}; border-color: #dddddd; }}

/* ---------- barre latérale : page active indiquée par un fond et une bordure complète */
section[data-testid="stSidebar"] {{ background: {SURFACE}; }}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{ {_t("title-2")} }}
a[data-testid="stSidebarNavLink"] {{ {_t("title-4")} border-radius: {r["md"]}; padding: {s["75"]} {s["100"]}; border: 1px solid transparent;
    transition: background-color 300ms var(--ease), border-color 300ms var(--ease); }}
a[data-testid="stSidebarNavLink"] span, a[data-testid="stSidebarNavLink"] p {{ {_t("title-4")} }}
a[data-testid="stSidebarNavLink"]:hover {{ background: {CANVAS}; }}
a[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {CANVAS}; border-color: {HAIRLINE}; }}
[data-testid="stNavSectionHeader"], [data-testid="stNavSectionHeader"] span {{ {_t("caption-3")} color: {INK_FAINT}; }}
.side-brand {{ display: flex; align-items: center; gap: {s["100"]}; padding: 0 0 {s["200"]} 0; }}
.side-brand .logo {{ width: {s["500"]}; height: {s["500"]}; border-radius: {r["lg"]}; background: {PRIMARY}; display: flex; align-items: center; justify-content: center; }}
.side-brand .logo img {{ width: {s["300"]}; height: {s["300"]}; }}
.side-brand b {{ {_t("title-3")} display: block; color: {INK}; }}
.side-brand small {{ {_t("caption-5")} color: {INK_MUTED}; }}
button[data-baseweb="tab"] {{ {_t("title-4")} }}

/* ---------- mobile (390 px) : pas de débordement, titres un cran plus petits */
@media (max-width: 640px) {{
  .block-container {{ padding: {s["300"]} {s["200"]} {s["600"]} {s["200"]}; }}
  .hero {{ padding: {s["300"]}; }}
  .hero h1 {{ {_t("heading-3")} }}
  .stApp h1 {{ {_t("heading-5")} }}
  .stats {{ grid-template-columns: 1fr; }}
}}
/* ---------- mouvement réduit : plus de déplacement, seulement un changement d'opacité */
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition-property: opacity !important; transition-duration: 200ms !important; }}
  .stButton > button:active, .stDownloadButton > button:active {{ transform: none !important; }}
}}
</style>
"""


# ------------------------------------------------------------------ Couleur principale par page
ACCENT = {"tint": "blue"}                       # couleur de la page en cours (lue aussi par les graphiques)


def _rgb(h): return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _hex(c): return "#%02x%02x%02x" % tuple(round(v) for v in c)


def _mix(a, b, t):
    """Mélange de deux couleurs hexadécimales : t = part de b."""
    return _hex(tuple(x + (y - x) * t for x, y in zip(_rgb(a), _rgb(b))))


def accent_css(tint: str) -> str:
    """Couleur principale d'une page : fond teinté, boutons, étapes, onglets, page active, mise au point.
    Seules des couleurs sont redéfinies (espacements et rayons restent ceux de build_css)."""
    bg, fg = STICKERS[tint]
    dark = _mix(fg, "#000000", .18)
    fond = _mix(CANVAS, bg, .6)
    return f"""<style>
.stApp {{ background: {fond}; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{ background: {fg}; border-color: {fg}; }}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {{ background: {dark}; border-color: {dark}; }}
.stButton > button[kind="secondary"]:hover, .stDownloadButton > button[kind="secondary"]:hover {{ border-color: {fg}; color: {fg}; }}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible, input:focus-visible,
[data-baseweb="select"]:focus-within {{ outline-color: {fg}; }}
.step .n {{ background: {fg}; }}
.empty .how b {{ color: {fg}; }}
.empty .how div {{ background: {bg}; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {fg}; }}
div[data-baseweb="tab-highlight"] {{ background-color: {fg}; }}
a[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {bg}; border-color: {fg}; }}
div[data-testid="stPlotlyChart"], div[data-testid="stExpander"], div[data-testid="stDataFrame"] {{ border-color: {_mix(HAIRLINE, fg, .35)}; }}
</style>"""


def set_accent(tint: str):
    ACCENT["tint"] = tint
    st.markdown(accent_css(tint), unsafe_allow_html=True)


def colorway():
    """Palette des graphiques : la couleur de la page d'abord, puis les autres couleurs."""
    fg = STICKERS[ACCENT["tint"]][1]
    return [fg] + [c for c in COLORWAY if c.lower() != fg.lower()]


@st.cache_data(show_spinner=False, max_entries=4000)
def vignette(path):
    """Miniature d'une image ; une image illisible (il y en a dans Paris6k) donne une vignette grise au lieu de bloquer la page."""
    try:
        im = load_rgb(path)
        im.thumbnail((200, 200))
        return im
    except Exception:
        return Image.new("RGB", (200, 150), (210, 210, 210))


# ------------------------------------------------------------------ Icônes
_FALLBACK = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
             'stroke-linecap="round"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/></svg>')


def _data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def icon_uri(name: str, color: str = PRIMARY) -> str:
    """Icône en URI de données. Priorité : assets/icons (vos SVG) > assets/phosphor (Phosphor, MIT) > secours."""
    svg = next((f.read_text(encoding="utf-8") for d in ICON_DIRS if (f := d / f"{name}.svg").exists()), _FALLBACK)
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    if "xmlns=" not in svg:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return _data_uri(svg.replace("currentColor", color))


def _tile(icon, tint="blue", fond=None):
    bg, fg = STICKERS[tint]
    return f'<span class="tile" style="background:{fond or bg}"><img src="{icon_uri(icon, fg)}" alt=""/></span>'


# ------------------------------------------------------------------ Composants
def inject_css():
    st.markdown(build_css(), unsafe_allow_html=True)


def sidebar_brand():
    st.sidebar.markdown(f'<div class="side-brand"><span class="logo"><img src="{icon_uri("search", "#ffffff")}" alt=""/></span>'
                        '<span><b>DeepCBIR</b><small>Master 2 SID, USTO MB</small></span></div>', unsafe_allow_html=True)


def page_header(title, subtitle="", icon="target", tint="blue"):
    set_accent(tint)
    st.markdown(f'<header class="ph">{_tile(icon, tint)}<div><h1>{title}</h1>' + (f"<p>{subtitle}</p>" if subtitle else "")
                + "</div></header>", unsafe_allow_html=True)


def hero(title_html, subtitle="", pills=()):
    """title_html : peut contenir <br> aux endroits où la phrase respire."""
    set_accent("blue")
    tints = list(STICKERS.values())
    pastilles = "".join(f'<span class="pill" style="background:{tints[i % len(tints)][0]};color:{tints[i % len(tints)][1]};'
                        f'border-color:{tints[i % len(tints)][1]}">{p}</span>' for i, p in enumerate(pills))
    st.markdown(f'<section class="hero"><h1>{title_html}</h1><p>{subtitle}</p><div class="pills">{pastilles}</div></section>',
                unsafe_allow_html=True)


def step(n, title, hint=""):
    st.markdown(f'<div class="step"><span class="n">{n}</span><span class="t">{title}</span>'
                + (f'<span class="s">{hint}</span>' if hint else "") + "</div>", unsafe_allow_html=True)


def stats(items):
    """items = [dict(label, value, sub='', icon='bars', tint='blue'), ...]"""
    html = "".join(f'<div class="stat" style="background:{STICKERS[i.get("tint", "blue")][0]};border-color:{STICKERS[i.get("tint", "blue")][1]}">'
                   f'{_tile(i.get("icon", "bars"), i.get("tint", "blue"), "#ffffff")}<div><div class="lab">{i["label"]}</div>'
                   f'<div class="val">{i["value"]}</div>' + (f'<div class="sub">{i["sub"]}</div>' if i.get("sub") else "") + "</div></div>"
                   for i in items)
    st.markdown(f'<div class="stats">{html}</div>', unsafe_allow_html=True)


def empty_state(title, text, steps=()):
    """Vue de démarrage : titre, explication, étapes à suivre."""
    how = "".join(f"<div><b>{i}. {h}</b><span>{d}</span></div>" for i, (h, d) in enumerate(steps, 1))
    st.markdown(f'<section class="empty"><h3>{title}</h3><p>{text}</p>' + (f'<div class="how">{how}</div>' if how else "") + "</section>",
                unsafe_allow_html=True)


def skeleton(lines=3, big=True):
    return '<div>' + ('<div class="skel big"></div>' if big else "") + '<div class="skel"></div>' * lines + '</div>'


# ------------------------------------------------------------------ Graphiques Plotly (tailles : 12, 14, 18 px)
def style(fig: go.Figure, height=None) -> go.Figure:
    fig.update_layout(
        template="plotly_white", colorway=colorway(), font=dict(family="Geist, system-ui, sans-serif", size=14, color=INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=80, b=8),
        title=dict(font=dict(size=18), x=0.0, xanchor="left", y=0.97, yanchor="top", pad=dict(l=4)),
        legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top", title_text="", bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=14))
    fig.update_xaxes(gridcolor="#efeeec", linecolor=HAIRLINE, zeroline=False, tickfont=dict(size=12))
    fig.update_yaxes(gridcolor="#efeeec", linecolor=HAIRLINE, zeroline=False, tickfont=dict(size=12))
    if height:
        fig.update_layout(height=height)
    return fig


def facets(fig: go.Figure, tickangle=-30) -> go.Figure:
    """Titres de facettes propres (« Corel-1K » et non « dataset_nom=Corel-1K »), un seul libellé d'axe, étiquettes inclinées."""
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=14, color=INK)))
    fig.update_xaxes(title_text="", tickangle=tickangle)
    return fig


def show(fig: go.Figure, height=None, **kw):
    st.plotly_chart(style(fig, height), width="stretch")


def category_axes(fig, x=None, y=None, big=40):
    """Axes catégoriels dans l'ordre donné (Corel-5K et Corel-10K : classes nommées '0', '1', ...)."""
    for axis, labels in (("x", x), ("y", y)):
        if labels is None:
            continue
        labels = [str(v) for v in labels]
        upd = dict(type="category", categoryorder="array", categoryarray=labels, tickfont=dict(size=12))
        if len(labels) > big:
            upd.update(tickmode="array", tickvals=labels[::5], ticktext=labels[::5])
        elif len(labels) > 1:
            upd.update(dtick=1)
        (fig.update_xaxes if axis == "x" else fig.update_yaxes)(**upd)
    return fig
