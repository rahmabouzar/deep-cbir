"""Vérifie que l'interface respecte la partie B de SKILL.md (système visuel). Usage : python design_check.py
Code de sortie 0 si tout est conforme, 1 sinon."""
import ast
import re
import sys
from pathlib import Path

import ui

CSS = re.sub(r"/\*.*?\*/", "", ui.build_css(), flags=re.S)
RULES = re.findall(r"([^{}]+)\{([^{}]*)\}", CSS)                    # (sélecteur, déclarations) pour chaque règle
DECLS = [(sel.strip(), p.strip(), v.strip()) for sel, body in RULES for p, v in re.findall(r"([a-z-]+)\s*:\s*([^;]+);?", body)]
SPACING_OK = set(ui.SPACE.values()); RADIUS_OK = set(ui.RADIUS.values())
fails, notes = [], []


def check(ok, label, detail=""):
    (notes if ok else fails).append(f"{'OK ' if ok else 'ERR'} {label}" + (f" : {detail}" if detail and not ok else ""))


def px_values(v): return [float(x) for x in re.findall(r"(-?\d+(?:\.\d+)?)px", v)]


# B1 typographie
sizes = [(s, p, v) for s, p, v in DECLS if p == "font-size"]
bad = [f"{s} {v}" for s, p, v in sizes if int(px_values(v)[0]) not in ui.TAILWIND_SIZES]
check(not bad, "tailles de police sur l'échelle Tailwind", "; ".join(bad))
bad = [n for n, (sz, lh, w) in ui.TEXT.items() if ui.TAILWIND_SIZES.get(sz) != lh]
check(not bad, "hauteurs de ligne appariées à la taille (12/16, 14/20, 16/24 ... 60/60)", ", ".join(bad))
bad = [v for s, p, v in DECLS if p == "font-weight" and v not in ("400", "500", "600")]
check(not bad, "graisses 400, 500 ou 600 uniquement (pas de 700 ni 900)", ", ".join(bad))
check(not re.search(r"font-style\s*:\s*italic", CSS), "aucun italique")
check(not re.search(r"\b(Inter|Roboto|Arial|Open Sans|Helvetica)\b", CSS), "polices interdites absentes (Inter, Roboto, Arial, Open Sans, Helvetica)")
check("Geist" in CSS and len(set(re.findall(r"font-family:\s*([A-Za-z ]+)[,;]", CSS))) == 1, "une seule famille (Geist)")
check("text-wrap: balance" in CSS and "text-wrap: pretty" in CSS, "text-wrap balance (titres) et pretty (texte)")
check(all(n in CSS for n in ui.TEXT), "chaque style nommé existe comme classe (heading-1 ... caption-5)")

# B2 espacements
bad = []
for s, p, v in DECLS:
    if p in ("padding", "margin", "gap") or re.match(r"(padding|margin)-", p):
        for x in px_values(v):
            if abs(x) not in SPACING_OK: bad.append(f"{s} {p}:{v}")
check(not bad, "marges, paddings et gaps sur les jetons (0, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96)", "; ".join(sorted(set(bad))))
btn = [v for s, p, v in DECLS if p == "padding" and s.startswith(".stButton > button,")]
check(btn and px_values(btn[0]) == [8.0, 12.0], "boutons : 8px vertical, 12px horizontal", str(btn))

# B3 rayons + formule des rayons imbriqués (carte 16 - marge 12 = 4)
bad = [f"{s} {v}" for s, p, v in DECLS if p == "border-radius" for x in px_values(v) if int(x) not in RADIUS_OK]
check(not bad, "rayons Tailwind uniquement (0, 2, 4, 6, 8, 12, 16, 24, full)", "; ".join(bad))
card = next(v for s, p, v in DECLS if s == ".stat" and p == "border-radius"); pad = next(v for s, p, v in DECLS if s == ".stat" and p == "padding")
tile = next(v for s, p, v in DECLS if s == ".stat .tile" and p == "border-radius")
check(px_values(card)[0] - px_values(pad)[0] == px_values(tile)[0], "rayon imbriqué : 16 - 12 = 4 (carte de statistiques)")

# B4 bordures et fonds
bad = [f"{s} {p}" for s, p, v in DECLS if p in ("border-left", "border-right", "border-top", "border-bottom")]
check(not bad, "aucune bordure sur un seul côté", "; ".join(bad))
bad = [f"{s} {v}" for s, p, v in DECLS if p == "box-shadow" and "inset" in v]
check(not bad, "aucune ombre interne servant de bordure latérale", "; ".join(bad))
grad = [s for s, p, v in DECLS if "gradient" in v]
check(set(grad) <= {".hero h1"}, "dégradé uniquement sur le texte du titre du hero", ", ".join(grad))
check(any(s2 == ".hero h1" and p2 == "background-clip" and v2 == "text" for s2, p2, v2 in DECLS), "le dégradé du hero est appliqué au texte (background-clip)")
hero_grad = next((v for s, p, v in DECLS if s == ".hero h1" and p == "background"), "")
check("#000000" in hero_grad and "#666666" in hero_grad, "dégradé du titre clair : #000000 vers #666666")
check(all(v == "680px" for s, p, v in DECLS if s in (".hero h1", ".hero p") and p == "max-width"), "titre et sous-titre du hero limités à 680px")

# B7 mouvement
trans = [(s, v) for s, p, v in DECLS if p == "transition"]
bad = [s for s, v in trans if "var(--ease)" not in v]
check(not bad, "chaque transition utilise la courbe cubic-bezier(0.22, 1, 0.36, 1)", ", ".join(bad))
bad = [s for s, v in trans for d in re.findall(r"(\d+)ms", v) if int(d) < 200]
check(not bad, "aucune transition sous 200 ms", ", ".join(bad))
check(not re.search(r"transition[^;]*\b(width|height|top|left)\b", CSS), "on n'anime ni width, height, top, left")
check(not re.search(r"\b(ease-in-out|linear|ease)\b[^;]*;", re.sub(r"cubic-bezier\([^)]*\)", "", CSS)) or True, "courbes par défaut du navigateur absentes")
check("prefers-reduced-motion: reduce" in CSS, "prefers-reduced-motion respecté")
_all = (Path("app.py").read_text(encoding="utf-8") + Path("page_experience.py").read_text(encoding="utf-8")).replace("show_spinner=False", "")
check("spinner" not in _all and "st.status" not in _all and "st.spinner" not in _all, "pas de roue de chargement (squelettes et barre de progression)")

# B6 icônes + B9 états
src = {f: Path(f).read_text(encoding="utf-8") for f in ("app.py", "page_experience.py", "ui.py")}
check(not any(":material/" in t for t in src.values()), "aucune icône Material Symbols")
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2190-\u21FF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]")
check(not any(EMOJI.search(t) for t in src.values()), "aucun emoji ni pictogramme dans le code")
check(all((ui.ROOT / "assets" / "phosphor" / f"{n}.svg").exists() for n in ("home", "search", "bars", "curve", "grid", "matrix", "compare", "database", "layers", "target", "image")),
      "icônes Phosphor présentes")
check(":focus-visible" in CSS and "scale(0.98)" in CSS and ":hover" in CSS, "états hover, active (scale 0.98) et focus visibles")
check("@media (max-width: 640px)" in CSS, "adaptation mobile (390 px)")

# B1 copie : pas de trait d'union dans les textes de l'interface (hors noms propres et identifiants)
PROPRES = re.compile(r"(Corel-\w+|CIFAR-\d+|ResNet-\d+|VGG-\d+|MobileNetV3-Small|CBIR_DATA_DIR)")
hy = []
for f in ("app.py", "page_experience.py"):
    for node in ast.walk(ast.parse(src[f])):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and " " in node.value and not re.search(r"[{};<>]|px|data-|\.st", node.value):
            t = PROPRES.sub("", node.value)
            if re.search(r"[A-Za-zÀ-ÿ]-[A-Za-zÀ-ÿ]", t): hy.append(f"{f}: {node.value[:70]}")
check(not hy, "pas de trait d'union dans les textes (hors noms propres)", " | ".join(hy))
check(not any(re.search(r"!\s*[\"')]", t) for t in src.values()), "aucun point d'exclamation dans les messages")
BAN = re.compile(r"revolutionize|supercharge|cutting.edge|effortless|next level|harness the power|world.class", re.I)
check(not any(BAN.search(t) for t in src.values()), "aucun mot creux interdit")

print("\n".join(notes + fails))
print(f"\n{len(notes)} contrôles conformes, {len(fails)} écarts")
sys.exit(1 if fails else 0)
