"""ÉTAPE 7 -- Images de résultats de chaque étape (PNG) + rapport PDF complet et PDF par étape.

Usage : python etape7_pdf.py
  -> results/figures/*.png                      (images, une série par étape)
  -> results/pdf/rapport_complet.pdf            (toutes les étapes, avec sommaire cliquable)
  -> results/pdf/etape1.pdf … etape4.pdf        (un PDF par étape)
Appelé automatiquement par etape6_site_html.py (les boutons de téléchargement du site pointent vers ces fichiers).
"""
import datetime, os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

import config as C
import analyses as A
from analyses import nd, nm, esc

PDFDIR = C.RESULTS / "pdf"; PDFDIR.mkdir(parents=True, exist_ok=True)
W = A4[0] - 3.4 * cm
ACC = colors.HexColor("#2f5bea"); GRY = colors.HexColor("#667085"); LIGHT = colors.HexColor("#f2f4f8"); GRN = colors.HexColor("#d6efe4")

# ------------------------------------------------------------------ polices (DejaVu est fournie avec matplotlib)
_fd = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for n, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-M", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(n, os.path.join(_fd, f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=26, leading=32, textColor=ACC, spaceAfter=8),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=12, leading=17, textColor=GRY, spaceAfter=14),
    "H1": ParagraphStyle("H1", fontName="DV-B", fontSize=17, leading=22, textColor=ACC, spaceBefore=6, spaceAfter=8, keepWithNext=1),
    "H2": ParagraphStyle("H2", fontName="DV-B", fontSize=12, leading=16, textColor=colors.HexColor("#1f2430"), spaceBefore=12, spaceAfter=5, keepWithNext=1),
    "sommaire": ParagraphStyle("sommaire", fontName="DV-B", fontSize=17, leading=22, textColor=ACC, spaceAfter=10),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9, leading=13.2, spaceAfter=5, alignment=TA_LEFT),
    "box": ParagraphStyle("box", fontName="DV", fontSize=8.8, leading=12.8, backColor=LIGHT, borderPadding=(6, 7, 6, 9),
                          borderColor=ACC, borderWidth=0, spaceBefore=6, spaceAfter=10, leftIndent=6, rightIndent=4),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=7.8, leading=10.5, textColor=GRY, spaceAfter=9),
    "th": ParagraphStyle("th", fontName="DV-B", fontSize=7.4, leading=9.4, textColor=colors.white),
    "td": ParagraphStyle("td", fontName="DV", fontSize=7.4, leading=9.4),
    "li": ParagraphStyle("li", fontName="DV", fontSize=9, leading=13, leftIndent=12, bulletIndent=2, spaceAfter=3),
    "code": ParagraphStyle("code", fontName="DV-M", fontSize=7.8, leading=10.5),
}
P_ = lambda t, st="body": Paragraph(t, S[st])
LI = lambda t: Paragraph(t, S["li"], bulletText="•")
BOX = lambda t, titre=None: Paragraph((f"<b>{titre}</b><br/>" if titre else "") + t, S["box"])


def fmtv(v, fmt=None):
    if isinstance(v, (float, np.floating)):
        return (fmt or "{:.4f}").format(v)
    return esc(v)


def tbl(df, fmt=None, weights=None, hl_max=(), hl_min=()):
    fmt = fmt or {}; cols = list(df.columns); weights = weights or [1] * len(cols)
    cw = [W * w / sum(weights) for w in weights]
    rows = [[Paragraph(esc(c), S["th"]) for c in cols]]
    for _, r in df.iterrows():
        rows.append([Paragraph(fmtv(r[c], fmt.get(c)), S["td"]) for c in cols])
    st = [("BACKGROUND", (0, 0), (-1, 0), ACC), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#d9dde6")),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("LEFTPADDING", (0, 0), (-1, -1), 3.5)]
    for i in range(2, len(rows), 2): st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    for c in hl_max + hl_min:
        if c in cols:
            j = cols.index(c); v = df[c].max() if c in hl_max else df[c].min()
            for i, x in enumerate(df[c], start=1):
                if x == v: st.append(("BACKGROUND", (j, i), (j, i), GRN))
    t = Table(rows, colWidths=cw, repeatRows=1); t.setStyle(TableStyle(st)); return [t, Spacer(1, 7)]


def fig(name, caption, width=W):
    path = C.FIGS / name
    if not path.exists(): return []
    if name.startswith(("echantillon", "pretraitement", "recherche")):          # photos : JPEG (PDF 3-4x plus léger)
        jd = C.FIGS / "_jpg"; jd.mkdir(exist_ok=True); jp = jd / (path.stem + ".jpg")
        if not jp.exists(): PILImage.open(path).convert("RGB").save(jp, quality=82, optimize=True)
        path = jp
    iw, ih = PILImage.open(path).size; h = width * ih / iw
    if h > 20.5 * cm: width, h = width * 20.5 * cm / h, 20.5 * cm
    return [KeepTogether([Image(str(path), width=width, height=h), P_(caption, "cap")])]


# ------------------------------------------------------------------ document avec sommaire cliquable
class Doc(BaseDocTemplate):
    def __init__(self, path, titre, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.9 * cm, bottomMargin=1.7 * cm,
                         title=titre, author="DeepCBIR", **kw)
        self.titre, self._hk = titre, 0
        self.addPageTemplates([PageTemplate(id="p", frames=[Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")], onPage=self.deco)])

    def handle_documentBegin(self):
        self._hk = 0                      # clés de signets identiques à chaque passe (sommaire stable)
        super().handle_documentBegin()

    def deco(self, c, d):
        c.saveState(); c.setFont("DV", 7.5); c.setFillColor(GRY)
        c.drawString(1.7 * cm, A4[1] - 1.1 * cm, "DeepCBIR · " + self.titre); c.drawRightString(A4[0] - 1.7 * cm, A4[1] - 1.1 * cm, datetime.date.today().strftime("%d/%m/%Y"))
        c.setStrokeColor(colors.HexColor("#d9dde6")); c.line(1.7 * cm, A4[1] - 1.25 * cm, A4[0] - 1.7 * cm, A4[1] - 1.25 * cm)
        c.drawCentredString(A4[0] / 2, 0.9 * cm, f"Page {d.page}"); c.restoreState()

    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and fl.style.name in ("H1", "H2"):
            lvl = 0 if fl.style.name == "H1" else 1; txt = fl.getPlainText(); key = f"h{self._hk}"; self._hk += 1
            self.canv.bookmarkPage(key); self.canv.addOutlineEntry(txt, key, lvl, 0)
            if lvl == 0: self.notify("TOCEntry", (lvl, txt, self.page, key))


def build_pdf(path, titre, story, toc=False):
    doc = Doc(path, titre)
    if toc:
        t = TableOfContents(); t.levelStyles = [ParagraphStyle("t0", fontName="DV-B", fontSize=10, leading=17, leftIndent=8)]
        i = next((k for k, f in enumerate(story) if isinstance(f, PageBreak)), -1) + 1     # après la couverture
        story = story[:i] + [P_("Sommaire", "sommaire"), t, PageBreak()] + story[i:]
        doc.multiBuild(story)
    else:
        doc.build(story)


# ================================================================== FIGURES (PNG)
def savefig(f, name):
    f.tight_layout(); f.savefig(C.FIGS / name, dpi=140); plt.close(f); return name


def make_figures(R, T, norms):
    G, DSS, MODS, MAN = R.G, R.DSS, R.MODS, R.MAN
    out = {}
    # ---- étape 1
    f, ax = plt.subplots(figsize=(7.5, 0.55 * len(DSS) + 1.6))
    used = [R.ST[d]["n_images"] for d in DSS]; tot = [R.ST[d].get("n_disponibles", R.ST[d]["n_images"]) for d in DSS]
    ax.barh([nd(d) for d in DSS], tot, color="#d0d5e2", label="images sur disque"); ax.barh([nd(d) for d in DSS], used, color="#2f5bea", label="images utilisées")
    for i, (u, t_) in enumerate(zip(used, tot)): ax.text(t_, i, f"  {100 * u / t_:.0f} %", va="center", fontsize=8)
    ax.set_title("Couverture des données"); ax.legend(fontsize=8); ax.invert_yaxis()
    out["etape1_couverture.png"] = savefig(f, "etape1_couverture.png")
    f, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    axs[0].boxplot([MAN[d].width for d in DSS], tick_labels=[nd(d) for d in DSS], showfliers=False); axs[0].set_title("Largeur des images (px)"); axs[0].set_yscale("log")
    axs[1].boxplot([MAN[d].size_kb for d in DSS], tick_labels=[nd(d) for d in DSS], showfliers=False); axs[1].set_title("Taille des fichiers (Ko)"); axs[1].set_yscale("log")
    for a in axs: a.tick_params(axis="x", labelsize=7, rotation=20)
    savefig(f, "etape1_formats.png")
    # ---- étape 2
    mean_ms = T.groupby("Modèle", sort=False)["ms/image"].mean()
    f, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].bar(mean_ms.index, mean_ms.values, color="#2f5bea"); axs[0].set_title("Temps de calcul par image (ms)"); axs[0].tick_params(axis="x", labelsize=7, rotation=20)
    for i, v in enumerate(mean_ms.values): axs[0].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    bot = np.zeros(len(DSS))
    for mdl in T.Modèle.unique():
        v = np.array([T[(T.dataset == d) & (T.Modèle == mdl)]["Temps total (min)"].sum() for d in DSS]); axs[1].bar([nd(d) for d in DSS], v, bottom=bot, label=mdl); bot += v
    axs[1].set_title("Temps total d'extraction par base (min)"); axs[1].legend(fontsize=7); axs[1].tick_params(axis="x", labelsize=7, rotation=20)
    savefig(f, "etape2_temps.png")
    f, ax = plt.subplots(figsize=(8.5, 3.8)); w = 0.8 / len(DSS)
    for i, d in enumerate(DSS):
        s = T[T.dataset == d].set_index("model").reindex(MODS); ax.bar(np.arange(len(MODS)) + i * w, s["Séparabilité"], w, label=nd(d))
    ax.set_xticks(np.arange(len(MODS)) + w * (len(DSS) - 1) / 2); ax.set_xticklabels([nm(m) for m in MODS], fontsize=8)
    ax.set_title("Séparabilité des descripteurs (cos. même classe − cos. classes différentes)"); ax.legend(fontsize=8)
    savefig(f, "etape2_separabilite.png")
    f, ax = plt.subplots(figsize=(7.5, 3.6)); ks = [m for m in MODS if m in norms]
    ax.boxplot([norms[m] for m in ks], tick_labels=[nm(m) for m in ks], showfliers=False); ax.set_title(f"Distribution des normes L2 des vecteurs – {nd(DSS[0])}")
    savefig(f, "etape2_normes.png")
    rng = np.random.default_rng(C.SEED)
    for d in DSS:
        bm = A.best_row(G, d, "cosinus").model; F = np.load(C.FEATURES / f"{d}__{bm}.npy"); s, P = A.pca2(F, rng); names = MAN[d].class_name.values[s]
        u = sorted(set(names)); cmap = plt.get_cmap("tab20" if len(u) <= 20 else "hsv"); f, ax = plt.subplots(figsize=(6.4, 4.6))
        for i, c in enumerate(u):
            m_ = names == c; ax.scatter(P[m_, 0], P[m_, 1], s=7, alpha=.75, color=cmap(i / max(len(u) - 1, 1) if len(u) > 20 else i), label=c if len(u) <= 20 else None)
        if len(u) <= 20: ax.legend(fontsize=6, ncol=2, loc="best", markerscale=1.5)
        ax.set_title(f"Projection 2D (ACP) – {nd(d)} – {nm(bm)}"); ax.set_xlabel("composante 1"); ax.set_ylabel("composante 2")
        savefig(f, f"etape2_pca_{d}.png")
    # ---- étape 3
    nc = min(2, len(DSS)); nr = -(-len(DSS) // nc); f, axs = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.2 * nr), squeeze=False)
    for a in axs.ravel()[len(DSS):]: a.axis("off")
    for ax, d in zip(axs.ravel(), DSS):
        s = G[G.dataset == d]; x = np.arange(len(MODS))
        for i, met in enumerate(C.METRICS):
            ax.bar(x + (i - .5) * .38, [s[(s.model == m) & (s.metrique == met)].ms_recherche_req.values[0] for m in MODS], .38, label=met)
        ax.set_xticks(x); ax.set_xticklabels([nm(m) for m in MODS], rotation=30, ha="right", fontsize=6.5); ax.set_title(nd(d), fontsize=9); ax.set_ylabel("ms / requête")
    axs[0][0].legend(fontsize=7); f.suptitle("Temps moyen d'une recherche exhaustive", fontsize=10)
    savefig(f, "etape3_latence.png")
    for d in DSS:
        bm = A.best_row(G, d, "cosinus").model; F = np.load(C.FEATURES / f"{d}__{bm}.npy")
        a, b = A.dist_samples(F, MAN[d].label.values, MAN[d], d, rng)
        f, ax = plt.subplots(figsize=(7, 3.4)); ax.hist(a, bins=60, alpha=.65, density=True, label="même classe (pertinentes)"); ax.hist(b, bins=60, alpha=.65, density=True, label="autres classes")
        ax.set_title(f"Distances cosinus requête → base – {nd(d)} – {nm(bm)}"); ax.set_xlabel("distance cosinus"); ax.legend(fontsize=8)
        savefig(f, f"etape3_distances_{d}.png")
    import etape3_recherche as E3
    ex = {}
    for d in DSS:
        r = A.best_row(G, d, "cosinus"); pc = R.PC[(R.PC.dataset == d) & (R.PC.model == r.model) & (R.PC.metrique == "cosinus")].sort_values("AP", ascending=False)
        for lab, cls in (("facile", pc.classe.iloc[0]), ("difficile", pc.classe.iloc[-1])):
            qi = int(rng.choice(MAN[d].index[MAN[d].class_name == cls])); p = E3.figure_recherche(d, r.model, qi, 8)
            ex.setdefault(d, []).append((lab, cls, os.path.basename(str(p))))
    out["exemples"] = ex
    # ---- étape 4
    nc = min(2, len(DSS)); nr = -(-len(DSS) // nc); f, axs = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.2 * nr), squeeze=False, sharey=True)
    for a in axs.ravel()[len(DSS):]: a.axis("off")
    for ax, d in zip(axs.ravel(), DSS):
        for m in MODS:
            s = G[(G.dataset == d) & (G.model == m) & (G.metrique == "cosinus")].iloc[0]; ax.plot(C.K_LIST, [s[f"P@{k}"] for k in C.K_LIST], marker="o", ms=3, label=nm(m))
        ax.set_title(nd(d), fontsize=9); ax.set_xlabel("K"); ax.grid(alpha=.3)
    axs[0][0].set_ylabel("Précision@K (cosinus)"); axs[0][0].legend(fontsize=6.5)
    savefig(f, "etape4_precision_k.png")
    f, ax = plt.subplots(figsize=(8, 3.6)); w = 0.8 / len(DSS)
    for i, d in enumerate(DSS):
        s = G[G.dataset == d].pivot_table(index="model", columns="metrique", values="mAP").reindex(MODS); ax.bar(np.arange(len(MODS)) + i * w, s["cosinus"] - s["euclidienne"], w, label=nd(d))
    ax.axhline(0, color="k", lw=.7); ax.set_xticks(np.arange(len(MODS)) + w * (len(DSS) - 1) / 2); ax.set_xticklabels([nm(m) for m in MODS], fontsize=8)
    ax.set_title("Écart de mAP : cosinus − euclidienne (positif = cosinus meilleur)"); ax.legend(fontsize=8)
    savefig(f, "etape4_cos_euc.png")
    import etape5_rapport as R5
    R5.fig_map_bar(); R5.fig_tradeoff()
    for d in DSS:
        r = A.best_row(G, d); R5.fig_pr(d); R5.fig_heat(d, r.metrique); R5.fig_conf(d, r.model, r.metrique)
    return out


def art(R, ex):
    """Catalogue des images par page du site : {page: [(titre, fichier, légende)]}."""
    DSS = R.DSS; E = lambda *a: [(t, f, c) for t, f, c in a if (C.FIGS / f).exists()]
    A_ = {"etape1": E(("Couverture des données", "etape1_couverture.png", "Images utilisées / images trouvées sur le disque."),
                      ("Formats et tailles", "etape1_formats.png", "Largeur et poids des fichiers par base.")),
          "etape2": E(("Temps d'extraction", "etape2_temps.png", "Temps par image et temps total par base."),
                      ("Séparabilité des descripteurs", "etape2_separabilite.png", "Plus l'écart est grand, mieux les classes sont regroupées."),
                      ("Normes des vecteurs", "etape2_normes.png", "La norme varie d'une image à l'autre : le cosinus l'ignore.")),
          "etape3": E(("Temps de recherche", "etape3_latence.png", "Recherche exhaustive, par requête.")),
          "resultats": E(("mAP par base, modèle et métrique", "rapport_map_barres.png", "Plus haut = meilleur."),
                         ("Précision@K", "etape4_precision_k.png", "Qualité des K premiers résultats (cosinus).")),
          "courbes": [], "classes": [], "confusion": [],
          "comparaisons": E(("Cosinus − euclidienne", "etape4_cos_euc.png", "Écart de mAP par modèle et par base."),
                            ("Compromis précision / vitesse", "rapport_compromis.png", "mAP en fonction du temps d'extraction."))}
    for d in DSS:
        A_["etape1"] += E((f"Échantillon – {nd(d)}", f"echantillon_{d}.png", "Une image par classe."), (f"Prétraitement – {nd(d)}", f"pretraitement_{d}.png", "Originale → 224×224 → normalisée."))
        A_["etape2"] += E((f"Projection 2D – {nd(d)}", f"etape2_pca_{d}.png", "ACP sur 2000 vecteurs (couleur = classe)."))
        A_["etape3"] += E((f"Distances – {nd(d)}", f"etape3_distances_{d}.png", "Distance aux images pertinentes et non pertinentes."))
        for lab, cls, fn in ex.get(d, []):
            A_["etape3"] += E((f"Exemple – {nd(d)} (classe {lab} : {cls})", fn, "Haut : cosinus, bas : euclidienne ; bordure verte = même classe."))
        A_["courbes"] += E((f"Courbes P-R – {nd(d)}", f"rapport_pr_{d}.png", "Précision interpolée en fonction du rappel."))
        A_["classes"] += E((f"AP par classe – {nd(d)}", f"rapport_classes_{d}.png", "AP de chaque classe, pour chaque modèle."))
        A_["confusion"] += E((f"Confusions – {nd(d)}", f"rapport_confusion_{d}.png", "Classe requête (lignes) vs classe retrouvée (colonnes)."))
    return A_


# ================================================================== SECTIONS DU PDF
def stage_head(n, titre, role, script, entree, sortie):
    return [P_(f"Étape {n} — {titre}", "H1"), BOX(f"<b>Rôle :</b> {role}<br/><b>Script :</b> {script}<br/><b>Entrée :</b> {entree}<br/><b>Sortie :</b> {sortie}")]


def s_etape1(R, ART):
    s = stage_head(1, "Préparation des données", "repérer les images de chaque base, en déduire la <b>classe</b>, vérifier les fichiers (format, résolution, lisibilité) et définir le prétraitement commun.",
                   "etape1_preparation.py", "dossiers d'images", "manifestes CSV, statistiques JSON, figures d'échantillon")
    s.append(P_("Couverture et statistiques des bases", "H2"))
    t = pd.DataFrame([{"Base": nd(d), "Sur disque": R.ST[d].get("n_disponibles", R.ST[d]["n_images"]), "Utilisées": R.ST[d]["n_images"], "Couv. (%)": R.ST[d].get("couverture_pct", 100.0),
                       "Classes": R.ST[d]["n_classes"], "Extension": ", ".join(R.ST[d]["extensions"]), "Résolution moy.": f"{R.ST[d]['largeur_moy']:.0f}×{R.ST[d]['hauteur_moy']:.0f}",
                       "Taille moy. (Ko)": R.ST[d]["taille_moy_ko"], "Illisibles": R.ST[d]["illisibles"]} for d in R.DSS])
    s += tbl(t, fmt={"Couv. (%)": "{:.1f}", "Taille moy. (Ko)": "{:.1f}"}, weights=[1.5, 1, 1, .9, .8, 1, 1.2, 1, .9])
    s += fig("etape1_couverture.png", "Figure 1.1 – Images utilisées par rapport aux images trouvées sur le disque.", W * .8)
    s += fig("etape1_formats.png", "Figure 1.2 – Largeur et poids des fichiers : CIFAR-100 (PNG 32×32) est très différent des JPEG de Corel et de Paris.")
    s.append(BOX("Chaque image est redimensionnée en <b>224×224</b> (sans recadrage) puis normalisée avec la moyenne (0,485 ; 0,456 ; 0,406) et l'écart-type (0,229 ; 0,224 ; 0,225) d'ImageNet : "
                 "c'est le prétraitement utilisé à l'entraînement des réseaux. Les petites images de CIFAR-100 sont donc fortement agrandies.", "Prétraitement"))
    for d in R.DSS:
        s += fig(f"echantillon_{d}.png", f"Échantillon – {nd(d)} (une image par classe).", W * .85) + fig(f"pretraitement_{d}.png", f"Prétraitement – {nd(d)} : image originale, redimensionnée, normalisée.", W * .85)
    return s


def s_etape2(R, ART, T):
    s = stage_head(2, "Extraction des vecteurs caractéristiques", "transformer chaque image en <b>un vecteur</b> avec un réseau pré-entraîné ; indexation faite une seule fois, hors-ligne.",
                   "etape2_extraction.py", "manifestes (images 224×224 normalisées)", "un fichier .npy (N × d) par base et par modèle")
    mm = T.groupby("Modèle", sort=False)["ms/image"].mean(); sp = T[["Séparabilité", "mAP (étape 4)"]].corr(method="spearman").iloc[0, 1] if len(T) >= 4 else float("nan")
    s.append(BOX(f"<b>{len(T)}</b> index créés, taille totale <b>{T['Index (Mo)'].sum():.0f} Mo</b>, temps de calcul cumulé <b>{T['Temps total (min)'].sum():.0f} min</b>. "
                 f"Le plus lent : <b>{mm.idxmax()}</b> ({mm.max():.1f} ms/image) ; le plus rapide : <b>{mm.idxmin()}</b> ({mm.min():.1f} ms/image), soit <b>{mm.max() / mm.min():.0f}×</b> d'écart.", "Constat"))
    s.append(P_("Tableau de l'extraction", "H2"))
    t = T[["Base", "Modèle", "Images", "Dimension", "Index (Mo)", "ms/image", "Temps total (min)", "Séparabilité", "mAP (étape 4)"]]
    s += tbl(t, fmt={"Index (Mo)": "{:.1f}", "ms/image": "{:.2f}", "Temps total (min)": "{:.1f}", "Séparabilité": "{:.3f}"}, weights=[1.4, 1.5, .8, .9, .9, .8, 1, 1, 1], hl_max=("Séparabilité", "mAP (étape 4)"), hl_min=("ms/image",))
    s.append(P_("<b>Index (Mo)</b> = N × dimension × 4 octets. <b>Séparabilité</b> = cosinus moyen entre images de même classe moins cosinus moyen entre classes différentes (échantillon de 1500 images)."))
    s += fig("etape2_temps.png", "Figure 2.1 – Coût de calcul : proportionnel au nombre d'opérations du réseau (VGG-16 ≈ 15 GFLOPs, MobileNetV3-Small ≈ 0,06).")
    s += fig("etape2_separabilite.png", "Figure 2.2 – Séparabilité des descripteurs.")
    if sp == sp:
        s.append(BOX(f"Cet indicateur, calculé dès l'étape 2, est corrélé au mAP de l'étape 4 (corrélation de Spearman ρ = <b>{sp:.2f}</b>) : il prédit le classement des configurations.", "Constat"))
    s += fig("etape2_normes.png", "Figure 2.3 – La longueur des vecteurs varie d'une image à l'autre : la distance cosinus l'ignore, l'euclidienne y est sensible.", W * .8)
    s.append(P_("Projection 2D des vecteurs", "H2")); s.append(P_("Chaque point est une image, la couleur est sa classe : des nuages séparés annoncent une recherche facile, des nuages mélangés une recherche difficile."))
    for d in R.DSS: s += fig(f"etape2_pca_{d}.png", f"Projection ACP – {nd(d)}.", W * .72)
    return s


def s_etape3(R, ART, ex):
    s = stage_head(3, "Recherche et mesure de similarité", "pour une image requête, calculer la distance à tous les vecteurs de la base et retourner les K plus proches.",
                   "etape3_recherche.py", "un vecteur requête + la base de vecteurs", "la liste ordonnée des K images les plus proches")
    s.append(P_("Algorithme : (1) extraire le vecteur q de la requête ; (2) calculer d(q, xᵢ) pour toutes les images ; (3) trier et garder les K premières. "
                "Cosinus : d = 1 − (q·x)/(‖q‖‖x‖). Euclidienne : d = √Σ(qᵢ − xᵢ)². Recherche exhaustive exacte, coût ∝ N × dimension."))
    s.append(P_("Qualité des premiers résultats", "H2"))
    t = pd.DataFrame([{"Base": nd(d), "Meilleur modèle": r.modele, "1er résultat correct (P@1)": r["P@1"], "Top-5 (P@5)": r["P@5"], "Top-10 (P@10)": r["P@10"]}
                      for d in R.DSS for r in [A.best_row(R.G, d, "cosinus")]])
    s += tbl(t, fmt={c: "{:.1%}" for c in t.columns[2:]}, weights=[1.5, 1.6, 1.4, 1, 1])
    s += fig("etape3_latence.png", "Figure 3.1 – Coût d'une recherche : croît avec la taille de la base et la dimension des descripteurs.")
    s.append(P_("Pourquoi la recherche réussit ou échoue", "H2")); s.append(P_("Si les distances aux images de même classe se séparent de celles des autres classes, la recherche est facile ; si les histogrammes se chevauchent, elle est difficile."))
    for d in R.DSS: s += fig(f"etape3_distances_{d}.png", f"Distribution des distances – {nd(d)}.", W * .8)
    s.append(P_("Exemples de recherche (cosinus et euclidienne)", "H2"))
    for d in R.DSS:
        for lab, cls, fn in ex.get(d, []):
            s += fig(fn, f"{nd(d)} – requête de la classe {lab} « {esc(cls)} » : haut = cosinus, bas = euclidienne ; bordure verte = même classe que la requête.")
    return s


def s_etape4(R, ART):
    G = R.G
    s = stage_head(4, "Évaluation", "mesurer la qualité de la recherche : chaque image est une requête (leave-one-out), on classe toute la base et on compare à la vérité (même classe).",
                   "etape4_evaluation.py", "vecteurs + manifestes", "metrics_global.csv, metrics_par_classe.csv, pr_curves.json, matrices de confusion")
    s.append(BOX("<b>P@K</b> = part de pertinentes dans le Top-K ; <b>R@K</b> = part des pertinentes retrouvées ; <b>AP</b> = moyenne des précisions aux rangs des images pertinentes ; "
                 "<b>mAP</b> = moyenne des AP sur toutes les requêtes (aire sous la courbe Précision-Rappel).", "Mesures"))
    s += fig("rapport_map_barres.png", "Figure 4.1 – mAP par base, modèle et métrique."); s += fig("etape4_precision_k.png", "Figure 4.2 – Précision@K (cosinus).")
    for d in R.DSS:
        br = A.best_row(G, d)
        s.append(P_(f"{nd(d)}", "H2"))
        s.append(P_(f"Meilleure configuration : <b>{br.modele}</b> ({br.metrique}), mAP = <b>{br.mAP:.4f}</b>, P@10 = <b>{br['P@10']:.4f}</b> (≈ {br['P@10'] * 10:.1f} images pertinentes sur 10). "
                    f"Un tirage aléatoire donnerait ≈ {1 / br.n_classes:.3f}."))
        t = G[G.dataset == d].sort_values("mAP", ascending=False)[["modele", "metrique", "mAP", "P@1", "P@5", "P@10", "P@20", "R@10", "R@50"]].rename(columns={"modele": "Modèle", "metrique": "Métrique"})
        s += tbl(t, weights=[1.6, 1.3, 1, 1, 1, 1, 1, 1, 1], hl_max=("mAP", "P@1", "P@5", "P@10", "P@20", "R@10", "R@50"))
        s += fig(f"rapport_pr_{d}.png", f"Courbes Précision-Rappel – {nd(d)}.")
        pc = R.PC[(R.PC.dataset == d) & (R.PC.model == br.model) & (R.PC.metrique == br.metrique)].sort_values("AP", ascending=False)
        a = pc.head(5)[["classe", "AP", "P@1", "P@10"]].rename(columns={"classe": "Classe (faciles)"}); b = pc.tail(5)[["classe", "AP", "P@1", "P@10"]].rename(columns={"classe": "Classe (difficiles)"})
        s += tbl(a, fmt={"AP": "{:.3f}", "P@1": "{:.3f}", "P@10": "{:.3f}"}, weights=[2, 1, 1, 1]) + tbl(b, fmt={"AP": "{:.3f}", "P@1": "{:.3f}", "P@10": "{:.3f}"}, weights=[2, 1, 1, 1])
        s += fig(f"rapport_classes_{d}.png", f"AP par classe – {nd(d)} (cosinus ou euclidienne selon la meilleure configuration).", W * .85)
        s += fig(f"rapport_confusion_{d}.png", f"Matrice de confusion (Top-10) – {nd(d)} : diagonale forte = recherche fiable.", W * .7)
    return s


def s_compar(R, ART):
    G = R.G
    s = [P_("Comparaisons croisées", "H1")]
    d_ = G.pivot_table(index=["dataset_nom", "modele"], columns="metrique", values="mAP").reset_index(); d_["Écart (cos − euc)"] = d_["cosinus"] - d_["euclidienne"]
    moy = d_["Écart (cos − euc)"].mean(); wins = int((d_["Écart (cos − euc)"] > 0).sum())
    s.append(P_("Cosinus vs euclidienne", "H2")); s.append(P_(f"Le cosinus gagne dans <b>{wins}/{len(d_)}</b> configurations ; écart moyen <b>{moy:+.4f}</b> mAP."))
    s += tbl(d_.rename(columns={"dataset_nom": "Base", "modele": "Modèle", "cosinus": "mAP cosinus", "euclidienne": "mAP euclidienne"}), weights=[1.6, 1.6, 1, 1, 1])
    s += fig("etape4_cos_euc.png", "Figure 5.1 – Écart de mAP entre cosinus et euclidienne.")
    s.append(P_("Compromis précision / vitesse", "H2")); s += fig("rapport_compromis.png", "Figure 5.2 – mAP en fonction du temps d'extraction (taille du point = dimension).", W * .8)
    s.append(P_("Types de fichiers et bases", "H2"))
    rows = [{"Base": nd(d), "Extension": ", ".join(R.ST[d]["extensions"]), "Résolution moy.": f"{R.ST[d]['largeur_moy']:.0f}×{R.ST[d]['hauteur_moy']:.0f}", "Classes": R.ST[d]["n_classes"],
             "mAP moyen": G[(G.dataset == d) & (G.metrique == "cosinus")].mAP.mean(), "P@10 moyen": G[(G.dataset == d) & (G.metrique == "cosinus")]["P@10"].mean()} for d in R.DSS]
    s += tbl(pd.DataFrame(rows), weights=[1.6, 1, 1.2, .8, 1, 1], hl_max=("mAP moyen",))
    s.append(P_("La difficulté d'une base dépend du <b>nombre de classes</b>, de leur proximité visuelle et de la <b>qualité des images</b> (PNG 32×32 agrandis, JPEG nets de Corel, grandes photos de monuments de Paris)."))
    return s


def s_conclusion(R):
    s = [P_("Conclusion", "H1")] + [LI(x) for x in A.conclusion_points(R)]
    s.append(BOX(f"Mode d'exécution : <b>{esc(R.PRESET)}</b> (full = toutes les images ; capN = au plus N images par classe). Les descripteurs sont utilisés sans fine-tuning ; la pertinence signifie « même classe » ; "
                 "CIFAR-100 est agrandi de 32 à 224 px.", "Limites"))
    return s


def s_cover(R, ART):
    dev = ""
    try:
        import json; dev = json.loads(next(C.FEATURES.glob("*.json")).read_text(encoding="utf-8")).get("device", "")
    except Exception: pass
    s = [Spacer(1, 3 * cm), P_("DeepCBIR", "title"), P_("Recherche d'images par le contenu avec des réseaux profonds pré-entraînés<br/>Rapport complet des résultats — toutes les étapes", "sub")]
    facts = pd.DataFrame([("Bases de données", ", ".join(f"{nd(d)} ({R.ST[d]['n_images']} img.)" for d in R.DSS)), ("Modèles", ", ".join(nm(m) for m in R.MODS)),
                          ("Similarités", "cosinus, euclidienne"), ("Mesures", "P@K, R@K, AP / mAP, courbes Précision-Rappel"), ("Mode d'exécution", R.PRESET),
                          ("Calcul", dev or "—"), ("Généré le", datetime.date.today().strftime("%d/%m/%Y"))], columns=["", ""])
    rows = [[Paragraph(f"<b>{esc(a)}</b>", S["td"]), Paragraph(esc(b), S["td"])] for a, b in facts.values]
    t = Table(rows, colWidths=[W * .25, W * .75]); t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#d9dde6")), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    s += [t, Spacer(1, 1 * cm)] + fig(f"echantillon_{R.DSS[0]}.png", f"Aperçu de la base {nd(R.DSS[0])}.", W * .6) + [PageBreak()]
    return s


def build_all(force=True):
    """Génère images + PDF ; retourne {'art': {page: [(titre, fichier, légende)]}, 'pdf': {nom: chemin}}."""
    R = A.load_results(); T, norms = A.extraction_table(R)
    out = make_figures(R, T, norms); ART = art(R, out["exemples"]); ex = out["exemples"]
    pdfs = {}
    build_pdf(PDFDIR / "rapport_complet.pdf", "Rapport complet", s_cover(R, ART) + s_etape1(R, ART) + [PageBreak()] + s_etape2(R, ART, T) + [PageBreak()] + s_etape3(R, ART, ex) + [PageBreak()] +
              s_etape4(R, ART) + [PageBreak()] + s_compar(R, ART) + [PageBreak()] + s_conclusion(R), toc=True)
    pdfs["rapport_complet"] = PDFDIR / "rapport_complet.pdf"
    for n, fn in [(1, lambda: s_etape1(R, ART)), (2, lambda: s_etape2(R, ART, T)), (3, lambda: s_etape3(R, ART, ex)), (4, lambda: s_etape4(R, ART))]:
        build_pdf(PDFDIR / f"etape{n}.pdf", f"Étape {n}", fn()); pdfs[f"etape{n}"] = PDFDIR / f"etape{n}.pdf"
    return dict(art=ART, pdf=pdfs)


if __name__ == "__main__":
    r = build_all()
    for k, v in r["pdf"].items(): print("[OK]", v, f"({os.path.getsize(v) / 1e6:.1f} Mo)")
