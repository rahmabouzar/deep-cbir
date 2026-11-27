"""Pages « Le projet » : architecture, technologies, formules. Appelées depuis app.py."""
import importlib.metadata as md
import math
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
import ui
from pages_outils import nb_lignes, montrer_code
from analyses import GFLOPS

DIMS = {"squeezenet1_1": 512, "mobilenet_v3_small": 576, "resnet18": 512, "resnet50": 2048, "vgg16": 4096}


# =========================================================================== ARCHITECTURE
ETAPES_CARTES = [
    dict(n=0, title="Planification", file="etape0_planification.py", icon="bars", tint="orange",
         text="Mesure la vitesse de votre machine et propose un plan qui tient dans votre budget de temps."),
    dict(n=1, title="Préparation", file="etape1_preparation.py", icon="image", tint="teal",
         text="Parcourt les bases, redimensionne en 224×224 et normalise avec les statistiques ImageNet."),
    dict(n=2, title="Extraction", file="etape2_extraction.py", icon="database", tint="purple",
         text="Indexation hors ligne : un vecteur par image et par réseau, avec reprise après interruption."),
    dict(n=3, title="Recherche", file="etape3_recherche.py", icon="search", tint="green",
         text="Distance cosinus ou euclidienne entre la requête et la base, puis les K images les plus proches."),
    dict(n=4, title="Évaluation", file="etape4_evaluation.py", icon="curve", tint="blue",
         text="Courbes de précision et de rappel, AP, mAP, P@K et R@K pour chaque requête de chaque base."),
    dict(n=5, title="Rapport", file="etape5_rapport.py", icon="compare", tint="pink",
         text="Rédige le rapport complet en Markdown et en HTML autonome."),
    dict(n=6, title="Site HTML", file="etape6_site_html.py", icon="grid", tint="orange",
         text="Site hors ligne : une page par étape, avec résultats, explications et images à télécharger."),
    dict(n=7, title="PDF", file="etape7_pdf.py", icon="matrix", tint="teal",
         text="Images de résultats de chaque étape et rapport PDF avec sommaire cliquable."),
]

COUCHES = [
    dict(title="Interface", sub="Application web", icon="home", tint="blue",
         files=["app.py", "page_experience.py", "pages_projet.py", "pages_etapes.py", "ui.py", "dialogue.py"]),
    dict(title="Logique", sub="Fonctions réutilisables", icon="layers", tint="purple",
         files=["common.py", "recherche_eval.py", "experience.py", "analyses.py"]),
    dict(title="Pipeline", sub="Scripts en ligne de commande", icon="curve", tint="orange",
         files=[f"etape{n}_{x}.py" for n, x in ((0, "planification"), (1, "preparation"), (2, "extraction"), (3, "recherche"),
                                                (4, "evaluation"), (5, "rapport"), (6, "site_html"), (7, "pdf"))]),
    dict(title="Configuration", sub="Une seule source de vérité", icon="target", tint="teal",
         files=["config.py", ".streamlit/config.toml", "requirements.txt"]),
    dict(title="Données", sub="Images et fichiers produits", icon="database", tint="green",
         files=["bases d'images", "workdir/manifests", "workdir/features", "results/"]),
]

ROLES = {
    "app.py": "application Streamlit, navigation et pages de résultats", "ui.py": "système visuel, CSS, composants et animations",
    "page_experience.py": "page Expérience sur échantillon", "pages_projet.py": "pages Architecture, Technologies, Formules",
    "pages_etapes.py": "une page de résultats par étape", "pages_outils.py": "lecture du code source et utilitaires de pages",
    "dialogue.py": "boîte de dialogue native pour choisir une image",
    "common.py": "scan des bases, prétraitement, modèles, distances", "recherche_eval.py": "précision, rappel, AP, RRF, robustesse d'une requête",
    "experience.py": "expérience aléatoire sur un échantillon de classes", "analyses.py": "calculs partagés par le site HTML et le PDF",
    "config.py": "bases, modèles, graine, tailles, métriques", "design_check.py": "contrôle automatique des règles de design",
    "etape0_planification.py": "plan de calcul selon la machine", "etape1_preparation.py": "manifestes, statistiques, prétraitement",
    "etape2_extraction.py": "vecteurs caractéristiques, reprise sur blocs", "etape3_recherche.py": "Top K cosinus et euclidienne",
    "etape4_evaluation.py": "métriques de recherche, incrémentale", "etape5_rapport.py": "rapport Markdown et HTML",
    "etape6_site_html.py": "site HTML multipages", "etape7_pdf.py": "figures et rapports PDF",
    "requirements.txt": "dépendances Python", "README.md": "présentation du projet",
}


def _base(_dossier, ST, ds_key=None, defaut=""):
    """Description d'une base à partir des statistiques de l'étape 1 si elles existent."""
    s = (ST or {}).get(ds_key) if ds_key else None
    return f"{s['n_images']:,} images, {s['n_classes']} classes".replace(",", " ") if s else defaut


def _dispo(chemin):
    return "utilisé" if (C.DATA_DIR / chemin).exists() else "introuvable"


def architecture(ctx):
    ui.page_header("Architecture du projet", "Comment le projet est construit : le pipeline en huit étapes, les couches du code, "
                   "l'organisation des dossiers et le trajet d'une image jusqu'à sa mesure de précision.", icon="layers", tint="blue")

    ui.step(1, "Le pipeline", "huit scripts, chacun produit les fichiers du suivant")
    ui.pipe(ETAPES_CARTES, cyc=8400)

    ui.step(2, "Du pixel à la mesure", "ce que devient une image au fil du pipeline")
    ui.chain([("Image", "JPEG ou PNG de la base, classe lue dans le nom du dossier"),
              ("Tenseur", "3 × 224 × 224, redimensionné et normalisé ImageNet"),
              ("Réseau", "convolutions puis pooling global ou couche fc7"),
              ("Vecteur", "d dimensions : 512, 576, 2048 ou 4096"),
              ("Index", "matrice N × d enregistrée en .npy"),
              ("Mesure", "distance, classement, précision, rappel, AP")], tint="blue")

    ui.step(3, "Les couches du code", "chaque couche ne dépend que de celles du dessous")
    ui.layers(COUCHES)

    ui.step(4, "Organisation des dossiers", "à gauche les données, à droite le code")
    st.markdown("Le dossier **MiniProject** contient les bases à côté du dossier `script`. "
                "Le chemin est modifiable avec la variable d'environnement `CBIR_DATA_DIR`.")
    ST = ctx.ST
    c1, c2 = st.columns(2)
    with c1:
        ui.cap("Données", f"dossier courant : {C.DATA_DIR.name}")
        ui.tree([
            dict(d=0, nm="MiniProject", kind="dir", desc="dossier parent"),
            dict(d=1, nm="CIFAR100", kind="dir", desc="trois partitions"),
            dict(d=2, nm="TEST", kind="dir", desc=_base("CIFAR100", ST, "cifar100", "10 000 images PNG 32×32, 100 classes"),
                 tag=_dispo("CIFAR100"), tag_cls="use" if (C.DATA_DIR / "CIFAR100").exists() else "bad"),
            dict(d=2, nm="TRAIN", kind="dir", desc="non utilisé", tag="ignoré", tag_cls="skip"),
            dict(d=2, nm="VALID", kind="dir", desc="non utilisé", tag="ignoré", tag_cls="skip"),
            dict(d=1, nm="Corel-1K / Corel-1K", kind="dir", desc=_base("Corel-1K", ST, "corel1k", "1 000 images, 10 classes"),
                 tag=_dispo("Corel-1K"), tag_cls="use" if (C.DATA_DIR / "Corel-1K").exists() else "bad"),
            dict(d=1, nm="Corel-5K / Corel-5K", kind="dir", desc=_base("Corel-5K", ST, "corel5k", "5 000 images, 50 classes"),
                 tag=_dispo("Corel-5K"), tag_cls="use" if (C.DATA_DIR / "Corel-5K").exists() else "bad"),
            dict(d=1, nm="Corel-10K", kind="dir", desc=_base("Corel-10K", ST, "corel10k", "10 000 images, 100 classes"),
                 tag=_dispo("Corel-10K"), tag_cls="use" if (C.DATA_DIR / "Corel-10K").exists() else "bad"),
            dict(d=1, nm="paris", kind="dir", desc=_base("paris", ST, "paris6k", "6 412 images, monuments"),
                 tag=_dispo("paris"), tag_cls="use" if (C.DATA_DIR / "paris").exists() else "bad"),
            dict(d=2, nm="paris_1", kind="dir", desc="paris_monument_NNNNNN.jpg", tag="utilisé", tag_cls="use"),
            dict(d=2, nm="paris_2", kind="dir", desc="classe lue dans le nom du fichier", tag="utilisé", tag_cls="use"),
            dict(d=2, nm="oxbuild_images", kind="dir", desc="Oxford5k", tag="ignoré", tag_cls="skip"),
            dict(d=1, nm="script", kind="dir", desc="le code du projet (à droite)"),
        ])
        st.caption("Les dossiers Corel sont parcourus récursivement : le sous dossier du même nom est donc trouvé automatiquement. "
                   "Pour CIFAR-100, seule la partition TEST est indexée.")
    with c2:
        ui.cap("Code", "script/")
        sources = [("app.py", "py"), ("ui.py", "py"), ("pages_projet.py", "py"), ("pages_etapes.py", "py"), ("pages_outils.py", "py"),
                   ("page_experience.py", "py"), ("common.py", "py"), ("recherche_eval.py", "py"), ("experience.py", "py"),
                   ("analyses.py", "py"), ("dialogue.py", "py"), ("config.py", "cfg"), ("design_check.py", "py")]
        rows = [dict(d=0, nm="script", kind="dir", desc="dossier du code")]
        rows += [dict(d=1, nm=f, kind=k, desc=f"{ROLES.get(f, '')} · {nb_lignes(f)} lignes") for f, k in sources]
        rows += [dict(d=1, nm=f"etape{n}_{x}.py", kind="py", desc=f"{ROLES.get(f'etape{n}_{x}.py', '')} · {nb_lignes(f'etape{n}_{x}.py')} lignes")
                 for n, x in ((0, "planification"), (1, "preparation"), (2, "extraction"), (3, "recherche"),
                              (4, "evaluation"), (5, "rapport"), (6, "site_html"), (7, "pdf"))]
        rows += [dict(d=1, nm="assets/", kind="img", desc="favicon et icônes Phosphor"),
                 dict(d=1, nm=".streamlit/config.toml", kind="cfg", desc="thème de l'application"),
                 dict(d=1, nm="workdir/", kind="dir", desc="manifestes et vecteurs", tag="généré", tag_cls="skip"),
                 dict(d=1, nm="results/", kind="dir", desc="métriques, figures, rapport, site, PDF", tag="généré", tag_cls="skip")]
        ui.tree(rows)

    ui.step(5, "Fichiers produits", "qui écrit quoi, et ce qui est déjà présent sur cette machine")
    prod = [("workdir/manifests/<base>.csv", "Étape 1", "Étapes 2, 3, 4, application", C.MANIFESTS, "*.csv"),
            ("results/etape1_stats.json", "Étape 1", "Accueil, étape 1", C.RESULTS, "etape1_stats.json"),
            ("results/figures/echantillon_*.png", "Étape 1", "Accueil, étape 1", C.FIGS, "echantillon_*.png"),
            ("workdir/features/<base>__<modèle>.npy", "Étape 2", "Étapes 3 et 4, application", C.FEATURES, "*.npy"),
            ("workdir/features/<base>__<modèle>.json", "Étape 2", "Étape 4 (dimension, temps)", C.FEATURES, "*.json"),
            ("results/figures/recherche_*.png", "Étape 3", "Étape 3", C.FIGS, "recherche_*.png"),
            ("results/metrics_global.csv", "Étape 4", "Résultats globaux, rapport", C.RESULTS, "metrics_global.csv"),
            ("results/metrics_par_classe.csv", "Étape 4", "Comparaison par classe", C.RESULTS, "metrics_par_classe.csv"),
            ("results/pr_curves.json", "Étape 4", "Courbes précision et rappel", C.RESULTS, "pr_curves.json"),
            ("results/confusion/*.npy", "Étape 4", "Matrices de confusion", C.CONF_DIR, "*.npy"),
            ("results/rapport.html", "Étape 5", "Livrable", C.RESULTS, "rapport.html"),
            ("results/site/index.html", "Étape 6", "Livrable", C.RESULTS / "site", "index.html"),
            ("results/pdf/rapport_complet.pdf", "Étape 7", "Livrable", C.RESULTS / "pdf", "rapport_complet.pdf")]
    df = pd.DataFrame([{"Fichier": f, "Produit par": p, "Lu par": u, "Présent": ("oui" if (d.exists() and any(d.glob(g))) else "non")} for f, p, u, d, g in prod])
    st.dataframe(df, width="stretch", hide_index=True)


# =========================================================================== TECHNOLOGIES
def _version(pkg, requis=""):
    try:
        return f"version {md.version(pkg)}"
    except md.PackageNotFoundError:
        return f"requis {requis}" if requis else "non installé ici"


def technologies(ctx):
    ui.page_header("Technologies utilisées", "La pile logicielle du projet, le rôle de chaque bibliothèque et les choix techniques "
                   "qui rendent l'indexation et l'évaluation rapides.", icon="grid", tint="purple")
    py = f"version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    ui.step(1, "Apprentissage profond", "les réseaux préentraînés et leur exécution")
    ui.tech_cards([
        dict(name="Python", version=py, icon="home", tint="blue", used=["tous les scripts"],
             text="Langage unique du projet : pipeline, calculs numériques, évaluation et interface web."),
        dict(name="PyTorch", version=_version("torch", "installé à part"), icon="layers", tint="purple", used=["etape0", "etape2", "common.py"],
             text="Exécute les réseaux en mode inférence : inference_mode, channels_last, tous les cœurs CPU, GPU choisi automatiquement."),
        dict(name="torchvision", version=_version("torchvision", "installé à part"), icon="image", tint="purple", used=["common.py"],
             text="SqueezeNet 1.1, MobileNetV3 Small, ResNet 18 et 50, VGG 16 avec poids ImageNet, plus Resize, ToTensor et Normalize."),
    ])

    ui.step(2, "Calcul et données", "les structures qui portent les vecteurs et les métriques")
    ui.tech_cards([
        dict(name="NumPy", version=_version("numpy", ">=1.26"), icon="matrix", tint="teal", used=["common.py", "etape4", "recherche_eval.py"],
             text="Distances par produit matriciel, classement, AP et courbes calculés pour des centaines de requêtes à la fois."),
        dict(name="pandas", version=_version("pandas", ">=2.1"), icon="database", tint="green", used=["etape1", "etape4", "app.py"],
             text="Manifestes CSV (chemin, classe, résolution), tableaux de métriques, pivots pour les comparaisons."),
        dict(name="Pillow", version=_version("pillow", ">=10.0"), icon="image", tint="orange", used=["common.py", "recherche_eval.py"],
             text="Lecture des images avec décodage JPEG réduit, et perturbations de la page Recherche (flou, bruit, rotation)."),
    ])

    ui.step(3, "Interface et visualisation", "ce que vous voyez dans le navigateur")
    ui.tech_cards([
        dict(name="Streamlit", version=_version("streamlit", ">=1.50"), icon="home", tint="blue", used=["app.py", "page_experience.py", "pages_projet.py"],
             text="Application web : navigation par groupes, widgets, cache des données et des figures."),
        dict(name="Plotly", version=_version("plotly", ">=5.20"), icon="curve", tint="orange", used=["app.py", "etape6"],
             text="Graphiques interactifs : barres, courbes précision et rappel, cartes de chaleur, nuages de points."),
        dict(name="Matplotlib", version=_version("matplotlib", ">=3.8"), icon="bars", tint="teal", used=["etape1", "etape3", "etape7"],
             text="Figures PNG de chaque étape : échantillons, prétraitement, résultats de recherche."),
        dict(name="tkinter", version="bibliothèque standard", icon="search", tint="pink", used=["dialogue.py"],
             text="Fenêtre native pour choisir l'image requête, lancée dans un processus séparé pour éviter les conflits."),
    ])

    ui.step(4, "Rapports", "les livrables des étapes 5 à 7")
    ui.tech_cards([
        dict(name="Markdown", version=_version("markdown", ">=3.5"), icon="compare", tint="pink", used=["etape5"],
             text="Convertit le rapport Markdown en page HTML autonome, images incluses."),
        dict(name="ReportLab", version=_version("reportlab", ">=4.0"), icon="grid", tint="purple", used=["etape7"],
             text="Génère le rapport PDF complet avec sommaire cliquable, et un PDF par étape."),
        dict(name="tqdm", version=_version("tqdm", ">=4.66"), icon="bars", tint="green", used=["etape1", "etape2"],
             text="Barres de progression en ligne de commande pour la lecture des images et l'extraction."),
    ])

    ui.step(5, "Système visuel", "un seul fichier définit le design")
    ui.tech_cards([
        dict(name="Geist", version="police unique", icon="target", tint="blue", used=["ui.py"],
             text="Une seule famille de caractères, trois graisses, échelle de tailles Tailwind avec styles nommés."),
        dict(name="Phosphor", version="icônes, licence MIT", icon="image", tint="teal", used=["assets/phosphor"],
             text="Icônes SVG intégrées en données dans le HTML, recolorées selon la couleur de la page."),
        dict(name="CSS et animations", version="opacité et transform", icon="curve", tint="orange", used=["ui.py", "design_check.py"],
             text="Jetons d'espacement et de rayon, une seule courbe de mouvement, mouvement réduit respecté, contrôle automatique."),
    ])

    ui.step(6, "Choix techniques", "ce qui rend le projet rapide et reproductible")
    ui.pipe([
        dict(title="Décodage unique", file="etape2_extraction.py", icon="image", tint="teal", n="a",
             text="Chaque image est lue une seule fois puis passée dans tous les réseaux du lot."),
        dict(title="Reprise sur blocs", file="etape2_extraction.py", icon="database", tint="purple", n="b",
             text="Sauvegarde tous les 1000 images : un arrêt ou une coupure ne fait rien perdre."),
        dict(title="Un produit matriciel", file="common.py", icon="matrix", tint="blue", n="c",
             text="Cosinus et euclidienne se calculent avec une seule multiplication de matrices."),
        dict(title="Top K partiel", file="common.py", icon="search", tint="green", n="d",
             text="argpartition isole les K plus proches sans trier toute la base."),
        dict(title="Évaluation par blocs", file="etape4_evaluation.py", icon="curve", tint="orange", n="e",
             text="La mémoire est bornée même pour 10 000 images, et les couples déjà évalués sont ignorés."),
        dict(title="Résultats reproductibles", file="config.py", icon="target", tint="pink", n="f",
             text="Graine fixe, exclusion de la requête elle même, pertinence définie par la classe."),
    ], cyc=6000)
    ui.note("Les versions affichées sont celles installées dans l'environnement qui exécute cette application. "
            "Sans version installée, la valeur minimale de requirements.txt est indiquée.", "blue")


# =========================================================================== FORMULES
def _legende(*lignes):
    st.markdown("\n".join(f"- {x}" for x in lignes))


def _fiche(cle, titre, sous, icon, tint, tag):
    c = st.container(border=True, key=f"fcard_{cle}")
    with c:
        ui.formula_head(titre, sous, icon, tint, tag)
    return c


def pr_exemple(rel_rangs, n_rel, n_liste):
    """Mêmes calculs que etape4_evaluation.evaluate pour UNE requête : précision, rappel et AP.
    rel_rangs : rangs (1 à n_liste) des images pertinentes ; n_rel : nombre total d'images pertinentes dans la base."""
    rel = np.zeros(n_liste, dtype=bool)
    for r in rel_rangs:
        rel[r - 1] = True
    cum = np.cumsum(rel)
    prec = cum / np.arange(1, n_liste + 1)
    rap = cum / max(n_rel, 1)
    pts = np.flatnonzero(rel)
    ph = prec[pts]                                                   # précision à chaque image pertinente
    ap = float(ph.sum() / max(n_rel, 1))                             # les pertinentes jamais retrouvées comptent 0
    pm = np.maximum.accumulate(ph[::-1])[::-1] if len(ph) else ph
    niveaux = np.array(C.RECALL_LEVELS)
    interp = []
    for lv in niveaux:                                               # précision interpolée : max des précisions à rappel >= niveau
        ok = [p for p, r in zip(prec[pts], rap[pts]) if r >= lv - 1e-9]
        interp.append(max(ok) if ok else 0.0)
    return dict(rel=rel, cum=cum, prec=prec, rap=rap, ap=ap, ph=ph, pts=pts, niveaux=niveaux, interp=np.array(interp), pm=pm)


def _svg_cosinus(theta0=60):
    """Animation : le vecteur x tourne autour de q ; l'angle (arc) et la corde (distance euclidienne) suivent.
    Animation CSS dans le SVG : elle s'arrête si l'utilisateur demande moins de mouvement (pose fixe = angle du curseur)."""
    R, cx, cy = 110, 130, 150
    angs = list(range(10, 171, 10)) + list(range(160, 10, -10))
    n = len(angs)
    blue, orange, purple, teal = ui.PRIMARY, "#dd5b00", "#7a4cc2", "#2a9d99"
    rot = lambda a: f"rotate({a}deg)"
    corde = lambda a: f"rotate({a / 2}deg) translateX({R * math.cos(math.radians(a / 2)):.2f}px) scaleY({math.sin(math.radians(a / 2)):.3f})"

    def kf(name, f):
        return f"@keyframes {name}{{" + "".join(f"{100 * i / (n - 1):.2f}%{{{f(a)}}}" for i, a in enumerate(angs)) + "}"
    th = theta0
    css = (kf("rx", lambda a: f"transform:{rot(a)}") + kf("ch", lambda a: f"transform:{corde(a)}") + kf("ar", lambda a: f"stroke-dasharray:{a} 360")
           + f".rx{{transform:{rot(th)}}}.ch{{transform:{corde(th)}}}.ar{{stroke-dasharray:{th} 360}}"
           + ".a{animation-duration:9s;animation-iteration-count:infinite;animation-timing-function:linear}"
           + ".rx{animation-name:rx}.ch{animation-name:ch}.ar{animation-name:ar}text{font-family:system-ui,sans-serif;font-size:16px;font-weight:600}"
           + "@media (prefers-reduced-motion: reduce){.a{animation:none}}")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300" width="300" height="300"><style>{css}</style>'
            f'<g transform="translate({cx} {cy}) scale(1 -1)" fill="none" stroke-linecap="round">'
            f'<circle r="{R}" stroke="#e6e6e6" stroke-width="1.5"/>'
            f'<circle class="a ar" r="34" pathLength="360" stroke="{teal}" stroke-width="4"/>'
            f'<line x1="0" y1="0" x2="{R}" y2="0" stroke="{blue}" stroke-width="4"/><circle cx="{R}" cy="0" r="6" fill="{blue}"/>'
            f'<g class="a ch"><line x1="0" y1="-{R}" x2="0" y2="{R}" stroke="{purple}" stroke-width="3"/></g>'
            f'<g class="a rx"><line x1="0" y1="0" x2="{R}" y2="0" stroke="{orange}" stroke-width="4"/><circle cx="{R}" cy="0" r="6" fill="{orange}"/></g>'
            f'<circle r="5" fill="#0d0d0d"/></g>'
            f'<text x="{cx + R + 10}" y="{cy + 5}" fill="{blue}">q</text></svg>')


def formules(ctx):
    ui.page_header("Formules et méthodes de calcul", "Chaque mesure du projet avec sa formule, ses symboles et l'endroit du code où elle est calculée. "
                   "Les exemples sont interactifs et reprennent exactement les calculs de l'évaluation.", icon="matrix", tint="orange")
    ui.pipe([dict(n=1, title="Descripteur", file="etape2", icon="database", tint="purple", text="Prétraitement, pooling global, normalisation L2."),
             dict(n=2, title="Similarité", file="etape3", icon="search", tint="green", text="Distance cosinus et distance euclidienne."),
             dict(n=3, title="Précision et rappel", file="etape4", icon="target", tint="blue", text="P@K, R@K pour un nombre K de résultats."),
             dict(n=4, title="AP et mAP", file="etape4", icon="curve", tint="orange", text="Précision moyenne par requête puis sur toute la base."),
             dict(n=5, title="Consensus", file="app.py", icon="compare", tint="pink", text="Fusion RRF, accord de Jaccard, vote des voisins."),
             dict(n=6, title="Temps", file="etape0", icon="bars", tint="teal", text="Estimation de la durée d'extraction et d'évaluation.")], cyc=6000)

    # ------------------------------------------------------------ 1 descripteur
    ui.step(1, "Du pixel au descripteur", "étapes 1 et 2")
    mu, sg = C.IMAGENET_MEAN, C.IMAGENET_STD
    with _fiche("prep", "Prétraitement ImageNet", "Même format d'entrée que celui des réseaux à l'entraînement", "image", "teal", "Étape 1"):
        st.latex(r"x_c=\frac{I_c}{255},\qquad x'_c=\frac{x_c-\mu_c}{\sigma_c},\qquad I\in\mathbb{R}^{H\times W\times 3}\rightarrow x'\in\mathbb{R}^{3\times 224\times 224}")
        st.latex(r"\mu=(%s),\qquad \sigma=(%s)" % (", ".join(f"{v}" for v in mu), ", ".join(f"{v}" for v in sg)))
        _legende("$I$ : image d'origine, redimensionnée en 224 par 224 pixels ; $c$ : canal rouge, vert ou bleu.",
                 "$\\mu_c$ et $\\sigma_c$ : moyenne et écart type de chaque canal sur ImageNet (valeurs de `config.py`).")
    with _fiche("gap", "Descripteur : pooling global", "Une valeur par carte de caractéristiques", "database", "purple", "Étape 2"):
        st.latex(r"v_c=\frac{1}{H'W'}\sum_{i=1}^{H'}\sum_{j=1}^{W'}F_c(i,j),\qquad v\in\mathbb{R}^{d}")
        _legende("$F_c(i,j)$ : activation de la carte $c$ à la position $(i,j)$ en sortie du dernier bloc convolutif.",
                 "Dimensions $d$ : " + ", ".join(f"{C.MODELS[m][0]} {d}" for m, d in DIMS.items()) + ".",
                 "VGG 16 n'utilise pas de pooling global : le descripteur est la sortie de la couche fc7, de dimension 4096.")
        st.latex(r"v_{\text{VGG}}=\mathrm{ReLU}\big(W_7\,\mathrm{ReLU}(W_6\,\mathrm{vec}(F)+b_6)+b_7\big)")
    with _fiche("l2", "Normalisation L2 et taille de l'index", "Rend la longueur des vecteurs sans effet", "layers", "blue", "Étapes 2 et 3"):
        st.latex(r"\hat v=\frac{v}{\lVert v\rVert_2},\qquad \lVert v\rVert_2=\sqrt{\sum_{i=1}^{d}v_i^{2}},\qquad \text{Taille de l'index}=N\times d\times 4\ \text{octets}")
        _legende("$N$ : nombre d'images de la base ; chaque valeur est un nombre à virgule flottante de 4 octets (float32).")

    # ------------------------------------------------------------ 2 similarité
    ui.step(2, "Mesures de similarité", "étape 3")
    with _fiche("cos", "Distance cosinus", "Compare la direction des vecteurs, pas leur longueur", "compare", "blue", "Étape 3"):
        st.latex(r"d_{\cos}(q,x)=1-\frac{q\cdot x}{\lVert q\rVert\,\lVert x\rVert}=1-\cos\theta")
        _legende("$q$ : vecteur de la requête ; $x$ : vecteur d'une image de la base ; $\\theta$ : angle entre les deux.",
                 "0 signifie même direction. Plus la valeur est petite, plus l'image est proche.")
    with _fiche("euc", "Distance euclidienne", "Distance en ligne droite entre les deux extrémités", "compare", "orange", "Étape 3"):
        st.latex(r"d_{2}(q,x)=\lVert q-x\rVert_2=\sqrt{\sum_{i=1}^{d}(q_i-x_i)^2}")
        st.latex(r"\lVert q-x\rVert^2=\lVert q\rVert^2+\lVert x\rVert^2-2\,q\cdot x\qquad\text{(un seul produit matriciel pour toute la base)}")
        _legende("Si les vecteurs sont normalisés : $\\lVert \\hat q-\\hat x\\rVert^2 = 2\\,d_{\\cos}(q,x)$. Les deux classements sont alors identiques.",
                 "Sans normalisation, la distance euclidienne dépend aussi de la longueur des vecteurs.")

    cA, cB = st.columns([1, 1])
    with cB:
        st.markdown("**Essayez avec deux vecteurs**")
        theta = st.slider("Angle entre q et x (degrés)", 0, 180, 60, key="f_theta")
        nx = st.slider("Longueur de x (celle de q vaut 1)", 0.5, 3.0, 1.5, step=0.1, key="f_nx")
        t = math.radians(theta)
        dcos = 1 - math.cos(t)
        deuc = math.sqrt(1 + nx ** 2 - 2 * nx * math.cos(t))
        dnorm = 2 * math.sin(t / 2)
        ui.stats([dict(label="Distance cosinus", value=f"{dcos:.3f}", sub="ne dépend pas de la longueur", icon="compare", tint="blue"),
                  dict(label="Distance euclidienne", value=f"{deuc:.3f}", sub=f"avec x de longueur {nx:.1f}", icon="compare", tint="orange"),
                  dict(label="Euclidienne normalisée", value=f"{dnorm:.3f}", sub=f"racine de 2 fois {dcos:.3f}", icon="compare", tint="green")])
    with cA:
        ui.figure_svg(_svg_cosinus(theta), "Bleu : q. Orange : x qui tourne. Vert : l'angle. Violet : la corde, soit la distance euclidienne.")

    # ------------------------------------------------------------ 3 précision rappel
    ui.step(3, "Précision et rappel", "étape 4, pour une requête")
    with _fiche("pr", "Précision@K et Rappel@K", "Qualité des K premiers résultats", "target", "blue", "Étape 4"):
        st.latex(r"P@K=\frac{\left|\{\text{pertinentes parmi les }K\text{ premières}\}\right|}{K},\qquad R@K=\frac{\left|\{\text{pertinentes parmi les }K\text{ premières}\}\right|}{N_{\mathrm{rel}}}")
        _legende("Une image est **pertinente** si elle appartient à la même classe que la requête.",
                 "$N_{\\mathrm{rel}}$ : nombre d'images de cette classe dans la base, la requête elle même exclue.",
                 "La précision répond à « parmi ce que je retourne, combien est juste ? », le rappel à « parmi ce qui existe, combien ai je retrouvé ? ».")

    st.markdown("**Exemple pas à pas.** Choisissez quelles images du classement sont pertinentes : la bande, le tableau et les courbes se mettent à jour.")
    e1, e2 = st.columns([2, 1])
    with e1:
        n_liste = 12
        sel = st.multiselect("Rangs des images pertinentes", list(range(1, n_liste + 1)), default=[1, 2, 4, 5, 8, 11], key="f_rel")
    with e2:
        K = st.slider("K", 1, n_liste, 5, key="f_K")
    n_rel = max(st.slider("Images pertinentes dans toute la base (N rel)", 1, 30, 8, key="f_nrel"), len(sel), 1)
    ex = pr_exemple(sorted(sel), n_rel, n_liste)
    ui.rank_strip(ex["rel"])
    st.caption("Sous chaque case : la précision calculée jusqu'à ce rang.")
    hits = int(ex["cum"][K - 1])
    ui.stats([dict(label=f"Précision@{K}", value=f"{hits / K:.3f}", sub=f"{hits} pertinentes sur {K} retournées", icon="target", tint="blue"),
              dict(label=f"Rappel@{K}", value=f"{hits / n_rel:.3f}", sub=f"{hits} retrouvées sur {n_rel} existantes", icon="grid", tint="teal"),
              dict(label="AP", value=f"{ex['ap']:.3f}", sub="voir l'étape 4 ci dessous", icon="bars", tint="purple"),
              dict(label="Premier résultat pertinent", value=(f"rang {int(ex['pts'][0]) + 1}" if len(ex["pts"]) else "aucun"), icon="search", tint="orange")])
    st.latex(rf"P@{K}=\frac{{{hits}}}{{{K}}}={hits / K:.3f}\qquad R@{K}=\frac{{{hits}}}{{{n_rel}}}={hits / n_rel:.3f}")

    # ------------------------------------------------------------ 4 AP mAP
    ui.step(4, "Précision moyenne (AP) et mAP", "étape 4, pour une requête puis pour toute la base")
    with _fiche("ap", "AP : Average Precision", "Moyenne des précisions aux rangs où une image pertinente apparaît", "curve", "orange", "Étape 4"):
        st.latex(r"AP(q)=\frac{1}{N_{\mathrm{rel}}}\sum_{k=1}^{n}P(k)\cdot\mathrm{rel}(k)")
        st.latex(r"mAP=\frac{1}{|Q|}\sum_{q\in Q}AP(q)")
        _legende("$P(k)$ : précision au rang $k$ ; $\\mathrm{rel}(k)$ vaut 1 si l'image au rang $k$ est pertinente, sinon 0.",
                 "$Q$ : l'ensemble des requêtes. Chaque image de la base est une requête, et elle s'exclut elle même du classement.",
                 "Une image pertinente jamais retrouvée compte pour 0 : l'AP baisse.")
    if len(ex["pts"]):
        termes = " + ".join(f"{p:.2f}" for p in ex["ph"])
        st.latex(rf"AP=\frac{{1}}{{{n_rel}}}\left({termes}\right)={ex['ap']:.3f}")
    st.dataframe(pd.DataFrame({"Rang": np.arange(1, n_liste + 1), "Pertinente": np.where(ex["rel"], "oui", "non"),
                               "Pertinentes jusqu'ici": ex["cum"], "Précision au rang": ex["prec"].round(3), "Rappel au rang": ex["rap"].round(3)}),
                 width="stretch", hide_index=True)

    # ------------------------------------------------------------ 5 courbe PR
    ui.step(5, "Courbe précision et rappel", "interpolation sur 11 niveaux de rappel")
    with _fiche("interp", "Précision interpolée", "Rend la courbe monotone et comparable entre requêtes", "curve", "green", "Étape 4"):
        st.latex(r"p_{\mathrm{interp}}(r)=\max_{r'\ge r}\,p(r'),\qquad r\in\{0,\,0.1,\,0.2,\,\dots,\,1\}")
        _legende("$p(r')$ : précision mesurée au niveau de rappel $r'$. La courbe affichée dans l'application est la moyenne de ces valeurs sur toutes les requêtes.")
    fig = go.Figure()
    fig.add_scatter(x=list(ex["rap"][ex["pts"]]), y=list(ex["prec"][ex["pts"]]), mode="markers+lines", name="Précision mesurée", line_shape="hv")
    fig.add_scatter(x=list(ex["niveaux"]), y=list(ex["interp"]), mode="lines+markers", name="Précision interpolée")
    fig.update_layout(title="Courbe précision et rappel de l'exemple", xaxis_title="Rappel", yaxis_title="Précision", xaxis_range=[0, 1.02], yaxis_range=[0, 1.05])
    ui.show(fig, height=380)

    # ------------------------------------------------------------ 6 autres mesures
    ui.step(6, "Autres mesures de la page Recherche", "précision à R, consensus, confusion")
    with _fiche("rprec", "Précision à R et rang du premier pertinent", "Deux lectures rapides du classement", "target", "teal", "Recherche"):
        st.latex(r"R\text{-précision}=P@N_{\mathrm{rel}},\qquad \text{premier}=\min\{k:\mathrm{rel}(k)=1\}")
    with _fiche("rrf", "Fusion RRF (Reciprocal Rank Fusion)", "Combine les classements de plusieurs réseaux", "layers", "pink", "Recherche"):
        st.latex(r"\mathrm{RRF}(d)=\sum_{m\in M}\frac{1}{c+\mathrm{rang}_m(d)},\qquad c=60")
        _legende("$\\mathrm{rang}_m(d)$ : position de l'image $d$ dans le classement du réseau $m$. Les images bien classées par plusieurs réseaux passent devant.")
    with _fiche("jac", "Accord de Jaccard et vote des voisins", "Les réseaux retrouvent ils les mêmes images ?", "compare", "purple", "Recherche"):
        st.latex(r"J(A,B)=\frac{|A\cap B|}{|A\cup B|},\qquad \hat y=\arg\max_{c}\sum_{i=1}^{K}\mathbb{1}[y_i=c]")
        _legende("$A$ et $B$ : les K premiers résultats de deux réseaux. $\\hat y$ : classe prédite par vote des K voisins ; la confiance est la part des voix.")
    with _fiche("conf", "Matrice de confusion", "Où vont les erreurs de recherche", "matrix", "pink", "Étape 4"):
        st.latex(r"M_{c,c'}=\frac{\#\{\text{résultats de classe }c'\text{ parmi les 10 premiers des requêtes de classe }c\}}{10\times\#\{\text{requêtes de classe }c\}}")

    # ------------------------------------------------------------ 7 temps
    ui.step(7, "Estimation du temps de calcul", "étape 0")
    with _fiche("temps", "Modèle de coût", "Pour choisir un plan qui tient dans le budget", "bars", "orange", "Étape 0"):
        st.latex(r"t_{\mathrm{extraction}}=\sum_{b}N_b\cdot\max\Big(\sum_{m}t_m,\ \frac{t^{\mathrm{déc}}_b}{w}\Big)")
        st.latex(r"t_{\mathrm{évaluation}}=\sum_{b}t_{\mathrm{réf}}\Big(\frac{N_b}{N_{\mathrm{réf}}}\Big)^{2}\cdot|M|\cdot 2,\qquad t_{\mathrm{total}}=1.03\,(t_{\mathrm{extraction}}+t_{\mathrm{évaluation}})")
        _legende("$N_b$ : images de la base $b$ ; $t_m$ : millisecondes par image du réseau $m$ ; $t^{\\mathrm{déc}}_b$ : temps de lecture d'une image ; $w$ : processus de décodage.",
                 "L'évaluation croît comme le carré de $N$ : chaque image est une requête comparée à toute la base. Le facteur 2 couvre les deux métriques.")
    montrer_code("recherche_eval.py", "metriques", "précision, rappel, AP, précision à R, rang du premier")
