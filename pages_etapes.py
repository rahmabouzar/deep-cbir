"""Une page de résultats par étape du pipeline (étapes 0 à 7). Chaque page : entrées, script, sorties, résultats réels, code réel.
Les résultats sont lus dans workdir/ et results/ ; sans fichiers, la page explique quelle commande lancer."""
import random
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as C
import ui
from analyses import GFLOPS
from common import load_meta, class_names, top_k
from pages_outils import montrer_code, commande

DIMS = {"squeezenet1_1": 512, "mobilenet_v3_small": 576, "resnet18": 512, "resnet50": 2048, "vgg16": 4096}
COUCHE = {"squeezenet1_1": "features puis pooling global", "mobilenet_v3_small": "features puis avgpool",
          "resnet18": "avant la couche fc (pooling global)", "resnet50": "avant la couche fc (pooling global)", "vgg16": "fc7, sortie de classifier[:5]"}
SIM, DIF = ":green[**Similaire**]", ":red[**Différente**]"


def _tete(n, titre, sous, icon, tint, label=None):
    ui.page_header(f"Étape {label or n} · {titre}", sous, icon=icon, tint=tint)
    ui.stepper(n, tint)


def _vide(texte, cmds):
    ui.empty_state("Pas encore de résultats pour cette étape", texte, cmds)


def _bases_label():
    return [C.DATASETS[k]["label"] for k in C.DEFAULT_DATASETS]


# =========================================================================== ÉTAPE 0
def etape0(ctx):
    _tete(0, "Planification", "Mesure la vitesse de la machine et choisit un plan de calcul qui tient dans le budget de temps. "
          "Aucun poids n'est téléchargé : la vitesse d'un réseau ne dépend pas de ses poids.", "bars", "orange")

    ui.step(1, "Ce que fait le script")
    ui.io_flow(["Bases d'images", "Réseaux de config.py", "Budget en minutes"], "etape0_planification.py",
               ["Plans proposés dans la console", "Commandes à lancer ensuite"], "python etape0_planification.py --budget-min 90", "orange")
    ui.pipe([dict(n=1, title="Vitesse des réseaux", icon="layers", tint="purple", text="Chaque réseau traite un lot d'images aléatoires : millisecondes par image."),
             dict(n=2, title="Vitesse de lecture", icon="image", tint="teal", text="Décodage et prétraitement d'un échantillon de chaque base."),
             dict(n=3, title="Vitesse d'évaluation", icon="curve", tint="blue", text="Évaluation d'une base synthétique pour mesurer le coût des métriques."),
             dict(n=4, title="Plans", icon="target", tint="orange", text="Test rapide, complet, sans VGG 16, modèles légers : plafond d'images par classe adapté.")], cyc=4800)
    commande("python etape0_planification.py                    # budget total de 90 minutes\npython etape0_planification.py --budget-min 120")

    ui.step(2, "Les réseaux du projet", "dimension du vecteur et coût de calcul")
    tab = pd.DataFrame([{"Réseau": C.MODELS[m][0], "Famille": C.MODELS[m][1], "Couche d'extraction": COUCHE[m], "Dimension": DIMS[m],
                         "GFLOPs (environ)": GFLOPS.get(m)} for m in C.MODELS])
    st.dataframe(tab, width="stretch", hide_index=True)

    ui.step(3, "Estimation sur vos mesures", "temps calculé avec les vitesses réellement mesurées par les étapes 2 et 4")
    G = ctx.G
    if G is None or G.empty:
        _vide("Les vitesses mesurées apparaissent après l'extraction et l'évaluation. Le script de planification, lui, s'exécute avant tout calcul.",
              [("Planifier", "python etape0_planification.py"), ("Extraire", "python etape2_extraction.py"), ("Évaluer", "python etape4_evaluation.py")])
    else:
        cos = G[G.metrique == "cosinus"]
        ms_mod = cos.groupby("model").ms_extraction_img.mean()
        ui.hbars([(ctx.nom_m(m), v, f"{v:.1f} ms") for m, v in ms_mod.items()], "orange", "Temps d'extraction moyen par image")
        a, b = st.columns(2)
        mods = a.multiselect("Réseaux", list(ms_mod.index), default=list(ms_mod.index), format_func=ctx.nom_m, key="e0_m")
        bases = b.multiselect("Bases", list(ctx.DSS), default=list(ctx.DSS), format_func=ctx.nom_ds, key="e0_d")
        sub = cos[cos.model.isin(mods) & cos.dataset.isin(bases)]
        t_ext = float((sub.ms_extraction_img * sub.n_images).sum() / 60000)
        ev = G[G.model.isin(mods) & G.dataset.isin(bases)]
        t_ev = float((ev.ms_recherche_req * ev.n_requetes).sum() / 60000)
        ui.stats([dict(label="Images indexées", value=f"{int(sub.n_images.sum()):,}".replace(",", " "), sub="somme sur les couples base et réseau", icon="image", tint="teal"),
                  dict(label="Extraction", value=f"{t_ext:.1f} min", icon="database", tint="purple"),
                  dict(label="Évaluation", value=f"{t_ev:.1f} min", sub="deux mesures de similarité", icon="curve", tint="blue"),
                  dict(label="Total estimé", value=f"{1.03 * (t_ext + t_ev):.1f} min", sub="avec 3 % de marge", icon="target", tint="orange")])
        st.caption("Le décodage des images se fait en parallèle de l'inférence : il est déjà compris dans les temps mesurés.")
    montrer_code("etape0_planification.py", "bench_models", "mesure des ms par image")


# =========================================================================== ÉTAPE 1
def etape1(ctx):
    _tete(1, "Préparation", "Parcourt chaque base, vérifie les images, calcule les statistiques et illustre le prétraitement : "
          "redimensionnement en 224 par 224 et normalisation ImageNet.", "image", "teal")
    ui.step(1, "Ce que fait le script")
    ui.io_flow(_bases_label(), "etape1_preparation.py",
               ["workdir/manifests/<base>.csv", "results/etape1_stats.json", "figures d'échantillon et de prétraitement"],
               "python etape1_preparation.py", "teal")
    ui.chain([("Lecture", "image RGB, décodage JPEG réduit pour aller plus vite"),
              ("Redimensionnement", f"{C.IMG_SIZE} × {C.IMG_SIZE} pixels"),
              ("Mise à l'échelle", "valeurs entre 0 et 1"),
              ("Normalisation", f"moyenne {C.IMAGENET_MEAN}, écart type {C.IMAGENET_STD}")], "teal")
    commande("python etape1_preparation.py                      # toutes les images\npython etape1_preparation.py --preset quick       # test rapide")

    ui.step(2, "Résultats", "statistiques lues dans results/etape1_stats.json")
    ST = ctx.ST
    if not ST:
        _vide("Cette page lit le fichier produit par l'étape 1.", [("Préparer", "python etape1_preparation.py")])
    else:
        ks = [k for k in C.DATASETS if k in ST]
        tot = sum(ST[k]["n_images"] for k in ks)
        ui.stats([dict(label="Bases préparées", value=len(ks), icon="database", tint="blue"),
                  dict(label="Images au total", value=f"{tot:,}".replace(",", " "), icon="image", tint="teal"),
                  dict(label="Couverture minimale", value=f"{min(ST[k].get('couverture_pct', 100) for k in ks):.0f} %", sub="images utilisées sur images trouvées", icon="target", tint="green"),
                  dict(label="Images illisibles", value=sum(ST[k].get("illisibles", 0) for k in ks), icon="grid", tint="orange")])
        ui.hbars([(ST[k]["nom"], ST[k]["n_images"], f"{ST[k]['n_images']:,}".replace(",", " ") + f" · {ST[k]['n_classes']} classes") for k in ks], "teal", "Images par base")
        st.dataframe(pd.DataFrame([{"Base": ST[k]["nom"], "Images": ST[k]["n_images"], "Classes": ST[k]["n_classes"], "Extensions": ", ".join(ST[k]["extensions"]),
                                    "Largeur moy.": ST[k]["largeur_moy"], "Hauteur moy.": ST[k]["hauteur_moy"], "Taille moy. (Ko)": ST[k]["taille_moy_ko"],
                                    "Taille totale (Mo)": ST[k]["taille_totale_mo"], "Couverture %": ST[k].get("couverture_pct", 100)} for k in ks]),
                     width="stretch", hide_index=True)
        ds = st.selectbox("Détail d'une base", ks, format_func=ctx.nom_ds, key="e1_ds")
        s = ST[ds]
        ui.stats([dict(label="Résolution moyenne", value=f"{s['largeur_moy']:.0f} × {s['hauteur_moy']:.0f}", sub=f"largeur de {s['largeur_min']} à {s['largeur_max']} px", icon="image", tint="teal"),
                  dict(label="Images par classe", value=f"{s['images_par_classe_min']} à {s['images_par_classe_max']}", icon="grid", tint="purple"),
                  dict(label="Mode", value=s["preset"], sub=f"{s.get('n_disponibles', s['n_images'])} images sur disque", icon="target", tint="orange")])
        try:
            df = ctx.manifest(ds)
            vc = df.groupby("class_name").size().reset_index(name="Images").rename(columns={"class_name": "Classe"})
            fig = px.bar(vc, x="Classe", y="Images", title=f"Images par classe · {ctx.nom_ds(ds)}")
            ui.show(ui.category_axes(fig, x=vc["Classe"]), height=340)
        except Exception:
            st.info("Le manifeste de cette base est introuvable.")
        c1, c2 = st.columns(2)
        for col, pref, leg in ((c1, "echantillon", "Une image par classe"), (c2, "pretraitement", "Avant et après le prétraitement")):
            f = C.FIGS / f"{pref}_{ds}.png"
            if f.exists():
                col.image(str(f), caption=leg)
    montrer_code("common.py", "get_transform", "transformations appliquées à chaque image")
    montrer_code("common.py", "scan_dataset", "parcours d'une base et lecture de la classe")


# =========================================================================== ÉTAPE 2
def etape2(ctx):
    _tete(2, "Extraction", "Indexation hors ligne : chaque image passe dans chaque réseau préentraîné, et le vecteur obtenu est enregistré. "
          "Le calcul est repris exactement là où il s'est arrêté.", "database", "purple")
    ui.step(1, "Ce que fait le script")
    ui.io_flow(["manifests/<base>.csv", "Poids ImageNet de torchvision", "Liste des réseaux"], "etape2_extraction.py",
               ["workdir/features/<base>__<réseau>.npy", "fichier .json : dimension, temps, paramètres"],
               "python etape2_extraction.py --models resnet50", "purple")
    ui.chain([("Image", "tenseur 3 × 224 × 224"), ("Convolutions", "caractéristiques locales de plus en plus abstraites"),
              ("Pooling global", "une valeur par carte : moyenne spatiale"), ("Vecteur", "d valeurs, 512 à 4096"), ("Index", "matrice N × d")], "purple")
    commande("python etape2_extraction.py\npython etape2_extraction.py --models squeezenet1_1 mobilenet_v3_small resnet18 --time-limit-min 40   # puis relancer la même commande")

    ui.step(2, "Résultats", "un fichier de métadonnées par base et par réseau")
    if not ctx.pairs:
        _vide("Aucun vecteur extrait pour le moment.", [("Préparer", "python etape1_preparation.py"), ("Extraire", "python etape2_extraction.py")])
    else:
        rows = []
        for d, m in ctx.pairs:
            try:
                mt = load_meta(d, m)
            except Exception:
                continue
            rows.append({"Base": ctx.nom_ds(d), "Réseau": ctx.nom_m(m), "model": m, "Images": mt["n_images"], "Dimension": mt["dim"],
                         "Paramètres (M)": mt.get("params_millions"), "Appareil": mt.get("device"), "ms par image": mt.get("ms_par_image"),
                         "Temps total (min)": round(mt.get("temps_inference_s", 0) / 60, 2), "Index (Mo)": round(mt["n_images"] * mt["dim"] * 4 / 1e6, 1)})
        T = pd.DataFrame(rows)
        if T.empty:
            _vide("Les fichiers de métadonnées sont introuvables.", [("Extraire", "python etape2_extraction.py --force")])
        else:
            ui.stats([dict(label="Index construits", value=len(T), sub="couples base et réseau", icon="database", tint="purple"),
                      dict(label="Vecteurs", value=f"{int(T.Images.sum()):,}".replace(",", " "), icon="grid", tint="blue"),
                      dict(label="Temps d'inférence", value=f"{T['Temps total (min)'].sum():.1f} min", icon="bars", tint="orange"),
                      dict(label="Espace disque", value=f"{T['Index (Mo)'].sum():.0f} Mo", icon="layers", tint="teal")])
            par = T.groupby("model").agg(ms=("ms par image", "mean"), dim=("Dimension", "first"), params=("Paramètres (M)", "first")).reset_index()
            c1, c2 = st.columns(2)
            with c1:
                ui.hbars([(ctx.nom_m(r.model), r.ms, f"{r.ms:.1f} ms") for r in par.itertuples()], "purple", "Millisecondes par image")
            with c2:
                ui.hbars([(ctx.nom_m(r.model), r.dim, f"{int(r.dim)} valeurs") for r in par.itertuples()], "blue", "Dimension du vecteur")
            st.dataframe(T.drop(columns="model"), width="stretch", hide_index=True)

            st.subheader("Voir le vecteur d'une image")
            a, b, c = st.columns([1, 1, 2])
            ds = a.selectbox("Base", list(ctx.DSS), format_func=ctx.nom_ds, key="e2_ds")
            ms = [m for d, m in ctx.pairs if d == ds]
            model = b.selectbox("Réseau", ms, format_func=ctx.nom_m, key="e2_m")
            df, F = ctx.manifest(ds), ctx.feats(ds, model)
            i = c.slider("Image numéro", 0, len(df) - 1, min(10, len(df) - 1), key="e2_i")
            v = F[i]
            g1, g2 = st.columns([1, 3])
            g1.image(ui.vignette(df.path[i]), caption=f"classe {df.class_name[i]}", width="stretch")
            n = min(len(v), 128)
            fig = go.Figure(go.Bar(x=list(range(n)), y=[float(x) for x in v[:n]]))
            fig.update_layout(title=f"{n} premières valeurs sur {len(v)}", xaxis_title="Dimension", yaxis_title="Valeur")
            with g2:
                ui.show(fig, height=300)
            ui.stats([dict(label="Dimension", value=len(v), icon="grid", tint="blue"),
                      dict(label="Norme L2", value=f"{float(np.linalg.norm(v)):.2f}", icon="target", tint="purple"),
                      dict(label="Valeurs nulles", value=f"{100 * float((v <= 1e-6).mean()):.0f} %", sub="la fonction ReLU met à zéro les valeurs négatives", icon="bars", tint="orange"),
                      dict(label="Valeur maximale", value=f"{float(v.max()):.2f}", icon="curve", tint="green")])
    montrer_code("common.py", "build_model", "coupe chaque réseau à la couche du descripteur")


# =========================================================================== ÉTAPE 3
def _hasard(n):
    st.session_state["e3_pos"] = random.randint(1, max(n, 1))


def etape3(ctx):
    _tete(3, "Recherche", "Pour une image requête : son vecteur est comparé à tous ceux de la base, avec la distance cosinus puis la distance euclidienne, "
          "et les K images les plus proches sont retournées.", "search", "green")
    ui.step(1, "Ce que fait le script")
    ui.io_flow(["Vecteur de la requête", "Index .npy de la base", "Mesure de similarité", "Nombre K"], "etape3_recherche.py",
               ["Top K : chemins, classes, distances", "figure recherche_<base>_<réseau>.png"],
               "python etape3_recherche.py --dataset corel1k --model resnet50 --k 10", "green")
    a, b = st.columns(2)
    with a:
        ui.note("Cosinus : compare l'angle entre les vecteurs. Une image plus lumineuse ou plus contrastée garde le même voisinage.", "blue")
    with b:
        ui.note("Euclidienne : mesure l'écart entre les extrémités. Elle est sensible à la longueur des vecteurs, donc à l'intensité des activations.", "orange")
    commande("python etape3_recherche.py --dataset corel1k --model resnet50 --query-class buses --k 10\npython etape3_recherche.py --dataset paris6k --model vgg16 --query-idx 123")

    ui.step(2, "Essayer une requête", "les K voisins pour les deux mesures, calculés avec les vecteurs enregistrés")
    if not ctx.pairs:
        _vide("Il faut au moins un index de vecteurs pour lancer une recherche.", [("Préparer", "python etape1_preparation.py"), ("Extraire", "python etape2_extraction.py")])
    else:
        c1, c2, c3 = st.columns(3)
        ds = c1.selectbox("Base", list(ctx.DSS), format_func=ctx.nom_ds, key="e3_ds")
        ms = [m for d, m in ctx.pairs if d == ds]
        model = c2.selectbox("Réseau", ms, format_func=ctx.nom_m, key="e3_m")
        k = c3.slider("Nombre de résultats K", 1, 20, 10, key="e3_k")
        df, F = ctx.manifest(ds), ctx.feats(ds, model)
        cls = st.selectbox("Classe de la requête", list(class_names(df)), key="e3_cls")
        sub = df[df.class_name == cls]
        d1, d2 = st.columns([3, 1])
        st.session_state["e3_pos"] = min(max(int(st.session_state.get("e3_pos", 1)), 1), len(sub))     # reste valide si la classe change
        pos = d1.slider("Image dans la classe", 1, len(sub), key="e3_pos") if len(sub) > 1 else 1
        d2.button("Image au hasard", on_click=_hasard, args=(len(sub),), key="e3_rnd")
        q = int(sub.index[pos - 1])
        h1, h2 = st.columns([1, 5])
        h1.image(ui.vignette(df.path[q]), caption="Requête", width="stretch")
        h2.markdown(f"**Classe de la requête :** {df.class_name[q]}  \n{SIM} : même classe, {DIF} : autre classe. "
                    "La requête s'exclut elle même de ses résultats.")
        for metric in C.METRICS:
            idx, dist = top_k(F, F[q], metric, k, exclude=q)
            ok = df.label.values[idx] == df.label.values[q]
            st.subheader(f"{metric.capitalize()} : {int(ok.sum())} similaires sur {k}")
            cols = st.columns(min(k, 5))
            for j, (i, d) in enumerate(zip(idx, dist)):
                col = cols[j % 5]
                col.image(ui.vignette(df.path[i]), width="stretch")
                col.caption(f"{SIM if ok[j] else DIF} · {df.class_name[i]} · d={d:.3f}")
        figs = sorted(C.FIGS.glob("recherche_*.png"))[:2]
        if figs:
            st.subheader("Figures enregistrées par le script")
            for f in figs:
                st.image(str(f), caption=f.name)
    montrer_code("common.py", "pairwise", "matrice des distances, cosinus ou euclidienne")
    montrer_code("common.py", "top_k", "les K plus proches sans trier toute la base")


# =========================================================================== ÉTAPE 4
def etape4(ctx):
    _tete(4, "Évaluation", "Chaque image de la base devient une requête. Le classement de toute la base est comparé aux classes réelles pour calculer "
          "précision, rappel, AP, mAP et courbes précision et rappel.", "curve", "blue")
    ui.step(1, "Ce que fait le script")
    ui.io_flow(["Index .npy de chaque base", "manifests/<base>.csv", "Métadonnées .json"], "etape4_evaluation.py",
               ["results/metrics_global.csv", "metrics_par_classe.csv", "pr_curves.json", "results/confusion/*.npy"],
               "python etape4_evaluation.py", "blue")
    ui.chain([("Requête", "chaque image de la base, sauf les distracteurs de Paris6k"),
              ("Classement", "toutes les autres images, de la plus proche à la plus éloignée"),
              ("Pertinence", "1 si même classe que la requête, sinon 0"),
              ("Mesures", "P@K, R@K, AP et précision interpolée"),
              ("Moyennes", "sur toutes les requêtes : mAP et courbes")], "blue")
    commande("python etape4_evaluation.py\npython etape4_evaluation.py --datasets corel1k --models resnet50 --force")
    ui.note("Les formules détaillées, avec un exemple interactif, sont dans la page Formules.", "blue")

    ui.step(2, "Résultats", "lus dans results/metrics_global.csv et results/pr_curves.json")
    G, PR = ctx.G, ctx.PR
    if G is None or G.empty:
        _vide("Cette page lit les fichiers produits par l'étape 4.", [("Extraire", "python etape2_extraction.py"), ("Évaluer", "python etape4_evaluation.py")])
    else:
        met = st.radio("Mesure de similarité", C.METRICS, horizontal=True, key="e4_met")
        g = G[G.metrique == met]
        best = g.sort_values("mAP", ascending=False).iloc[0]
        ui.stats([dict(label="Configurations évaluées", value=len(G), sub="base, réseau et mesure", icon="grid", tint="blue"),
                  dict(label="Requêtes évaluées", value=f"{int(G.drop_duplicates(['dataset', 'model']).n_requetes.sum()):,}".replace(",", " "), icon="search", tint="teal"),
                  dict(label="Meilleur mAP", value=f"{best.mAP:.3f}", sub=f"{best.modele} · {best.dataset_nom}", icon="target", tint="green"),
                  dict(label="Mesure comparée", value=met.capitalize(), icon="compare", tint="orange")])
        piv = g.pivot_table(index="modele", columns="dataset_nom", values="mAP")
        fig = px.imshow(piv, text_auto=".3f", aspect="auto", color_continuous_scale="Blues", zmin=0, zmax=1,
                        labels=dict(color="mAP", x="Base", y="Réseau"), title=f"mAP par réseau et par base · {met}")
        ui.category_axes(fig, x=piv.columns, y=piv.index)
        ui.show(fig, height=max(320, 60 * len(piv) + 140))

        ds = st.selectbox("Base", list(ctx.DSS), format_func=ctx.nom_ds, key="e4_ds")
        l, r = st.columns(2)
        f1 = go.Figure()
        if PR:
            for m in ctx.MODS:
                key = f"{ds}|{m}|{met}"
                if key in PR["courbes"]:
                    f1.add_scatter(x=PR["niveaux_rappel"], y=PR["courbes"][key], mode="lines+markers", name=ctx.nom_m(m))
        f1.update_layout(title=f"Courbes précision et rappel · {ctx.nom_ds(ds)}", xaxis_title="Rappel", yaxis_title="Précision")
        with l:
            ui.show(f1, height=400)
        pk = g[g.dataset == ds].melt(id_vars=["modele"], value_vars=[f"P@{k}" for k in C.K_LIST], var_name="K", value_name="Précision")
        f2 = px.line(pk, x="K", y="Précision", color="modele", markers=True, title=f"Précision@K · {ctx.nom_ds(ds)}")
        with r:
            ui.show(f2, height=400)
        cols = ["modele", "mAP"] + [f"P@{k}" for k in C.K_LIST] + [f"R@{k}" for k in C.K_LIST]
        st.dataframe(g[g.dataset == ds][cols].rename(columns={"modele": "Réseau"}), width="stretch", hide_index=True)
    montrer_code("etape4_evaluation.py", "evaluate", "classement, P@K, R@K, AP et précision interpolée par blocs de requêtes")


# =========================================================================== ÉTAPES 5 À 7
def _fichier(chemin):
    if chemin.exists():
        s = chemin.stat()
        return "oui", f"{s.st_size / 1e6:.2f} Mo", datetime.fromtimestamp(s.st_mtime).strftime("%d/%m/%Y %H:%M")
    return "non", "", ""


def livrables(ctx):
    _tete([5, 6, 7], "Livrables", "Les trois dernières étapes assemblent les résultats : rapport rédigé, site HTML hors ligne et rapports PDF.", "compare", "pink", label="5 à 7")
    ui.step(1, "Trois scripts, trois formats")
    ui.pipe([dict(n=5, title="Rapport", file="etape5_rapport.py", icon="compare", tint="pink", text="Rapport Markdown et page HTML autonome avec images intégrées."),
             dict(n=6, title="Site HTML", file="etape6_site_html.py", icon="grid", tint="orange", text="Une page par étape, graphiques interactifs, images et PDF à télécharger. Lance aussi l'étape 7."),
             dict(n=7, title="PDF", file="etape7_pdf.py", icon="matrix", tint="teal", text="Figures de chaque étape, rapport complet avec sommaire cliquable et un PDF par étape.")], cyc=4800)
    commande("python etape5_rapport.py\npython etape6_site_html.py      # produit aussi les PDF\npython etape7_pdf.py")
    ui.step(2, "Fichiers produits", "présence, taille et date sur cette machine")
    liste = [("Rapport Markdown", C.RESULTS / "rapport.md"), ("Rapport HTML", C.RESULTS / "rapport.html"), ("Site, page d'accueil", C.RESULTS / "site" / "index.html"),
             ("PDF complet", C.RESULTS / "pdf" / "rapport_complet.pdf")] + [(f"PDF de l'étape {n}", C.RESULTS / "pdf" / f"etape{n}.pdf") for n in (1, 2, 3, 4)]
    rows = []
    for nom, p in liste:
        ok, taille, date = _fichier(p)
        rows.append({"Livrable": nom, "Chemin": str(p.relative_to(C.BASE)), "Présent": ok, "Taille": taille, "Modifié le": date})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    figs = sorted(C.FIGS.glob("*.png")) if C.FIGS.exists() else []
    ui.stats([dict(label="Figures PNG", value=len(figs), sub="results/figures", icon="image", tint="orange"),
              dict(label="Livrables présents", value=f"{sum(r['Présent'] == 'oui' for r in rows)} sur {len(rows)}", icon="target", tint="pink")])
    c1, c2 = st.columns(2)
    for col, (nom, p, mime) in ((c1, ("Télécharger le rapport HTML", C.RESULTS / "rapport.html", "text/html")),
                                (c2, ("Télécharger le PDF complet", C.RESULTS / "pdf" / "rapport_complet.pdf", "application/pdf"))):
        if p.exists():
            col.download_button(nom, p.read_bytes(), p.name, mime)
    if not any(r["Présent"] == "oui" for r in rows):
        _vide("Les livrables sont générés une fois les étapes 1, 2 et 4 terminées.", [("Rapport", "python etape5_rapport.py"), ("Site et PDF", "python etape6_site_html.py")])
