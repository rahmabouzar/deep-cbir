"""Application web (Streamlit) – CBIR par réseaux profonds. Lancer : python -m streamlit run app.py"""
import json
import os
import types
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import config as C
import ui
from common import (load_manifest, load_features, class_names, top_k, load_rgb,
                    available_pairs, pairwise, vecteur_image, relabel_G, relabel_ST)
import recherche_eval as E
import pages_etapes
import pages_projet
from dialogue import choisir_image

st.set_page_config(page_title="DeepCBIR, IRDM", page_icon=Image.open(ui.ROOT / "assets" / "favicon.png"), layout="wide")
ui.inject_css()


@st.cache_data
def lire_csv(name):
    p = C.RESULTS / name
    return pd.read_csv(p, dtype={"classe": str}) if p.exists() else None


@st.cache_data
def lire_json(name):
    p = C.RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def manifest(ds): return load_manifest(ds)


@st.cache_data
def feats(ds, m): return load_features(ds, m)


nom_ds = lambda d: C.DATASETS[d]["label"]
nom_m = lambda m: C.MODELS[m][0]
G, PC, PR, ST = (relabel_G(lire_csv("metrics_global.csv")), lire_csv("metrics_par_classe.csv"),
                 lire_json("pr_curves.json"), relabel_ST(lire_json("etape1_stats.json")))
pairs = available_pairs()
DSS = sorted({d for d, _ in pairs}, key=list(C.DATASETS).index)
MODS = [m for m in C.MODELS if m in {m for _, m in pairs}]
CTX = types.SimpleNamespace(G=G, PC=PC, PR=PR, ST=ST, pairs=pairs, DSS=DSS, MODS=MODS,
                            manifest=manifest, feats=feats, nom_ds=nom_ds, nom_m=nom_m)   # lu par pages_projet.py et pages_etapes.py


def etat_vide():
    ui.empty_state("Aucun résultat à afficher",
                   "Cette page lit les fichiers produits par les étapes 1 à 4 du projet. Exécutez ces étapes une fois, puis rechargez la page.",
                   [("Préparer", "python etape1_preparation.py"), ("Extraire", "python etape2_extraction.py"),
                    ("Rechercher", "python etape3_recherche.py"), ("Évaluer", "python etape4_evaluation.py")])


def besoin_resultats():
    """Les pages de résultats exigent les étapes 1 à 4 ; sinon vue de démarrage et arrêt de la page."""
    if not pairs or G is None:
        ui.page_header("Résultats indisponibles", "Les caractéristiques extraites ou les métriques évaluées sont introuvables.", icon="bars", tint="orange")
        etat_vide()
        st.stop()


def pairwise_row(F, qvec, metric, idx):
    """Distances de la requête aux images d'indices idx."""
    return pairwise(qvec[None, :], F[idx], metric)[0]


def choisir_ds(key="ds"):
    return st.sidebar.selectbox("Base de données", DSS, format_func=nom_ds, key=key)


# ------------------------------------------------------------------ Accueil
def page_accueil():
    ui.hero("Recherche d'images<br>par le contenu",
            "Descripteurs extraits de réseaux convolutifs préentraînés, comparaison cosinus et euclidienne, "
            "évaluation par courbes de précision et de rappel et mAP.",
            pills=[nom_m(m) for m in MODS] or ["SqueezeNet", "ResNet", "VGG"])
    ui.pipe(pages_projet.ETAPES_CARTES[:5], cyc=6000)
    if not pairs:
        etat_vide()
        return
    cards = [dict(label="Bases", value=len(DSS), icon="database", tint="blue"),
             dict(label="Modèles", value=len(MODS), icon="layers", tint="purple"),
             dict(label="Mesures de similarité", value=len(C.METRICS), icon="compare", tint="orange")]
    if G is not None:
        b = G.sort_values("mAP", ascending=False).iloc[0]
        cards.append(dict(label="Meilleur mAP", value=f"{b.mAP:.3f}", sub=f"{b.modele} · {b.dataset_nom}", icon="target", tint="green"))
    ui.stats(cards)
    if ST:
        df = pd.DataFrame([{"Base": s["nom"], "Images utilisées": s["n_images"],
                            "Sur disque": s.get("n_disponibles", s["n_images"]), "Couverture %": s.get("couverture_pct", 100),
                            "Classes": s["n_classes"], "Extensions": ", ".join(s["extensions"]),
                            "Résolution moy.": f"{s['largeur_moy']:.0f}×{s['hauteur_moy']:.0f}",
                            "Taille moy. (Ko)": s["taille_moy_ko"], "Taille totale (Mo)": s["taille_totale_mo"],
                            "Mode": s["preset"]} for k, s in ST.items() if k in DSS])
        st.subheader("Bases de données et types de fichiers")
        st.dataframe(df, width="stretch", hide_index=True)
    if G is not None:
        best = G.sort_values("mAP", ascending=False).groupby("dataset").head(1)
        st.subheader("Meilleure configuration par base")
        st.dataframe(best[["dataset_nom", "modele", "metrique", "mAP", "P@10"]].rename(
            columns={"dataset_nom": "Base", "modele": "Modèle", "metrique": "Métrique"}), width="stretch", hide_index=True)
    figs = [(ds, C.FIGS / f"echantillon_{ds}.png") for ds in DSS if (C.FIGS / f"echantillon_{ds}.png").exists()]
    if figs:
        st.subheader("Échantillons des bases")
        for ds, f in figs:
            st.image(str(f), caption=f"Échantillon – {nom_ds(ds)}")


# ------------------------------------------------------------------ Expérience aléatoire / manuelle
def page_experience():
    from page_experience import afficher
    afficher()


# ------------------------------------------------------------------ Recherche
def _ouvrir_dialogue(dossier):
    """Ouvre la boîte de dialogue native, mémorise le fichier et sélectionne la base qui le contient."""
    chemin = choisir_image(dossier)
    if chemin is None:
        st.session_state["dlg_indispo"] = True
    elif chemin:
        st.session_state["req_path"] = chemin
        st.session_state["dlg_indispo"] = False
        st.session_state.pop("ds_auto", None)
        for d in DSS:                                              # l'image vient peut être d'une autre base que celle affichée
            if _position_dans_base(manifest(d), chemin) is not None:
                if st.session_state.get("ds") != d:
                    st.session_state["ds"] = d
                    st.session_state["ds_auto"] = d
                break


def _choisir_idx(i):
    st.session_state["req_idx"] = i


def _position_dans_base(df, chemin):
    """Indice de l'image dans la base si ce fichier en fait partie (comparaison des chemins), sinon None."""
    cible = os.path.normcase(os.path.abspath(chemin))
    chemins = df.path.map(lambda q: os.path.normcase(os.path.abspath(q))).values
    hit = np.flatnonzero(chemins == cible)
    return int(hit[0]) if len(hit) else None


def page_recherche():
    ui.page_header("Recherche d'images similaires", icon="search", tint="teal",
                   subtitle="Choisissez l'image requête avec une boîte de dialogue ou parmi les images de la base, "
                            "comparez les K images les plus proches et mesurez la précision de la recherche.")
    if not pairs:
        besoin_resultats()
    ds = choisir_ds(); df = manifest(ds)
    ms = [m for d, m in pairs if d == ds]
    model = st.sidebar.selectbox("Modèle", ms, format_func=nom_m)
    k = st.sidebar.slider("Nombre de résultats (K)", 1, 20, 10)
    noms = dict(zip(df.label.astype(int), df.class_name))
    inverse = {v: k_ for k_, v in noms.items()}
    F = feats(ds, model)
    classes = list(class_names(df))

    # ---------------------------------------------------- choix de l'image requête
    mode = st.radio("Image requête", ["dialogue", "base", "upload"], horizontal=True, key="req_mode",
                    format_func=lambda m: {"dialogue": "Boîte de dialogue", "base": "Parcourir la base",
                                           "upload": "Téléverser dans le navigateur"}[m])
    q_idx, qvec, qimg, qlab, hors_base = None, None, None, None, False

    if mode == "dialogue":
        b, t = st.columns([1, 3])
        b.button("Parcourir les fichiers", type="primary", on_click=_ouvrir_dialogue, args=(str(C.DATA_DIR),))
        chemin = st.session_state.get("req_path")
        if st.session_state.get("dlg_indispo"):
            t.warning("La boîte de dialogue ne peut pas s'ouvrir sur ce serveur (pas d'écran, ou tkinter absent). "
                      "Utilisez « Parcourir la base » ou « Téléverser dans le navigateur ».")
        elif chemin:
            t.markdown(f"Fichier choisi : **{Path(chemin).name}**  \n{chemin}")
            if st.session_state.get("ds_auto") == ds:
                t.info(f"Cette image appartient à la base {nom_ds(ds)} : elle a été sélectionnée automatiquement.")
        else:
            t.info("Cliquez sur « Parcourir les fichiers » : une fenêtre s'ouvre pour choisir l'image requête.")
        if not chemin or not Path(chemin).exists():
            if chemin:
                st.error("Ce fichier est introuvable. Choisissez un autre fichier.")
            return
        try:
            qimg = Image.open(chemin).convert("RGB")
        except Exception as e:
            st.error(f"Image illisible : {e}")
            return
        pos = _position_dans_base(df, chemin)
        if pos is not None:
            q_idx = pos; qvec = F[pos]; qlab = int(df.label.values[pos])

    elif mode == "base":
        cls = st.selectbox("Classe", classes)
        sub = df[df.class_name == cls]
        if st.session_state.get("req_idx") not in set(df.index):
            st.session_state["req_idx"] = int(sub.index[0])
        npages = max(1, -(-len(sub) // 12))
        pg = st.number_input("Page", 1, npages, 1) if npages > 1 else 1
        cols = st.columns(6)
        for j, i in enumerate(sub.index[(pg - 1) * 12: pg * 12]):
            with cols[j % 6]:
                st.image(ui.vignette(df.path[i]), width="stretch")
                if i == st.session_state["req_idx"]:
                    st.caption("**Requête choisie**")
                else:
                    st.button("Choisir", key=f"pick_{i}", on_click=_choisir_idx, args=(int(i),))
        q_idx = int(st.session_state["req_idx"]); qvec = F[q_idx]; qimg = load_rgb(df.path[q_idx]); qlab = int(df.label[q_idx])

    else:
        up = st.file_uploader("Image (jpg ou png)", type=["jpg", "jpeg", "png"])
        if not up:
            return
        qimg = Image.open(up).convert("RGB")

    if qvec is None:                       # image hors de la base affichée : extraction du descripteur
        try:
            qvec = vecteur_image(model, qimg)
        except Exception as e:
            st.error(f"Extraction impossible, PyTorch doit être installé : {e}")
            return
        d0 = pairwise(qvec[None, :], F, "cosinus")[0]; i0 = int(np.argmin(d0))     # même image déjà présente dans la base ?
        if np.linalg.norm(F[i0] - qvec) <= 1e-3 * max(float(np.linalg.norm(qvec)), 1e-12):
            q_idx = i0; qlab = int(df.label.values[i0])
            st.info("Cette image fait partie de la base : elle est exclue des résultats et sa classe est connue.")
        else:
            hors_base = True
            att = st.selectbox("Classe attendue (facultatif, pour calculer la vraie précision)", ["Aucune"] + classes)
            if att != "Aucune":
                qlab = int(df.label[df.class_name == att].iloc[0])

    # ---------------------------------------------------- classement et mesures
    rangs = {m: E.classement(F, qvec, m, q_idx) for m in C.METRICS}               # classement complet de la base
    pred, conf = E.vote(df.label.values[rangs[C.METRICS[0]][:k]], noms)
    estime = qlab is None                                                          # pas de classe connue : référence = classe prédite
    ref = qlab if qlab is not None else inverse[pred]
    rel = {m: df.label.values[rangs[m]] == ref for m in C.METRICS}
    mes = {m: E.metriques(rel[m], k) for m in C.METRICS}
    SIM, DIF = ":green[**Similaire**]", ":red[**Différente**]"

    c0, c1 = st.columns([1, 5]); c0.image(qimg, caption="Requête", width=160)
    if qlab is not None:
        c1.markdown(f"**Classe {'attendue' if hors_base else 'réelle'} :** {noms[qlab]}")
    c1.markdown(f"**Classe prédite par vote des {k} voisins :** {pred}, confiance {conf:.0%}"
                + ("" if qlab is None else (" : :green[**correct**]" if pred == noms[qlab] else " : :red[**incorrect**]")))
    c1.markdown(f"{SIM} : classe {noms[ref]}, {DIF} : autre classe.")
    m0 = C.METRICS[0]
    nom_p = f"Cohérence@{k}" if estime else f"Précision@{k}"
    ui.stats([dict(label=f"{nom_p} ({m0})", value=f"{mes[m0]['P']:.2f}", sub=f"{mes[m0]['hits']} sur {k} similaires", icon="target", tint="green"),
              dict(label=f"Rappel@{k}", value=f"{mes[m0]['R']:.2f}", sub=f"sur {mes[m0]['nrel']} images de la classe", icon="grid", tint="teal"),
              dict(label="AP sur le classement complet", value=f"{mes[m0]['AP']:.3f}", icon="bars", tint="purple"),
              dict(label="Premier résultat similaire", value=f"rang {mes[m0]['premier']}", icon="search", tint="orange")])
    if estime:
        st.caption("Aucune classe attendue : la référence est la classe prédite par vote. Ces valeurs mesurent la cohérence des résultats, "
                   "pas une vraie précision. Choisissez la classe attendue pour obtenir la précision réelle.")

    t1, t2, t3, t4 = st.tabs(["Résultats", "Précision et rappel", "Consensus des modèles", "Robustesse"])

    with t1:
        for metric in C.METRICS:
            idx = rangs[metric][:k]
            dist = pairwise_row(F, qvec, metric, idx)
            ok = rel[metric][:k]
            st.subheader(f"{metric.capitalize()} : {int(ok.sum())} similaires sur {k}")
            cols = st.columns(min(k, 5))
            for j, (i, d) in enumerate(zip(idx, dist)):
                col = cols[j % 5]
                col.image(ui.vignette(df.path[i]), width="stretch")
                col.caption(f"{SIM if ok[j] else DIF} · {df.class_name[i]} · d={d:.3f}")

    with t2:
        tab = pd.DataFrame({m.capitalize(): {f"{nom_p}": mes[m]["P"], f"Rappel@{k}": mes[m]["R"], "AP": mes[m]["AP"],
                                             "Précision à R": mes[m]["RPrec"], "Rang du premier similaire": mes[m]["premier"],
                                             f"Similaires dans les {k} premiers": mes[m]["hits"]} for m in C.METRICS}).T
        st.dataframe(tab.style.format({c: "{:.3f}" for c in tab.columns[:4]} | {tab.columns[4]: "{:.0f}", tab.columns[5]: "{:.0f}"}),
                     width="stretch")
        st.caption("L'AP et la précision à R sont calculées sur le classement complet de la base, comme le mAP des pages de résultats. "
                   "La précision à R est la précision au rang égal au nombre d'images de la classe.")
        g1, g2 = st.columns(2)
        f1, f2 = go.Figure(), go.Figure()
        for m in C.METRICS:
            p, r, pp = E.courbes(rel[m])
            f1.add_scatter(x=list(range(1, len(p) + 1)), y=p, mode="lines+markers", name=m.capitalize())
            f2.add_scatter(x=r, y=pp, mode="lines+markers", name=m.capitalize())
        f1.update_layout(title="Précision en fonction du rang", xaxis_title="Rang", yaxis_title="Précision au rang", yaxis_range=[0, 1.02])
        f2.update_layout(title="Courbe précision et rappel de cette requête", xaxis_title="Rappel", yaxis_title="Précision",
                         xaxis_range=[0, 1.02], yaxis_range=[0, 1.02])
        with g1: ui.show(f1, height=420)
        with g2: ui.show(f2, height=420)

    with t3:
        if len(ms) < 2:
            st.info("Il faut au moins deux modèles extraits pour cette base.")
        else:
            st.markdown("Les différents modèles retrouvent ils les mêmes images ? Une **fusion RRF** (Reciprocal Rank Fusion) combine leurs classements "
                        "et montre si le consensus fait mieux que chaque modèle seul.")
            met_c = st.radio("Mesure de similarité", C.METRICS, horizontal=True, key="met_c")
            try:
                res = {}
                for m in ms:
                    Fm = feats(ds, m)
                    res[m] = E.classement(Fm, Fm[q_idx] if q_idx is not None else vecteur_image(m, qimg), met_c, q_idx)
            except Exception as e:
                st.error(f"Extraction impossible, PyTorch doit être installé : {e}")
                res = None
            if res:
                res_f = dict(res); res_f["Fusion RRF"] = E.fusion_rrf([res[m] for m in ms])
                rows = []
                for m, r in res_f.items():
                    x = E.metriques(df.label.values[r] == ref, k)
                    rows.append({"Modèle": nom_m(m) if m in C.MODELS else m, nom_p: x["P"], "AP": x["AP"], "Précision à R": x["RPrec"]})
                d = pd.DataFrame(rows)
                fig = px.bar(d.melt(id_vars="Modèle", var_name="Mesure", value_name="Valeur"), x="Modèle", y="Valeur", color="Mesure",
                             barmode="group", range_y=[0, 1.05], title="Modèles seuls et fusion RRF")
                ui.show(ui.facets(fig, tickangle=0), height=400)
                top = {m: set(map(int, res[m][:k])) for m in ms}
                J = pd.DataFrame([[E.jaccard(res[a][:k], res[b][:k]) for b in ms] for a in ms], index=[nom_m(m) for m in ms], columns=[nom_m(m) for m in ms])
                hm = px.imshow(J, text_auto=".2f", zmin=0, zmax=1, color_continuous_scale="Blues", title=f"Accord entre modèles (Jaccard des {k} premiers résultats)")
                ui.category_axes(hm, x=J.columns, y=J.index)
                ui.show(hm, height=380)
                commun = set.intersection(*top.values())
                st.caption(f"{len(commun)} image(s) sont retrouvées parmi les {k} premiers résultats par tous les modèles.")
                if commun:
                    cc = st.columns(min(len(commun), 6))
                    for col, i in zip(cc, sorted(commun)[:6]):
                        col.image(ui.vignette(df.path[i]), width="stretch"); col.caption(df.class_name[i])

    with t4:
        st.markdown("L'image requête est dégradée (rotation, flou, bruit, compression) puis cherchée à nouveau : "
                    "un bon descripteur garde une précision stable.")
        met_r = st.radio("Mesure de similarité", C.METRICS, horizontal=True, key="met_r")
        pert = E.perturbations(qimg)
        choix = st.multiselect("Perturbations", list(pert), default=list(pert))
        if st.button("Lancer le test de robustesse", type="primary") and choix:
            try:
                rows = [dict(Perturbation="Image originale", **E.metriques(df.label.values[E.classement(F, qvec, met_r, q_idx)] == ref, k))]
                for nm in choix:
                    v = vecteur_image(model, pert[nm])
                    rows.append(dict(Perturbation=nm, **E.metriques(df.label.values[E.classement(F, v, met_r, q_idx)] == ref, k)))
            except Exception as e:
                st.error(f"Extraction impossible, PyTorch doit être installé : {e}")
            else:
                d = pd.DataFrame(rows)
                d["Perte"] = d.iloc[0]["P"] - d["P"]
                fig = px.bar(d.melt(id_vars="Perturbation", value_vars=["P", "AP"], var_name="Mesure", value_name="Valeur")
                             .replace({"P": nom_p}), x="Perturbation", y="Valeur", color="Mesure", barmode="group",
                             range_y=[0, 1.05], title=f"Robustesse : {nom_m(model)}, {met_r}")
                ui.show(ui.facets(fig, tickangle=-30), height=440)
                st.dataframe(d.rename(columns={"P": nom_p, "premier": "Rang du premier similaire", "Perte": f"Perte de {nom_p.lower()}"})
                             [["Perturbation", nom_p, "AP", f"Perte de {nom_p.lower()}", "Rang du premier similaire"]],
                             width="stretch", hide_index=True)
                cols = st.columns(5)
                for n_, nm in enumerate(choix):
                    cols[n_ % 5].image(pert[nm], caption=nm, width="stretch")


# ---------------------------------------------------------------- Résultats globaux
def page_resultats():
    besoin_resultats()
    ui.page_header("Résultats globaux", "mAP et précision@K de chaque modèle, pour chaque base et chaque mesure de similarité.",
                   icon="bars", tint="blue")
    met = st.sidebar.multiselect("Métrique", C.METRICS, default=C.METRICS)
    bases = st.sidebar.multiselect("Bases", DSS, default=DSS, format_func=nom_ds)
    if not met or not bases:
        st.info("Sélectionnez au moins une métrique et une base dans la barre latérale."); return
    d = G[G.metrique.isin(met) & G.dataset.isin(bases)]
    wrap = 3 if len(bases) > 3 else len(bases)
    nrows = -(-len(bases) // wrap)
    fig = px.bar(d, x="modele", y="mAP", color="metrique", barmode="group", facet_col="dataset_nom", facet_col_wrap=wrap,
                 facet_row_spacing=0.22, labels={"modele": "Modèle", "metrique": "Métrique"}, title="mAP par modèle")
    ui.show(ui.facets(fig), height=330 * nrows + 110)
    pk = d.melt(id_vars=["dataset_nom", "modele", "metrique"], value_vars=[f"P@{k}" for k in C.K_LIST],
                var_name="K", value_name="Précision")
    fig = px.line(pk[pk.metrique == met[0]], x="K", y="Précision", color="modele", markers=True, facet_col="dataset_nom",
                  facet_col_wrap=wrap, facet_row_spacing=0.18, title=f"Précision@K ({met[0]})", labels={"modele": "Modèle"})
    ui.show(ui.facets(fig, tickangle=0), height=300 * nrows + 110)
    st.dataframe(d.drop(columns=["dataset", "model"]), width="stretch", hide_index=True)
    st.download_button("Télécharger (CSV)", G.to_csv(index=False).encode("utf-8"), "metrics_global.csv")


# ---------------------------------------------------------------- Courbes PR
def page_courbes():
    besoin_resultats()
    ui.page_header("Courbes précision et rappel", "Moyenne sur toutes les requêtes de la base.", icon="curve", tint="green")
    ds = choisir_ds(); met = st.sidebar.radio("Métrique", C.METRICS)
    fig = go.Figure()
    for m in MODS:
        key = f"{ds}|{m}|{met}"
        if key in PR["courbes"]:
            fig.add_scatter(x=PR["niveaux_rappel"], y=PR["courbes"][key], mode="lines+markers", name=nom_m(m))
    fig.update_layout(xaxis_title="Rappel", yaxis_title="Précision", title=f"{nom_ds(ds)} · {met}")
    ui.show(fig, height=520)


# ---------------------------------------------------------------- Par classe
def page_classes():
    besoin_resultats()
    ui.page_header("Comparaison entre classes", "Difficulté de chaque classe (AP moyenne) selon le modèle.", icon="grid", tint="purple")
    ds = choisir_ds(); met = st.sidebar.radio("Métrique", C.METRICS)
    sub = PC[(PC.dataset == ds) & (PC.metrique == met)]
    if sub.empty:
        st.info("Aucun résultat par classe pour cette base et cette métrique."); return
    piv = sub.assign(classe=sub.classe.astype(str)).pivot_table(index="classe", columns="model", values="AP", aggfunc="mean")
    piv = piv[[m for m in MODS if m in piv.columns]]
    piv.columns = [nom_m(c) for c in piv.columns]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]        # classes les plus faciles en haut
    moy = piv.mean(axis=1)
    n = len(piv)

    ui.stats([dict(label="Classes", value=n, icon="grid", tint="blue"),
              dict(label="AP moyenne", value=f"{moy.mean():.3f}", icon="bars", tint="purple"),
              dict(label="Plus facile", value=moy.index[0], sub=f"AP {moy.iloc[0]:.3f}", icon="target", tint="green"),
              dict(label="Plus difficile", value=moy.index[-1], sub=f"AP {moy.iloc[-1]:.3f}", icon="target", tint="orange")])

    # Beaucoup de classes (Corel-5K, Corel-10K, CIFAR-100) : on propose une vue lisible
    vue = "Toutes les classes"
    if n > 30:
        vue = st.radio("Classes affichées", ["20 plus faciles", "20 plus difficiles", "Toutes les classes"], horizontal=True)
    vp = piv.head(20) if vue == "20 plus faciles" else piv.tail(20) if vue == "20 plus difficiles" else piv
    fig = px.imshow(vp, aspect="auto", color_continuous_scale="Blues", zmin=0, zmax=1, labels=dict(color="AP", x="Modèle", y="Classe"),
                    title=f"AP par classe et par modèle · {nom_ds(ds)} · {met}")
    ui.category_axes(fig, x=vp.columns, y=vp.index)
    fig.update_xaxes(side="top")
    ui.show(fig, height=max(380, 22 * len(vp) + 140))

    st.subheader("Détail d'une classe")
    cl = st.selectbox("Classe", list(piv.index), label_visibility="collapsed")
    det = piv.loc[cl].rename("AP").reset_index().rename(columns={"index": "Modèle", "model": "Modèle"})
    det.columns = ["Modèle", "AP"]
    fig = px.bar(det, x="Modèle", y="AP", color="Modèle", range_y=[0, 1], text_auto=".3f", title=f"AP de la classe « {cl} »")
    fig.update_layout(showlegend=False)
    ui.show(fig, height=380)

    c1, c2 = st.columns(2)
    c1.markdown("**Classes les plus faciles**"); c1.dataframe(moy.head(10).rename("AP moy.").to_frame(), width="stretch")
    c2.markdown("**Classes les plus difficiles**"); c2.dataframe(moy.tail(10).iloc[::-1].rename("AP moy.").to_frame(), width="stretch")


# ---------------------------------------------------------------- Confusion
def page_confusion():
    besoin_resultats()
    ui.page_header("Matrices de confusion", "Part des 10 premiers résultats appartenant à chaque classe, pour chaque classe requête.",
                   icon="matrix", tint="pink")
    ds = choisir_ds(); model = st.sidebar.selectbox("Modèle", [m for d, m in pairs if d == ds], format_func=nom_m)
    met = st.sidebar.radio("Métrique", C.METRICS)
    f = C.CONF_DIR / f"{ds}__{model}__{met}.npy"
    if not f.exists():
        st.info("Matrice non calculée pour cette combinaison (relancer l'étape 4)."); return
    M = np.load(f); names = [str(x) for x in class_names(manifest(ds))]
    fig = px.imshow(M, x=names, y=names, color_continuous_scale="Purples",
                    labels=dict(x="Classe retrouvée", y="Classe requête", color="Part"),
                    title=f"{nom_ds(ds)} · {nom_m(model)} · {met}")
    ui.category_axes(fig, x=names, y=names)
    ui.show(fig, height=760)
    M2 = M.copy(); np.fill_diagonal(M2, 0)
    top = np.dstack(np.unravel_index(np.argsort(-M2.ravel())[:15], M2.shape))[0]
    st.subheader("Paires les plus confondues")
    st.dataframe(pd.DataFrame([(names[i], names[j], f"{M2[i, j]:.1%}") for i, j in top],
                              columns=["Requête", "Retrouvée", "Part du Top-10"]), width="stretch", hide_index=True)


# --------------------------------------------------------- Comparaisons croisées
def page_croisees():
    besoin_resultats()
    ui.page_header("Comparaisons croisées", "Mesures de similarité, bases, types de fichiers et efficacité.", icon="compare", tint="orange")
    t1, t2, t3, t4 = st.tabs(["Cosinus vs Euclidienne", "Base vs Base", "Type de fichier / résolution", "Efficacité"])
    with t1:
        p = G.pivot_table(index=["dataset_nom", "modele"], columns="metrique", values="mAP").reset_index()
        p["Écart (cos − euc)"] = p["cosinus"] - p["euclidienne"]
        ui.show(px.bar(p, x="modele", y="Écart (cos − euc)", color="dataset_nom", barmode="group",
                       labels={"modele": "Modèle", "dataset_nom": "Base"}))
        st.dataframe(p, width="stretch", hide_index=True)
    with t2:
        met = st.radio("Métrique", C.METRICS, horizontal=True, key="m2")
        ui.show(px.bar(G[G.metrique == met], x="dataset_nom", y="mAP", color="modele", barmode="group",
                       labels={"dataset_nom": "Base", "modele": "Modèle"}))
    with t3:
        rows = [{"Base": s["nom"], "Format": ", ".join(s["extensions"]), "Largeur moy.": s["largeur_moy"],
                 "Taille moy. (Ko)": s["taille_moy_ko"],
                 "mAP moyen (cosinus)": G[(G.dataset == k) & (G.metrique == "cosinus")].mAP.mean()}
                for k, s in ST.items() if k in DSS]
        r = pd.DataFrame(rows)
        st.dataframe(r, width="stretch", hide_index=True)
        ui.show(px.scatter(r, x="Largeur moy.", y="mAP moyen (cosinus)", color="Format", text="Base", size="Taille moy. (Ko)",
                           log_x=True))
    with t4:
        e = G[G.metrique == "cosinus"]
        ui.show(px.scatter(e, x="ms_extraction_img", y="mAP", color="dataset_nom", size="dim", text="modele", log_x=True,
                           labels={"ms_extraction_img": "ms / image", "dataset_nom": "Base"}))


# ------------------------------------------------------------------ Pages du projet et des étapes (pages_projet.py, pages_etapes.py)
def page_architecture(): pages_projet.architecture(CTX)
def page_technologies(): pages_projet.technologies(CTX)
def page_formules(): pages_projet.formules(CTX)
def page_etape0(): pages_etapes.etape0(CTX)
def page_etape1(): pages_etapes.etape1(CTX)
def page_etape2(): pages_etapes.etape2(CTX)
def page_etape3(): pages_etapes.etape3(CTX)
def page_etape4(): pages_etapes.etape4(CTX)
def page_livrables(): pages_etapes.livrables(CTX)


# ------------------------------------------------------------------ Navigation
ui.sidebar_brand()
nav = st.navigation({
    "Vue d'ensemble": [st.Page(page_accueil, title="Accueil", url_path="accueil", default=True)],
    "Le projet": [
        st.Page(page_architecture, title="Architecture", url_path="architecture"),
        st.Page(page_technologies, title="Technologies", url_path="technologies"),
        st.Page(page_formules, title="Formules et calculs", url_path="formules"),
    ],
    "Étapes du pipeline": [
        st.Page(page_etape0, title="Étape 0 · Planification", url_path="etape0"),
        st.Page(page_etape1, title="Étape 1 · Préparation", url_path="etape1"),
        st.Page(page_etape2, title="Étape 2 · Extraction", url_path="etape2"),
        st.Page(page_etape3, title="Étape 3 · Recherche", url_path="etape3"),
        st.Page(page_etape4, title="Étape 4 · Évaluation", url_path="etape4"),
        st.Page(page_livrables, title="Étapes 5 à 7 · Livrables", url_path="livrables"),
    ],
    "Expérimentation": [
        st.Page(page_experience, title="Expérience sur échantillon", url_path="experience"),
        st.Page(page_recherche, title="Recherche d'images", url_path="recherche"),
    ],
    "Résultats": [
        st.Page(page_resultats, title="Résultats globaux", url_path="resultats"),
        st.Page(page_courbes, title="Courbes précision et rappel", url_path="courbes"),
        st.Page(page_classes, title="Comparaison par classe", url_path="classes"),
        st.Page(page_confusion, title="Matrices de confusion", url_path="confusion"),
        st.Page(page_croisees, title="Comparaisons croisées", url_path="croisees"),
    ],
})
nav.run()
