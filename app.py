"""Application web (Streamlit) – CBIR par réseaux profonds. Lancer : python -m streamlit run app.py"""
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import config as C
import ui
from common import (load_manifest, load_features, class_names, top_k, load_rgb,
                    available_pairs)

st.set_page_config(page_title="DeepCBIR · IRDM", page_icon=":material/image_search:", layout="wide")
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


@st.cache_resource
def modele(m):
    from common import build_model
    return build_model(m)


nom_ds = lambda d: C.DATASETS[d]["label"]
nom_m = lambda m: C.MODELS[m][0]
G, PC, PR, ST = (lire_csv("metrics_global.csv"), lire_csv("metrics_par_classe.csv"),
                 lire_json("pr_curves.json"), lire_json("etape1_stats.json"))
pairs = available_pairs()
DSS = sorted({d for d, _ in pairs}, key=list(C.DATASETS).index)
MODS = [m for m in C.MODELS if m in {m for _, m in pairs}]


def besoin_resultats():
    """Les pages de résultats exigent les étapes 1 à 4 ; sinon message et arrêt de la page."""
    if not pairs or G is None:
        ui.page_header("Résultats indisponibles")
        st.warning("Aucune caractéristique extraite ou aucun résultat évalué. Lancer les étapes 1 à 4 (voir README).",
                   icon=":material/warning:")
        st.stop()


def choisir_ds(key="ds"):
    return st.sidebar.selectbox("Base de données", DSS, format_func=nom_ds, key=key)


# ------------------------------------------------------------------ Accueil
def page_accueil():
    ui.hero("Recherche d'images par le contenu",
            "Descripteurs extraits de réseaux convolutifs pré-entraînés, comparaison cosinus / euclidienne, "
            "évaluation par courbes Précision-Rappel et mAP.",
            pills=[nom_m(m) for m in MODS] or ["SqueezeNet", "ResNet", "VGG"])
    if not pairs:
        st.warning("Aucune caractéristique extraite. Lancer les étapes 1 à 4 (voir README).", icon=":material/warning:")
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
def page_recherche():
    ui.page_header("Recherche d'images similaires", icon="search", tint="blue", subtitle="Choisissez une image requête (dans la base ou téléversée) et comparez "
                   "les K images les plus proches pour chaque mesure de similarité.")
    if not pairs:
        besoin_resultats()
    ds = choisir_ds(); df = manifest(ds)
    ms = [m for d, m in pairs if d == ds]
    model = st.sidebar.selectbox("Modèle", ms, format_func=nom_m)
    k = st.sidebar.slider("Nombre de résultats (K)", 1, 20, 10)
    mode = st.radio("Image requête", ["Choisir dans la base", "Téléverser une image"], horizontal=True)
    F = feats(ds, model); q_idx, qvec, qimg = None, None, None
    if mode == "Choisir dans la base":
        cls = st.selectbox("Classe", class_names(df))
        sub = df[df.class_name == cls]
        pos = st.slider("Image n°", 0, len(sub) - 1, 0)
        q_idx = int(sub.index[pos]); qvec = F[q_idx]; qimg = load_rgb(df.path[q_idx])
    else:
        up = st.file_uploader("Image (jpg/png)", type=["jpg", "jpeg", "png"])
        if up:
            try:
                import torch
                from common import get_transform
                qimg = Image.open(up).convert("RGB")
                with torch.inference_mode():
                    qvec = modele(model)(get_transform()(qimg)[None]).numpy()[0]
            except Exception as e:
                st.error(f"Extraction impossible (PyTorch installé ?) : {e}", icon=":material/error:")
    if qvec is not None:
        c0, c1 = st.columns([1, 5]); c0.image(qimg, caption="Requête", width=160)
        if q_idx is not None:
            c1.markdown(f"**Classe réelle :** {df.class_name[q_idx]}")
        for metric in C.METRICS:
            idx, dist = top_k(F, qvec, metric, k, exclude=q_idx)
            ok = (df.label.values[idx] == df.label[q_idx]) if q_idx is not None else None
            st.subheader(metric.capitalize() + (f" · {int(ok.sum())}/{k} pertinentes" if ok is not None else ""))
            cols = st.columns(min(k, 5))
            for j, (i, d) in enumerate(zip(idx, dist)):
                col = cols[j % 5]
                col.image(load_rgb(df.path[i]), width="stretch")
                tag = "" if ok is None else (":green[Pertinente] · " if ok[j] else ":red[Non pertinente] · ")
                col.caption(f"{tag}{df.class_name[i]} · d={d:.3f}")


# ---------------------------------------------------------------- Résultats globaux
def page_resultats():
    besoin_resultats()
    ui.page_header("Résultats globaux", "mAP et précision@K de chaque modèle, pour chaque base et chaque mesure de similarité.",
                   icon="bars", tint="blue")
    met = st.sidebar.multiselect("Métrique", C.METRICS, default=C.METRICS)
    bases = st.sidebar.multiselect("Bases", DSS, default=DSS, format_func=nom_ds)
    if not met or not bases:
        st.info("Sélectionnez au moins une métrique et une base dans la barre latérale.", icon=":material/info:"); return
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
    st.download_button("Télécharger (CSV)", G.to_csv(index=False).encode("utf-8"), "metrics_global.csv",
                       icon=":material/download:")


# ---------------------------------------------------------------- Courbes PR
def page_courbes():
    besoin_resultats()
    ui.page_header("Courbes Précision-Rappel", "Moyenne sur toutes les requêtes de la base.", icon="curve", tint="teal")
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
        st.info("Aucun résultat par classe pour cette base et cette métrique.", icon=":material/info:"); return
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
        st.info("Matrice non calculée pour cette combinaison (relancer l'étape 4).", icon=":material/info:"); return
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


# ------------------------------------------------------------------ Navigation
ui.sidebar_brand()
nav = st.navigation({
    "Vue d'ensemble": [st.Page(page_accueil, title="Accueil", icon=":material/home:", url_path="accueil", default=True)],
    "Expérimentation": [
        st.Page(page_experience, title="Expérience sur échantillon", icon=":material/science:", url_path="experience"),
        st.Page(page_recherche, title="Recherche d'images", icon=":material/image_search:", url_path="recherche"),
    ],
    "Résultats": [
        st.Page(page_resultats, title="Résultats globaux", icon=":material/bar_chart:", url_path="resultats"),
        st.Page(page_courbes, title="Courbes Précision-Rappel", icon=":material/show_chart:", url_path="courbes"),
        st.Page(page_classes, title="Comparaison par classe", icon=":material/grid_view:", url_path="classes"),
        st.Page(page_confusion, title="Matrices de confusion", icon=":material/table_chart:", url_path="confusion"),
        st.Page(page_croisees, title="Comparaisons croisées", icon=":material/compare_arrows:", url_path="croisees"),
    ],
})
nav.run()
