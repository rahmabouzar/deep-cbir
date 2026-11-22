"""Application web (Streamlit) – CBIR par réseaux profonds. Lancer : streamlit run app.py"""
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import config as C
from common import (load_manifest, load_features, class_names, top_k, load_rgb,
                    available_pairs)

st.set_page_config(page_title="CBIR – Réseaux profonds", page_icon="🔎", layout="wide")


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
G, PC, PR, ST = lire_csv("metrics_global.csv"), lire_csv("metrics_par_classe.csv"), lire_json("pr_curves.json"), lire_json("etape1_stats.json")

st.sidebar.title("🔎 CBIR – IRDM")
st.sidebar.caption("Master 2 SID · USTO-MB")
page = st.sidebar.radio("Navigation", ["🏠 Accueil", "🎲 Expérience aléatoire", "🔎 Recherche d'images", "📊 Résultats globaux",
                                       "📈 Courbes Précision-Rappel", "🧩 Comparaison par classe",
                                       "🔀 Matrices de confusion", "⚖️ Comparaisons croisées", "📄 Rapport"])
if page.startswith("🎲"):                      # page autonome : extrait elle-même les descripteurs de l'échantillon
    from page_experience import afficher
    afficher()
    st.stop()

pairs = available_pairs()
if not pairs:
    st.warning("Aucune caractéristique extraite. Lancer les étapes 1 à 4 (voir README).")
    st.stop()
DSS = sorted({d for d, _ in pairs}, key=list(C.DATASETS).index)
MODS = [m for m in C.MODELS if m in {m for _, m in pairs}]


def choisir_ds(key="ds"): return st.sidebar.selectbox("Base de données", DSS, format_func=nom_ds, key=key)


# ------------------------------------------------------------------ Accueil
if page.startswith("🏠"):
    st.title("Recherche d'images par le contenu (CBIR) avec des CNN pré-entraînés")
    st.write("Descripteurs extraits de **SqueezeNet 1.1, MobileNetV3, ResNet-18/50, VGG-16** ; similarité **cosinus** vs **euclidienne** ; "
             "évaluation par **courbes Précision-Rappel** et **mAP**.")
    if ST:
        df = pd.DataFrame([{"Base": s["nom"], "Images utilisées": s["n_images"], "Sur disque": s.get("n_disponibles", s["n_images"]), "Couverture %": s.get("couverture_pct", 100), "Classes": s["n_classes"],
                            "Extensions": ", ".join(s["extensions"]),
                            "Résolution moy.": f"{s['largeur_moy']:.0f}×{s['hauteur_moy']:.0f}",
                            "Taille moy. (Ko)": s["taille_moy_ko"], "Taille totale (Mo)": s["taille_totale_mo"],
                            "Mode": s["preset"]} for k, s in ST.items() if k in DSS])
        st.subheader("Bases de données et types de fichiers")
        st.dataframe(df, width="stretch", hide_index=True)
    for ds in DSS:
        f = C.FIGS / f"echantillon_{ds}.png"
        if f.exists():
            st.image(str(f), caption=f"Échantillon – {nom_ds(ds)}")
    if G is not None:
        best = G.sort_values("mAP", ascending=False).groupby("dataset").head(1)
        st.subheader("Meilleure configuration par base")
        st.dataframe(best[["dataset_nom", "modele", "metrique", "mAP", "P@10"]].rename(
            columns={"dataset_nom": "Base", "modele": "Modèle", "metrique": "Métrique"}), hide_index=True)

# ------------------------------------------------------------------ Recherche
elif page.startswith("🔎"):
    st.title("Recherche d'images similaires")
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
                st.error(f"Extraction impossible (PyTorch installé ?) : {e}")
    if qvec is not None:
        c0, c1 = st.columns([1, 5]); c0.image(qimg, caption="Requête", width=160)
        if q_idx is not None: c1.write(f"**Classe réelle :** {df.class_name[q_idx]}")
        for metric in C.METRICS:
            idx, dist = top_k(F, qvec, metric, k, exclude=q_idx)
            ok = (df.label.values[idx] == df.label[q_idx]) if q_idx is not None else None
            st.subheader(f"{metric.capitalize()}" + (f" – {int(ok.sum())}/{k} pertinentes" if ok is not None else ""))
            cols = st.columns(min(k, 5))
            for j, (i, d) in enumerate(zip(idx, dist)):
                mark = "" if ok is None else ("✅ " if ok[j] else "❌ ")
                cols[j % 5].image(load_rgb(df.path[i]), caption=f"{mark}{df.class_name[i]} · d={d:.3f}", width="stretch")

# ---------------------------------------------------------------- Résultats
elif page.startswith("📊"):
    st.title("Résultats globaux")
    met = st.sidebar.multiselect("Métrique", C.METRICS, default=C.METRICS)
    d = G[G.metrique.isin(met)]
    st.plotly_chart(px.bar(d, x="modele", y="mAP", color="metrique", barmode="group", facet_col="dataset_nom",
                           labels={"modele": "Modèle"}), width="stretch")
    pk = d.melt(id_vars=["dataset_nom", "modele", "metrique"], value_vars=[f"P@{k}" for k in C.K_LIST],
                var_name="K", value_name="Précision")
    st.plotly_chart(px.line(pk[pk.metrique == met[0]], x="K", y="Précision", color="modele", markers=True,
                            facet_col="dataset_nom", title=f"Précision@K ({met[0]})"), width="stretch")
    st.dataframe(d.drop(columns=["dataset", "model"]), width="stretch", hide_index=True)
    st.download_button("⬇ Télécharger (CSV)", G.to_csv(index=False).encode("utf-8"), "metrics_global.csv")

# ---------------------------------------------------------------- Courbes PR
elif page.startswith("📈"):
    st.title("Courbes Précision-Rappel (moyenne sur toutes les requêtes)")
    ds = choisir_ds(); met = st.sidebar.radio("Métrique", C.METRICS)
    fig = go.Figure()
    for m in MODS:
        key = f"{ds}|{m}|{met}"
        if key in PR["courbes"]:
            fig.add_scatter(x=PR["niveaux_rappel"], y=PR["courbes"][key], mode="lines+markers", name=nom_m(m))
    fig.update_layout(xaxis_title="Rappel", yaxis_title="Précision", title=f"{nom_ds(ds)} – {met}")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Par classe
elif page.startswith("🧩"):
    st.title("Comparaison entre classes")
    ds = choisir_ds(); met = st.sidebar.radio("Métrique", C.METRICS)
    sub = PC[(PC.dataset == ds) & (PC.metrique == met)]
    piv = sub.pivot(index="classe", columns="model", values="AP")[[m for m in MODS if m in set(sub.model)]]
    piv.columns = [nom_m(c) for c in piv.columns]; piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    st.plotly_chart(px.imshow(piv, aspect="auto", color_continuous_scale="Viridis", labels=dict(color="AP"),
                              height=max(400, 18 * len(piv))), width="stretch")
    cl = st.selectbox("Détail d'une classe", list(piv.index))
    st.plotly_chart(px.bar(piv.loc[cl].reset_index().rename(columns={"index": "Modèle", cl: "AP"}), x="Modèle", y="AP",
                           title=f"AP de la classe « {cl} »"), width="stretch")
    c1, c2 = st.columns(2)
    c1.write("**Classes les plus faciles**"); c1.dataframe(piv.mean(axis=1).head(10).rename("AP moy."))
    c2.write("**Classes les plus difficiles**"); c2.dataframe(piv.mean(axis=1).tail(10).rename("AP moy."))

# ---------------------------------------------------------------- Confusion
elif page.startswith("🔀"):
    st.title("Matrices de confusion (Top-10)")
    ds = choisir_ds(); model = st.sidebar.selectbox("Modèle", [m for d, m in pairs if d == ds], format_func=nom_m)
    met = st.sidebar.radio("Métrique", C.METRICS)
    M = np.load(C.CONF_DIR / f"{ds}__{model}__{met}.npy"); names = class_names(manifest(ds))
    st.plotly_chart(px.imshow(M, x=names, y=names, color_continuous_scale="Magma", height=700,
                              labels=dict(x="Classe retrouvée", y="Classe requête", color="Part")), width="stretch")
    M2 = M.copy(); np.fill_diagonal(M2, 0)
    top = np.dstack(np.unravel_index(np.argsort(-M2.ravel())[:15], M2.shape))[0]
    st.write("**Paires les plus confondues**")
    st.dataframe(pd.DataFrame([(names[i], names[j], f"{M2[i, j]:.1%}") for i, j in top],
                              columns=["Requête", "Retrouvée", "Part du Top-10"]), hide_index=True)

# --------------------------------------------------------- Comparaisons croisées
elif page.startswith("⚖️"):
    st.title("Comparaisons croisées")
    t1, t2, t3, t4 = st.tabs(["Cosinus vs Euclidienne", "Base vs Base", "Type de fichier / résolution", "Efficacité"])
    with t1:
        p = G.pivot_table(index=["dataset_nom", "modele"], columns="metrique", values="mAP").reset_index()
        p["Écart (cos − euc)"] = p["cosinus"] - p["euclidienne"]
        st.plotly_chart(px.bar(p, x="modele", y="Écart (cos − euc)", color="dataset_nom", barmode="group"), width="stretch")
        st.dataframe(p, hide_index=True)
    with t2:
        met = st.radio("Métrique", C.METRICS, horizontal=True, key="m2")
        st.plotly_chart(px.bar(G[G.metrique == met], x="dataset_nom", y="mAP", color="modele", barmode="group"), width="stretch")
    with t3:
        rows = [{"Base": s["nom"], "Format": ", ".join(s["extensions"]), "Largeur moy.": s["largeur_moy"],
                 "Taille moy. (Ko)": s["taille_moy_ko"],
                 "mAP moyen (cosinus)": G[(G.dataset == k) & (G.metrique == "cosinus")].mAP.mean()}
                for k, s in ST.items() if k in DSS]
        r = pd.DataFrame(rows)
        st.dataframe(r, hide_index=True)
        st.plotly_chart(px.scatter(r, x="Largeur moy.", y="mAP moyen (cosinus)", color="Format", text="Base", size="Taille moy. (Ko)",
                                   log_x=True), width="stretch")
    with t4:
        e = G[G.metrique == "cosinus"]
        st.plotly_chart(px.scatter(e, x="ms_extraction_img", y="mAP", color="dataset_nom", size="dim", text="modele", log_x=True,
                                   labels={"ms_extraction_img": "ms / image"}), width="stretch")

# ------------------------------------------------------------------- Rapport
else:
    st.title("Rapport complet")
    f = C.RESULTS / "rapport.html"
    if f.exists():
        html = f.read_text(encoding="utf-8")
        st.download_button("⬇ Télécharger le rapport (HTML, ouvrable dans Word)", html.encode("utf-8"), "rapport.html", "text/html")
        st.iframe(html, height=900)
    else:
        st.info("Lancer `python etape5_rapport.py` pour générer le rapport.")
