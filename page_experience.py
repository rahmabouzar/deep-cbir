"""Page web « Expérience sur échantillon » (appelée depuis app.py).

Flux : 1) choisir base / nb de classes / images par classe / modèles  ->  2) « Tirer l'échantillon » (aperçu)
       ->  3) « Lancer l'évaluation » (le code Python tourne en arrière-plan, progression affichée)  ->  4) tableau des résultats.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

import config as C
import experience as X
import ui
from common import load_rgb


CONSEILS = """
**Pour que le tableau soit réellement informatif :**

1. **Au moins 5 classes, idéalement 10** (Corel-1K : prenez les 10). Avec 2 classes, le hasard donne déjà 50 % et la tâche est triviale.
2. **Mélangez des classes faciles ET des classes proches.** Les paires très différentes (bus et fleurs) donnent 1.000 pour tous les réseaux
   et ne permettent pas de les comparer. Ajoutez des classes qui se ressemblent visuellement : plage / montagnes / Afrique (paysages),
   éléphants / chevaux / Afrique (animaux dans la nature), monuments / Afrique (bâtiments).
3. **Même nombre d'images par classe** (écart faible) et **au moins 30 à 50 images par classe** : sinon P@10 et mAP dépendent surtout de la taille de la classe.
4. **Ne choisissez pas seulement les « belles » images** : le choix manuel des meilleures images gonfle les scores. Pour un résultat honnête,
   cochez « Tout » ou « n au hasard », et gardez le choix manuel pour illustrer des cas précis (classe difficile, images ambiguës).
5. **Répétez avec 3 à 5 graines différentes** (mode aléatoire) et indiquez la moyenne dans le rapport : un seul tirage peut être chanceux.
6. **Autres bases :** prenez 10 à 20 classes qui mélangent plusieurs « familles » (CIFAR-100 : arbres chêne/érable/saule, véhicules bus/tramway, personnes garçon/fille).
   Sur Paris6k, gardez les distracteurs `general` : ils rendent la recherche plus réaliste.
7. **Signalez dans le rapport** comment l'échantillon a été choisi (graine, classes, images par classe).
"""


@st.cache_data(show_spinner="Lecture de la base…")
def _scan(ds):
    return X.scanner(ds)


@st.cache_data(show_spinner=False, max_entries=4000)
def _thumb(path):
    """Miniature ; une image illisible/corrompue (il y en a dans Paris6k) donne une vignette grise au lieu de planter la page."""
    try:
        im = load_rgb(path); im.thumbnail((200, 200)); return im
    except Exception:
        return Image.new("RGB", (200, 150), (210, 210, 210))


def _sync(p, k):
    """Case cochée/décochée -> met à jour l'ensemble des images choisies."""
    if k not in st.session_state:          # état déjà réinitialisé par un bouton groupé : rien à synchroniser
        return
    S = st.session_state["manual_sel"]
    (S.add if st.session_state[k] else S.discard)(p)


def _maj(paths, S, flag, vk, n=None, seed=0):
    """Boutons groupés : tout cocher / tout décocher / tirer n images au hasard.
    Incrémente la version de la classe -> les cases sont recréées avec la bonne valeur (pas de conflit d'état)."""
    paths = list(paths)
    S.difference_update(paths)
    if flag:
        S.update(paths if n is None else random.Random(seed).sample(paths, min(n, len(paths))))
    st.session_state[vk] = st.session_state.get(vk, 0) + 1


def _cocher_classes(keys, flag):
    for k in list(keys):
        st.session_state[k] = flag


def _manuel(ds, df, excl, npc):
    PAGE = 24
    if st.session_state.get("manual_ds") != ds:                 # changement de base -> sélection vide
        st.session_state["manual_sel"] = set(); st.session_state["manual_ds"] = ds
    S = st.session_state["manual_sel"]
    pool = X._tri_classes([c for c in df.class_name.unique() if c not in excl])
    # Choix des classes par cases à cocher (visibles directement, sans liste déroulante)
    st.markdown("**Classes à utiliser (au moins 2)**")
    ck = {c: f"cls_{ds}_{c}" for c in pool}
    t = st.columns(4)
    t[0].button("Toutes les classes", icon=":material/select_all:", key=f"cls_all_{ds}", on_click=_cocher_classes, args=(ck.values(), True))
    t[1].button("Aucune classe", icon=":material/deselect:", key=f"cls_none_{ds}", on_click=_cocher_classes, args=(ck.values(), False))
    gc = st.columns(4)
    for i, c in enumerate(pool):
        gc[i % 4].checkbox(c, key=ck[c])
    chosen = [c for c in pool if st.session_state.get(ck[c], False)]
    by = {c: df[df.class_name == c].path.tolist() for c in chosen}
    for cl in chosen:
        paths = by[cl]; n_sel = sum(p in S for p in paths)
        vk = f"ver_{ds}_{cl}"; ver = st.session_state.get(vk, 0)
        with st.expander(f"{cl} · {n_sel} / {len(paths)} images sélectionnées", icon=":material/folder:", expanded=len(chosen) <= 2):
            b = st.columns(4)
            b[0].button("Tout", icon=":material/select_all:", key=f"all_{cl}", on_click=_maj, args=(paths, S, True, vk))
            b[1].button("Aucune", icon=":material/deselect:", key=f"none_{cl}", on_click=_maj, args=(paths, S, False, vk))
            b[2].button(f"{npc} au hasard", icon=":material/casino:", key=f"rnd_{cl}", on_click=_maj,
                        args=(paths, S, True, vk, npc, random.randint(1, 10**6)))
            npages = max(1, -(-len(paths) // PAGE))
            pg = b[3].number_input("Page", 1, npages, 1, key=f"pg_{cl}") if npages > 1 else 1
            page = paths[(pg - 1) * PAGE: pg * PAGE]
            cols = st.columns(6)
            for i, p in enumerate(page):
                with cols[i % 6]:
                    st.image(_thumb(p), width="stretch")
                    k = f"cb_{ver}_{p}"
                    st.checkbox(Path(p).name, value=p in S, key=k, on_change=_sync, args=(p, k))
    # résumé + validation
    counts = {c: sum(p in S for p in by[c]) for c in chosen}
    if chosen:
        st.markdown("**Sélection :** " + " · ".join(f"{c} ({n})" for c, n in counts.items()))
    if st.button("Valider ma sélection", icon=":material/check_circle:", type="primary"):
        ok = [c for c in chosen if counts[c] >= 2]
        if len(ok) < 2:
            st.error("Choisissez au moins 2 classes avec au moins 2 images sélectionnées chacune.", icon=":material/error:")
        else:
            if len(ok) < len(chosen):
                st.warning("Classes ignorées (moins de 2 images) : " + ", ".join(c for c in chosen if c not in ok), icon=":material/warning:")
            if len(ok) < 5:
                st.warning("Moins de 5 classes : l'expérience risque d'être trop facile (scores proches de 1.000 pour tous les "
                           "modèles). Voir les conseils ci-dessus.", icon=":material/warning:")
            sel = [p for c in ok for p in by[c] if p in S]
            sm, classes = X.echantillon_manuel(ds, sel, df, st.session_state["graine"])
            st.session_state["ech"] = dict(ds=ds, s=sm, classes=classes, seed=st.session_state["graine"],
                                           npc=int(sm[~sm.class_name.isin(excl)].groupby("class_name").size().mean()),
                                           manuel=True)
            st.session_state.pop("res", None)


def _hex(c): return "#%02x%02x%02x" % tuple(int(v) for v in c)


def _mix(a, b, t): return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def _relatif(t):
    """0 -> rose pâle, 0.5 -> crème, 1 -> vert pâle (échelle douce, texte toujours lisible)."""
    lo, mid, hi = (253, 228, 224), (255, 247, 224), (214, 238, 220)
    return _hex(_mix(lo, mid, t * 2) if t < .5 else _mix(mid, hi, (t - .5) * 2))


def _style_table(tab, mode):
    """Met en évidence ce qui compte au lieu de tout colorer :
    best = le meilleur modèle de chaque colonne (par indicateur) ; rel = dégradé relatif entre les modèles ; none = aucun."""
    css = pd.DataFrame("", index=tab.index, columns=tab.columns)
    css[X.TOUTES] = "font-weight:700;border-left:1px solid #e6e6e6;"
    if mode != "none":
        for ind in X.INDICATEURS:
            sub = tab.xs(ind, level="Indicateur", drop_level=False)
            lo, hi = sub.min(), sub.max()
            for idx in sub.index:
                for col in tab.columns:
                    v = sub.loc[idx, col]
                    if hi[col] <= lo[col]:
                        continue
                    if mode == "best":
                        if np.isclose(v, hi[col]):
                            css.loc[idx, col] += "background-color:#e1eefc;color:#0b4f9e;font-weight:700;"
                    else:
                        css.loc[idx, col] += f"background-color:{_relatif((v - lo[col]) / (hi[col] - lo[col]))};"
    return tab.style.format("{:.3f}").apply(lambda _: css, axis=None)


def _nouvelle_graine():
    st.session_state["graine"] = random.randint(1, 99999)


def afficher():
    ui.page_header("Expérience sur échantillon",
                   "Tirez au hasard (ou choisissez vous-même) des classes et des images, puis lancez l'évaluation : "
                   "extraction des descripteurs, recherche par similarité, P@10 et mAP par classe.")

    bases = X.bases_disponibles()
    if not bases:
        st.error(f"Aucune base trouvée dans {C.DATA_DIR}. Placer Corel-1K, Corel-5K, Corel-10K, CIFAR100, paris à côté "
                 "du dossier du script (ou définir la variable CBIR_DATA_DIR).", icon=":material/error:")
        return

    # ------------------------------------------------------------ paramètres
    sb = st.sidebar
    sb.divider(); sb.subheader("Paramètres de l'expérience", anchor=False)
    ds = sb.selectbox("Base de données", bases, format_func=lambda d: C.DATASETS[d]["label"])
    df = _scan(ds)
    excl = C.DATASETS[ds].get("query_exclude", [])
    n_pool = df[~df.class_name.isin(excl)].class_name.nunique()
    par_classe_max = int(df[~df.class_name.isin(excl)].groupby("class_name").size().min())

    if X.NB_CLASSES[ds] is None:
        nc = None
        sb.info(f"Toutes les classes de la base sont utilisées ({n_pool}).", icon=":material/info:")
    else:
        nc = sb.radio("Nombre de classes tirées au hasard", [10, 20], index=[10, 20].index(X.NB_CLASSES[ds]),
                      horizontal=True, help=f"La base contient {n_pool} classes.")
    defaut = min(X.NB_PAR_CLASSE[ds], par_classe_max)
    npc = sb.slider("Images tirées par classe", min(5, par_classe_max), par_classe_max, defaut,
                    help="Plus l'échantillon est grand, plus les résultats sont stables (et plus le calcul est long).")
    modeles = sb.multiselect("Modèles", list(C.MODELS), default=X.MODELES_DEFAUT, format_func=X.nom_modele)
    metrique = sb.radio("Mesure de similarité", C.METRICS, horizontal=True)

    if "graine" not in st.session_state:
        _nouvelle_graine()
    c1, c2 = sb.columns([3, 2])
    c1.number_input("Graine aléatoire", 1, 99999, key="graine")
    c2.write(""); c2.write("")
    c2.button("Nouvelle", icon=":material/casino:", help="Tirer une nouvelle graine", on_click=_nouvelle_graine)

    # ------------------------------------------------- étape 1 : tirage
    ui.step(1, "Échantillon", "classes et images utilisées pour le test")
    mode = st.radio("Mode de choix", ["alea", "manuel"], horizontal=True,
                    format_func=lambda m: ":material/casino: Aléatoire" if m == "alea"
                    else ":material/touch_app: Manuel (je choisis moi-même les images)")
    with st.expander("Conseils pour bien choisir les classes et les images", icon=":material/lightbulb:"):
        st.markdown(CONSEILS)
    if mode == "alea":
        if st.button("Tirer l'échantillon", icon=":material/ads_click:", type="primary"):
            s, classes = X.tirer_echantillon(ds, nc, npc, st.session_state["graine"], df)
            st.session_state["ech"] = dict(ds=ds, s=s, classes=classes, seed=st.session_state["graine"], npc=npc)
            st.session_state.pop("res", None)
    else:
        _manuel(ds, df, excl, npc)

    E = st.session_state.get("ech")
    if not E:
        st.info("Choisissez le mode et les paramètres, puis tirez ou validez l'échantillon.", icon=":material/info:")
        return
    s, classes = E["s"], E["classes"]
    if E["ds"] != ds:
        st.warning("L'échantillon affiché correspond à une autre base : retirez un échantillon pour cette base.", icon=":material/warning:")

    nm = C.DATASETS[E["ds"]]["label"]
    st.success(f"**{nm}** · graine **{E['seed']}** · **{len(classes)} classes** · **{len(s)} images** "
               + (f"(~{E['npc']} par classe, choisies manuellement)" if E.get("manuel") else f"({E['npc']} par classe)") + (f" + {int(s.class_name.isin(excl).sum())} distracteurs" if excl and E["ds"] == ds else ""),
               icon=":material/check_circle:")
    st.markdown("**Classes de l'échantillon :** " + " · ".join(classes))
    with st.expander("Aperçu des images de l'échantillon (4 par classe)", icon=":material/photo_library:", expanded=True):
        for cl in classes:
            sub = s[s.class_name == cl].head(4)
            st.markdown(f"**{cl}** — {int((s.class_name == cl).sum())} images")
            cols = st.columns(4)
            for col, p in zip(cols, sub.path):
                try:
                    col.image(load_rgb(p), width="stretch")
                except Exception:
                    col.caption("Image illisible")

    # ------------------------------------------- étape 2 : évaluation
    ui.step(2, "Évaluation", "extraction, recherche et métriques")
    if not modeles:
        st.warning("Sélectionnez au moins un modèle.", icon=":material/warning:")
        return
    if st.button("Lancer l'évaluation", icon=":material/rocket_launch:", type="primary"):
        bar = st.progress(0.0, text="Démarrage…")
        with st.status("Le code Python s'exécute en arrière-plan…", expanded=False) as status:
            try:
                tab, info = X.evaluer(E["ds"], s, classes, modeles, metrique,
                                      progress=lambda f, msg: bar.progress(min(max(f, 0.0), 1.0), text=msg))
            except ImportError as e:
                status.update(label="PyTorch manquant", state="error")
                st.error(f"PyTorch/torchvision n'est pas installé : {e}", icon=":material/error:")
                return
            except Exception as e:
                status.update(label="Erreur", state="error")
                st.exception(e)
                return
            status.update(label="Terminé", state="complete")
        bar.empty()
        X.sauver(tab, E["ds"], E["seed"])
        st.session_state["res"] = dict(tab=tab, info=info, metrique=metrique, seed=E["seed"], ds=E["ds"])

    R = st.session_state.get("res")
    if not R:
        return

    # ----------------------------------------------------- résultats
    tab, info = R["tab"], R["info"]
    ui.step(3, "Résultats", f"{C.DATASETS[R['ds']]['label']} · graine {R['seed']} · similarité {R['metrique']}")
    mm = tab.xs("mAP moyen", level="Indicateur")[X.TOUTES]
    ui.stats([dict(label="Classes", value=tab.shape[1] - 1, icon="grid", tint="blue"),
              dict(label="Images / requêtes", value=f"{info['n_images']} / {info['n_requetes']}", icon="image", tint="purple"),
              dict(label="Meilleur modèle (mAP)", value=mm.idxmax(), sub=f"mAP {mm.max():.3f}", icon="target", tint="green"),
              dict(label="P@10 moyen (meilleur)", value=f"{tab.xs('P@10 moyen', level='Indicateur')[X.TOUTES].max():.3f}", icon="bars", tint="orange")])
    mise = st.radio("Mise en évidence", ["best", "rel", "none"], horizontal=True,
                    format_func=lambda m: {"best": "Meilleur modèle par colonne", "rel": "Dégradé relatif par colonne", "none": "Aucune"}[m])
    st.dataframe(_style_table(tab, mise), width="stretch", height=min(700, 38 + 35 * len(tab)))
    st.caption("Requêtes en leave-one-out (pertinent = même classe que la requête). P@10 moyen = précision moyenne des 10 premiers résultats ; "
               "mAP moyen = moyenne des AP des requêtes de la classe ; mAP max = meilleure AP d'une requête de la classe.")

    if tab.xs("mAP moyen", level="Indicateur")[X.TOUTES].min() >= 0.98:
        st.info("Tous les modèles obtiennent un mAP supérieur ou égal à 0.98 : l'échantillon est trop facile pour les départager "
                "(classes très différentes ou trop peu de classes). Ajoutez des classes proches visuellement ou "
                "augmentez le nombre de classes (voir les conseils).", icon=":material/info:")
    d1, d2 = st.columns(2)
    d1.download_button("Télécharger (CSV)", tab.to_csv(encoding="utf-8-sig").encode("utf-8-sig"),
                       f"resultats_{R['ds']}_seed{R['seed']}.csv", "text/csv", icon=":material/download:")
    d2.markdown(":material/timer: " + " · ".join(f"{k} : {v}s" for k, v in info["temps"].items()))

    with st.expander("Carte de chaleur : mAP moyen par modèle et par classe", icon=":material/grid_view:", expanded=True):
        m = tab.xs("mAP moyen", level="Indicateur").drop(columns=X.TOUTES)
        fig = px.imshow(m, text_auto=".2f", aspect="auto", color_continuous_scale="Blues", zmin=0, zmax=1,
                        labels=dict(x="Classe", y="Modèle", color="mAP"))
        ui.category_axes(fig, x=m.columns, y=m.index)
        ui.show(fig, height=max(260, 70 * len(m) + 120))
