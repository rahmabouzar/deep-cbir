"""EXPÉRIENCE ALÉATOIRE -- tirage d'un échantillon (classes + images) puis évaluation CBIR sur cet échantillon.

Principe (à chaque lancement, tout est tiré au hasard, graine affichée donc reproductible) :
  1. Classes : Corel-1K -> les 10 classes ; Paris6k -> tous les monuments ; Corel-5K/10K/CIFAR-100 -> 10 ou 20 classes tirées.
  2. Images : n images tirées au hasard dans chaque classe retenue  => « échantillon » (= la base de test).
  3. Descripteurs : repris de l'étape 2 s'ils existent, sinon extraits à la volée (avec cache disque par image).
  4. Évaluation : chaque image de l'échantillon est une requête (leave-one-out, pertinent = même classe).
  5. Tableau final : lignes = (modèle x {P@10 moyen, mAP max, mAP moyen}), colonnes = classes tirées (+ « Toutes classes »).

Définitions (par classe c, sur les requêtes de la classe c) :
  P@10 moyen = moyenne des P@10 ; mAP moyen = moyenne des AP ; mAP max = meilleure AP obtenue par une requête de la classe.

Usage en ligne de commande :  python experience.py --dataset corel1k [--seed 7] [--models vgg16 resnet50 squeezenet1_1 resnet18]
Usage web                  :  streamlit run app.py   -> page « Expérience sur échantillon »
"""
import argparse, hashlib, time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from common import (scan_dataset, find_root, load_rgb, get_transform, build_model,
                    load_manifest, load_features, feat_path)
from etape4_evaluation import evaluate

# Modèles demandés, dans l'ordre d'affichage du tableau
MODELES_DEFAUT = ["vgg16", "resnet50", "squeezenet1_1", "resnet18"]
BASES = ["corel1k", "corel5k", "corel10k", "cifar100", "paris6k"]

# Règle « 10 ou 20 classes selon la base » (None = toutes les classes de la base)
NB_CLASSES = {"corel1k": None, "paris6k": None, "corel5k": 10, "corel10k": 20, "cifar100": 20}
# Images tirées par classe (par défaut) ; plafonné par la taille réelle de la classe
NB_PAR_CLASSE = {"corel1k": 100, "corel5k": 50, "corel10k": 50, "cifar100": 50, "paris6k": 50}

CACHE = C.WORK / "cache_echantillons"        # un petit .npy par (modèle, image) -> les tirages suivants sont rapides
INDICATEURS = ["P@10 moyen", "mAP max", "mAP moyen"]
TOUTES = "Toutes classes"


def nom_modele(m): return C.MODELS[m][0]


def _tri_classes(noms):
    return sorted(noms, key=lambda s: (len(s), s) if str(s).isdigit() else (0, s))


def bases_disponibles():
    out = []
    for k in BASES:
        try:
            find_root(k); out.append(k)
        except FileNotFoundError:
            pass
    return out


def scanner(ds: str) -> pd.DataFrame:
    """Toutes les images de la base (path, class_name)."""
    return scan_dataset(ds, "full")


# --------------------------------------------------------------- 1) Tirage
def tirer_echantillon(ds: str, n_classes=None, n_par_classe=None, seed: int = 0, df: pd.DataFrame = None):
    """Retourne (échantillon, classes_tirées). Tout est aléatoire et reproductible avec `seed`."""
    rng = np.random.default_rng(seed)
    df = scanner(ds) if df is None else df
    excl = [c for c in C.DATASETS[ds].get("query_exclude", []) if c in set(df.class_name)]
    pool = _tri_classes([c for c in df.class_name.unique() if c not in excl])
    k = len(pool) if not n_classes else min(int(n_classes), len(pool))
    classes = _tri_classes([str(c) for c in rng.choice(pool, size=k, replace=False)])
    n_par_classe = int(n_par_classe or NB_PAR_CLASSE.get(ds, 50))
    parts = []
    for c in classes + excl:                                   # distracteurs (Paris « general ») : dans la base, jamais requêtes
        sub = df[df.class_name == c]
        n = min(n_par_classe, len(sub))
        parts.append(sub.iloc[np.sort(rng.choice(len(sub), size=n, replace=False))])
    s = pd.concat(parts).reset_index(drop=True)
    names = _tri_classes(s.class_name.unique())
    s["label"] = s.class_name.map({n: i for i, n in enumerate(names)})
    return s, classes


def echantillon_manuel(ds: str, paths, df: pd.DataFrame, seed: int = 0, n_distracteurs: int = None):
    """Échantillon construit à partir des images CHOISIES par l'utilisateur (liste de chemins)."""
    rng = np.random.default_rng(seed)
    excl = [c for c in C.DATASETS[ds].get("query_exclude", []) if c in set(df.class_name)]
    s = df[df.path.isin(set(paths))]
    s = s[~s.class_name.isin(excl)]
    classes = _tri_classes(s.class_name.unique())
    parts = [s]
    for c in excl:                                             # distracteurs (Paris) : ajoutés au hasard, jamais requêtes
        sub = df[df.class_name == c]
        n = min(n_distracteurs or int(s.groupby("class_name").size().mean()), len(sub))
        parts.append(sub.iloc[np.sort(rng.choice(len(sub), size=n, replace=False))])
    s = pd.concat(parts).sort_values(["class_name", "path"]).reset_index(drop=True)
    names = _tri_classes(s.class_name.unique())
    s["label"] = s.class_name.map({n: i for i, n in enumerate(names)})
    return s, classes


# --------------------------------------------------------- 2) Descripteurs
def _depuis_etape2(ds, model, paths):
    """Réutilise les vecteurs déjà extraits (étape 2) si tous les chemins y figurent."""
    try:
        if not feat_path(ds, model).exists():
            return None
        man, F = load_manifest(ds), load_features(ds, model)
    except Exception:
        return None
    if len(F) != len(man):
        return None
    pos = {p: i for i, p in enumerate(man.path)}
    if not all(p in pos for p in paths):
        return None
    return F[[pos[p] for p in paths]].astype(np.float32)


def _key(p): return hashlib.md5(str(p).encode()).hexdigest()


def _extraire_a_la_volee(model, paths, progress=None, batch=32):
    """Extrait (avec cache disque) les vecteurs des images `paths` pour `model`."""
    import torch
    cdir = CACHE / model; cdir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in paths if not (cdir / f"{_key(p)}.npy").exists()]
    if todo:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = build_model(model).to(device); tf = get_transform()
        with torch.inference_mode():
            for s in range(0, len(todo), batch):
                chunk = todo[s:s + batch]; xs = []
                for p in chunk:
                    try:
                        xs.append(tf(load_rgb(p)))
                    except Exception:                          # image illisible -> vecteur nul
                        xs.append(torch.zeros(3, C.IMG_SIZE, C.IMG_SIZE))
                f = net(torch.stack(xs).to(device)).float().cpu().numpy()
                for p, v in zip(chunk, f):
                    np.save(cdir / f"{_key(p)}.npy", v.astype(np.float32))
                if progress:
                    progress(min(s + batch, len(todo)) / len(todo))
    return np.stack([np.load(cdir / f"{_key(p)}.npy") for p in paths])


def descripteurs(ds, model, paths, progress=None):
    F = _depuis_etape2(ds, model, list(paths))
    return F if F is not None else _extraire_a_la_volee(model, list(paths), progress)


# ---------------------------------------------------------- 3) Évaluation
def evaluer(ds, echantillon, classes, modeles=None, metrique="cosinus", progress=None, extracteur=None):
    """Retourne (tableau, infos). `progress(fraction, message)` est appelé au fil du calcul.
    `extracteur(ds, model, paths, progress)` peut remplacer `descripteurs` (tests)."""
    modeles = modeles or MODELES_DEFAUT
    extracteur = extracteur or descripteurs
    y = echantillon.label.values
    names = _tri_classes(echantillon.class_name.unique()); n_cls = len(names)
    lab = {n: i for i, n in enumerate(names)}
    excl = C.DATASETS[ds].get("query_exclude", [])
    query_ok = ~echantillon.class_name.isin(excl).values
    cols = {}                                                  # (modèle, indicateur) -> {classe: valeur}
    infos = dict(temps={}, n_images=len(echantillon), n_requetes=0)
    for i, m in enumerate(modeles):
        def cb(frac, i=i, m=m):
            if progress:
                progress((i + 0.8 * frac) / len(modeles), f"Extraction des descripteurs – {nom_modele(m)} ({int(100 * frac)} %)")
        t0 = time.perf_counter()
        F = extracteur(ds, m, echantillon.path.tolist(), cb)
        if progress:
            progress((i + 0.8) / len(modeles), f"Recherche + calcul des métriques – {nom_modele(m)}")
        r = evaluate(F, y, metrique, query_ok, n_cls)
        q, ap, p10 = r["q"], r["ap"], r["pk"][10]
        yq = y[q]; infos["n_requetes"] = len(q)
        res = {k: {} for k in INDICATEURS}
        for c in classes:
            msk = yq == lab[c]
            if msk.any():
                res["P@10 moyen"][c] = float(p10[msk].mean())
                res["mAP max"][c] = float(ap[msk].max())
                res["mAP moyen"][c] = float(ap[msk].mean())
        res["P@10 moyen"][TOUTES] = float(p10.mean())
        res["mAP max"][TOUTES] = float(ap.max())
        res["mAP moyen"][TOUTES] = float(ap.mean())
        for k in INDICATEURS:
            cols[(nom_modele(m), k)] = res[k]
        infos["temps"][nom_modele(m)] = round(time.perf_counter() - t0, 1)
        if progress:
            progress((i + 1) / len(modeles), f"{nom_modele(m)} terminé")
    tab = pd.DataFrame(cols).T                                 # lignes = (modèle, indicateur)
    tab = tab[[c for c in classes if c in tab.columns] + [TOUTES]]
    tab.index = pd.MultiIndex.from_tuples(tab.index, names=["Modèle", "Indicateur"])
    return tab, infos


def sauver(tab, ds, seed):
    d = C.RESULTS / "experiences"; d.mkdir(parents=True, exist_ok=True)
    p = d / f"{ds}_seed{seed}.csv"
    tab.to_csv(p, encoding="utf-8-sig")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="corel1k", choices=BASES)
    ap.add_argument("--classes", type=int, choices=[10, 20], help="10 ou 20 (ignoré pour Corel-1K / Paris6k)")
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=MODELES_DEFAUT, choices=list(C.MODELS))
    ap.add_argument("--metric", default="cosinus", choices=C.METRICS)
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    a = ap.parse_args()
    nc = None if NB_CLASSES[a.dataset] is None else (a.classes or NB_CLASSES[a.dataset])
    s, classes = tirer_echantillon(a.dataset, nc, a.per_class, a.seed)
    print(f"Graine {a.seed} | {len(classes)} classes : {', '.join(classes)} | {len(s)} images")
    tab, info = evaluer(a.dataset, s, classes, a.models, a.metric,
                        progress=lambda f, msg: print(f"  [{int(100 * f):3d} %] {msg}"))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(tab.round(3).to_string())
    print("[OK] enregistré :", sauver(tab, a.dataset, a.seed))


if __name__ == "__main__":
    main()
