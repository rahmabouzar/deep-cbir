"""ÉTAPE 3 -- Recherche et mesure de similarité (online retrieval).

Pour une image requête : on prend son vecteur, on calcule la distance à tous les vecteurs
de la base (Cosinus ET Euclidienne) et on retourne les K images les plus proches.

Usage : python etape3_recherche.py --dataset corel1k --model resnet50 --query-class buses --k 10
        python etape3_recherche.py --dataset paris6k --model vgg16 --query-idx 123
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from common import load_manifest, load_features, top_k, load_rgb


def search(ds, model, query_idx, metric, k):
    df, F = load_manifest(ds), load_features(ds, model)
    idx, dist = top_k(F, F[query_idx], metric, k, exclude=query_idx)
    res = df.iloc[idx].copy()
    res["distance"] = dist
    res["pertinent"] = res.label.values == df.label.iloc[query_idx]
    return res


def figure_recherche(ds, model, query_idx, k, path=None):
    df = load_manifest(ds)
    fig, axs = plt.subplots(2, k + 1, figsize=(1.9 * (k + 1), 4.8))
    for row, metric in enumerate(C.METRICS):
        res = search(ds, model, query_idx, metric, k)
        axs[row, 0].imshow(load_rgb(df.path[query_idx]))
        axs[row, 0].set_title(f"REQUÊTE\n{df.class_name[query_idx]}", fontsize=8)
        for j, (_, r) in enumerate(res.iterrows(), start=1):
            axs[row, j].imshow(load_rgb(r.path))
            axs[row, j].set_title(f"{r.class_name}\nd={r.distance:.3f}", fontsize=7)
            col = "#2ca02c" if r.pertinent else "#d62728"
            for s in axs[row, j].spines.values():
                s.set_edgecolor(col); s.set_linewidth(3)
        axs[row, 0].set_ylabel(metric.capitalize(), fontsize=11)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{C.DATASETS[ds]['label']} – {C.MODELS[model][0]} – Top-{k} "
                 "(vert = même classe, rouge = autre classe)")
    fig.tight_layout()
    path = path or C.FIGS / f"recherche_{ds}_{model}_q{query_idx}.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="corel1k", choices=list(C.DATASETS))
    ap.add_argument("--model", default="resnet50", choices=list(C.MODELS))
    ap.add_argument("--query-idx", type=int)
    ap.add_argument("--query-class", help="nom de classe : tire une requête au hasard dedans")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    df = load_manifest(a.dataset)
    rng = np.random.default_rng(C.SEED)
    if a.query_idx is None:
        pool = df.index[df.class_name == a.query_class] if a.query_class else df.index
        a.query_idx = int(rng.choice(pool))
    print(f"Requête #{a.query_idx} : {df.path[a.query_idx]} (classe {df.class_name[a.query_idx]})")
    for metric in C.METRICS:
        res = search(a.dataset, a.model, a.query_idx, metric, a.k)
        print(f"\n-- {metric} : {int(res.pertinent.sum())}/{a.k} pertinentes")
        print(res[["path", "class_name", "distance", "pertinent"]].to_string(index=False))
    print("\n✔ figure :", figure_recherche(a.dataset, a.model, a.query_idx, a.k))


if __name__ == "__main__":
    main()
