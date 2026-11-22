"""Calculs partagés par le site HTML (étape 6) et le PDF (étape 7) : mêmes chiffres, mêmes explications."""
import html, json, types
import numpy as np
import pandas as pd

import config as C
from common import load_manifest, load_features, pairwise

nd = lambda d: C.DATASETS[d]["label"]
nm = lambda m: C.MODELS[m][0]
esc = lambda x: html.escape(str(x))
GFLOPS = {"squeezenet1_1": 0.35, "mobilenet_v3_small": 0.06, "resnet18": 1.81, "resnet50": 4.09, "vgg16": 15.47}  # ≈ docs torchvision


def load_results():
    need = ["metrics_global.csv", "metrics_par_classe.csv", "pr_curves.json", "etape1_stats.json"]
    manq = [n for n in need if not (C.RESULTS / n).exists()]
    if manq:
        raise SystemExit("\n❌ Fichier(s) manquant(s) dans results\\ : " + ", ".join(manq) +
                         "\n   -> lancer d'abord : etape1_preparation.py | etape2_extraction.py | etape4_evaluation.py\n")
    r = types.SimpleNamespace()
    r.G = pd.read_csv(C.RESULTS / "metrics_global.csv"); r.PC = pd.read_csv(C.RESULTS / "metrics_par_classe.csv", dtype={"classe": str})
    r.PR = json.loads((C.RESULTS / "pr_curves.json").read_text(encoding="utf-8"))
    r.ST = json.loads((C.RESULTS / "etape1_stats.json").read_text(encoding="utf-8"))
    r.DSS = [d for d in C.DATASETS if d in set(r.G.dataset)]
    r.MODS = [m for m in C.MODELS if m in set(r.G.model) and all(((r.G.dataset == d) & (r.G.model == m)).any() for d in r.DSS)]
    r.MAN = {d: load_manifest(d) for d in r.DSS}
    r.PRESET = next(iter(r.ST.values()))["preset"] if r.ST else "?"
    return r


def best_row(G, ds, metric=None):
    s = G[G.dataset == ds] if metric is None else G[(G.dataset == ds) & (G.metrique == metric)]
    return s.sort_values("mAP", ascending=False).iloc[0]


# ---------------------------------------------------------------- étape 2
def feat_info(F, y, rng):
    n = len(F); nr = np.linalg.norm(F, axis=1)
    s = rng.choice(n, size=min(1500, n), replace=False)
    Fn = F[s] / (nr[s][:, None] + 1e-12); S = Fn @ Fn.T
    same = y[s][:, None] == y[s][None, :]; eye = np.eye(len(s), dtype=bool)
    intra, inter = S[same & ~eye].mean(), S[~same].mean()
    return dict(mb=F.nbytes / 1e6, norm_mean=float(nr.mean()), norm_cv=float(100 * nr.std() / (nr.mean() + 1e-12)),
                zeros=float(100 * (F <= 1e-6).mean()), intra=float(intra), inter=float(inter), sep=float(intra - inter), norms=nr[s])


def extraction_table(R):
    rng = np.random.default_rng(C.SEED); rows, norms = [], {}
    for d in R.DSS:
        for m in R.MODS:
            r = R.G[(R.G.dataset == d) & (R.G.model == m) & (R.G.metrique == "cosinus")]
            if r.empty: continue
            r = r.iloc[0]; fi = feat_info(load_features(d, m), R.MAN[d].label.values, rng); tot = r.ms_extraction_img * r.n_images / 1000
            rows.append({"Base": nd(d), "Modèle": nm(m), "dataset": d, "model": m, "Images": int(r.n_images), "Dimension": int(r.dim),
                         "Index (Mo)": fi["mb"], "Temps total (min)": tot / 60, "ms/image": float(r.ms_extraction_img),
                         "images/s": 1000 / float(r.ms_extraction_img), "Norme moy.": fi["norm_mean"], "Var. des normes (%)": fi["norm_cv"],
                         "Valeurs nulles (%)": fi["zeros"], "cos. même classe": fi["intra"], "cos. classes diff.": fi["inter"],
                         "Séparabilité": fi["sep"], "mAP (étape 4)": float(r.mAP)})
            if d == R.DSS[0]: norms[m] = fi["norms"]
    return pd.DataFrame(rows), norms


def pca2(F, rng, n=2000):
    """ACP 2D par astuce de Gram (rapide) sur un échantillon de n vecteurs normalisés. Retourne (indices, coordonnées)."""
    s = rng.choice(len(F), size=min(n, len(F)), replace=False)
    X = F[s] / (np.linalg.norm(F[s], axis=1, keepdims=True) + 1e-12); X = X - X.mean(0)
    w, V = np.linalg.eigh(X @ X.T); idx = np.argsort(w)[::-1][:2]
    return s, V[:, idx] * np.sqrt(np.maximum(w[idx], 0))


# ---------------------------------------------------------------- étape 3
def dist_samples(F, y, df, ds, rng, nq=200, cap=20000):
    """Distances cosinus requête->base : (même classe, autres classes)."""
    okq = np.where(~df.class_name.isin(C.DATASETS[ds].get("query_exclude", [])).values)[0]
    qs = rng.choice(okq, size=min(nq, len(okq)), replace=False)
    D = pairwise(F[qs], F, "cosinus"); same = y[qs][:, None] == y[None, :]; D[np.arange(len(qs)), qs] = np.nan
    a, b = D[same & ~np.isnan(D)], D[~same & ~np.isnan(D)]
    return rng.choice(a, size=min(cap, len(a)), replace=False), rng.choice(b, size=min(cap, len(b)), replace=False)


# ---------------------------------------------------------------- conclusion
def conclusion_points(R):
    G, PC, DSS, MODS = R.G, R.PC, R.DSS, R.MODS
    cm = G[G.metrique == "cosinus"].groupby("model").mAP.mean(); cm = cm[[m for m in MODS if m in cm.index]]
    best_m, worst_m = cm.idxmax(), cm.idxmin()
    lat = G[G.metrique == "cosinus"].groupby("model").ms_extraction_img.mean()
    good = [m for m in cm.index if cm[m] >= 0.95 * cm.max()]; compromise = min(good, key=lambda m: lat[m])
    leg = G[(G.metrique == "cosinus") & (G.famille == "Léger")]; cla = G[(G.metrique == "cosinus") & (G.famille == "Classique")]
    d_ = G.pivot_table(index=["dataset", "model"], columns="metrique", values="mAP"); wins = int((d_["cosinus"] > d_["euclidienne"]).sum())
    perds = G[G.metrique == "cosinus"].groupby("dataset").mAP.mean().sort_values()
    L = [f"<b>Meilleur modèle en moyenne</b> (cosinus, toutes bases) : <b>{nm(best_m)}</b> (mAP moyen {cm.max():.4f}) ; le plus faible : {nm(worst_m)} ({cm.min():.4f})."]
    for d in DSS:
        r = best_row(G, d); L.append(f"<b>{nd(d)}</b> : meilleur = {r.modele} / {r.metrique} (mAP {r.mAP:.4f}, P@10 {r['P@10']:.4f}).")
    L.append(f"<b>Cosinus vs euclidienne</b> : le cosinus gagne dans {wins}/{len(d_)} cas (écart moyen {(d_['cosinus'] - d_['euclidienne']).mean():+.4f} mAP).")
    if len(leg) and len(cla):
        L.append(f"<b>Léger vs classique</b> : mAP moyen {leg.mAP.mean():.4f} (léger) contre {cla.mAP.mean():.4f} (classique) ; temps d'extraction moyen "
                 f"{leg.ms_extraction_img.mean():.1f} ms contre {cla.ms_extraction_img.mean():.1f} ms.")
    L.append(f"<b>Meilleur compromis</b> (≥ 95 % du meilleur mAP, le plus rapide) : <b>{nm(compromise)}</b> ({cm[compromise]:.4f} mAP, {lat[compromise]:.1f} ms/image).")
    L.append("<b>Difficulté des bases</b> (mAP moyen) : " + " &lt; ".join(f"{nd(d)} ({v:.3f})" for d, v in perds.items()) + ".")
    for d in DSS:
        r = best_row(G, d); pc = PC[(PC.dataset == d) & (PC.model == r.model) & (PC.metrique == r.metrique)].sort_values("AP", ascending=False)
        L.append(f"<b>{nd(d)} – classes</b> : plus facile « {esc(pc.classe.iloc[0])} » (AP {pc.AP.iloc[0]:.3f}), plus difficile « {esc(pc.classe.iloc[-1])} » (AP {pc.AP.iloc[-1]:.3f}).")
    return L
