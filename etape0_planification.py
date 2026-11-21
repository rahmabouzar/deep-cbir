"""ÉTAPE 0 -- Planification : mesure la vitesse de VOTRE machine et propose un plan dans votre budget de temps.

Ne télécharge aucun poids (la vitesse ne dépend pas des poids). Durée : 1 à 3 minutes.

Usage : python etape0_planification.py                    (budget total 90 min)
        python etape0_planification.py --budget-min 120
"""
import argparse, os, time
import numpy as np
import torch

import config as C
from common import scan_dataset, build_model, load_rgb, get_transform
import etape4_evaluation as E4

LIGHT = ["squeezenet1_1", "mobilenet_v3_small", "resnet18"]


def bench_models(models, bs=8, reps=2):
    x = torch.randn(bs, 3, C.IMG_SIZE, C.IMG_SIZE).contiguous(memory_format=torch.channels_last)
    out = {}
    for m in models:
        net = build_model(m, pretrained=False).to(memory_format=torch.channels_last)
        with torch.inference_mode():
            net(x[:2]); t = time.perf_counter()
            for _ in range(reps): net(x)
        out[m] = 1000 * (time.perf_counter() - t) / (reps * bs)
        print(f"  {C.MODELS[m][0]:20s} {out[m]:8.1f} ms / image")
    return out


def bench_decode(df, n=20):
    tf = get_transform(); paths = df.sample(min(n, len(df)), random_state=C.SEED).path
    for p in paths[:3]: tf(load_rgb(p))                       # échauffement (cache disque)
    t = time.perf_counter()
    for p in paths: tf(load_rgb(p))
    return 1000 * (time.perf_counter() - t) / len(paths)


def bench_eval(n=2500, d=1024):
    rng = np.random.default_rng(0); y = rng.integers(0, 50, n); F = rng.normal(size=(n, d)).astype(np.float32)
    t = time.perf_counter(); E4.evaluate(F, y, "cosinus", np.ones(n, bool), 50)
    return time.perf_counter() - t, n


def fmt(sec):
    m = sec / 60
    return f"{m:.0f} min" if m < 90 else f"{m / 60:.1f} h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-min", type=float, default=90, help="budget TOTAL (extraction + évaluation) en minutes")
    ap.add_argument("--test-min", type=float, default=8, help="durée visée du test préliminaire")
    ap.add_argument("--datasets", nargs="+", default=C.DEFAULT_DATASETS, choices=list(C.DATASETS))
    ap.add_argument("--models", nargs="+", default=list(C.MODELS), choices=list(C.MODELS))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    print(f"Machine : {os.cpu_count()} cœurs logiques, torch utilise {torch.get_num_threads()} threads\n")
    print("1) Vitesse des réseaux (calcul seul, par image) :")
    ms = bench_models(a.models)
    print("\n2) Vitesse de lecture/prétraitement des images :")
    counts, dec = {}, {}
    for k in a.datasets:
        try:
            df = scan_dataset(k, "full")
        except FileNotFoundError as e:
            print(f"  (ignoré) {e}"); continue
        counts[k] = df.groupby("class_name").size().values; dec[k] = bench_decode(df)
        print(f"  {C.DATASETS[k]['label']:20s} {len(df):>6} images, {dec[k]:6.1f} ms / image")
    t_ref, n_ref = bench_eval()

    def n_imgs(k, cap): return int(np.minimum(counts[k], cap).sum()) if cap else int(counts[k].sum())
    def t_extract(models, cap):
        s = sum(ms[m] for m in models)
        return sum(n_imgs(k, cap) * max(s, dec[k] / a.workers) / 1000 for k in counts)
    def t_eval(models, cap):
        return sum(t_ref * (n_imgs(k, cap) / n_ref) ** 2 * len(models) * 2 for k in counts)
    def total(models, cap): return 1.03 * (t_extract(models, cap) + t_eval(models, cap))
    full_n = sum(n_imgs(k, None) for k in counts); maxc = max(int(c.max()) for c in counts.values())

    def best_cap(models, budget_s):
        if total(models, None) <= budget_s: return None
        lo, hi = 1, maxc
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if total(models, mid) <= budget_s: lo = mid
            else: hi = mid - 1
        return lo

    light = [m for m in LIGHT if m in a.models] or a.models[:1]
    no_vgg = [m for m in a.models if m != "vgg16"] or a.models
    plans = [("TEST préliminaire", light, a.test_min), ("COMPLET – tous les modèles", a.models, a.budget_min)]
    if no_vgg != a.models: plans.append(("COMPLET – sans VGG-16", no_vgg, a.budget_min))
    if light != no_vgg and light != a.models: plans.append(("COMPLET – modèles légers", light, a.budget_min))
    print(f"\n3) Estimation pour {full_n} images au total (toutes bases, couverture 100 %) :")
    for name, models, bud in plans[1:]:
        print(f"   {name:30s} extraction {fmt(t_extract(models, None)):>8} + évaluation {fmt(t_eval(models, None)):>7}"
              f" = {fmt(total(models, None))}")
    print(f"\n{'='*86}\nPLANS (budget : test {a.test_min:.0f} min ; complet {a.budget_min:.0f} min)\n{'='*86}")
    best = None
    for name, models, bud in plans:
        cap = best_cap(models, bud * 60); n = sum(n_imgs(k, cap) for k in counts); cov = 100 * n / full_n
        print(f"\n▶ {name}\n   modèles   : {', '.join(models)}\n   images    : {n} / {full_n} ({cov:.0f} %)"
              + (f"  [plafond {cap} images/classe]" if cap else "  [toutes les images]")
              + f"\n   durée est.: extraction {fmt(t_extract(models, cap))} + évaluation {fmt(t_eval(models, cap))}")
        cmd = (f"python etape1_preparation.py" + (f" --cap-per-class {cap}" if cap else "") +
               f"\n      python etape2_extraction.py --models {' '.join(models)}\n      python etape4_evaluation.py\n      python etape6_site_html.py")
        print(f"   commandes : {cmd}")
        if name.startswith("COMPLET") and (best is None or (cov, len(models)) > (best[0], best[1])):
            best = (cov, len(models), name)
    if best:
        print(f"\n★ Recommandé pour votre budget : « {best[2]} » ({best[0]:.0f} % des images).")
    if "vgg16" in a.models:
        print(f"  VGG-16 seul sur toutes les images : ≈ {fmt(sum(n_imgs(k, None) * ms['vgg16'] / 1000 for k in counts))} "
              "(à lancer une nuit : python etape2_extraction.py --models vgg16).")
    print("\nAstuce : ajoutez --time-limit-min 45 à l'étape 2 pour travailler par sessions ; relancer la même commande reprend là où ça s'est arrêté.")
    print("Les durées sont des estimations (±30 %) : elles servent à choisir, pas à promettre.")


if __name__ == "__main__":
    main()
