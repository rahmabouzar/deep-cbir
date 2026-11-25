"""ÉTAPE 1 -- Préparation des données et prétraitement.

- Parcourt chaque base, construit le manifeste (chemin, classe, label, format, résolution, taille)
- Vérifie les images illisibles, calcule les statistiques (type de fichier, résolution, classes)
- Illustre le prétraitement : redimensionnement 224x224 + normalisation ImageNet

Usage : python etape1_preparation.py                      (toutes les bases, toutes les images)
        python etape1_preparation.py --preset quick       (test rapide, sous-ensemble)
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config as C
from common import scan_dataset, load_rgb, LAST_TOTAL


def image_info(path):
    try:
        with Image.open(path) as im:
            w, h = im.size
            return im.format, im.mode, w, h, round(Path(path).stat().st_size / 1024, 2), True
    except Exception:
        return None, None, 0, 0, 0.0, False


def figure_echantillon(key, df):
    """1 image par classe (max 20) : aperçu de la base."""
    first = df.drop_duplicates("label").head(20)
    n = len(first); cols = min(5, n); rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows))
    for ax in np.atleast_1d(axs).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axs).ravel(), first.iterrows()):
        ax.imshow(load_rgb(r.path)); ax.set_title(r.class_name, fontsize=9)
    fig.suptitle(f"Échantillon – {C.DATASETS[key]['label']}")
    fig.tight_layout(); fig.savefig(C.FIGS / f"echantillon_{key}.png", dpi=110); plt.close(fig)


def figure_pretraitement(key, df):
    """Avant / après : image originale, 224x224, tenseur normalisé (ImageNet)."""
    r = df.iloc[len(df) // 2]
    with Image.open(r.path) as im:
        orig = im.convert("RGB")
    res = orig.resize((C.IMG_SIZE, C.IMG_SIZE))
    arr = (np.asarray(res, dtype=np.float32) / 255 - np.array(C.IMAGENET_MEAN)) / np.array(C.IMAGENET_STD)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
    axs[0].imshow(orig); axs[0].set_title(f"Originale {orig.size[0]}x{orig.size[1]}")
    axs[1].imshow(res); axs[1].set_title("Redimensionnée 224x224")
    axs[2].imshow(np.clip((arr - arr.min()) / (arr.max() - arr.min()), 0, 1))
    axs[2].set_title("Normalisée ImageNet\n(ré-échelonnée pour affichage)")
    for a in axs: a.axis("off")
    fig.suptitle(f"Prétraitement – {C.DATASETS[key]['label']} (classe « {r.class_name} »)")
    fig.tight_layout(); fig.savefig(C.FIGS / f"pretraitement_{key}.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=C.DEFAULT_DATASETS, choices=list(C.DATASETS))
    ap.add_argument("--preset", default="full", choices=list(C.PRESETS),
                    help="full = TOUTES les images (défaut) ; quick = sous-ensemble pour tester")
    ap.add_argument("--cap-per-class", type=int, default=None,
                    help="plafond d'images par classe (ex. 120) : prioritaire sur --preset (voir etape0)")
    a = ap.parse_args()
    label = f"cap{a.cap_per_class}" if a.cap_per_class else a.preset
    stats = {}
    for key in a.datasets:
        print(f"\n=== {C.DATASETS[key]['label']} ===")
        df = scan_dataset(key, a.preset, a.cap_per_class)
        info = [image_info(p) for p in tqdm(df.path, desc="lecture des en-têtes")]
        df[["format", "mode", "width", "height", "size_kb", "ok"]] = pd.DataFrame(info, index=df.index)
        bad = int((~df.ok).sum())
        if bad:
            print(f"  [!] {bad} image(s) illisible(s) ignorée(s)")
        df = df[df.ok].drop(columns="ok").reset_index(drop=True)
        # ré-indexation des labels après suppression éventuelle
        names = sorted(df.class_name.unique(), key=lambda s: (len(s), s) if s.isdigit() else (0, s))
        df["label"] = df.class_name.map({n: i for i, n in enumerate(names)})
        df["ext"] = df.path.str.rsplit(".", n=1).str[-1].str.lower()
        df.to_csv(C.MANIFESTS / f"{key}.csv", index=False)
        stats[key] = dict(
            nom=C.DATASETS[key]["label"], n_images=len(df), n_classes=len(names),
            extensions=df.ext.value_counts().to_dict(), formats=df.format.value_counts().to_dict(),
            largeur_moy=round(df.width.mean(), 1), hauteur_moy=round(df.height.mean(), 1),
            largeur_min=int(df.width.min()), largeur_max=int(df.width.max()),
            taille_moy_ko=round(df.size_kb.mean(), 2), taille_totale_mo=round(df.size_kb.sum() / 1024, 1),
            images_par_classe_min=int(df.groupby("label").size().min()),
            images_par_classe_max=int(df.groupby("label").size().max()),
            preset=label, illisibles=bad, n_disponibles=LAST_TOTAL[key],
            couverture_pct=round(100 * len(df) / LAST_TOTAL[key], 1))
        figure_echantillon(key, df); figure_pretraitement(key, df)
        print(json.dumps(stats[key], ensure_ascii=False, indent=2))
    print("\n=== COUVERTURE : images utilisées / images trouvées sur le disque ===")
    for k, s in stats.items():
        print(f"  {s['nom']:20s} {s['n_images']:>6} / {s['n_disponibles']:<6} = {s['couverture_pct']:>5.1f} %")
    if any(s["couverture_pct"] < 100 for s in stats.values()):
        print("  -> couverture < 100 % : relancer avec --preset full pour tout utiliser.")
    path = C.RESULTS / "etape1_stats.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(stats)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Étape 1 terminée -> {C.MANIFESTS}")


if __name__ == "__main__":
    main()
