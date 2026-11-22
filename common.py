"""Fonctions partagées : scan des bases, prétraitement, modèles, distances."""
import json, re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config as C


# ------------------------------------------------------------------ Données
LAST_TOTAL = {}   # clé -> nb d'images trouvées sur le disque (avant échantillonnage)


def find_root(key: str) -> Path:
    cfg = C.DATASETS[key]
    for r in cfg["roots"]:
        p = C.DATA_DIR / r
        if p.exists():
            if cfg.get("find_dir"):               # ex. CIFAR100/**/TEST (casse ignorée)
                for d in sorted(p.rglob("*")):
                    if d.is_dir() and d.name.lower() == cfg["find_dir"]:
                        return d
                continue
            return p
    raise FileNotFoundError(
        f"[{key}] dossier introuvable. Cherché dans {C.DATA_DIR} : {cfg['roots']}")


def scan_dataset(key: str, preset: str = "full", cap: int = None) -> pd.DataFrame:
    """Parcourt la base et retourne un DataFrame (path, class_name, label, ...)."""
    cfg, root = C.DATASETS[key], find_root(key)
    rx = re.compile(cfg["class_regex"]) if "class_regex" in cfg else None
    dirs = [root / s for s in cfg["subdirs"] if (root / s).exists()] if cfg.get("subdirs") else [root]
    if not dirs:
        raise FileNotFoundError(f"[{key}] sous-dossiers {cfg['subdirs']} introuvables dans {root}")
    rows = []
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in cfg["exts"]:
                continue
            cls = p.parent.name
            if rx:
                m = rx.search(p.name.lower())
                if m:
                    cls = m.group(1)
                elif cfg.get("strict_regex"):
                    continue
            rows.append((str(p), cls))
    df = pd.DataFrame(rows, columns=["path", "class_name"]).drop_duplicates("path")
    if df.empty:
        raise RuntimeError(f"[{key}] aucune image trouvée dans {root}")
    LAST_TOTAL[key] = len(df)
    cap = cap if cap else C.PRESETS[preset].get(key)   # --cap-per-class prioritaire
    if cap:
        df = (df.sample(frac=1, random_state=C.SEED)      # mélange reproductible
                .groupby("class_name").head(cap)
                .sort_values("path"))
    df = df.reset_index(drop=True)
    names = sorted(df["class_name"].unique(), key=lambda s: (len(s), s) if s.isdigit() else (0, s))
    df["label"] = df["class_name"].map({n: i for i, n in enumerate(names)})
    return df


def load_manifest(key: str) -> pd.DataFrame:
    return pd.read_csv(C.MANIFESTS / f"{key}.csv", dtype={"class_name": str})   # classes numériques (Corel-5K/10K) = texte


def class_names(df: pd.DataFrame) -> list:
    return df.drop_duplicates("label").sort_values("label")["class_name"].tolist()


# ------------------------------------------------------------- Prétraitement
def load_rgb(path: str, size: int = C.IMG_SIZE) -> Image.Image:
    img = Image.open(path)
    if img.format == "JPEG":              # décodage JPEG réduit : gros gain de temps (Paris)
        img.draft("RGB", (size, size))
    return img.convert("RGB")


def get_transform():
    from torchvision import transforms as T
    return T.Compose([T.Resize((C.IMG_SIZE, C.IMG_SIZE)), T.ToTensor(),
                      T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)])


# ------------------------------------------------------------------- Modèles
def build_model(name: str, pretrained: bool = True):
    """Retourne un nn.Module qui sort un vecteur (B, d) -- couche GAP / fc7."""
    import torch.nn as nn
    import torchvision.models as M
    if name == "squeezenet1_1":
        m = M.squeezenet1_1(weights=(M.SqueezeNet1_1_Weights.DEFAULT if pretrained else None))
        net = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())      # 512
    elif name == "mobilenet_v3_small":
        m = M.mobilenet_v3_small(weights=(M.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None))
        net = nn.Sequential(m.features, m.avgpool, nn.Flatten())                    # 576
    elif name == "resnet18":
        m = M.resnet18(weights=(M.ResNet18_Weights.DEFAULT if pretrained else None)); m.fc = nn.Identity(); net = m   # 512
    elif name == "resnet50":
        m = M.resnet50(weights=(M.ResNet50_Weights.DEFAULT if pretrained else None)); m.fc = nn.Identity(); net = m   # 2048
    elif name == "vgg16":
        m = M.vgg16(weights=(M.VGG16_Weights.DEFAULT if pretrained else None))
        net = nn.Sequential(m.features, m.avgpool, nn.Flatten(), *list(m.classifier)[:5])   # 4096 (fc7)
    else:
        raise ValueError(name)
    return net.eval()


# ------------------------------------------------------- Caractéristiques
def feat_path(ds, model): return C.FEATURES / f"{ds}__{model}.npy"
def meta_path(ds, model): return C.FEATURES / f"{ds}__{model}.json"


def load_features(ds: str, model: str) -> np.ndarray:
    return np.load(feat_path(ds, model))


def load_meta(ds: str, model: str) -> dict:
    return json.loads(meta_path(ds, model).read_text(encoding="utf-8"))


def available_pairs():
    out = []
    for p in sorted(C.FEATURES.glob("*__*.npy")):
        ds, m = p.stem.split("__", 1)
        out.append((ds, m))
    return out


# ------------------------------------------------------------------ Distances
def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def pairwise(Q: np.ndarray, DB: np.ndarray, metric: str) -> np.ndarray:
    """Matrice (nQ, nDB) de distances. cosinus = 1 - cos ; euclidienne = L2."""
    Q, DB = Q.astype(np.float32), DB.astype(np.float32)
    if metric == "cosinus":
        return 1.0 - l2norm(Q) @ l2norm(DB).T
    d2 = (Q ** 2).sum(1)[:, None] + (DB ** 2).sum(1)[None, :] - 2.0 * Q @ DB.T
    return np.sqrt(np.maximum(d2, 0.0))


def top_k(F: np.ndarray, qvec: np.ndarray, metric: str, k: int, exclude: int = None):
    """Étape 3 : indices + distances des k images les plus proches du vecteur requête."""
    d = pairwise(qvec[None, :], F, metric)[0]
    if exclude is not None:
        d[exclude] = np.inf
    idx = np.argpartition(d, min(k, len(d) - 1))[:k]
    idx = idx[np.argsort(d[idx])]
    return idx, d[idx]
