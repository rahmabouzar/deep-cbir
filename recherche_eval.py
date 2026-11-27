"""Évaluation d'UNE requête : précision/rappel, AP, consensus entre modèles (fusion RRF), robustesse aux perturbations.
Fonctions pures (numpy / PIL), sans Streamlit, donc testables."""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from common import top_k


def classement(F, qvec, metric, exclude=None):
    """Classement COMPLET de la base (du plus proche au plus éloigné), la requête elle-même exclue."""
    n = len(F) - (0 if exclude is None else 1)
    idx, _ = top_k(F, qvec, metric, n, exclude=exclude)
    return idx[:n]


def metriques(rel, k):
    """rel : booléens de pertinence dans l'ordre du classement complet. Retourne les mesures de la requête."""
    rel = np.asarray(rel, dtype=bool)
    nrel = int(rel.sum())
    if nrel == 0:
        return dict(P=np.nan, R=np.nan, AP=np.nan, RPrec=np.nan, premier=np.nan, nrel=0, hits=0)
    cum = np.cumsum(rel)
    prec = cum / (np.arange(len(rel)) + 1)
    return dict(P=float(rel[:k].mean()), R=float(cum[k - 1] / nrel), AP=float((prec * rel).sum() / nrel),
                RPrec=float(prec[nrel - 1]), premier=int(np.argmax(rel)) + 1, nrel=nrel, hits=int(rel[:k].sum()))


def courbes(rel, rang_max=50):
    """Précision@rang (1..rang_max) et courbe Précision-Rappel complète."""
    rel = np.asarray(rel, dtype=bool); cum = np.cumsum(rel); nrel = max(int(rel.sum()), 1)
    prec = cum / (np.arange(len(rel)) + 1)
    pts = np.where(rel)[0]
    return prec[:rang_max], (cum[pts] / nrel if len(pts) else np.array([])), (prec[pts] if len(pts) else np.array([]))


def vote(labels_topk, noms):
    """Classe prédite par vote des K voisins + part des voix (confiance)."""
    v, c = np.unique(np.asarray(labels_topk), return_counts=True)
    j = int(np.argmax(c))
    return noms[int(v[j])], float(c[j] / len(labels_topk))


def fusion_rrf(classements, c=60):
    """Reciprocal Rank Fusion : fusionne plusieurs classements (indices d'images) en un seul."""
    n = max(len(r) for r in classements)
    score = {}
    for r in classements:
        for pos, i in enumerate(r):
            score[int(i)] = score.get(int(i), 0.0) + 1.0 / (c + pos + 1)
    return np.array(sorted(score, key=score.get, reverse=True)[:n])


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / max(len(a | b), 1)


def perturbations(img: Image.Image) -> dict:
    """Perturbations usuelles pour tester la robustesse d'un descripteur."""
    w, h = img.size
    out = {"Miroir horizontal": ImageOps.mirror(img),
           "Rotation 20°": img.rotate(20, expand=True, fillcolor=(0, 0, 0)),
           "Recadrage central 60 %": img.crop((int(.2 * w), int(.2 * h), int(.8 * w), int(.8 * h))),
           "Flou gaussien": img.filter(ImageFilter.GaussianBlur(max(w, h) / 40)),
           "Luminosité +50 %": ImageEnhance.Brightness(img).enhance(1.5),
           "Contraste −50 %": ImageEnhance.Contrast(img).enhance(.5),
           "Niveaux de gris": ImageOps.grayscale(img).convert("RGB")}
    arr = np.asarray(img, dtype=np.float32)
    rng = np.random.default_rng(0)
    out["Bruit gaussien"] = Image.fromarray(np.clip(arr + rng.normal(0, 25, arr.shape), 0, 255).astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=12); buf.seek(0)
    out["JPEG qualité 12"] = Image.open(buf).convert("RGB")
    return out
