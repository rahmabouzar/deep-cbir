"""ÉTAPE 5 -- Rapport complet en français (Markdown + HTML autonome) généré à partir des résultats.

Usage : python etape5_rapport.py   ->  results/rapport.md  et  results/rapport.html
"""
import base64, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import markdown

import config as C
from common import load_manifest, class_names

def _exiger(*noms):
    manq = [n for n in noms if not (C.RESULTS / n).exists()]
    if manq:
        raise SystemExit("\n[ERREUR] Fichier(s) manquant(s) dans results\\ : " + ", ".join(manq) +
                         "\n   -> lancer d'abord, dans l'ordre :  python etape1_preparation.py  |  python etape2_extraction.py  |  python etape4_evaluation.py\n")


_exiger("metrics_global.csv", "metrics_par_classe.csv", "pr_curves.json", "etape1_stats.json")
G = pd.read_csv(C.RESULTS / "metrics_global.csv")
PC = pd.read_csv(C.RESULTS / "metrics_par_classe.csv", dtype={"classe": str})
PR = json.loads((C.RESULTS / "pr_curves.json").read_text(encoding="utf-8"))
ST = json.loads((C.RESULTS / "etape1_stats.json").read_text(encoding="utf-8"))
DSS = list(G.dataset.unique())
MODS_ALL = [m for m in C.MODELS if m in set(G.model)]
# seuls les modèles évalués sur TOUTES les bases sont comparés dans les graphiques/tableaux croisés
MODS = [m for m in MODS_ALL if all(((G.dataset == d) & (G.model == m)).any() for d in DSS)]
PARTIELS = [m for m in MODS_ALL if m not in MODS]
nom_ds = lambda d: C.DATASETS[d]["label"]
nom_m = lambda m: C.MODELS[m][0]
fmt = lambda x: f"{x:.4f}"


def md_table(df, floatfmt=fmt):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt(v) if isinstance(v, (float, np.floating)) else str(v)
                                      for v in r) + " |")
    return "\n".join(out)


def save(fig, name):
    fig.tight_layout(); fig.savefig(C.FIGS / name, dpi=120); plt.close(fig)
    return f"figures/{name}"


# ----------------------------------------------------------------- figures
def fig_map_bar():
    fig, axs = plt.subplots(1, len(DSS), figsize=(5 * len(DSS), 3.8), squeeze=False)
    for ax, ds in zip(axs[0], DSS):
        sub = G[G.dataset == ds]; w = 0.38; x = np.arange(len(MODS))
        for i, met in enumerate(C.METRICS):
            v = [sub[(sub.model == m) & (sub.metrique == met)].mAP.values[0] for m in MODS]
            ax.bar(x + (i - .5) * w, v, w, label=met)
        ax.set_xticks(x); ax.set_xticklabels([nom_m(m) for m in MODS], rotation=30, ha="right", fontsize=8)
        ax.set_title(nom_ds(ds)); ax.set_ylabel("mAP"); ax.legend(fontsize=8)
    return save(fig, "rapport_map_barres.png")


def fig_pr(ds):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, met in zip(axs, C.METRICS):
        for m in MODS:
            ax.plot(PR["niveaux_rappel"], PR["courbes"][f"{ds}|{m}|{met}"], marker="o", ms=3, label=nom_m(m))
        ax.set_title(f"{nom_ds(ds)} – {met}"); ax.set_xlabel("Rappel"); ax.grid(alpha=.3)
    axs[0].set_ylabel("Précision (interpolée)"); axs[0].legend(fontsize=8)
    return save(fig, f"rapport_pr_{ds}.png")


def fig_heat(ds, met):
    sub = PC[(PC.dataset == ds) & (PC.metrique == met)].pivot(index="classe", columns="model", values="AP")
    sub = sub[[m for m in MODS if m in sub.columns]]
    sub = sub.sort_values(sub.columns[0])
    if len(sub) > 40:                                    # CIFAR-100 : 20 pires + 20 meilleures classes
        sub = sub.loc[sub.mean(axis=1).sort_values().index]; sub = pd.concat([sub.head(20), sub.tail(20)])
    fig, ax = plt.subplots(figsize=(1.3 * len(sub.columns) + 3, 0.28 * len(sub) + 1.6))
    im = ax.imshow(sub.values, aspect="auto", cmap="viridis", vmin=0)
    ax.set_xticks(range(len(sub.columns))); ax.set_xticklabels([nom_m(m) for m in sub.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.index, fontsize=7)
    for i in range(len(sub)):
        for j in range(len(sub.columns)):
            ax.text(j, i, f"{sub.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color=("w" if sub.values[i, j] < 0.6 * np.nanmax(sub.values) else "k"))
    ax.set_title(f"AP par classe – {nom_ds(ds)} ({met})"); fig.colorbar(im, ax=ax, fraction=.04)
    return save(fig, f"rapport_classes_{ds}.png")


def fig_conf(ds, model, met):
    M = np.load(C.CONF_DIR / f"{ds}__{model}__{met}.npy"); names = class_names(load_manifest(ds))
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); im = ax.imshow(M, cmap="magma")
    if len(names) <= 20:
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Classe retrouvée (Top-10)"); ax.set_ylabel("Classe de la requête")
    ax.set_title(f"Confusion – {nom_ds(ds)} – {nom_m(model)} ({met})"); fig.colorbar(im, ax=ax, fraction=.04)
    return save(fig, f"rapport_confusion_{ds}.png")


def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for ds in DSS:
        s = G[(G.dataset == ds) & (G.metrique == "cosinus")]
        ax.scatter(s.ms_extraction_img, s.mAP, s=40 + s.dim / 30, alpha=.7, label=nom_ds(ds))
        for _, r in s.iterrows():
            ax.annotate(nom_m(r.model), (r.ms_extraction_img, r.mAP), fontsize=6)
    ax.set_xscale("log"); ax.set_xlabel("Temps d'extraction (ms/image, échelle log)"); ax.set_ylabel("mAP (cosinus)")
    ax.set_title("Compromis précision / vitesse (taille du point = dimension)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    return save(fig, "rapport_compromis.png")


# ----------------------------------------------------------------- rapport
def top_pairs(ds, model, met, n=5):
    M = np.load(C.CONF_DIR / f"{ds}__{model}__{met}.npy"); names = class_names(load_manifest(ds))
    M = M.copy(); np.fill_diagonal(M, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-M.ravel())[:n], M.shape))[0]
    return [(names[i], names[j], M[i, j]) for i, j in idx]


def build():
    L = []; A = L.append
    A("# Rapport – CBIR par réseaux profonds pré-entraînés\n")
    A("**Module IRDM – Master 2 SID – USTO-MB** · *Rapport généré automatiquement à partir des résultats mesurés.*\n")
    if PARTIELS:
        A("> [!] Modèle(s) évalué(s) sur une partie seulement des bases, donc exclus des comparaisons : "
          + ", ".join(C.MODELS[m][0] for m in PARTIELS) + ".\n")
    A("## 1. Protocole\n")
    A("- **Prétraitement** : redimensionnement 224×224 et normalisation ImageNet (moyenne/écart-type).")
    A("- **Descripteurs** : sortie de la couche Global Average Pooling (SqueezeNet, MobileNetV3, ResNet) ou de fc7 (VGG-16).")
    A("- **Similarité** : distance cosinus (1 − cos) et distance euclidienne ; chaque image de la base sert de requête (leave-one-out) ; une image est pertinente si elle est de la même classe.")
    A("- **Mesures** : courbe Précision-Rappel (11 niveaux interpolés), AP/mAP, P@K, R@K.\n")
    A("### 1.1 Bases de données et types de fichiers\n")
    rows = [dict(Base=s["nom"], Images=s["n_images"], Classes=s["n_classes"], Extensions=", ".join(s["extensions"]),
                 **{"Résolution moy.": f"{s['largeur_moy']:.0f}×{s['hauteur_moy']:.0f}"},
                 **{"Taille moy. (Ko)": s["taille_moy_ko"]}, Illisibles=s["illisibles"]) for k, s in ST.items() if k in DSS]
    A(md_table(pd.DataFrame(rows)) + "\n")
    for ds in DSS:
        A(f"![Échantillon {nom_ds(ds)}](figures/echantillon_{ds}.png)\n")
    A("### 1.2 Modèles\n")
    mm = G[(G.metrique == "cosinus")].groupby("model").agg(dim=("dim", "first"), params=("params_M", "first"),
                                                          ms=("ms_extraction_img", "mean")).reindex(MODS).reset_index()
    mm["Modèle"] = mm.model.map(nom_m); mm["Famille"] = mm.model.map(lambda m: C.MODELS[m][1])
    A(md_table(mm[["Modèle", "Famille", "dim", "params", "ms"]].rename(
        columns={"dim": "Dimension", "params": "Paramètres (M)", "ms": "ms/image (moy.)"})) + "\n")

    A("## 2. Résultats globaux (comparaison par base, modèle et métrique)\n")
    piv = G.pivot_table(index="modele", columns=["dataset_nom", "metrique"], values="mAP").reindex([nom_m(m) for m in MODS])
    piv.columns = [f"{a} – {b}" for a, b in piv.columns]
    A("**mAP**\n"); A(md_table(piv.reset_index().rename(columns={"modele": "Modèle"})) + "\n")
    A(f"![mAP]({fig_map_bar()})\n")
    for ds in DSS:
        s = G[(G.dataset == ds) & (G.metrique == "cosinus")].sort_values("mAP", ascending=False)
        A(f"**Précision@K / Rappel@K – {nom_ds(ds)} (cosinus)**\n")
        A(md_table(s[["modele"] + [f"P@{k}" for k in C.K_LIST] + [f"R@{k}" for k in (10, 50)]]
                   .rename(columns={"modele": "Modèle"})) + "\n")

    A("## 3. Cosinus vs Euclidienne\n")
    d = G.pivot_table(index=["dataset_nom", "modele"], columns="metrique", values="mAP").reset_index()
    d["Écart (cos − euc)"] = d["cosinus"] - d["euclidienne"]
    A(md_table(d.rename(columns={"dataset_nom": "Base", "modele": "Modèle"})) + "\n")
    wins = int((d["Écart (cos − euc)"] > 0).sum())
    moy = d["Écart (cos − euc)"].mean()
    expl = ("Ce résultat s'explique par la normalisation L2 du cosinus, qui neutralise la norme des vecteurs "
            "(elle varie avec le contenu de l'image et pénalise l'euclidienne brute)." if moy > 0 else
            "Ici l'euclidienne brute fait au moins aussi bien : la norme des descripteurs porte une information utile pour ces bases.")
    A(f"La distance cosinus est meilleure dans **{wins}/{len(d)}** configurations ; écart moyen = **{moy:+.4f}** mAP. {expl}\n")

    A("## 4. Courbes Précision-Rappel\n")
    for ds in DSS:
        A(f"![PR {nom_ds(ds)}]({fig_pr(ds)})\n")

    A("## 5. Comparaison entre classes\n")
    best_cfg = {}
    for ds in DSS:
        s = G[G.dataset == ds].sort_values("mAP", ascending=False).iloc[0]
        best_cfg[ds] = (s.model, s.metrique)
        sub = PC[(PC.dataset == ds) & (PC.model == s.model) & (PC.metrique == s.metrique)].sort_values("AP", ascending=False)
        A(f"### {nom_ds(ds)} – meilleure configuration : {nom_m(s.model)} / {s.metrique} (mAP = {s.mAP:.4f})\n")
        A("**Classes les plus faciles**\n"); A(md_table(sub.head(5)[["classe", "AP", "P@1", "P@10"]]) + "\n")
        A("**Classes les plus difficiles**\n"); A(md_table(sub.tail(5)[["classe", "AP", "P@1", "P@10"]]) + "\n")
        A(f"![AP par classe]({fig_heat(ds, s.metrique)})\n")
        A(f"![Confusion]({fig_conf(ds, s.model, s.metrique)})\n")
        pr = top_pairs(ds, s.model, s.metrique)
        A("**Paires de classes les plus confondues (Top-10)** : " +
          " ; ".join(f"{a} → {b} ({v:.1%})" for a, b, v in pr) + ".\n")

    A("## 6. Efficacité (dimension, temps)\n")
    A(f"![Compromis]({fig_tradeoff()})\n")
    A(md_table(G[G.metrique == "cosinus"][["dataset_nom", "modele", "dim", "ms_extraction_img", "ms_recherche_req", "mAP"]]
               .rename(columns={"dataset_nom": "Base", "modele": "Modèle", "dim": "Dim.",
                                "ms_extraction_img": "Extraction (ms/img)", "ms_recherche_req": "Recherche (ms/requête)"})) + "\n")

    A("## 7. Comparaison entre types de fichiers et bases\n")
    rows = []
    for ds in DSS:
        s_ = G[(G.dataset == ds) & (G.metrique == "cosinus")]
        rows.append({"Base": nom_ds(ds), "Format": ", ".join(ST[ds]["extensions"]),
                     "Résolution": f"{ST[ds]['largeur_moy']:.0f}×{ST[ds]['hauteur_moy']:.0f}",
                     "mAP moyen": s_.mAP.mean(), "mAP max": s_.mAP.max(), "P@10 moyen": s_["P@10"].mean()})
    t = pd.DataFrame(rows)
    A(md_table(t) + "\n")
    easiest = t.sort_values("mAP moyen", ascending=False).iloc[0]; hardest = t.sort_values("mAP moyen").iloc[0]
    A(f"La base la plus facile est **{easiest.Base}** (mAP moyen {easiest['mAP moyen']:.3f}) et la plus difficile "
      f"**{hardest.Base}** ({hardest['mAP moyen']:.3f}). Les images PNG 32×32 de CIFAR-100 sont agrandies à 224×224 : "
      "les réseaux voient des images floues, hors de la distribution ImageNet, et les 100 classes sont fines. "
      "Les JPEG Corel (≈ 384×256) ont des classes visuellement très séparées ; Paris6k contient de grandes photos JPEG "
      "de monuments, avec forte variation de point de vue.\n")

    A("## 8. Conclusion\n")
    for ds in DSS:
        s = G[G.dataset == ds].sort_values("mAP", ascending=False)
        l = s[s.metrique == "cosinus"].sort_values("mAP")
        A(f"- **{nom_ds(ds)}** : meilleur = **{nom_m(s.iloc[0].model)}** ({s.iloc[0].metrique}, mAP {s.iloc[0].mAP:.4f}) ; "
          f"moins bon = {nom_m(l.iloc[0].model)} (mAP {l.iloc[0].mAP:.4f}).")
    A("\n*Ces conclusions sont calculées sur les résultats réels des étapes 2 et 4 ; consulter les tableaux ci-dessus pour les valeurs exactes. "
      "Mode d'exécution utilisé : voir `results/etape1_stats.json` (champ `preset`).*")
    return "\n".join(L)


def main():
    md = build()
    (C.RESULTS / "rapport.md").write_text(md, encoding="utf-8")
    html = markdown.markdown(md, extensions=["tables"])
    for p in set(__import__("re").findall(r'src="(figures/[^"]+)"', html)):
        b64 = base64.b64encode((C.RESULTS / p).read_bytes()).decode()
        html = html.replace(f'src="{p}"', f'src="data:image/png;base64,{b64}"')
    css = ("body{font-family:Segoe UI,Arial;max-width:1000px;margin:2rem auto;padding:0 1rem;line-height:1.5}"
           "table{border-collapse:collapse;margin:1rem 0;font-size:.85rem}td,th{border:1px solid #ccc;padding:4px 8px}"
           "th{background:#f0f0f0}img{max-width:100%}")
    (C.RESULTS / "rapport.html").write_text(
        f"<html><head><meta charset='utf-8'><title>Rapport CBIR</title><style>{css}</style></head><body>{html}</body></html>",
        encoding="utf-8")
    print("[OK] Rapport :", C.RESULTS / "rapport.html")


if __name__ == "__main__":
    main()
