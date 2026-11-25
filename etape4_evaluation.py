"""ÉTAPE 4 -- Évaluation : courbe Précision-Rappel, AP/mAP, P@K, R@K (rapide et INCRÉMENTALE).

Protocole : chaque image de la base est une requête (leave-one-out : elle-même est exclue
du classement) ; pertinent = même classe. Pour Paris6k, la classe « general » (distracteurs)
reste dans la base mais n'est pas utilisée comme requête.
Incrémental : les couples (base, modèle) déjà évalués et inchangés sont ignorés
(--force pour tout recalculer). On peut donc évaluer au fur et à mesure de l'extraction.

Usage : python etape4_evaluation.py   [--datasets corel1k ...] [--models resnet50 ...] [--force]
"""
import argparse, json, time
import numpy as np
import pandas as pd

import config as C
from common import available_pairs, load_manifest, load_features, load_meta, class_names, l2norm

LEVELS = np.array(C.RECALL_LEVELS)


def evaluate(F, y, metric, query_ok, n_cls, chunk=None, topn=10):
    N = len(y)
    chunk = chunk or int(max(50, min(500, 4_000_000 // N)))     # borne la mémoire (N grand)
    F = F.astype(np.float32); y32 = y.astype(np.int32)
    nrel_all = np.bincount(y, minlength=n_cls)[y] - 1
    q = np.where(query_ok & (nrel_all > 0))[0]; nq = len(q)
    ap = np.zeros(nq); pk = {k: np.zeros(nq) for k in C.K_LIST}; rk = {k: np.zeros(nq) for k in C.K_LIST}
    pr = np.zeros((nq, len(LEVELS))); conf = np.zeros(n_cls * n_cls)
    if metric == "cosinus":
        Fn = l2norm(F)                                          # normalisé UNE seule fois
    else:
        sq = (F ** 2).sum(1)
    maxk = min(max(C.K_LIST), N); t0 = time.perf_counter()
    for s in range(0, nq, chunk):
        qi = q[s:s + chunk]; b = len(qi); sl = slice(s, s + b)
        if metric == "cosinus":
            D = 1.0 - Fn[qi] @ Fn.T
        else:
            D = np.sqrt(np.maximum(sq[qi][:, None] + sq[None, :] - 2.0 * (F[qi] @ F.T), 0.0))
        D[np.arange(b), qi] = np.inf                            # la requête s'exclut elle-même
        order = np.argsort(D, axis=1)                           # soi-même = dernière position
        yq = y32[qi]
        rel = y32[order] == yq[:, None]; rel[:, -1] = False
        csk = np.cumsum(rel[:, :maxk], axis=1)
        nrel = nrel_all[qi]
        for k in C.K_LIST:
            kk = min(k, N - 1)
            pk[k][sl] = csk[:, kk - 1] / k; rk[k][sl] = csk[:, kk - 1] / nrel
        conf += np.bincount(np.repeat(yq, topn) * n_cls + y32[order[:, :topn]].ravel(), minlength=n_cls * n_cls)
        rows, cols = np.nonzero(rel)                            # rangs (0-based) des images pertinentes
        off = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=b))))
        for i in range(b):                                      # AP et PR sur les seules positions pertinentes
            c = cols[off[i]:off[i + 1]]; n = len(c)
            ph = np.arange(1, n + 1) / (c + 1.0)                # précision à chaque image pertinente
            ap[s + i] = ph.mean()
            pm = np.maximum.accumulate(ph[::-1])[::-1]          # précision interpolée
            pr[s + i] = pm[np.maximum(np.ceil(LEVELS * n - 1e-9).astype(int) - 1, 0)]
    ms_q = 1000 * (time.perf_counter() - t0) / max(nq, 1)
    conf = conf.reshape(n_cls, n_cls); conf = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    return dict(q=q, ap=ap, pk=pk, rk=rk, pr=pr, conf=conf, ms_q=ms_q)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--datasets", nargs="+", choices=list(C.DATASETS))
    ap_.add_argument("--models", nargs="+", choices=list(C.MODELS))
    ap_.add_argument("--force", action="store_true")
    a = ap_.parse_args()
    gp, cp, jp = C.RESULTS / "metrics_global.csv", C.RESULTS / "metrics_par_classe.csv", C.RESULTS / "pr_curves.json"
    G = pd.read_csv(gp) if gp.exists() else pd.DataFrame()
    P = pd.read_csv(cp) if cp.exists() else pd.DataFrame()
    prc = json.loads(jp.read_text(encoding="utf-8"))["courbes"] if jp.exists() else {}
    n_new = 0
    for ds, model in available_pairs():
        if (a.datasets and ds not in a.datasets) or (a.models and model not in a.models):
            continue
        df = load_manifest(ds); names = class_names(df); y = df.label.values
        F = load_features(ds, model); meta = load_meta(ds, model)
        if len(F) != len(df):
            print(f"[!] {ds}/{model} : {len(F)} vecteurs ≠ {len(df)} images du manifeste -> relancer l'étape 2"); continue
        h = meta.get("manifest_md5", "")
        if (not a.force and len(G) and ((G.dataset == ds) & (G.model == model) & (G.get("manifest_md5") == h)).any()):
            print(f"= {ds:9s} {model:19s} déjà évalué"); continue
        excl = C.DATASETS[ds].get("query_exclude", []); query_ok = ~df.class_name.isin(excl).values
        if len(G):
            G = G[~((G.dataset == ds) & (G.model == model))]
        if len(P):
            P = P[~((P.dataset == ds) & (P.model == model))]
        rows, crow = [], []
        for metric in C.METRICS:
            r = evaluate(F, y, metric, query_ok, len(names))
            row = dict(dataset=ds, dataset_nom=C.DATASETS[ds]["label"], model=model, modele=C.MODELS[model][0],
                       famille=C.MODELS[model][1], metrique=metric, n_images=len(df), n_classes=len(names),
                       n_requetes=len(r["q"]), dim=meta["dim"], params_M=meta["params_millions"],
                       ms_extraction_img=meta["ms_par_image"], ms_recherche_req=round(r["ms_q"], 3), mAP=r["ap"].mean(),
                       manifest_md5=h)
            for k in C.K_LIST:
                row[f"P@{k}"] = r["pk"][k].mean(); row[f"R@{k}"] = r["rk"][k].mean()
            rows.append(row); yq = y[r["q"]]
            for c, name in enumerate(names):
                m = yq == c
                if m.any():
                    crow.append(dict(dataset=ds, model=model, metrique=metric, classe=name, AP=r["ap"][m].mean(),
                                     **{f"P@{k}": r["pk"][k][m].mean() for k in (1, 10)}))
            prc[f"{ds}|{model}|{metric}"] = r["pr"].mean(0).tolist()
            np.save(C.CONF_DIR / f"{ds}__{model}__{metric}.npy", r["conf"].astype(np.float32))
            print(f"{ds:9s} {model:19s} {metric:11s} mAP={row['mAP']:.4f}  P@10={row['P@10']:.4f}  ({r['ms_q']:.2f} ms/requête)")
        G = pd.concat([G, pd.DataFrame(rows)], ignore_index=True); P = pd.concat([P, pd.DataFrame(crow)], ignore_index=True)
        n_new += 1
        G.to_csv(gp, index=False); P.to_csv(cp, index=False)       # sauvegarde après chaque couple
        jp.write_text(json.dumps(dict(niveaux_rappel=C.RECALL_LEVELS, courbes=prc)), encoding="utf-8")
    if G.empty:
        raise SystemExit("Aucune caractéristique trouvée : lancer l'étape 2 d'abord.")
    print(f"\n[OK] Étape 4 terminée ({n_new} couple(s) évalué(s)) ->", C.RESULTS)


if __name__ == "__main__":
    main()
