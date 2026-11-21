"""ÉTAPE 2 -- Extraction des vecteurs caractéristiques (indexation hors-ligne) : RAPIDE et REPRENABLE.

- Chaque image est décodée UNE seule fois et passée dans tous les modèles du lot.
- Points de sauvegarde tous les ~1000 images : on peut arrêter (Ctrl+C, veille, coupure) et relancer,
  le calcul reprend exactement là où il s'est arrêté.
- --time-limit-min N : s'arrête proprement après N minutes (puis relancer la même commande plus tard).
- channels_last + tous les cœurs CPU ; GPU automatique si disponible.

Usage : python etape2_extraction.py
        python etape2_extraction.py --models squeezenet1_1 mobilenet_v3_small resnet18 --time-limit-min 40
"""
import argparse, hashlib, json, math, os, shutil, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

import config as C
from common import load_manifest, load_rgb, get_transform, build_model, feat_path, meta_path, load_meta


class ImgDataset(Dataset):
    def __init__(self, paths):
        self.paths, self.tf = list(paths), get_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return self.tf(load_rgb(self.paths[i])), i
        except Exception as e:                       # image corrompue -> vecteur nul
            print(f"\n⚠ image illisible {self.paths[i]} : {e}")
            return torch.zeros(3, C.IMG_SIZE, C.IMG_SIZE), i


def manifest_hash(df):
    return hashlib.md5("\n".join(df.path).encode()).hexdigest()[:10]


def extract_dataset(ds, models, batch, workers, device, force, t_end):
    """Retourne False si la limite de temps a interrompu le travail."""
    df = load_manifest(ds); n = len(df); h = manifest_hash(df)
    todo = [m for m in models if force or not (feat_path(ds, m).exists() and meta_path(ds, m).exists()
                                               and load_meta(ds, m).get("manifest_md5") == h)]
    if not todo:
        print(f"[{ds}] déjà extrait pour : {', '.join(models)}"); return True
    shard = batch * max(1, round(int(os.environ.get("CBIR_SHARD", 1000)) / batch)); ns = math.ceil(n / shard)
    sdir = C.FEATURES / "_shards" / f"{ds}_{h}"; sdir.mkdir(parents=True, exist_ok=True)
    sf = lambda m, s: sdir / f"{m}_{s:04d}.npz"
    if force:
        for m in todo:
            for s in range(ns): sf(m, s).unlink(missing_ok=True)
    missing = {s: [m for m in todo if not sf(m, s).exists()] for s in range(ns)}
    work = [s for s in range(ns) if missing[s]]
    print(f"\n=== {ds} : {n} images | modèles : {', '.join(todo)} | {device} | "
          f"{ns - len(work)}/{ns} blocs déjà faits ===")
    if work:
        need = sorted({m for s in work for m in missing[s]}, key=list(C.MODELS).index)
        nets = {m: build_model(m).to(device).to(memory_format=torch.channels_last) for m in need}
        idx = [i for s in work for i in range(s * shard, min((s + 1) * shard, n))]
        loader = DataLoader(Subset(ImgDataset(df.path), idx), batch_size=batch, shuffle=False,
                            num_workers=workers, pin_memory=(device == "cuda"))
        buf, tm, pos = {}, {}, 0
        with torch.inference_mode(), tqdm(total=len(idx), desc=ds, unit="img") as bar:
            for x, _ in loader:
                s = idx[pos] // shard; end = pos + len(x)
                x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                for m in missing[s]:
                    t0 = time.perf_counter(); f = nets[m](x)
                    if device == "cuda": torch.cuda.synchronize()
                    tm[(s, m)] = tm.get((s, m), 0.0) + time.perf_counter() - t0
                    buf.setdefault((s, m), []).append(f.float().cpu().numpy())
                pos = end; bar.update(len(x))
                if end == len(idx) or idx[end] // shard != s:        # fin d'un bloc -> sauvegarde
                    for m in missing[s]:
                        np.savez(sf(m, s), F=np.concatenate(buf.pop((s, m))), t=tm.pop((s, m)))
                    if t_end and time.time() > t_end and end < len(idx):
                        print("\n⏸ limite de temps atteinte : progression sauvegardée. "
                              "Relancer la MÊME commande pour continuer."); return False
    for m in todo:                                                    # assemblage final
        parts = [np.load(sf(m, s)) for s in range(ns)]
        F = np.concatenate([p["F"] for p in parts]).astype(np.float32); t = float(sum(p["t"] for p in parts))
        assert len(F) == n, f"{m}: {len(F)} vecteurs pour {n} images"
        np.save(feat_path(ds, m), F)
        n_par = sum(p.numel() for p in build_model(m, pretrained=False).parameters()) / 1e6
        meta = dict(dataset=ds, model=m, n_images=int(n), dim=int(F.shape[1]), params_millions=round(n_par, 2),
                    device=device, temps_inference_s=round(t, 2), ms_par_image=round(1000 * t / n, 3), manifest_md5=h)
        meta_path(ds, m).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  ✔ {m}: {F.shape}  ({meta['ms_par_image']} ms/image)")
    shutil.rmtree(sdir, ignore_errors=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=C.DEFAULT_DATASETS, choices=list(C.DATASETS))
    ap.add_argument("--models", nargs="+", default=list(C.MODELS), choices=list(C.MODELS))
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2 if (os.cpu_count() or 1) > 4 else 1,
                    help="processus de décodage (le reste des cœurs sert au calcul)")
    ap.add_argument("--time-limit-min", type=float, default=0, help="arrêt propre après N minutes (0 = illimité)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(C.SEED); torch.set_flush_denormal(True)
    t_end = time.time() + 60 * a.time_limit_min if a.time_limit_min else None
    for ds in a.datasets:
        if not extract_dataset(ds, a.models, a.batch, a.workers, device, a.force, t_end):
            return
    print("\n✔ Étape 2 terminée ->", C.FEATURES)


if __name__ == "__main__":
    main()
